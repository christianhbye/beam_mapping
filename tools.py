import astropy
import croissant as cro
import croissant.jax as crojax
from functools import partial
import jax
import jax.numpy as jnp
import lunarsky
import s2fft
import yaml


class Simulator:

    def __init__(self, fname="sim_params.yaml"):
        with open(fname) as f:
            config = yaml.safe_load(f)
        self.lmax = config["lmax"]
        self.world = config["world"]
        self.ntimes = config["ntimes"]
        self.nside = config["nside"]
        fmin = config["fmin"]
        fmax = config["fmax"]
        nchan = config["nchan"]
        self.freq = jnp.linspace(fmin, fmax, nchan)
        if self.world == "earth":
            time_func = astropy.time.Time
            loc_func = astropy.coordinates.EarthLocation
            topo_class = astropy.coordinates.AltAz
            self.eq_frame = "fk5"
        elif self.world == "moon":
            time_func = lunarsky.Time
            loc_func = lunarsky.MoonLocation
            topo_class = lunarsky.LunarTopo
            self.eq_frame = "mcmf"
        else:
            raise ValueError("world must be 'earth' or 'moon'")

        time = time_func(config["t_start"])
        lon = config["lon"]
        lat = config["lat"]
        loc = loc_func(lon=lon, lat=lat)
        self.topo = topo_class(obstime=time, location=loc)

        self.theta = s2fft.sampling.s2_samples.thetas(
            L=self.lmax+1, sampling="mwss"
        )
        self.phi = s2fft.sampling.s2_samples.phis_equiang(
            L=self.lmax+1, sampling="mwss"
        )

        dt = cro.constants.sidereal_day[self.world] / self.ntimes
        self.phases = crojax.simulator.rot_alm_z(
            self.lmax, self.ntimes, dt, world=self.world
        )

        horizon = jnp.where(self.theta < jnp.pi/2, 1, 0)
        ones = jnp.ones((self.freq.size, self.theta.size, self.phi.size))
        self.horizon = horizon[None, :, None] * ones
       
        # scattering parameters
        self.epsilon = config["epsilon"]
        c = 299792458  # speed of light in m/s
        self.tau = 2 / c * config["height"]

        # blackbody temp
        self.T_bb = config["T_bb"]

        # clebsch-gordan coefficients
        self.clebsch = None  # XXX
        self.get_horizon_coeffs()

       

    def beam(self, return_xy=False):
        phi, theta = jnp.meshgrid(self.phi, self.theta)
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)
        beamX = (y**2 + z**2) ** (3/2)
        beamY = (x**2 + z**2) ** (3/2)

        if return_xy:
            return beamX, beamY

        return 1/2 * (beamX + beamY)

    @property
    def topo2eq(self):
        eul, dl = crojax.rotations.generate_euler_dl(
            self.lmax, self.topo, self.eq_frame
        )

        return partial(
            s2fft.utils.rotation.rotate_flms,
            L=self.lmax+1,
            rotation=eul,
            dl_array=dl,
        )

    @property
    def gal2eq(self):
        eul, dl = crojax.rotations.generate_euler_dl(
            self.lmax, "galactic", self.eq_frame,
        )

        return partial(
            s2fft.utils.rotation.rotate_flms,
            L=self.lmax+1,
            rotation=eul,
            dl_array=dl,
        )

    @property
    def mwss2alm(self):
        return partial(
            s2fft.forward_jax,
            L=self.lmax+1,
            spin=0,
            nside=None,
            sampling="mwss",
            reality=True,
        )

    @property
    def hp2alm(self):
        return partial(
            s2fft.forward_jax,
            L=self.lmax+1,
            spin=0,
            nside=self.nside,
            sampling="healpix",
            reality=True,
        )

    def get_horizon_coeffs(self):
        hlm = jax.vmap(self.mwss2alm)(self.horizon)
        self.hlm = jax.vmap(self.topo2eq)(hlm)
        h_clebsch = jnp.einsum("abcdef, ef -> abcd", self.clebsch, self.hlm)


    def compute_alms(self, beam, sky):
        """
        Compute the alms of the beam, sky, and horizon mask and transform
        them to the equatorial frame.

        Parameters
        ----------
        beam : jnp.ndarray
            Beam pattern in mwss sampling (see self.theta, self.phi) in
            topocentric coordinates.
        sky : jnp.ndarray
            Sky temperature in healpix sampling in galactic coordinates.

        Returns
        -------
        alm : jnp.ndarray
            Alm of the beam.
        blm : jnp.ndarray
            Alm of the sky.

        """
        alm = jax.vmap(self.mwss2alm)(beam)
        blm = jax.vmap(self.hp2alm)(sky)

        alm = jax.vmap(self.topo2eq)(alm)
        blm = jax.vmap(self.gal2eq)(blm)

        return alm, blm

    @jax.jit
    def compute_t_scatt(self, blm):
        nu = self.freq[:, None, None] * 1e6  # convert to Hz from MHz
        dly_phase = 2 * jnp.pi * nu * self.tau
        conv = crojax.simulator.convolve(self.hlm, blm, self.phases)
        return self.epsilon * jnp.exp(-1j * dly_phase) * conv

    @jax.jit
    def compute_tlm(self, blm):
        t_scatt = self.compute_t_scatt(blm)
        t_const = t_scatt + self.T_bb
        # sky * phases, shape is time, freq, ell', m'
        sky_phase = blm.conj()[None] * self.phases[:, None, None]
        tlm = jnp.einsum("tfij, lmij -> tflm", sky_phase, self.h_clebsch)
        tlm -= t_const * self.hlm.conj()[None]
        return tlm, t_const
    
    @jax.jit
    def compute_t_ant(self, alm, blm):
        tlm, t_const = self.compute_tlm(blm)
        t_ant = jnp.einsum("tflm, tflm -> tf", alm, tlm)
        norm = crojax.alm.total_power(alm, self.lmax)
        t_ant = t_ant / norm + t_const
        return t_ant


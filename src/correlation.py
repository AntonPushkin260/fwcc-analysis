import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from .dm_model import hubble_parameter, dm_igm

def projection_kernel(z, w, params):
    H0 = params.get('H0', 67.4)
    Omega_b_h2 = params.get('Omega_b_h2', 0.0224)
    h = H0 / 100.0
    Omega_b = Omega_b_h2 / (h**2)
    c_km_s = 2.998e5
    G_km = 6.674e-11 * 1e3
    m_p_kg = 1.673e-27
    Mpc_to_pc = 1e6
    C = (c_km_s * H0 * Omega_b) / (4 * np.pi * G_km * m_p_kg) * 1e-3 * Mpc_to_pc
    f_IGM = params.get('f_IGM', 0.84)
    chi_e = params.get('chi_e', 1.0)
    H_z = hubble_parameter(z, w, params.get('Omega_m', 0.315), H0)
    return C * f_IGM * chi_e * (1 + z) / H_z

def calculate_cross_correlation(frb, desi_maps, z_centers, w, params):
    density_profile, z_vals = get_angular_density(
        frb, desi_maps, z_centers, 
        aperture_radius=params.get('aperture_radius', 5.0)
    )
    valid_mask = ~np.isnan(density_profile)
    if not np.any(valid_mask):
        return 0.0
    density_interp = interp1d(
        z_centers[valid_mask], 
        density_profile[valid_mask],
        bounds_error=False, 
        fill_value=0.0
    )
    z_frb = frb.get('z', frb.get('z_mean', 0.5))
    def integrand(z_prime):
        if z_prime > z_frb:
            return 0.0
        kernel = projection_kernel(z_prime, w, params)
        density = density_interp(z_prime)
        return kernel * density
    integral, _ = quad(integrand, 0, z_frb, limit=100)
    dm_host_mean, _ = dm_host_model(
        z_frb,
        params.get('mu_host', 50.0),
        params.get('sigma_host', 30.0)
    )
    alpha = params.get('alpha', 1.0)
    return integral * alpha + dm_host_mean

def get_angular_density(frb, desi_maps, z_centers, aperture_radius=5.0):
    import healpy as hp
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    coord = SkyCoord(ra=frb['ra']*u.deg, dec=frb['dec']*u.deg)
    theta = np.pi/2 - coord.dec.rad
    phi = coord.ra.rad
    aperture_pixels = hp.query_disc(
        nside=256, 
        vec=hp.ang2vec(theta, phi),
        radius=np.radians(aperture_radius),
        inclusive=True
    )
    density_profiles = []
    for (z_min, z_max), density_map in desi_maps.items():
        aperture_density = density_map[aperture_pixels]
        valid_density = aperture_density[~np.isnan(aperture_density)]
        if len(valid_density) > 0:
            density_profiles.append(np.mean(valid_density) - 1)
        else:
            density_profiles.append(np.nan)
    return np.array(density_profiles), z_centers

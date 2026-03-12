import numpy as np
import pandas as pd
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
import warnings

def load_chime_data(catalog_path):
    frb_data = pd.read_csv(catalog_path)
    frb_data['DM_excess'] = frb_data['DM'] - frb_data['DM_MW_NE2001']
    frb_data['z_type'] = frb_data.apply(
        lambda row: 'spec' if pd.notna(row.get('spec_z', np.nan)) and row['spec_z'] > 0 else 'photo', 
        axis=1
    )
    if 'z_mean' not in frb_data.columns:
        frb_data['z_mean'] = frb_data.get('photo_z', 0.5)
    if 'z_error' not in frb_data.columns:
        frb_data['z_error'] = 0.1
    return frb_data

def load_desi_data(desi_path, nside=256):
    desi_data = pd.read_csv(desi_path)
    n_pix = hp.nside2npix(nside)
    density_maps = {}
    z_bins = np.arange(0.05, 1.5, 0.02)
    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        desi_bin = desi_data[(desi_data['z'] >= z_min) & (desi_data['z'] < z_max)]
        theta = np.pi/2 - np.radians(desi_bin['dec'])
        phi = np.radians(desi_bin['ra'])
        pix = hp.ang2pix(nside, theta, phi)
        counts = np.bincount(pix, minlength=n_pix)
        mean_count = np.mean(counts[counts > 0]) if np.any(counts > 0) else 1.0
        density = counts / mean_count
        density[counts == 0] = np.nan
        density_maps[(z_min, z_max)] = density
    return density_maps, z_bins

def get_angular_density(frb, desi_maps, z_bins, aperture_radius=5.0):
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
    z_centers = []
    for (z_min, z_max), density_map in desi_maps.items():
        aperture_density = density_map[aperture_pixels]
        valid_density = aperture_density[~np.isnan(aperture_density)]
        if len(valid_density) > 0:
            density_profiles.append(np.mean(valid_density) - 1)
        else:
            density_profiles.append(np.nan)
        z_centers.append((z_min + z_max) / 2)
    return np.array(density_profiles), np.array(z_centers)

def generate_ska_catalog(n_frb=10**6, f_spec=0.2, z_max=2.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    ra = np.random.uniform(0, 360, n_frb)
    dec = np.arcsin(np.random.uniform(-1, 1, n_frb)) * 180 / np.pi
    z = np.random.gamma(2.5, 0.4, n_frb)
    z = z[z < z_max]
    n_actual = len(z)
    ra = ra[:n_actual]
    dec = dec[:n_actual]
    n_spec = int(n_actual * f_spec)
    z_type = np.array(['photo'] * n_actual)
    spec_indices = np.random.choice(n_actual, n_spec, replace=False)
    z_type[spec_indices] = 'spec'
    dm_error = np.random.normal(5, 2, n_actual)
    dm_error = np.clip(dm_error, 1, 15)
    dm_mw = np.random.exponential(50, n_actual)
    z_mean = z.copy()
    z_error = np.where(
        z_type == 'spec',
        0.001,
        np.random.uniform(0.05, 0.15, n_actual)
    )
    frb_data = pd.DataFrame({
        'ra': ra,
        'dec': dec,
        'z': z,
        'z_type': z_type,
        'z_mean': z_mean,
        'z_error': z_error,
        'DM_error': dm_error,
        'DM_MW_NE2001': dm_mw,
        'LSS_noise': np.random.normal(10, 3, n_actual),
        'MW_noise': np.random.normal(5, 2, n_actual)
    })
    return frb_data

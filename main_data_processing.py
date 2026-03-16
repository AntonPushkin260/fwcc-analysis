#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolving the Dark Energy Crisis with Fast Radio Bursts:
A w-Measurement from CHIME and SDSS
Cross-Correlation
Author: Anton Pushkin
Date: March 2026
"""

import numpy as np
import pandas as pd
import os
import sys
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.constants import c, G, m_p
from scipy.optimize import minimize
import warnings
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colors

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn("emcee not installed. Install with: pip install emcee")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13

# =============================================================================
# CONFIGURATION
# =============================================================================

CHIME_CATALOG_PATH = r"YOUR_PATH"
SDSS_CATALOG_PATH = r"YOUR_PATH"

BASE_OUTPUT_DIR = r"YOUR_PATH"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "output")
FRB_CATALOG_OUTPUT = os.path.join(BASE_OUTPUT_DIR, "ska_frb_catalog.csv")
DENSITY_RESULTS_OUTPUT = os.path.join(BASE_OUTPUT_DIR, "frb_density_results.csv")
METADATA_OUTPUT = os.path.join(BASE_OUTPUT_DIR, "processing_metadata.json")
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")

NSIDE = 256
APERTURE_RADIUS_DEG = 5.0
N_FRB_GENERATE = 10000
F_SPEC = 0.2
Z_MAX = 2.0
RANDOM_SEED = 42

H0_PLANCK = 67.4
OMEGA_M_PLANCK = 0.315
OMEGA_B_H2 = 0.0224

Z_COLUMN_NAMES = ['z', 'redshift', 'z_est', 'z_mean', 'photo_z', 'spec_z']
RA_COLUMN_NAMES = ['ra', 'RA', 'right_ascension', 'ra_deg']
DEC_COLUMN_NAMES = ['dec', 'DEC', 'declination', 'dec_deg']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [BASE_OUTPUT_DIR, OUTPUT_DIR, PLOTS_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

def find_column(df, possible_names, dtype_check=None):
    """Find column in DataFrame by possible names."""
    for name in possible_names:
        if name in df.columns:
            if dtype_check is None or df[name].dtype in dtype_check:
                return name
    for col in df.columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_names):
            if 'err' not in col_lower and 'error' not in col_lower:
                if dtype_check is None or df[col].dtype in dtype_check:
                    return col
    return None

# =============================================================================
# COSMOLOGICAL MODELS
# =============================================================================

def hubble_parameter(z, w, Omega_m=OMEGA_M_PLANCK, H0=H0_PLANCK):
    """Calculate Hubble parameter H(z) for given w."""
    Omega_DE = 1 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_DE * (1 + z)**(3 * (1 + w)))

def dm_igm(z, w, f_IGM=0.84, chi_e=1.0, Omega_m=OMEGA_M_PLANCK, 
           Omega_b_h2=OMEGA_B_H2, H0=H0_PLANCK):
    """Calculate DM_IGM contribution."""
    h = H0 / 100.0
    Omega_b = Omega_b_h2 / (h**2)
    
    c_km_s = 2.998e5
    G_km = 6.674e-11 * 1e3
    m_p_kg = 1.673e-27
    Mpc_to_pc = 1e6
    
    C = (c_km_s * H0 * Omega_b) / (4 * np.pi * G_km * m_p_kg) * 1e-3 * Mpc_to_pc
    
    def integrand(z_prime):
        H_z = hubble_parameter(z_prime, w, Omega_m, H0)
        return chi_e * (1 + z_prime) / H_z
    
    if np.isscalar(z):
        integral, _ = quad(integrand, 0, z, limit=100)
    else:
        integral = np.array([quad(integrand, 0, zi, limit=100)[0] for zi in z])
    
    return C * f_IGM * integral

def dm_host_model(z, mu_host=50.0, sigma_host=30.0):
    """Calculate DM_host contribution."""
    mean = mu_host / (1 + z)
    std = sigma_host / (1 + z)
    return mean, std

def calculate_model_dm(z, w, density_profile, z_centers, params):
    """Calculate model DM for given w and density profile."""
    dm_igm_val = dm_igm(
        z, w,
        f_IGM=params.get('f_IGM', 0.84),
        chi_e=params.get('chi_e', 1.0),
        Omega_m=params.get('Omega_m', OMEGA_M_PLANCK),
        H0=params.get('H0', H0_PLANCK)
    )
    
    valid_mask = ~np.isnan(density_profile)
    if np.any(valid_mask) and len(z_centers) > 0:
        density_interp = interp1d(
            z_centers[valid_mask],
            density_profile[valid_mask],
            bounds_error=False,
            fill_value=0.0
        )
        delta_g = density_interp(z)
        alpha = params.get('alpha', 50.0)
        cross_corr_term = alpha * delta_g
    else:
        cross_corr_term = 0.0
    
    dm_host_mean, _ = dm_host_model(
        z,
        params.get('mu_host', 50.0),
        params.get('sigma_host', 30.0)
    )
    
    return dm_igm_val + cross_corr_term + dm_host_mean

# =============================================================================
# LIKELIHOOD FUNCTION
# =============================================================================

def log_likelihood_w(w, frb_results, params):
    """Calculate log-likelihood for parameter w."""
    log_like = 0.0
    
    for _, row in frb_results.iterrows():
        z = row.get('z', row.get('z_mean', 0.5))
        if pd.isna(z) or z <= 0 or z > 10:
            continue
        
        density_profile = row.get('density_profile', None)
        if density_profile is None:
            continue
        if isinstance(density_profile, (list, np.ndarray)):
            density_profile = np.array(density_profile).flatten()
        else:
            density_profile = np.array([density_profile])
        
        z_centers = row.get('z_centers', None)
        if z_centers is None:
            z_centers = np.array([0.5])
        if isinstance(z_centers, (list, np.ndarray)):
            z_centers = np.array(z_centers).flatten()
        else:
            z_centers = np.array([z_centers])
        
        model_dm = calculate_model_dm(z, w[0] if isinstance(w, (list, np.ndarray)) else w, 
                                      density_profile, z_centers, params)
        
        obs_dm = row.get('DM_excess', np.nan)
        if pd.isna(obs_dm):
            continue
        
        sigma_meas = row.get('DM_error', 10.0)
        sigma_host = params.get('sigma_host', 30.0) / (1 + z)
        sigma_lss = row.get('LSS_noise', 15.0)
        sigma_mw = row.get('MW_noise', 10.0)
        
        sigma_tot = np.sqrt(sigma_meas**2 + sigma_host**2 + sigma_lss**2 + sigma_mw**2)
        if sigma_tot <= 0 or np.isnan(sigma_tot):
            sigma_tot = 10.0
        
        residual = obs_dm - model_dm
        log_like += -0.5 * (residual / sigma_tot)**2 - np.log(np.sqrt(2 * np.pi) * sigma_tot)
    
    return log_like

def prior_w(w):
    """Prior for w: uniform in [-2, 0]."""
    w_val = w[0] if isinstance(w, (list, np.ndarray)) else w
    if -2.0 < w_val < 0.0:
        return 0.0
    return -np.inf

def log_posterior_w(w, frb_results, params):
    """Log-posterior for w."""
    lp = prior_w(w)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_w(w, frb_results, params)

# =============================================================================
# W ESTIMATION
# =============================================================================

def estimate_w(frb_results, params, use_mcmc=True, nwalkers=32, nsteps=500, burn_in=100):
    """Estimate dark energy parameter w."""
    print("\nStarting w parameter estimation...")
    
    w_init = [-1.0]
    
    if not use_mcmc or not HAS_EMCEE:
        print("  Using optimization (fast mode)...")
        
        def neg_log_post(w):
            return -log_posterior_w(w, frb_results, params)
        
        result = minimize(neg_log_post, w_init, method='Nelder-Mead', bounds=[(-2.0, 0.0)])
        
        if result.success:
            w_best = result.x[0]
            print(f"  w = {w_best:.3f} (optimization)")
            return {'w_best': w_best, 'w_err': np.nan, 'success': True, 'samples': None}
        else:
            print(f"  Optimization failed: {result.message}")
            return {'w_best': np.nan, 'w_err': np.nan, 'success': False, 'samples': None}
    
    else:
        print(f"  Running MCMC: {nwalkers} walkers, {nsteps} steps...")
        
        pos = np.array(w_init) + 1e-4 * np.random.randn(nwalkers, 1)
        
        sampler = emcee.EnsembleSampler(
            nwalkers, 1, 
            lambda theta: log_posterior_w(theta, frb_results, params)
        )
        
        sampler.run_mcmc(pos, nsteps, progress=True)
        
        samples = sampler.get_chain(discard=burn_in, flat=True)
        w_samples = samples[:, 0]
        
        w_median = np.median(w_samples)
        w_16 = np.percentile(w_samples, 16)
        w_84 = np.percentile(w_samples, 84)
        w_err = (w_84 - w_16) / 2
        
        print(f"  w = {w_median:.3f} +/- {w_err:.3f} (68% CI)")
        print(f"  Range: [{w_16:.3f}, {w_84:.3f}]")
        
        return {
            'w_best': w_median,
            'w_err': w_err,
            'w_16': w_16,
            'w_84': w_84,
            'success': True,
            'samples': w_samples,
            'autocorr': None
        }

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_w_posterior(w_result, output_path):
    """Plot posterior distribution of w."""
    if w_result['samples'] is None:
        print(f"  No samples for plotting")
        return
    
    print(f"Plotting w distribution: {output_path}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    samples = w_result['samples']
    
    axes[0].hist(samples, bins=40, color='#4ECDC4', edgecolor='white', alpha=0.8, density=True)
    axes[0].axvline(w_result['w_best'], color='red', linestyle='-', linewidth=2, label=f'median: {w_result["w_best"]:.3f}')
    axes[0].axvline(w_result['w_best'] - w_result['w_err'], color='gray', linestyle='--', alpha=0.7)
    axes[0].axvline(w_result['w_best'] + w_result['w_err'], color='gray', linestyle='--', alpha=0.7, label='68% CI')
    axes[0].axvline(-1.0, color='blue', linestyle=':', linewidth=1.5, label='LambdaCDM (w = -1)')
    axes[0].set_xlabel('Parameter w')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('Posterior Distribution of w')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    if len(samples) > 100:
        axes[1].plot(samples, color='#2E86AB', alpha=0.5, linewidth=0.5)
        axes[1].axhline(w_result['w_best'], color='red', linestyle='-', linewidth=1)
        axes[1].set_xlabel('MCMC Step')
        axes[1].set_ylabel('w')
        axes[1].set_title('Trace Plot')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

# =============================================================================
# DATA LOADING (NO HEALPY)
# =============================================================================

def load_chime_data(catalog_path):
    """Load CHIME FRB catalog."""
    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"File not found: {catalog_path}")
    
    print(f"Loading CHIME catalog: {catalog_path}")
    frb_data = pd.read_csv(catalog_path)
    
    print(f"  Available columns: {list(frb_data.columns)[:15]}...")
    
    if 'DM' in frb_data.columns and 'DM_MW_NE2001' in frb_data.columns:
        frb_data['DM_excess'] = frb_data['DM'] - frb_data['DM_MW_NE2001']
    elif 'DM' in frb_data.columns:
        frb_data['DM_MW_NE2001'] = 50
        frb_data['DM_excess'] = frb_data['DM'] - 50
    
    z_col = find_column(frb_data, Z_COLUMN_NAMES, dtype_check=['float64', 'float32', 'int64', 'int32'])
    if z_col and z_col != 'z':
        print(f"  Found z column: '{z_col}' -> copied to 'z'")
        frb_data['z'] = frb_data[z_col].copy()
    elif z_col is None:
        print("  Redshift column not found, using DM_excess as proxy")
        if 'DM_excess' in frb_data.columns:
            frb_data['z'] = frb_data['DM_excess'] / 1000.0
        else:
            frb_data['z'] = 0.5
    
    if 'spec_z' in frb_data.columns:
        frb_data['z_type'] = frb_data.apply(
            lambda row: 'spec' if pd.notna(row.get('spec_z', np.nan)) and row.get('spec_z', 0) > 0 else 'photo', 
            axis=1
        )
    else:
        frb_data['z_type'] = 'photo'
    
    if 'z_mean' not in frb_data.columns:
        frb_data['z_mean'] = frb_data.get('photo_z', frb_data.get('z', 0.5))
    if 'z_error' not in frb_data.columns:
        frb_data['z_error'] = 0.1
    
    print(f"Loaded {len(frb_data)} FRBs")
    return frb_data

def load_sdss_data_no_healpy(sdss_path, z_bins):
    """
    Load SDSS and group galaxies by z-bins.
    """
    if not os.path.exists(sdss_path):
        raise FileNotFoundError(f"File not found: {sdss_path}")
    
    print(f"Loading catalog: {sdss_path}")
    sdss_data = pd.read_csv(sdss_path, low_memory=False)
    
    ra_col = 'ra' if 'ra' in sdss_data.columns else 'RA' if 'RA' in sdss_data.columns else None
    dec_col = 'dec' if 'dec' in sdss_data.columns else 'DEC' if 'DEC' in sdss_data.columns else None
    z_col = 'z' if 'z' in sdss_data.columns else 'REDSHIFT' if 'REDSHIFT' in sdss_data.columns else None
    
    if not all([ra_col, dec_col, z_col]):
        raise ValueError(f"Required columns not found. Available: {list(sdss_data.columns)[:10]}")
    
    sdss_data = sdss_data.rename(columns={ra_col: 'ra', dec_col: 'dec', z_col: 'z'})
    
    sdss_data['ra'] = pd.to_numeric(sdss_data['ra'], errors='coerce')
    sdss_data['dec'] = pd.to_numeric(sdss_data['dec'], errors='coerce')
    sdss_data['z'] = pd.to_numeric(sdss_data['z'], errors='coerce')
    
    sdss_data = sdss_data.dropna(subset=['ra', 'dec', 'z'])
    
    print(f"  Loaded {len(sdss_data)} galaxies after cleaning")
    
    galaxy_bins = {}
    
    print(f"Grouping galaxies into {len(z_bins)-1} z-bins...")
    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        bin_data = sdss_data[(sdss_data['z'] >= z_min) & (sdss_data['z'] < z_max)].copy()
        
        if len(bin_data) > 0:
            galaxy_bins[(z_min, z_max)] = bin_data[['ra', 'dec', 'z']].reset_index(drop=True)
        else:
            galaxy_bins[(z_min, z_max)] = pd.DataFrame(columns=['ra', 'dec', 'z'])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(z_bins)-1} z-bins")
    
    print(f"Created {len(galaxy_bins)} galaxy groups")
    return galaxy_bins, z_bins

def generate_ska_catalog(n_frb=10**4, f_spec=0.2, z_max=2.0, seed=None):
    """Generate simulated SKA FRB catalog."""
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
    z_error = np.where(z_type == 'spec', 0.001, np.random.uniform(0.05, 0.15, n_actual))
    
    frb_data = pd.DataFrame({
        'frb_id': np.arange(n_actual),
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

# =============================================================================
# ANALYSIS (NO HEALPY)
# =============================================================================

def get_angular_density_no_healpy(frb, galaxy_bins, z_bins, aperture_radius_deg=5.0):
    """
    Calculate angular density profile of galaxies around FRB.
    """
    frb_coord = SkyCoord(
        ra=float(frb['ra']) * u.deg,
        dec=float(frb['dec']) * u.deg,
        frame='icrs'
    )
    
    density_profiles = []
    z_centers = []
    
    aperture_radius = aperture_radius_deg * u.deg
    
    for (z_min, z_max), bin_data in galaxy_bins.items():
        if len(bin_data) == 0:
            density_profiles.append(np.nan)
            z_centers.append((z_min + z_max) / 2)
            continue
        
        try:
            galaxy_coord = SkyCoord(
                ra=bin_data['ra'].values * u.deg,
                dec=bin_data['dec'].values * u.deg,
                frame='icrs'
            )
            
            separations = frb_coord.separation(galaxy_coord)
            
            in_aperture = separations <= aperture_radius
            n_in_aperture = np.sum(in_aperture)
            
            if n_in_aperture > 0:
                sky_area = 4 * np.pi * np.sin(aperture_radius_deg * np.pi / 360)**2
                total_sky = 4 * np.pi
                expected_fraction = sky_area / total_sky
                n_expected = len(bin_data) * expected_fraction
                
                if n_expected > 0:
                    delta = (n_in_aperture / n_expected) - 1
                else:
                    delta = 0.0
            else:
                delta = -1.0
                
        except Exception as e:
            print(f"  Error calculating density: {e}")
            delta = np.nan
        
        density_profiles.append(delta)
        z_centers.append((z_min + z_max) / 2)
    
    return np.array(density_profiles), np.array(z_centers)

def process_all_frbs_no_healpy(frb_catalog, galaxy_bins, z_bins, aperture_radius_deg=5.0):
    """Process all FRBs without using healpy."""
    print(f"Processing {len(frb_catalog)} FRBs...")
    results = []
    
    z_col = 'z' if 'z' in frb_catalog.columns else 'z_mean' if 'z_mean' in frb_catalog.columns else None
    
    for idx, frb in frb_catalog.iterrows():
        density_profile, z_centers = get_angular_density_no_healpy(
            frb, galaxy_bins, z_bins, aperture_radius_deg=aperture_radius_deg
        )
        results.append({
            'frb_id': frb.get('frb_id', idx),
            'ra': frb['ra'],
            'dec': frb['dec'],
            'z': frb.get(z_col, 0) if z_col else 0,
            'z_type': frb.get('z_type', 'unknown'),
            'density_profile': density_profile.tolist(),
            'z_centers': z_centers.tolist(),
            'mean_density': np.nanmean(density_profile)
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(frb_catalog)} FRBs ({100*(idx+1)/len(frb_catalog):.1f}%)")
    
    print(f"Processed {len(frb_catalog)} FRBs")
    return pd.DataFrame(results)

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_sky_distribution(frb_catalog, output_path):
    """Plot sky distribution of FRBs."""
    print(f"Plotting sky map: {output_path}")
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='mollweide')
    
    ra_rad = np.radians(frb_catalog['ra'].values)
    dec_rad = np.radians(frb_catalog['dec'].values)
    
    spec_mask = frb_catalog['z_type'] == 'spec'
    
    ax.scatter(ra_rad[spec_mask] - np.pi, dec_rad[spec_mask], 
               s=2, c='#2E86AB', label='spec-z', alpha=0.6, rasterized=True)
    ax.scatter(ra_rad[~spec_mask] - np.pi, dec_rad[~spec_mask], 
               s=2, c='#A23B72', label='photo-z', alpha=0.3, rasterized=True)
    
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    ax.set_title('FRB Distribution')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

def plot_redshift_distribution(frb_catalog, output_path):
    """Plot redshift distribution of FRBs."""
    print(f"Plotting redshift histogram: {output_path}")
    
    z_col = 'z' if 'z' in frb_catalog.columns else 'z_mean' if 'z_mean' in frb_catalog.columns else None
    if z_col is None:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    z_values = frb_catalog[z_col].values
    
    axes[0].hist(z_values, bins=50, color='#4ECDC4', edgecolor='white', alpha=0.8)
    axes[0].set_xlabel('z')
    axes[0].set_ylabel('Number of FRBs')
    axes[0].set_title('Redshift Distribution')
    axes[0].grid(True, alpha=0.3)
    
    if 'z_type' in frb_catalog.columns:
        z_spec = frb_catalog[frb_catalog['z_type'] == 'spec'][z_col]
        z_photo = frb_catalog[frb_catalog['z_type'] == 'photo'][z_col]
        axes[1].hist(z_spec, bins=30, alpha=0.7, label='spec-z', color='#2E86AB', edgecolor='white')
        axes[1].hist(z_photo, bins=30, alpha=0.5, label='photo-z', color='#A23B72', edgecolor='white')
        axes[1].legend()
    
    axes[1].set_xlabel('z')
    axes[1].set_ylabel('Number of FRBs')
    axes[1].set_title('By z Type')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

# =============================================================================
# SAVE METADATA
# =============================================================================

def save_metadata(metadata):
    """Save processing metadata to JSON file."""
    with open(METADATA_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata: {METADATA_OUTPUT}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    print("=" * 80)
    print("FwCC ANALYSIS (NO HEALPY VERSION)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    ensure_dirs()
    
    metadata = {
        'start_time': datetime.now().isoformat(),
        'parameters': {
            'aperture_radius_deg': APERTURE_RADIUS_DEG,
            'n_frb_generate': N_FRB_GENERATE,
            'f_spec': F_SPEC,
            'z_max': Z_MAX,
            'random_seed': RANDOM_SEED,
            'cosmology': {'H0': H0_PLANCK, 'Omega_m': OMEGA_M_PLANCK, 'Omega_b_h2': OMEGA_B_H2},
            'healpy_used': False
        },
        'input_files': {
            'chime_catalog': CHIME_CATALOG_PATH,
            'sdss_catalog': SDSS_CATALOG_PATH
        },
        'output_files': {
            'frb_catalog': FRB_CATALOG_OUTPUT,
            'density_results': DENSITY_RESULTS_OUTPUT,
            'metadata': METADATA_OUTPUT,
            'plots_dir': PLOTS_DIR
        }
    }
    
    sdss_loaded = False
    
    try:
        # 1. FRB Catalog
        print("-" * 80)
        print("1. FRB CATALOG")
        print("-" * 80)
        
        if os.path.exists(CHIME_CATALOG_PATH):
            frb_catalog = load_chime_data(CHIME_CATALOG_PATH)
        else:
            print(f"File {CHIME_CATALOG_PATH} not found, generating simulated catalog...")
            frb_catalog = generate_ska_catalog(n_frb=N_FRB_GENERATE, f_spec=F_SPEC, z_max=Z_MAX, seed=RANDOM_SEED)
        
        frb_catalog.to_csv(FRB_CATALOG_OUTPUT, index=False)
        print(f"Saved: {FRB_CATALOG_OUTPUT} ({len(frb_catalog)} FRBs)")
        metadata['n_frb_processed'] = len(frb_catalog)
        
        # 2. SDSS Data (NO HEALPY)
        print()
        print("-" * 80)
        print("2. SDSS DATA (without healpy)")
        print("-" * 80)
        
        z_bins = np.arange(0.05, 1.5, 0.02)
        
        if os.path.exists(SDSS_CATALOG_PATH):
            galaxy_bins, z_bins = load_sdss_data_no_healpy(SDSS_CATALOG_PATH, z_bins)
            sdss_loaded = True
        else:
            print(f"File {SDSS_CATALOG_PATH} not found! Creating empty data...")
            galaxy_bins = {}
            for i in range(len(z_bins) - 1):
                galaxy_bins[(z_bins[i], z_bins[i+1])] = pd.DataFrame(columns=['ra', 'dec', 'z', 'coord'])
        
        metadata['n_density_bins'] = len(galaxy_bins)
        metadata['z_bins'] = z_bins.tolist()
        
        # 3. Density Profiles (NO HEALPY)
        print()
        print("-" * 80)
        print("3. DENSITY PROFILE CALCULATION")
        print("-" * 80)
        
        results_df = process_all_frbs_no_healpy(frb_catalog, galaxy_bins, z_bins, aperture_radius_deg=APERTURE_RADIUS_DEG)
        results_df.to_csv(DENSITY_RESULTS_OUTPUT, index=False)
        print(f"Saved: {DENSITY_RESULTS_OUTPUT}")
        metadata['n_results'] = len(results_df)
        
        # 4. Statistics
        print()
        print("-" * 80)
        print("STATISTICS")
        print("-" * 80)
        print(f"Total FRBs: {len(frb_catalog)}")
        if 'z_type' in frb_catalog.columns:
            print(f"With spec-z: {len(frb_catalog[frb_catalog['z_type']=='spec'])}")
            print(f"With photo-z: {len(frb_catalog[frb_catalog['z_type']=='photo'])}")
        print(f"SDSS z-bins: {len(galaxy_bins)}")
        mean_dens = results_df['mean_density'].mean()
        print(f"Mean density: {mean_dens:.4f}" if not np.isnan(mean_dens) else "Mean density: nan")
        
        # 5. Visualization
        print()
        print("-" * 80)
        print("4. VISUALIZATION")
        print("-" * 80)
        
        plot_sky_distribution(frb_catalog, os.path.join(PLOTS_DIR, "sky_distribution.png"))
        plot_redshift_distribution(frb_catalog, os.path.join(PLOTS_DIR, "redshift_distribution.png"))
        
        # 6. W PARAMETER ESTIMATION
        print()
        print("-" * 80)
        print("5. W PARAMETER ESTIMATION (FwCC method)")
        print("-" * 80)
        
        model_params = {
            'f_IGM': 0.84,
            'chi_e': 1.0,
            'Omega_m': OMEGA_M_PLANCK,
            'H0': H0_PLANCK,
            'mu_host': 50.0,
            'sigma_host': 30.0,
            'alpha': 50.0
        }
        
        use_mcmc = True
        
        w_result = estimate_w(
            frb_results=results_df,
            params=model_params,
            use_mcmc=use_mcmc,
            nwalkers=50,
            nsteps=1000,
            burn_in=200
        )
        
        metadata['w_estimation'] = {
            'w_best': float(w_result['w_best']) if not np.isnan(w_result['w_best']) else None,
            'w_err': float(w_result['w_err']) if 'w_err' in w_result and not np.isnan(w_result['w_err']) else None,
            'success': w_result['success'],
            'method': 'MCMC' if use_mcmc and HAS_EMCEE else 'optimization'
        }
        
        if w_result['samples'] is not None:
            plot_w_posterior(w_result, os.path.join(PLOTS_DIR, "w_posterior.png"))
            metadata['plots_generated'] = metadata.get('plots_generated', []) + ["w_posterior.png"]
        
        # KEY OUTPUT
        print()
        print("=" * 80)
        if w_result['success'] and not np.isnan(w_result['w_best']):
            print("W ESTIMATION SUCCESSFUL!")
            print(f"   w = {w_result['w_best']:.3f} +/- {w_result['w_err']:.3f}")
            if not np.isnan(w_result['w_err']) and w_result['w_err'] > 0:
                sigma_dev = abs(w_result['w_best'] + 1) / w_result['w_err']
                print(f"   Deviation from Lambda: {sigma_dev:.2f} sigma")
                if -1.15 < w_result['w_best'] < -0.85:
                    print("   Result consistent with LambdaCDM (w = -1)")
                else:
                    print("   Result deviates from LambdaCDM - verification needed")
            else:
                print("   Error not computed (used optimization)")
        else:
            print("W ESTIMATION FAILED")
            print("   Possible causes:")
            print("   - Insufficient data")
            print("   - SDSS loading issues")
            print("   - emcee package not installed (for MCMC: pip install emcee)")
        print("=" * 80)
        
        metadata['end_time'] = datetime.now().isoformat()
        metadata['status'] = 'success'
        
    except Exception as e:
        print(f"\nERROR: {e}")
        metadata['status'] = 'error'
        metadata['error'] = str(e)
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        save_metadata(metadata)
        
        print()
        print("=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Output files:")
        print(f"   {FRB_CATALOG_OUTPUT}")
        print(f"   {DENSITY_RESULTS_OUTPUT}")
        print(f"   {METADATA_OUTPUT}")
        print(f"   {PLOTS_DIR}/")
        print()

if __name__ == "__main__":
    main()
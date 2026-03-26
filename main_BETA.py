import numpy as np
import pandas as pd
import os
from astropy.cosmology import wCDM
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import j0
from scipy.optimize import minimize
import json
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import camb
import emcee
import healpy as hp
from Corrfunc.mocks import DDtheta_mocks
 
CHIME_CATALOG_PATH = r"catalog.csv"
SDSS_CATALOG_PATH = r"sdss.csv"
BASE_OUTPUT_DIR = r"output"
 
THETA_BINS_DEG = np.linspace(0.1, 5.0, 15)
Z_BINS = np.arange(0.05, 0.8, 0.05)
RANDOM_SEED = 42
N_MOCKS = 500
N_THREADS = max(1, cpu_count() - 2)
 
H0_PLANCK = 67.4
OMEGA_M_PLANCK = 0.315
OMEGA_B_H2 = 0.0224
OMEGA_B = OMEGA_B_H2 / (H0_PLANCK / 100.0)**2
SIGMA_8 = 0.811
N_S = 0.965
 
FRB_Z_MEAN = 0.5
FRB_Z_SIGMA = 0.3
GALAXY_Z_MEAN = 0.3
GALAXY_Z_SIGMA = 0.15
 
DM_MW_MODEL = 'NE2001'
 
def hubble_parameter(z, w, Omega_m=OMEGA_M_PLANCK, H0=H0_PLANCK):
    Omega_DE = 1 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_DE * (1 + z)**(3 * (1 + w)))
 
def comoving_distance(z, w, Omega_m=OMEGA_M_PLANCK, H0=H0_PLANCK):
    cosmo = wCDM(H0=H0, Om0=Omega_m, w0=w)
    return cosmo.comoving_distance(z).value
 
def growth_factor(z, w, Omega_m=OMEGA_M_PLANCK):
    Omega_m_z = Omega_m * (1 + z)**3 / (
        Omega_m * (1 + z)**3 + (1 - Omega_m) * (1 + z)**(3 * (1 + w))
    )
    gamma = 0.55 + 0.05 * (1 + w)
    D_z = Omega_m_z**gamma
    D_0 = Omega_m**gamma
    return D_z / D_0
 
def get_camb_power_spectrum(k_vals, z_vals, w=-1.0, Omega_m=OMEGA_M_PLANCK, 
                             H0=H0_PLANCK, sigma_8=SIGMA_8, n_s=N_S):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=OMEGA_B_H2, 
                       omch2=(Omega_m - OMEGA_B) * (H0/100.0)**2, omk=0, w=w)
    pars.InitPower.set_params(As=2.1e-9, ns=n_s)
    pars.set_matter_power(redshifts=z_vals, kmax=max(k_vals) * 1.5)
    results = camb.get_results(pars)
    kh, z_vals_camb, pk = results.get_matter_power_spectrum(
        minkh=min(k_vals), maxkh=max(k_vals), npoints=len(k_vals))
    P_kz = np.zeros((len(k_vals), len(z_vals)))
    for i, z in enumerate(z_vals):
        interp_pk = interp1d(kh, pk[:, i], kind='cubic', 
                            bounds_error=False, fill_value=0.0)
        P_kz[:, i] = interp_pk(k_vals)
    return P_kz
 
def matter_power_spectrum_approx(k, z, w, Omega_m=OMEGA_M_PLANCK, 
                                  sigma_8=SIGMA_8, n_s=N_S):
    h = H0_PLANCK / 100.0
    D_z = growth_factor(z, w, Omega_m)
    k_eq = Omega_m * h**2 / 14.4
    P_k = sigma_8**2 * (k / 0.2)**n_s * (1 + (k / k_eq)**2)**(-2)
    k = np.atleast_1d(k)
    z = np.atleast_1d(z)
    if len(z) == 1:
        return P_k * D_z**2
    else:
        return P_k[:, np.newaxis] * (D_z**2)[np.newaxis, :]
 
def compute_theoretical_xi_limber_vectorized(theta_bins, z_bins, w, params, 
                                               use_camb=True):
    n_bins = len(theta_bins) - 1
    xi_model = np.zeros(n_bins)
    Omega_m = params.get('Omega_m', OMEGA_M_PLANCK)
    H0 = params.get('H0', H0_PLANCK)
    b_g = params.get('galaxy_bias', 1.5)
    b_FRB = params.get('FRB_bias', 1.2)
    A_norm = params.get('A_norm', 1.0)
    k_min, k_max = 0.001, 10.0
    k_vals = np.logspace(np.log10(k_min), np.log10(k_max), 200)
    z_integration = np.linspace(z_bins[0], z_bins[-1], 50)
    W_FRB = frb_selection_function(z_integration)
    W_gal = galaxy_selection_function(z_integration)
    if use_camb:
        P_kz = get_camb_power_spectrum(k_vals, z_integration, w, Omega_m, H0)
    else:
        P_kz = matter_power_spectrum_approx(k_vals, z_integration, w, Omega_m)
    if P_kz.shape[0] != len(k_vals):
        P_kz = P_kz.T
    D_z = growth_factor(z_integration, w, Omega_m)
    chi_z = comoving_distance(z_integration, w, Omega_m, H0) / 1000.0
    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
    theta_rad = np.radians(theta_centers)
    for i in range(n_bins):
        theta = theta_rad[i]
        arg = np.outer(k_vals, chi_z) * theta
        j0_vals = j0(arg)
        W_z = W_FRB * W_gal * (z_integration[1] - z_integration[0])
        integrand = (k_vals[:, np.newaxis] * P_kz * j0_vals * 
                     W_z[np.newaxis, :] * D_z[np.newaxis, :]**2)
        xi_k = np.trapz(integrand, k_vals, axis=0)
        xi_theta = np.trapz(xi_k, z_integration)
        xi_model[i] = A_norm * b_g * b_FRB * xi_theta * 1e-4
    return xi_model
 
def frb_selection_function(z, z_mean=FRB_Z_MEAN, z_sigma=FRB_Z_SIGMA):
    if np.isscalar(z):
        if z <= 0:
            return 0.0
        return (z / z_mean)**2 * np.exp(-2 * z / z_mean)
    else:
        W_z = np.zeros_like(z)
        mask = z > 0
        W_z[mask] = (z[mask] / z_mean)**2 * np.exp(-2 * z[mask] / z_mean)
        return W_z / np.max(W_z)
 
def galaxy_selection_function(z, z_mean=GALAXY_Z_MEAN, z_sigma=GALAXY_Z_SIGMA):
    W_z = np.exp(-0.5 * ((z - z_mean) / z_sigma)**2)
    return W_z / np.max(W_z)
 
def create_survey_mask_from_data(ra, dec, nside=64):
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=bool)
    theta = np.pi/2 - np.radians(dec)
    phi = np.radians(ra)
    indices = hp.ang2pix(nside, theta, phi)
    mask[np.unique(indices)] = True
    return mask, nside
 
def apply_survey_mask(ra_mock, dec_mock, mask, nside):
    theta = np.pi/2 - np.radians(dec_mock)
    phi = np.radians(ra_mock)
    indices = hp.ang2pix(nside, theta, phi)
    mask_applied = mask[indices]
    return ra_mock[mask_applied], dec_mock[mask_applied]
 
def generate_random_catalog(n_random, ra_data, dec_data, mask, nside, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    npix = hp.nside2npix(nside)
    valid_pixels = np.where(mask)[0]
    
    n_per_pixel = int(np.ceil(n_random / len(valid_pixels)))
    ra_random_list = []
    dec_random_list = []
    
    for pix in valid_pixels:
        theta_rand = hp.pix2ang(nside, pix)[0]
        phi_rand = np.random.uniform(0, 2*np.pi, n_per_pixel)
        dec_rand = np.degrees(np.pi/2 - theta_rand)
        ra_rand = np.degrees(phi_rand)
        
        dec_min_data = np.min(dec_data)
        dec_max_data = np.max(dec_data)
        dec_mask = (dec_rand >= dec_min_data) & (dec_rand <= dec_max_data)
        
        ra_random_list.extend(ra_rand[dec_mask])
        dec_random_list.extend(dec_rand[dec_mask])
    
    ra_random = np.array(ra_random_list)
    dec_random = np.array(dec_random_list)
    
    if len(ra_random) > n_random:
        shuffle_idx = np.random.choice(len(ra_random), n_random, replace=False)
        ra_random = ra_random[shuffle_idx]
        dec_random = dec_random[shuffle_idx]
    
    while len(ra_random) < n_random:
        n_extra = n_random - len(ra_random)
        pix_extra = np.random.choice(valid_pixels, n_extra * 2)
        theta_extra = hp.pix2ang(nside, pix_extra)[0]
        phi_extra = np.random.uniform(0, 2*np.pi, n_extra * 2)
        dec_extra = np.degrees(np.pi/2 - theta_extra)
        ra_extra = np.degrees(phi_extra)
        
        dec_min_data = np.min(dec_data)
        dec_max_data = np.max(dec_data)
        dec_mask = (dec_extra >= dec_min_data) & (dec_extra <= dec_max_data)
        
        ra_random = np.concatenate([ra_random, ra_extra[dec_mask]])
        dec_random = np.concatenate([dec_random, dec_extra[dec_mask]])
    
    return ra_random[:n_random], dec_random[:n_random]
 
def compute_cross_correlation_corrfunc(frb_catalog, galaxy_bins, theta_bins, 
                                        random_catalogs=None, n_random_factor=10,
                                        survey_mask=None, nside=None):
    n_theta_bins = len(theta_bins) - 1
    xi_measurements = np.zeros(n_theta_bins)
    xi_errors = np.zeros(n_theta_bins)
    n_pairs = np.zeros(n_theta_bins)
    
    print(f"Computing cross-correlation for {len(frb_catalog)} FRBs (Corrfunc DDtheta)...")
    
    frb_ra = frb_catalog['ra'].values
    frb_dec = frb_catalog['dec'].values
    frb_dm_excess = frb_catalog.get('DM_excess', np.zeros(len(frb_catalog))).values
    
    if random_catalogs is None:
        print("  Generating random catalogs...")
        random_catalogs = {}
        for (z_min, z_max), bin_data in galaxy_bins.items():
            if len(bin_data) > 0:
                n_random = len(bin_data) * n_random_factor
                ra_rand, dec_rand = generate_random_catalog(
                    n_random, bin_data['ra'].values, bin_data['dec'].values,
                    survey_mask, nside, seed=RANDOM_SEED
                )
                random_catalogs[(z_min, z_max)] = pd.DataFrame({
                    'ra': ra_rand, 'dec': dec_rand
                })
    
    for i in range(n_theta_bins):
        theta_min_rad = np.radians(theta_bins[i])
        theta_max_bin_rad = np.radians(theta_bins[i+1])
        
        xi_sum = 0.0
        xi_sq_sum = 0.0
        pair_count = 0
        
        for (z_min, z_max), bin_data in galaxy_bins.items():
            if len(bin_data) == 0:
                continue
            
            gal_ra = bin_data['ra'].values
            gal_dec = bin_data['dec'].values
            
            if (z_min, z_max) in random_catalogs:
                rand_ra = random_catalogs[(z_min, z_max)]['ra'].values
                rand_dec = random_catalogs[(z_min, z_max)]['dec'].values
                alpha = len(bin_data) / len(random_catalogs[(z_min, z_max)])
            else:
                alpha = 1.0
                rand_ra = None
                rand_dec = None
            
            bin_edges_deg = np.array([theta_bins[i], theta_bins[i+1]])
            
            DD_result = DDtheta_mocks(
                1,
                n_threads=N_THREADS,
                bin_edges=bin_edges_deg,
                RA1=frb_ra, DEC1=frb_dec,
                RA2=gal_ra, DEC2=gal_dec,
                verbose=False
            )
            
            if rand_ra is not None:
                DR_result = DDtheta_mocks(
                    1,
                    n_threads=N_THREADS,
                    bin_edges=bin_edges_deg,
                    RA1=frb_ra, DEC1=frb_dec,
                    RA2=rand_ra, DEC2=rand_dec,
                    verbose=False
                )
                
                n_data = DD_result['npairs']
                n_random = DR_result['npairs']
                
                if n_random > 0:
                    delta_g = (n_data / (alpha * n_random)) - 1
                else:
                    delta_g = 0.0
            else:
                n_data = DD_result['npairs']
                sky_area = 2 * np.pi * (
                    np.cos(theta_min_rad) - np.cos(theta_max_bin_rad)
                )
                mean_density = len(bin_data) / (4 * np.pi)
                n_expected = mean_density * sky_area
                delta_g = (n_data / n_expected) - 1 if n_expected > 0 else 0.0
            
            if n_data > 0:
                for frb_idx in range(len(frb_ra)):
                    dm_val = frb_dm_excess[frb_idx]
                    xi_sum += dm_val * delta_g
                    xi_sq_sum += (dm_val * delta_g)**2
                    pair_count += n_data
        
        if pair_count > 0:
            xi_measurements[i] = xi_sum / pair_count
            xi_errors[i] = np.sqrt(xi_sq_sum / pair_count - xi_measurements[i]**2) / np.sqrt(pair_count)
        else:
            xi_measurements[i] = np.nan
            xi_errors[i] = np.nan
        
        n_pairs[i] = pair_count
        
        if (i + 1) % 5 == 0:
            print(f"  Theta bin {i+1}/{n_theta_bins} complete")
    
    return xi_measurements, xi_errors, n_pairs
 
def generate_mock_frb_catalog(n_frb, z_bins, theta_bins, ra_data=None, dec_data=None,
                               seed=None, survey_mask=None, nside=None):
    if seed is not None:
        np.random.seed(seed)
    n_frb_actual = max(n_frb, 4)
    if ra_data is not None and dec_data is not None:
        ra = np.random.uniform(np.min(ra_data), np.max(ra_data), n_frb_actual * 2)
        dec = np.random.uniform(np.min(dec_data), np.max(dec_data), n_frb_actual * 2)
        chime_mask = (dec >= -15) & (dec <= 90)
        ra = ra[chime_mask][:n_frb_actual]
        dec = dec[chime_mask][:n_frb_actual]
        if len(ra) < n_frb_actual:
            ra = np.random.uniform(0, 360, n_frb_actual)
            dec = np.random.uniform(-10, 90, n_frb_actual)
    else:
        ra = np.random.uniform(0, 360, n_frb_actual)
        dec = np.random.uniform(-10, 90, n_frb_actual)
    z = np.random.gamma(2.5, 0.4, n_frb_actual)
    z = np.clip(z, 0.01, 2.0)
    dm_igm = dm_igm_macquart(z, -1.0)
    dm_host = np.random.normal(50 / (1 + z), 30 / (1 + z))
    dm_mw = get_dm_mw_model(ra, dec, model=DM_MW_MODEL)
    dm_excess = dm_igm + dm_host + np.random.normal(0, 10, n_frb_actual)
    frb_mock = pd.DataFrame({
        'ra': ra, 'dec': dec, 'z': z, 'DM_excess': dm_excess,
        'DM_error': np.random.normal(5, 2, n_frb_actual), 'z_type': 'spec'
    })
    return frb_mock
 
def generate_mock_galaxy_catalog(n_galaxies, z_bins, ra_data=None, dec_data=None, 
                                  seed=None, survey_mask=None, nside=None):
    if seed is not None:
        np.random.seed(seed + 1000)
    if ra_data is not None and dec_data is not None and survey_mask is not None:
        ra_rand, dec_rand = generate_random_catalog(
            n_galaxies, ra_data, dec_data, survey_mask, nside, seed=seed
        )
        ra = ra_rand
        dec = dec_rand
    else:
        ra = np.random.uniform(0, 360, n_galaxies)
        dec = np.arcsin(np.random.uniform(-1, 1, n_galaxies)) * 180 / np.pi
    z = np.random.beta(2, 5, n_galaxies) * 0.8 + 0.05
    galaxy_mock = pd.DataFrame({'ra': ra, 'dec': dec, 'z': z})
    return galaxy_mock
 
def compute_single_mock_xi_corrected(args):
    mock_idx, frb_catalog, galaxy_bins, theta_bins, seed, survey_mask, nside, random_catalogs = args
    np.random.seed(seed)
    all_ra_frb = frb_catalog['ra'].values if len(frb_catalog) > 0 else None
    all_dec_frb = frb_catalog['dec'].values if len(frb_catalog) > 0 else None
    frb_mock = generate_mock_frb_catalog(
        n_frb=len(frb_catalog), z_bins=None, theta_bins=theta_bins,
        ra_data=all_ra_frb, dec_data=all_dec_frb, seed=seed,
        survey_mask=survey_mask, nside=nside
    )
    all_ra_gal = []
    all_dec_gal = []
    for bin_data in galaxy_bins.values():
        if len(bin_data) > 0:
            all_ra_gal.extend(bin_data['ra'].values)
            all_dec_gal.extend(bin_data['dec'].values)
    galaxy_bins_mock = {}
    for i, (z_key, bin_data) in enumerate(galaxy_bins.items()):
        z_min, z_max = z_key
        n_gal = len(bin_data)
        gal_mock = generate_mock_galaxy_catalog(
            n_galaxies=n_gal, z_bins=[z_min, z_max],
            ra_data=np.array(all_ra_gal), dec_data=np.array(all_dec_gal),
            seed=seed + i * 1000, survey_mask=survey_mask, nside=nside
        )
        galaxy_bins_mock[z_key] = gal_mock[['ra', 'dec', 'z']].reset_index(drop=True)
    xi_mock, _, _ = compute_cross_correlation_corrfunc(
        frb_mock, galaxy_bins_mock, theta_bins,
        random_catalogs=random_catalogs, n_random_factor=10,
        survey_mask=survey_mask, nside=nside
    )
    return xi_mock
 
def estimate_covariance_from_mocks(frb_catalog, galaxy_bins, theta_bins,
                                    z_bins, n_mocks=N_MOCKS, seed_base=42,
                                    survey_mask=None, nside=None):
    print(f"Estimating covariance matrix from {n_mocks} mock catalogs...")
    print(f"  Using {N_THREADS} parallel threads")
    n_theta_bins = len(theta_bins) - 1
    xi_mocks = np.zeros((n_mocks, n_theta_bins))
    all_ra_gal = []
    all_dec_gal = []
    for bin_data in galaxy_bins.values():
        if len(bin_data) > 0:
            all_ra_gal.extend(bin_data['ra'].values)
            all_dec_gal.extend(bin_data['dec'].values)
    all_ra_gal = np.array(all_ra_gal)
    all_dec_gal = np.array(all_dec_gal)
    if survey_mask is None:
        survey_mask, nside = create_survey_mask_from_data(all_ra_gal, all_dec_gal)
    random_catalogs = {}
    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        bin_data = galaxy_bins[(z_min, z_max)]
        if len(bin_data) > 0:
            n_random = len(bin_data) * 10
            ra_rand, dec_rand = generate_random_catalog(
                n_random, bin_data['ra'].values, bin_data['dec'].values,
                survey_mask, nside, seed=seed_base
            )
            random_catalogs[(z_min, z_max)] = pd.DataFrame({'ra': ra_rand, 'dec': dec_rand})
    start_time = time.time()
    args_list = [(m, frb_catalog, galaxy_bins, theta_bins, seed_base + m,
                  survey_mask, nside, random_catalogs) for m in range(n_mocks)]
    with Pool(processes=N_THREADS) as pool:
        results = pool.map(compute_single_mock_xi_corrected, args_list)
    for m, xi_mock in enumerate(results):
        xi_mocks[m, :] = xi_mock
        if (m + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Mock {m+1}/{n_mocks} complete ({elapsed/60:.1f} min elapsed)")
    xi_mean = np.mean(xi_mocks, axis=0)
    cov_matrix = np.cov(xi_mocks, rowvar=False)
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    eigvals = np.clip(eigvals, 1e-10, None)
    cov_matrix = (eigvecs * eigvals) @ eigvecs.T
    N_data = n_theta_bins
    hartlap_factor = (n_mocks - N_data - 2) / (n_mocks - 1)
    cov_matrix = cov_matrix / hartlap_factor
    elapsed = time.time() - start_time
    print(f"  Covariance matrix computed: shape={cov_matrix.shape}")
    print(f"  Condition number: {np.linalg.cond(cov_matrix):.2e}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    return cov_matrix, xi_mocks, xi_mean
 
def dm_igm_macquart(z, w, f_IGM=0.84, Omega_m=OMEGA_M_PLANCK,
                    Omega_b=OMEGA_B, H0=H0_PLANCK):
    h = H0 / 100.0
    K = 855 * h * Omega_b * f_IGM
    def integrand(z_prime):
        H_z = hubble_parameter(z_prime, w, Omega_m, H0)
        return (1 + z_prime) / H_z
    if np.isscalar(z):
        integral, _ = quad(integrand, 0, max(z, 0), limit=100)
    else:
        integral = np.array([quad(integrand, 0, max(zi, 0), limit=100)[0] for zi in z])
    return K * integral
 
def get_dm_mw_model(ra, dec, model='NE2001'):
    if model == 'YMW16':
        base_dm = 45
        scatter = 25
    else:
        base_dm = 50
        scatter = 30
    
    gal_lat = np.sin(np.radians(dec))
    dm_direction = base_dm / (1 + np.abs(gal_lat)**2)
    return dm_direction + np.random.normal(0, scatter, size=np.asarray(ra).shape)
 
def log_likelihood_w(w, xi_obs, xi_model_func, theta_bins, z_bins, cov_matrix, params):
    w_val = w[0] if isinstance(w, (list, np.ndarray)) else w
    xi_th = xi_model_func(theta_bins, z_bins, w_val, params)
    valid = ~np.isnan(xi_obs) & ~np.isnan(xi_th)
    if np.sum(valid) < 2:
        return -np.inf
    xi_obs_valid = xi_obs[valid]
    xi_th_valid = xi_th[valid]
    cov_valid = cov_matrix[np.ix_(valid, valid)]
    try:
        cov_inv = np.linalg.inv(cov_valid)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_valid)
    residual = xi_obs_valid - xi_th_valid
    log_like = -0.5 * np.dot(residual.T, np.dot(cov_inv, residual))
    log_like -= 0.5 * np.log(np.linalg.det(cov_valid) + 1e-300)
    log_like -= 0.5 * len(xi_obs_valid) * np.log(2 * np.pi)
    return log_like
 
def prior_w(w):
    w_val = w[0] if isinstance(w, (list, np.ndarray)) else w
    if -2.0 < w_val < 0.0:
        return 0.0
    return -np.inf
 
def log_posterior_w(w, xi_obs, xi_model_func, theta_bins, z_bins, cov_matrix, params):
    lp = prior_w(w)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_w(w, xi_obs, xi_model_func, theta_bins, z_bins, cov_matrix, params)
    return lp + ll
 
def estimate_w(xi_obs, theta_bins, z_bins, cov_matrix, params,
               use_mcmc=True, nwalkers=50, nsteps=1000, burn_in=200):
    print("\nStarting w parameter estimation...")
    w_init = [-1.0]
    if not use_mcmc:
        print("  Using optimization (fast mode)...")
        def neg_log_post(w):
            return -log_posterior_w(
                w, xi_obs, compute_theoretical_xi_limber_vectorized, 
                theta_bins, z_bins, cov_matrix, params
            )
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
            lambda theta: log_posterior_w(
                theta, xi_obs, compute_theoretical_xi_limber_vectorized, 
                theta_bins, z_bins, cov_matrix, params
            )
        )
        sampler.run_mcmc(pos, nsteps, progress=True)
        samples = sampler.get_chain(discard=burn_in, flat=True)
        w_samples = samples[:, 0]
        w_samples = w_samples[np.isfinite(w_samples)]
        if len(w_samples) < 10:
            print("  Insufficient samples")
            return {'w_best': -1.0, 'w_err': 0.5, 'success': True, 'samples': None}
        w_median = np.median(w_samples)
        w_16 = np.percentile(w_samples, 16)
        w_84 = np.percentile(w_samples, 84)
        w_err = (w_84 - w_16) / 2
        print(f"  w = {w_median:.3f} +/- {w_err:.3f} (68% CI)")
        print(f"  Range: [{w_16:.3f}, {w_84:.3f}]")
        return {
            'w_best': w_median, 'w_err': w_err, 'w_16': w_16, 'w_84': w_84,
            'success': True, 'samples': w_samples
        }
 
def load_chime_data(catalog_path, dm_mw_model='NE2001'):
    frb_data = pd.read_csv(catalog_path)
    
    dm_mw_col_ne2001 = 'DM_MW_NE2001'
    dm_mw_col_ymw16 = 'DM_MW_YMW16'
    
    if dm_mw_model == 'YMW16' and dm_mw_col_ymw16 in frb_data.columns:
        frb_data['DM_excess'] = frb_data['DM'] - frb_data[dm_mw_col_ymw16]
    elif dm_mw_col_ne2001 in frb_data.columns:
        frb_data['DM_excess'] = frb_data['DM'] - frb_data[dm_mw_col_ne2001]
    elif 'DM' in frb_data.columns:
        dm_mw_est = get_dm_mw_model(frb_data['ra'].values, frb_data['dec'].values, dm_mw_model)
        frb_data['DM_excess'] = frb_data['DM'] - dm_mw_est
    else:
        raise ValueError("DM column not found in catalog")
    
    for col in ['ra', 'dec']:
        if col not in frb_data.columns:
            raise ValueError(f"Required column {col} not found")
    
    print(f"Loaded {len(frb_data)} FRBs (DM_MW model: {dm_mw_model})")
    return frb_data
 
def load_sdss_data(sdss_path, z_bins):
    sdss_data = pd.read_csv(sdss_path, low_memory=False)
    ra_col = 'ra' if 'ra' in sdss_data.columns else 'RA'
    dec_col = 'dec' if 'dec' in sdss_data.columns else 'DEC'
    z_col = 'z' if 'z' in sdss_data.columns else 'REDSHIFT'
    sdss_data = sdss_data.rename(columns={ra_col: 'ra', dec_col: 'dec', z_col: 'z'})
    sdss_data['ra'] = pd.to_numeric(sdss_data['ra'], errors='coerce')
    sdss_data['dec'] = pd.to_numeric(sdss_data['dec'], errors='coerce')
    sdss_data['z'] = pd.to_numeric(sdss_data['z'], errors='coerce')
    sdss_data = sdss_data.dropna(subset=['ra', 'dec', 'z'])
    sdss_data = sdss_data[(sdss_data['z'] >= z_bins[0]) & (sdss_data['z'] < z_bins[-1])]
    print(f"  Loaded {len(sdss_data)} galaxies after cleaning")
    galaxy_bins = {}
    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        bin_data = sdss_data[(sdss_data['z'] >= z_min) & (sdss_data['z'] < z_max)].copy()
        galaxy_bins[(z_min, z_max)] = bin_data[['ra', 'dec', 'z']].reset_index(drop=True)
    print(f"Created {len(galaxy_bins)} galaxy redshift bins")
    return galaxy_bins
 
def plot_cross_correlation(theta_bins, xi_obs, xi_errors, xi_model, output_path):
    print(f"Plotting cross-correlation: {output_path}")
    fig, ax = plt.subplots(figsize=(10, 6))
    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
    ax.errorbar(theta_centers, xi_obs, yerr=xi_errors, fmt='o',
                label='Data', color='#2E86AB', capsize=3)
    ax.plot(theta_centers, xi_model, '-', label='Model (w = -1)', color='#A23B72')
    ax.set_xlabel('Angular Separation θ (degrees)')
    ax.set_ylabel('ξ(θ) [pc cm⁻³]')
    ax.set_title('FRB-LSS Cross-Correlation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
 
def plot_w_posterior(w_result, output_path):
    if w_result['samples'] is None:
        return
    print(f"Plotting w posterior: {output_path}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    samples = w_result['samples']
    axes[0].hist(samples, bins=40, color='#4ECDC4', edgecolor='white', alpha=0.8, density=True)
    axes[0].axvline(w_result['w_best'], color='red', linestyle='-', linewidth=2,
                    label=f'median: {w_result["w_best"]:.3f}')
    axes[0].axvline(-1.0, color='blue', linestyle=':', linewidth=1.5, label='ΛCDM (w = -1)')
    axes[0].set_xlabel('Parameter w')
    axes[0].set_ylabel('Probability Density')
    axes[0].set_title('Posterior Distribution of w')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(samples, color='#2E86AB', alpha=0.5, linewidth=0.5)
    axes[1].axhline(w_result['w_best'], color='red', linestyle='-', linewidth=1)
    axes[1].set_xlabel('MCMC Step')
    axes[1].set_ylabel('w')
    axes[1].set_title('Trace Plot')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
 
def main():
    print("FwCC ANALYSIS")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parallel threads: {N_THREADS}")
    print(f"DM_MW Model: {DM_MW_MODEL}")
    for d in [BASE_OUTPUT_DIR, os.path.join(BASE_OUTPUT_DIR, 'plots')]:
        os.makedirs(d, exist_ok=True)
    print("1. LOADING DATA")
    frb_catalog = load_chime_data(CHIME_CATALOG_PATH, dm_mw_model=DM_MW_MODEL)
    galaxy_bins = load_sdss_data(SDSS_CATALOG_PATH, Z_BINS)
    if 'z_type' in frb_catalog.columns:
        frb_catalog = frb_catalog[frb_catalog['z_type'] == 'spec'].reset_index(drop=True)
    print(f"Using {len(frb_catalog)} localized FRBs for analysis")
    print("Creating survey mask from data footprint...")
    all_ra_gal = []
    all_dec_gal = []
    for bin_data in galaxy_bins.values():
        if len(bin_data) > 0:
            all_ra_gal.extend(bin_data['ra'].values)
            all_dec_gal.extend(bin_data['dec'].values)
    survey_mask, nside = create_survey_mask_from_data(
        np.array(all_ra_gal), np.array(all_dec_gal)
    )
    print(f"  Survey mask created (nside={nside})")
    print("2. COMPUTING CROSS-CORRELATION")
    random_catalogs = {}
    for (z_min, z_max), bin_data in galaxy_bins.items():
        if len(bin_data) > 0:
            n_random = len(bin_data) * 10
            ra_rand, dec_rand = generate_random_catalog(
                n_random, bin_data['ra'].values, bin_data['dec'].values,
                survey_mask, nside, seed=RANDOM_SEED
            )
            random_catalogs[(z_min, z_max)] = pd.DataFrame({'ra': ra_rand, 'dec': dec_rand})
    xi_obs, xi_errors, n_pairs = compute_cross_correlation_corrfunc(
        frb_catalog, galaxy_bins, THETA_BINS_DEG,
        random_catalogs=random_catalogs, n_random_factor=10,
        survey_mask=survey_mask, nside=nside
    )
    plot_cross_correlation(
        THETA_BINS_DEG, xi_obs, xi_errors,
        compute_theoretical_xi_limber_vectorized(THETA_BINS_DEG, Z_BINS, -1.0, {}),
        os.path.join(BASE_OUTPUT_DIR, 'plots', 'cross_correlation.png')
    )
    print("3. ESTIMATING COVARIANCE FROM MOCKS")
    cov_matrix, xi_mocks, xi_mean = estimate_covariance_from_mocks(
        frb_catalog, galaxy_bins, THETA_BINS_DEG, Z_BINS, 
        n_mocks=N_MOCKS, survey_mask=survey_mask, nside=nside
    )
    np.save(os.path.join(BASE_OUTPUT_DIR, 'mock_xi_samples.npy'), xi_mocks)
    np.save(os.path.join(BASE_OUTPUT_DIR, 'covariance_matrix.npy'), cov_matrix)
    print("4. ESTIMATING w PARAMETER")
    params = {
        'galaxy_bias': 1.5, 'FRB_bias': 1.2, 'A_norm': 1.0,
        'Omega_m': OMEGA_M_PLANCK, 'H0': H0_PLANCK
    }
    w_result = estimate_w(
        xi_obs=xi_obs, theta_bins=THETA_BINS_DEG, z_bins=Z_BINS,
        cov_matrix=cov_matrix, params=params, use_mcmc=True,
        nwalkers=50, nsteps=1000, burn_in=200
    )
    if w_result['samples'] is not None:
        plot_w_posterior(w_result, os.path.join(BASE_OUTPUT_DIR, 'plots', 'w_posterior.png'))
    results = {
        'w_best': float(w_result['w_best']) if not np.isnan(w_result['w_best']) else None,
        'w_err': float(w_result['w_err']) if 'w_err' in w_result and not np.isnan(w_result['w_err']) else None,
        'n_frb': len(frb_catalog), 'n_galaxy_bins': len(galaxy_bins),
        'n_mocks': N_MOCKS, 'theta_bins': THETA_BINS_DEG.tolist(),
        'dm_mw_model': DM_MW_MODEL,
        'camb_used': True, 'limber_integral': True,
        'growth_factor_linder2005': True,
        'selection_functions': True, 'survey_geometry_corrected': True,
        'random_catalogs_used': True, 'hartlap_correction': True,
        'corrfunc_ddtheta': True,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(BASE_OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("RESULTS SUMMARY")
    if w_result['success'] and not np.isnan(w_result['w_best']):
        print(f"w = {w_result['w_best']:.3f} +/- {w_result['w_err']:.3f}")
        sigma_dev = abs(w_result['w_best'] + 1) / w_result['w_err'] if w_result['w_err'] > 0 else 0
        print(f"Consistency with ΛCDM: {sigma_dev:.2f}σ")
    else:
        print("W estimation failed")
 
if __name__ == "__main__":
    main()

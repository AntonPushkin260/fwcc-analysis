#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FwCC: FRB-w Cross-Correlation Pipeline
=======================================

End-to-end pipeline for measuring the angular cross-correlation between
the extragalactic FRB dispersion measure (DM) field and foreground
galaxy overdensity maps, with application to CHIME/FRB Catalogue 2
and DESI DR1 tracers.

Reference:
    Pushkin, A. 2026, "FwCC: An FRB-Galaxy Cross-Correlation Pipeline
    and its Application to CHIME/FRB and DESI DR1", ApJ (submitted).
    arXiv: [to be added]

Key Features:
    - Event-weighted DM map construction
    - Pseudo-C_ell estimation via NaMaster (mask mode-coupling correction)
    - PCA compression of jackknife covariance
    - Covariance-marginalised Sellentin-Heavens likelihood
    - Cosmology-dependent N(z) precomputation on (w, Omega_m, f_IGM) grid
    - Host-DM calibration from localized anchor FRBs

Author: Anton Pushkin
License: MIT
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import logging
import time
import copy
import warnings
import multiprocessing
import pickle
import pathlib
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import eval_legendre, gammaln
from scipy.stats import norm, lognorm
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.stats import gaussian_kde
import emcee
from tqdm import tqdm

from astropy.constants import c as c_const, G as G_const, m_p
from astropy import units as u

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('default', category=RuntimeWarning)

# Required dependencies
try:
    import pymaster as nmt
except ImportError:
    raise ImportError("[FATAL] pymaster (NaMaster) is required. Install via: pip install pymaster")

try:
    import pyccl as ccl
except ImportError:
    raise ImportError("[FATAL] pyccl is required. Install via: pip install pyccl")

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output_FwCC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for subdir in ["matrices", "systematics", "covariance", "plots", "chains"]:
    (OUTPUT_DIR / subdir).mkdir(exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
CHAINS_DIR = OUTPUT_DIR / "chains"

# Input data paths
CATALOG_FRB = BASE_DIR / "final_catalog_v2.csv"
DESI_MAPS_DIR = BASE_DIR
DESI_NZ_DIR = BASE_DIR

# Physical and numerical constants
C_KM_S = 299792.458          # Speed of light [km/s]
PC_IN_CM = 3.085677581e18    # Parsec in centimeters

# Step 1 priors: host-DM distribution calibration
PRIORS_STEP1 = {
    'mu_host':    {'type': 'gauss',   'loc': 100.0, 'scale': 50.0},
    'sig_host':   {'type': 'lognorm', 'loc': 50.0,  'scale': 40.0},
    'gamma_host': {'type': 'gauss',   'loc': 0.0,   'scale': 1.0},
}

# Step 2 priors: cosmological and nuisance parameters
PRIORS_STEP2 = {
    'w':                 {'type': 'uniform', 'min': -2.0,  'max': 0.5},
    'Om0':               {'type': 'gauss',   'loc': 0.315, 'scale': 0.010},
    'f_IGM':             {'type': 'uniform', 'min': 0.60,  'max': 1.00},
    'b_bgs':             {'type': 'gauss',   'loc': 1.35,  'scale': 0.10},
    'b_lrg':             {'type': 'gauss',   'loc': 2.10,  'scale': 0.15},
    'sigma_loc_arcmin':  {'type': 'lognorm', 'loc': 10.0,  'scale': 5.0},
}

# Fixed cosmological and astrophysical parameters
FIXED_PARAMS = {
    'H0':                67.4,     # Hubble constant [km/s/Mpc]
    'ombh2':             0.0224,   # Physical baryon density
    'Y_p':               0.245,    # Primordial helium fraction
    'z_re_H':            7.68,     # Hydrogen reionization redshift
    'dz_re_H':           0.5,      # Hydrogen reionization width
    'z_re_He':           3.5,      # Helium reionization redshift
    'dz_re_He':          0.5,      # Helium reionization width
    'f_cosmic_scatter':  0.20,     # Fractional IGM sightline scatter
}

# Pipeline configuration
CONFIG = {
    'nside':                64,
    'lmax':                 3 * 64 - 1,
    'kmax_limber':          1.0,
    'nside_jk':             16,
    'target_n_jk_regions':  250,
    'use_lrg':              True,
    'mcmc_step1': {
        'n_walkers':      32,
        'max_steps':      8000,
        'burn_in':        2000,
        'seed':           42,
        'ndim':           3,
        'check_interval': 4000,
    },
    'mcmc_step2': {
        'n_walkers':      160,
        'max_steps':      25000,
        'burn_in':        2500,
        'seed':           43,
        'ndim':           6,
        'check_interval': 9000,
    },
}

# Global variable for multiprocessing theory arguments
GLOBAL_THEORY_ARGS = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging():
    """Configure pipeline logging to both file and stdout."""
    logger = logging.getLogger('FwCC_v52.0')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    fh = logging.FileHandler(OUTPUT_DIR / 'run.log', mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def compute_dm_norm_exact(h0, ombh2, Y_p=0.245):
    """Compute the DM normalization constant (Eq. 14 in the paper).
    
    DM_norm = 3 * c * H0 * Omega_b / (8 * pi * G * m_p)
    
    Returns value in units of pc/cm^3.
    """
    h = h0 / 100.0
    H0 = h0 * u.km / u.s / u.Mpc
    Omega_b = ombh2 / (h ** 2)
    prefactor = (3 * c_const * H0 * Omega_b) / (8 * np.pi * G_const * m_p)
    return prefactor.cgs.value / PC_IN_CM


def compute_fe_z(z_grid, Y_p=0.245, z_re_H=7.68, dz_re_H=0.5,
                 z_re_He=3.5, dz_re_He=0.5):
    """Compute cosmic free-electron fraction f_e(z).
    
    Models hydrogen and helium reionization with tanh transitions.
    """
    x_e_H = 0.5 * (1.0 + np.tanh((z_re_H - z_grid) / dz_re_H))
    x_e_He = 0.5 * (1.0 + np.tanh((z_re_He - z_grid) / dz_re_He))
    
    f_e_H = (1.0 - Y_p) * x_e_H
    f_e_He = (Y_p / 4.0) * 2.0 * x_e_He
    
    return np.clip(f_e_H + f_e_He, 1e-4, 1.0)


def make_z_grid(z_min=0.001, z_max=3.0, n_low=200, n_high=150,
                z_transition=0.5):
    """Construct a non-uniform redshift grid with higher resolution at low z."""
    return np.concatenate([
        np.linspace(z_min, z_transition, n_low, endpoint=False),
        np.linspace(z_transition, z_max, n_high)
    ])


def compute_split_rhat(chains, n_walkers):
    """Compute split-Rhat convergence diagnostic for MCMC chains."""
    n_steps = chains.shape[0]
    if n_steps < 8:
        return np.inf
    
    half = n_steps // 2
    chains_split = np.concatenate(
        [chains[:half, :, :], chains[half:2*half, :, :]],
        axis=1
    )
    chains_split = np.transpose(chains_split, (1, 0, 2))
    
    m, n = chains_split.shape[0], chains_split.shape[1]
    chain_means = np.mean(chains_split, axis=1)
    B = n * np.var(chain_means, axis=0, ddof=1)
    W = np.mean(np.var(chains_split, axis=1, ddof=1), axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        var_plus = (n - 1) / n * W + (1 / n) * B
        rhat = np.sqrt(var_plus / W)
    
    return np.nanmax(rhat)


def gaussian_beam_window(ell_array, sigma_loc_arcmin):
    """Gaussian beam window function for FRB localization uncertainty (Eq. 11).
    
    W_loc(ell) = exp[-ell*(ell+1)*sigma_loc^2 / 2]
    """
    sigma_loc_rad = (sigma_loc_arcmin / 60.0) * (np.pi / 180.0)
    return np.exp(-0.5 * ell_array * (ell_array + 1) * sigma_loc_rad**2)


def catalog_detrend(frb_df, weights=None, l_max=1, logger=None):
    """Remove monopole and dipole from FRB DM catalog to suppress exposure systematics.
    
    Fits and subtracts spherical-harmonic components using weighted
    least-squares regression with a cos^2(dec - dec_zenith) exposure model.
    """
    ra_rad = np.radians(frb_df['RA_deg'].values)
    dec_rad = np.radians(frb_df['Dec_deg'].values)
    dm_obs = frb_df['DM_excess'].values.copy()
    
    theta = np.pi/2 - dec_rad
    phi = ra_rad
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    basis_cols = ['monopole']
    B_list = [np.ones(len(frb_df))]
    
    if l_max >= 1:
        basis_cols.extend(['dipole_x', 'dipole_y', 'dipole_z'])
        B_list.extend([x, y, z])
    
    B = np.column_stack(B_list)
    
    # Default: analytic CHIME exposure model
    if weights is None:
        dec_zenith_rad = np.radians(45.0)
        w_raw = np.cos(dec_rad - dec_zenith_rad)**2
        w_raw = np.clip(w_raw, 0.05, 1.0)
        weights = w_raw
    
    sqrt_W = np.sqrt(weights)
    B_w = B * sqrt_W[:, None]
    dm_w = dm_obs * sqrt_W
    
    Q, R = np.linalg.qr(B_w, mode='reduced')
    coeffs_raw = np.linalg.solve(R, Q.T @ dm_w)
    
    dm_trend = B @ coeffs_raw
    dm_clean = dm_obs - dm_trend
    
    coeffs = {name: float(c) for name, c in zip(basis_cols, coeffs_raw)}
    
    if logger:
        logger.info("[DETREND] Catalog detrending complete:")
        for name, val in coeffs.items():
            logger.info(f"   {name:15s} = {val:+.4f} pc/cm^3")
    
    return dm_clean, coeffs


# ============================================================================
# PROBABILISTIC REDSHIFT DISTRIBUTION
# ============================================================================

def compute_pz_exact_2d(dm_obs, dm_err, z_grid, dm_igm_pred, f_scatter,
                        mu_host, sig_host, gamma_host):
    """Compute per-FRB posterior P(z | DM_obs) via 2D numerical integration.
    
    Marginalizes over IGM and host-galaxy DM contributions using
    Gamma (IGM) and log-normal (host) priors (Eq. 16).
    """
    N_x, N_y = 80, 80
    x_grid = np.linspace(1.0, 1500.0, N_x).astype(np.float32)
    y_grid = np.linspace(1.0, 1000.0, N_y).astype(np.float32)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
    
    n_frb = len(dm_obs)
    pz = np.zeros((n_frb, len(z_grid)), dtype=np.float32)
    
    z_dep = (1.0 + z_grid)**(-0.3)
    mu_igm = dm_igm_pred
    sig_igm = np.maximum(f_scatter * mu_igm * z_dep, 1e-3)
    
    k_igm = (mu_igm / sig_igm)**2
    theta_igm = sig_igm**2 / mu_igm
    
    mu_h_rest = mu_host * (1.0 + z_grid)**gamma_host
    sig_h_rest = sig_host * np.ones_like(z_grid)
    sigma_ln = np.sqrt(np.log(1.0 + (sig_h_rest / mu_h_rest)**2))
    mu_ln = np.log(mu_h_rest) - 0.5 * sigma_ln**2
    
    chunk_size = 50
    for i in range(0, n_frb, chunk_size):
        end = min(i + chunk_size, n_frb)
        dm_o = dm_obs[i:end, None, None, None]
        dm_e = dm_err[i:end, None, None, None]
        
        k_3d = k_igm[:, None, None]
        th_3d = theta_igm[:, None, None]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p_igm = ((k_3d - 1) * np.log(X[None, :, :])
                         - X[None, :, :] / th_3d
                         - k_3d * np.log(th_3d)
                         - gammaln(k_3d))
            p_igm = np.where(np.isfinite(log_p_igm), np.exp(log_p_igm), 0.0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p_host = (-np.log(Y[None, :, :])
                          - np.log(sigma_ln[:, None, None])
                          - 0.5 * np.log(2 * np.pi)
                          - 0.5 * ((np.log(Y[None, :, :]) - mu_ln[:, None, None])
                                   / sigma_ln[:, None, None])**2)
            p_host = np.where(np.isfinite(log_p_host), np.exp(log_p_host), 0.0)
        
        p_joint = p_igm * p_host
        
        prob_chunk = np.zeros((end - i, len(z_grid)), dtype=np.float32)
        for j_z in range(len(z_grid)):
            dp = X + Y / (1.0 + z_grid[j_z])
            diff = dm_o[:, 0, :, :] - dp[None, :, :]
            L = (np.exp(-0.5 * (diff / dm_e[:, 0, :, :])**2)
                 / (np.sqrt(2 * np.pi) * dm_e[:, 0, :, :]))
            prob_chunk[:, j_z] = np.sum(p_joint[j_z, :, :] * L,
                                         axis=(1, 2)) * dx * dy
        
        norm = trapezoid(prob_chunk, z_grid, axis=1)
        norm = np.where(norm > 0, norm, 1.0)
        pz[i:end] = prob_chunk / norm[:, None]
    
    return pz


def compute_anchor_likelihood(anchors, mu_host, sig_host, gamma_host,
                               dm_igm_pred, z_grid, f_scatter):
    """Compute log-likelihood for the localized anchor FRB sample (Eq. 20).
    
    Constrains host-DM nuisance parameters while marginalizing over
    the cosmology-dependent IGM contribution.
    """
    z_spec = anchors['z_spec'].values
    if len(z_spec) == 0:
        return 0.0
    
    dm_obs = anchors['DM_excess'].values
    dm_err = anchors['DM_obs_err'].values
    
    mu_igm_arr = np.interp(z_spec, z_grid, dm_igm_pred)
    sig_igm_arr = np.maximum(
        f_scatter * mu_igm_arr * (1.0 + z_spec)**(-0.3), 1e-3
    )
    
    mu_host_arr = np.maximum(mu_host * (1 + z_spec)**(gamma_host - 1), 1e-3)
    sig_host_arr = np.maximum(sig_host / (1 + z_spec), 1e-3)
    
    dm_host_grid = np.linspace(0.5, 500, 500)
    sigma_ln = np.sqrt(np.log(1 + (sig_host_arr / mu_host_arr)**2))
    mu_ln = np.log(mu_host_arr) - 0.5 * sigma_ln**2
    
    p_dm_host = lognorm.pdf(
        dm_host_grid[None, :],
        s=sigma_ln[:, None],
        scale=np.exp(mu_ln[:, None])
    )
    
    dm_total_grid = mu_igm_arr[:, None] + dm_host_grid[None, :]
    total_err = np.sqrt(sig_igm_arr[:, None]**2 + dm_err[:, None]**2)
    
    p_dm_obs_unnorm = norm.pdf(dm_obs[:, None], loc=dm_total_grid, scale=total_err)
    norm_factor = (norm.cdf(4000.0, loc=dm_total_grid, scale=total_err)
                   - norm.cdf(10.0, loc=dm_total_grid, scale=total_err))
    p_dm_obs = p_dm_obs_unnorm / np.maximum(norm_factor, 1e-10)
    
    return np.sum(np.log(
        trapezoid(p_dm_host * p_dm_obs, dm_host_grid, axis=1) + 1e-120
    ))


# ============================================================================
# MCMC LOG-PROBABILITY FUNCTIONS
# ============================================================================

def ln_prob_step1(p, anchors, z_grid, f_e_z, fixed, priors):
    """Log-posterior for Step 1: host-DM parameter calibration."""
    mu_host, sig_host, gamma_host = p
    
    # Hard bounds
    if not (10.0 < mu_host < 300.0 and
            5.0 < sig_host < 200.0 and
            -2.0 < gamma_host < 2.0):
        return -np.inf
    
    lp = 0.0
    lp += norm.logpdf(mu_host,
                      loc=priors['mu_host']['loc'],
                      scale=priors['mu_host']['scale'])
    lp += norm.logpdf(gamma_host,
                      loc=priors['gamma_host']['loc'],
                      scale=priors['gamma_host']['scale'])
    
    s = np.sqrt(np.log(1 + (priors['sig_host']['scale']
                            / priors['sig_host']['loc'])**2))
    sc = priors['sig_host']['loc'] / np.exp(0.5 * s**2)
    lp += lognorm.logpdf(sig_host, s=s, scale=sc)
    
    if not np.isfinite(lp):
        return -np.inf
    
    # Compute IGM prediction at fiducial f_IGM = 0.83
    h0 = fixed['H0']
    dm_norm = compute_dm_norm_exact(h0, fixed['ombh2'], fixed['Y_p'])
    Ez = np.sqrt(0.315 * (1 + z_grid)**3 + 0.685)
    dm_int = cumulative_trapezoid(
        f_e_z * (1 + z_grid)**2 / Ez, z_grid, initial=0.0
    )
    dm_igm_pred = dm_norm * 0.83 * np.maximum(dm_int, 1e-4)
    
    lp_anchors = compute_anchor_likelihood(
        anchors, mu_host, sig_host, gamma_host,
        dm_igm_pred, z_grid, fixed['f_cosmic_scatter']
    )
    
    if not np.isfinite(lp_anchors):
        return -np.inf
    
    return lp + lp_anchors


def _init_worker(theory_args):
    """Initializer for multiprocessing pool: share theory arguments."""
    global GLOBAL_THEORY_ARGS
    GLOBAL_THEORY_ARGS = theory_args


def _nz_grid_worker(args):
    """Worker for parallel N(z) grid precomputation."""
    (iw, iom, ifigm, w, Om0, f_IGM, z_grid, f_e_z, fixed,
     frb_arrays, mu_host_base, sig_host, gamma_host) = args
    
    h0 = fixed['H0']
    dm_norm = compute_dm_norm_exact(h0, fixed['ombh2'], fixed['Y_p'])
    Ez = np.sqrt(Om0 * (1 + z_grid)**3
                 + (1 - Om0) * (1 + z_grid)**(3 * (1 + w)))
    dm_int = cumulative_trapezoid(
        f_e_z * (1 + z_grid)**2 / Ez, z_grid, initial=0.0
    )
    dm_igm_pred = dm_norm * f_IGM * np.maximum(dm_int, 1e-4)
    
    pz_all = compute_pz_exact_2d(
        frb_arrays['dm_obs'], frb_arrays['dm_err'],
        z_grid, dm_igm_pred, fixed['f_cosmic_scatter'],
        mu_host_base, sig_host, gamma_host
    )
    nz = pz_all.sum(axis=0)
    norm_nz = trapezoid(nz, z_grid)
    if norm_nz > 0:
        nz /= norm_nz
    
    return iw, iom, ifigm, nz


def precompute_nz_grid(w_grid_nz, Om0_grid_nz, f_IGM_grid, z_grid, f_e_z,
                       fixed, frb_arrays, mu_host_base, sig_host, gamma_host,
                       logger, cache_suffix=""):
    """Precompute N(z) on a 3D grid over (w, Omega_m, f_IGM) with checkpointing."""
    cache_file = OUTPUT_DIR / f"nz_grid_cache_v52{cache_suffix}.npz"
    checkpoint_file = OUTPUT_DIR / f"nz_grid_checkpoint{cache_suffix}.pkl"
    
    if cache_file.exists():
        logger.info("[PRECOMPUTE] Loading cached N(z) grid...")
        data = np.load(cache_file, allow_pickle=True)
        return (data['nz_grid'], data['w_grid'],
                data['Om0_grid'], data['f_IGM_grid'])
    
    nz_grid = np.zeros((
        len(w_grid_nz), len(Om0_grid_nz), len(f_IGM_grid), len(z_grid)
    ))
    completed_tasks = set()
    
    if checkpoint_file.exists():
        logger.info("[PRECOMPUTE] Resuming from checkpoint...")
        with open(checkpoint_file, "rb") as f:
            saved_state = pickle.load(f)
            nz_grid = saved_state["nz_grid"]
            completed_tasks = saved_state["completed"]
    
    tasks = []
    for iw, w in enumerate(w_grid_nz):
        for iom, Om0 in enumerate(Om0_grid_nz):
            for ifigm, f_IGM in enumerate(f_IGM_grid):
                task_id = (iw, iom, ifigm)
                if task_id not in completed_tasks:
                    tasks.append((
                        iw, iom, ifigm, w, Om0, f_IGM,
                        z_grid, f_e_z, fixed, frb_arrays,
                        mu_host_base, sig_host, gamma_host
                    ))
    
    pending_tasks = len(tasks)
    
    if pending_tasks == 0:
        np.savez(cache_file, nz_grid=nz_grid, w_grid=w_grid_nz,
                 Om0_grid=Om0_grid_nz, f_IGM_grid=f_IGM_grid)
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        return nz_grid, w_grid_nz, Om0_grid_nz, f_IGM_grid
    
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"[PRECOMPUTE] Dispatching {pending_tasks} tasks "
                f"to {n_cores} cores...")
    
    checkpoint_interval = 50
    tasks_since_checkpoint = 0
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        iterator = pool.imap_unordered(_nz_grid_worker, tasks, chunksize=1)
        with tqdm(total=pending_tasks, desc="[PRECOMPUTE] N(z) grid",
                  unit="cosmo", ncols=100) as pbar:
            for iw, iom, ifigm, nz in iterator:
                nz_grid[iw, iom, ifigm, :] = nz
                completed_tasks.add((iw, iom, ifigm))
                tasks_since_checkpoint += 1
                pbar.update(1)
                
                if tasks_since_checkpoint >= checkpoint_interval:
                    with open(checkpoint_file, "wb") as f:
                        pickle.dump({
                            "nz_grid": nz_grid,
                            "completed": completed_tasks
                        }, f)
                    tasks_since_checkpoint = 0
    
    np.savez(cache_file, nz_grid=nz_grid, w_grid=w_grid_nz,
             Om0_grid=Om0_grid_nz, f_IGM_grid=f_IGM_grid)
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    logger.info("[PRECOMPUTE] N(z) grid completed and cached.")
    return nz_grid, w_grid_nz, Om0_grid_nz, f_IGM_grid


# ============================================================================
# POWER SPECTRUM PRECOMPUTATION
# ============================================================================

def precompute_pk_grid(w_grid, Om0_grid, z_grid, fixed, cfg, logger):
    """Precompute non-linear matter power spectrum P(k, z) grid using pyccl."""
    cache_file = OUTPUT_DIR / "pk_grid_cache_v52.npz"
    
    if cache_file.exists():
        logger.info("[PRECOMPUTE] Loading cached P(k) grid...")
        data = np.load(cache_file, allow_pickle=True)
        return (data['pk_grid'], data['w_grid'],
                data['Om0_grid'], data['k_arr'])
    
    logger.info(f"[PRECOMPUTE] Building P(k, a) grid "
                f"({len(w_grid)}x{len(Om0_grid)})...")
    
    h0 = fixed['H0']
    h = h0 / 100.0
    omb = fixed['ombh2'] / h**2
    sigma8_fid = 0.811
    
    k_arr = np.geomspace(1e-4, cfg['kmax_limber'], 200)
    a_arr = 1.0 / (1.0 + z_grid)
    pk_grid = np.zeros((
        len(w_grid), len(Om0_grid), len(z_grid), len(k_arr)
    ))
    
    for iw, w in enumerate(w_grid):
        for iom, Om0 in enumerate(Om0_grid):
            omc = Om0 - omb
            if omc < 0.001:
                pk_grid[iw, iom] = np.nan
                continue
            try:
                cosmo = ccl.Cosmology(
                    Omega_c=omc, Omega_b=omb, h=h,
                    sigma8=sigma8_fid, n_s=0.965,
                    w0=w, wa=0,
                    transfer_function='eisenstein_hu',
                    matter_power_spectrum='halofit'
                )
                pk_2d_raw = ccl.nonlin_matter_power(cosmo, k_arr, a_arr)
                if pk_2d_raw.shape == (len(k_arr), len(a_arr)):
                    pk_grid[iw, iom] = pk_2d_raw.T
                else:
                    pk_grid[iw, iom] = pk_2d_raw
            except Exception:
                pk_grid[iw, iom] = np.nan
    
    np.savez(cache_file, pk_grid=pk_grid, w_grid=w_grid,
             Om0_grid=Om0_grid, k_arr=k_arr)
    logger.info("[PRECOMPUTE] P(k) grid completed and cached.")
    return pk_grid, w_grid, Om0_grid, k_arr


# ============================================================================
# THEORETICAL CROSS-CORRELATION PREDICTION
# ============================================================================

def compute_theory_xi(w, Om0, f_IGM, b_bgs, b_lrg, sigma_loc_arcmin,
                      z_grid, f_e_z, nzi_bgs_tuple, nzi_lrg_tuple, n_frb_z,
                      pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr,
                      fixed, cfg, bpws, ells, eff_ells,
                      leg_matrix_eff, leg_matrix_blind=None):
    """Compute theoretical cross-correlation xi(theta) for given cosmology.
    
    Returns concatenated [xi_BGS(theta), xi_LRG(theta)] array.
    """
    bgs_x, bgs_y = nzi_bgs_tuple
    lrg_x, lrg_y = nzi_lrg_tuple
    
    nzi_bgs_func_local = interp1d(bgs_x, bgs_y, bounds_error=False, fill_value=0.0)
    nzi_lrg_func_local = interp1d(lrg_x, lrg_y, bounds_error=False, fill_value=0.0)
    
    h0 = fixed['H0']
    h = h0 / 100.0
    Ez = np.sqrt(Om0 * (1 + z_grid)**3
                 + (1 - Om0) * (1 + z_grid)**(3 * (1 + w)))
    chi_Mpc = (C_KM_S / h0) * cumulative_trapezoid(
        1.0 / Ez, z_grid, initial=0.0
    )
    chi_safe = np.maximum(chi_Mpc, 1.0)
    
    # Bilinear interpolation in (w, Omega_m) for P(k)
    w_clamped = np.clip(w, pk_w_grid.min(), pk_w_grid.max())
    Om0_clamped = np.clip(Om0, pk_Om0_grid.min(), pk_Om0_grid.max())
    
    iw = np.clip(np.searchsorted(pk_w_grid, w_clamped) - 1,
                 0, len(pk_w_grid) - 2)
    iom = np.clip(np.searchsorted(pk_Om0_grid, Om0_clamped) - 1,
                  0, len(pk_Om0_grid) - 2)
    
    w0, w1 = pk_w_grid[iw], pk_w_grid[iw + 1]
    Om0_0, Om0_1 = pk_Om0_grid[iom], pk_Om0_grid[iom + 1]
    tw = (w_clamped - w0) / (w1 - w0) if w1 != w0 else 0.0
    tOm0 = (Om0_clamped - Om0_0) / (Om0_1 - Om0_0) if Om0_1 != Om0_0 else 0.0
    
    pk_interp_base = ((1 - tw) * (1 - tOm0) * pk_grid_data[iw, iom]
                      + (1 - tw) * tOm0 * pk_grid_data[iw, iom + 1]
                      + tw * (1 - tOm0) * pk_grid_data[iw + 1, iom]
                      + tw * tOm0 * pk_grid_data[iw + 1, iom + 1])
    
    if np.any(np.isnan(pk_interp_base)):
        return None
    
    pk_interp = RegularGridInterpolator(
        (z_grid, pk_k_arr), pk_interp_base,
        bounds_error=False, fill_value=0.0
    )
    
    ell_full = np.arange(0, cfg['lmax'] + 1)
    Hz = h0 * Ez
    cdf_nfrb = cumulative_trapezoid(n_frb_z, z_grid, initial=0.0)
    
    W_DM_g_raw = (f_e_z * (1 + z_grid)**2 / Ez
                  * np.maximum(1.0 - cdf_nfrb, 0.0))
    W_DM_g_norm = trapezoid(W_DM_g_raw, z_grid)
    
    if W_DM_g_norm <= 0:
        return None
    
    W_DM_g = W_DM_g_raw / W_DM_g_norm
    
    def compute_cl_tracer_limber(bias, nzi_func):
        """Compute C_ell via Limber approximation for one tracer."""
        W_g = bias * nzi_func(z_grid)
        k_grid = np.clip(
            (ell_full[:, None] + 0.5) / chi_safe[None, :],
            1e-4, cfg['kmax_limber']
        )
        pts = np.empty((len(ell_full), len(z_grid), 2))
        pts[:, :, 0] = z_grid[None, :]
        pts[:, :, 1] = k_grid
        pk_grid = pk_interp(pts)
        prefactor_z = (Hz[None, :] / C_KM_S) / (chi_safe[None, :]**2)
        integrand = prefactor_z * W_DM_g[None, :] * W_g[None, :] * pk_grid
        return cumulative_trapezoid(
            integrand, z_grid, axis=1, initial=0.0
        )[:, -1]
    
    cl_bgs = compute_cl_tracer_limber(b_bgs, nzi_bgs_func_local)
    cl_lrg = compute_cl_tracer_limber(b_lrg, nzi_lrg_func_local)
    
    # Apply localization beam window
    beam_window = gaussian_beam_window(ell_full, sigma_loc_arcmin)
    cl_bgs *= beam_window
    cl_lrg *= beam_window
    
    # Bin in ell-space
    nmt_bins = nmt.NmtBin(bpws=bpws, ells=ells)
    cl_bgs_binned = nmt_bins.bin_cell(cl_bgs)
    cl_lrg_binned = nmt_bins.bin_cell(cl_lrg)
    
    if leg_matrix_blind is None:
        leg_matrix_blind = leg_matrix_eff
    
    # Transform to real space
    xi_bgs = np.sum(((2 * eff_ells + 1) / (4 * np.pi) * cl_bgs_binned)[:, None]
                    * leg_matrix_blind, axis=0)
    xi_lrg = np.sum(((2 * eff_ells + 1) / (4 * np.pi) * cl_lrg_binned)[:, None]
                    * leg_matrix_blind, axis=0)
    
    dm_norm = compute_dm_norm_exact(h0, fixed['ombh2'], fixed['Y_p'])
    
    return np.concatenate([xi_bgs, xi_lrg]) * dm_norm * f_IGM * W_DM_g_norm


# ============================================================================
# STEP 2 LOG-POSTERIOR
# ============================================================================

def ln_prob_step2(p, xi_obs_comp, cov_inv_comp, P, valid_bins, n_jk, priors,
                  debug_first, anchors, mu_host_base, sig_host, gamma_host,
                  z_grid, f_e_z, fixed, use_anchors=False):
    """Log-posterior for Step 2: cosmological parameter inference."""
    global GLOBAL_THEORY_ARGS
    theory_args = GLOBAL_THEORY_ARGS
    
    w, Om0, f_IGM, b_bgs, b_lrg, sigma_loc_arcmin = p
    
    # Prior on w
    w_prior = priors['w']
    if w_prior['type'] == 'uniform':
        if not (w_prior['min'] < w < w_prior['max']):
            return -np.inf
    else:
        if not (-3.0 < w < 1.0):
            return -np.inf
    
    # Prior on f_IGM
    fg_prior = priors['f_IGM']
    fg_min = fg_prior.get('min', 0.1)
    fg_max = fg_prior.get('max', 1.5)
    if not (fg_min < f_IGM < fg_max):
        return -np.inf
    
    # Hard bounds on nuisance parameters
    if not (0.05 < Om0 < 0.95): return -np.inf
    if not (0.1 < b_bgs < 4.0): return -np.inf
    if not (0.1 < b_lrg < 5.0): return -np.inf
    if not (1.0 < sigma_loc_arcmin < 30.0): return -np.inf
    
    # Compute log-prior
    lp = 0.0
    lp += norm.logpdf(Om0,
                      loc=priors['Om0']['loc'],
                      scale=priors['Om0']['scale'])
    lp += norm.logpdf(b_bgs,
                      loc=priors['b_bgs']['loc'],
                      scale=priors['b_bgs']['scale'])
    lp += norm.logpdf(b_lrg,
                      loc=priors['b_lrg']['loc'],
                      scale=priors['b_lrg']['scale'])
    
    if priors['w']['type'] == 'gauss':
        lp += norm.logpdf(w,
                          loc=priors['w']['loc'],
                          scale=priors['w']['scale'])
    
    if priors['f_IGM']['type'] == 'gauss':
        lp += norm.logpdf(f_IGM,
                          loc=priors['f_IGM']['loc'],
                          scale=priors['f_IGM']['scale'])
    
    mean_loc = priors['sigma_loc_arcmin']['loc']
    std_loc = priors['sigma_loc_arcmin']['scale']
    s_loc = np.sqrt(np.log(1 + (std_loc / mean_loc)**2))
    sc_loc = mean_loc / np.exp(0.5 * s_loc**2)
    lp += lognorm.logpdf(sigma_loc_arcmin, s=s_loc, scale=sc_loc)
    
    if not np.isfinite(lp):
        return -np.inf
    
    # Optional anchor likelihood
    lp_anchors = 0.0
    if use_anchors:
        h0 = fixed['H0']
        dm_norm = compute_dm_norm_exact(h0, fixed['ombh2'], fixed['Y_p'])
        Ez = np.sqrt(Om0 * (1 + z_grid)**3
                     + (1 - Om0) * (1 + z_grid)**(3 * (1 + w)))
        dm_int = cumulative_trapezoid(
            f_e_z * (1 + z_grid)**2 / Ez, z_grid, initial=0.0
        )
        dm_igm_pred = dm_norm * f_IGM * np.maximum(dm_int, 1e-4)
        lp_anchors = compute_anchor_likelihood(
            anchors, mu_host_base, sig_host, gamma_host,
            dm_igm_pred, z_grid, fixed['f_cosmic_scatter']
        )
        if not np.isfinite(lp_anchors):
            return -np.inf
    
    # Unpack theory arguments
    (z_grid_th, f_e_z_th, nzi_bgs_func, nzi_lrg_func,
     nz_grid_data, nz_w_grid, nz_Om0_grid, nz_f_IGM_grid,
     pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr,
     fixed_th, cfg, bpws, ells, eff_ells,
     leg_matrix_eff, *rest) = theory_args
    
    leg_matrix_blind = rest[0] if rest else leg_matrix_eff
    
    # Trilinear interpolation of N(z) at current cosmology
    w_clamped_nz = np.clip(w, nz_w_grid.min(), nz_w_grid.max())
    Om0_clamped_nz = np.clip(Om0, nz_Om0_grid.min(), nz_Om0_grid.max())
    f_IGM_clamped_nz = np.clip(f_IGM, nz_f_IGM_grid.min(), nz_f_IGM_grid.max())
    
    iw_nz = np.clip(np.searchsorted(nz_w_grid, w_clamped_nz) - 1,
                    0, len(nz_w_grid) - 2)
    iom_nz = np.clip(np.searchsorted(nz_Om0_grid, Om0_clamped_nz) - 1,
                     0, len(nz_Om0_grid) - 2)
    ifigm_nz = np.clip(np.searchsorted(nz_f_IGM_grid, f_IGM_clamped_nz) - 1,
                       0, len(nz_f_IGM_grid) - 2)
    
    w0, w1 = nz_w_grid[iw_nz], nz_w_grid[iw_nz + 1]
    Om0_0, Om0_1 = nz_Om0_grid[iom_nz], nz_Om0_grid[iom_nz + 1]
    f0, f1 = nz_f_IGM_grid[ifigm_nz], nz_f_IGM_grid[ifigm_nz + 1]
    
    tw = (w_clamped_nz - w0) / (w1 - w0) if w1 != w0 else 0.0
    tOm0 = (Om0_clamped_nz - Om0_0) / (Om0_1 - Om0_0) if Om0_1 != Om0_0 else 0.0
    tf = (f_IGM_clamped_nz - f0) / (f1 - f0) if f1 != f0 else 0.0
    
    n_frb_z_interp = np.zeros(len(z_grid_th))
    for iw_i in range(2):
        for iom_i in range(2):
            for if_i in range(2):
                weight = (((1 - tw) if iw_i == 0 else tw)
                          * ((1 - tOm0) if iom_i == 0 else tOm0)
                          * ((1 - tf) if if_i == 0 else tf))
                n_frb_z_interp += weight * nz_grid_data[
                    iw_nz + iw_i, iom_nz + iom_i, ifigm_nz + if_i, :
                ]
    
    norm_nfrb = trapezoid(n_frb_z_interp, z_grid_th)
    if norm_nfrb <= 0:
        return -np.inf
    n_frb_z_interp /= norm_nfrb
    
    # Compute theory vector
    xi_th_full = compute_theory_xi(
        w, Om0, f_IGM, b_bgs, b_lrg, sigma_loc_arcmin,
        z_grid_th, f_e_z_th, nzi_bgs_func, nzi_lrg_func, n_frb_z_interp,
        pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr,
        fixed_th, cfg, bpws, ells, eff_ells,
        leg_matrix_eff, leg_matrix_blind
    )
    
    if xi_th_full is None:
        return -np.inf
    
    xi_th_valid = xi_th_full[valid_bins]
    if not np.all(np.isfinite(xi_th_valid)):
        return -np.inf
    
    # Project into PCA-compressed basis
    xi_th_comp = P @ xi_th_valid
    diff = xi_obs_comp - xi_th_comp
    chi2_xi = diff @ cov_inv_comp @ diff
    
    if not np.isfinite(chi2_xi):
        return -np.inf
    
    chi2_xi = max(chi2_xi, 0.0)
    
    # Sellentin-Heavens likelihood (Eq. 22)
    lp_xi = -0.5 * (n_jk - 1) * np.log(1.0 + chi2_xi / (n_jk - 1.0))
    lp_total = lp + lp_xi + lp_anchors
    
    # Debug output for first evaluation
    dbg = debug_first[0]
    if dbg:
        print(f"[DEBUG] w={w:.3f}, Om0={Om0:.3f}, f_IGM={f_IGM:.3f}, "
              f"b_bgs={b_bgs:.3f}, b_lrg={b_lrg:.3f}, "
              f"sigma_loc={sigma_loc_arcmin:.1f}, chi2={chi2_xi:.2f}",
              flush=True)
        debug_first[0] = False
    
    return lp_total if np.isfinite(lp_total) else -np.inf


# ============================================================================
# JACKKNIFE COVARIANCE
# ============================================================================

def create_equal_area_jackknife(combined_mask, nside, nside_jk,
                                 target_n_regions, logger):
    """Construct approximately equal-area jackknife regions."""
    npix_jk = hp.nside2npix(nside_jk)
    valid_pix = np.where(combined_mask > 0.5)[0]
    theta_pix, phi_pix = hp.pix2ang(nside, valid_pix, nest=False)
    ipix_jk = hp.ang2pix(nside_jk, theta_pix, phi_pix, nest=False)
    
    jk_areas = np.bincount(ipix_jk, minlength=npix_jk)
    target_area = np.sum(jk_areas) / target_n_regions
    
    regions = np.full(len(valid_pix), -1, dtype=int)
    current_region = 0
    current_area = 0
    
    sorted_superpixels = np.argsort(jk_areas)
    for spix in sorted_superpixels:
        if jk_areas[spix] == 0:
            continue
        local_indices = np.where(ipix_jk == spix)[0]
        regions[local_indices] = current_region
        current_area += jk_areas[spix]
        if current_area >= target_area:
            current_region += 1
            current_area = 0
    
    unique_regions = np.unique(regions[regions >= 0])
    n_jk = len(unique_regions)
    
    logger.info(f"[JK] Equal-area Jackknife: {n_jk} regions")
    return regions, unique_regions


def _jackknife_worker(args):
    """Worker for parallel jackknife cross-correlation measurement."""
    (region, valid_pix, regions, combined_mask, npix, nside,
     pix_frb, dm_cosmic, n_data, bgs_delta, bgs_mask,
     lrg_delta, lrg_mask, bpws, ells, eff_ells,
     leg_matrix_eff, lambda_bar) = args
    
    mask_jk = combined_mask.copy()
    mask_jk[valid_pix[regions == region]] = 0.0
    
    if mask_jk.sum() == 0:
        return np.zeros(n_data)
    
    dm_sum = np.zeros(npix, dtype=np.float64)
    np.add.at(dm_sum, pix_frb, dm_cosmic)
    
    mask_jk_binary = (mask_jk > 0.1).astype(np.float64)
    
    # Event-weighted map construction (Eq. 8-10)
    frb_mask = mask_jk_binary.copy()
    
    if frb_mask.sum() < 10:
        return np.zeros(n_data)
    
    # Normalize by mean event density lambda_bar
    dm_map = dm_sum / lambda_bar
    
    # Subtract mean over valid pixels
    valid_full = frb_mask > 0.5
    dm_mean = np.sum(dm_map[valid_full]) / valid_full.sum()
    dm_map = (dm_map - dm_mean) * frb_mask
    
    nmt_bins = nmt.NmtBin(bpws=bpws, ells=ells)
    f_frb = nmt.NmtField(frb_mask, [dm_map])
    
    xi_out = []
    for delta, mask in [(bgs_delta, bgs_mask), (lrg_delta, lrg_mask)]:
        gal_mask = mask * mask_jk_binary
        gal_map = delta * gal_mask
        f_gal = nmt.NmtField(gal_mask, [gal_map])
        
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(f_frb, f_gal, nmt_bins)
        cl_c = nmt.compute_coupled_cell(f_frb, f_gal)
        cl_d = wsp.decouple_cell(cl_c)[0]
        
        xi = np.sum(((2 * eff_ells + 1) / (4 * np.pi) * cl_d)[:, None]
                    * leg_matrix_eff, axis=0)
        xi_out.extend(xi)
    
    return np.array(xi_out)


# ============================================================================
# PLOTTING ROUTINES
# ============================================================================

def plot_posterior_w(chains, results, logger):
    """Generate marginalized posterior distribution plot for w."""
    logger.info("[PLOT] Generating posterior distribution of w...")
    
    w_chain = chains[:, 0]
    w_med = results['w']['median']
    w_err = results['w']['error']
    
    q16, q50, q84 = np.percentile(w_chain, [16, 50, 84])
    q025, q975 = np.percentile(w_chain, [2.5, 97.5])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(w_chain, bins=50, density=True, alpha=0.4,
            color='#1f77b4', edgecolor='white', label='MCMC samples')
    
    kde = gaussian_kde(w_chain, bw_method=0.15)
    w_grid = np.linspace(w_chain.min(), w_chain.max(), 500)
    kde_pdf = kde(w_grid)
    ax.plot(w_grid, kde_pdf, color='#d62728', linewidth=2.5,
            label='KDE smoothed')
    
    mask_1sigma = (w_grid >= q16) & (w_grid <= q84)
    ax.fill_between(w_grid, 0, kde_pdf, where=mask_1sigma,
                    color='#1f77b4', alpha=0.3, label=r'$1\sigma$ (68%)')
    
    mask_2sigma = (w_grid >= q025) & (w_grid <= q975)
    ax.fill_between(w_grid, 0, kde_pdf,
                    where=mask_2sigma & ~mask_1sigma,
                    color='#1f77b4', alpha=0.15, label=r'$2\sigma$ (95%)')
    
    ax.axvline(w_med, color='k', linestyle='-', linewidth=2.0,
               label=f'Median = {w_med:.3f}')
    ax.axvline(q16, color='#2ca02c', linestyle='--',
               linewidth=1.5, alpha=0.8)
    ax.axvline(q84, color='#2ca02c', linestyle='--', linewidth=1.5,
               alpha=0.8, label=f'$1\\sigma$: [{q16:.3f}, {q84:.3f}]')
    ax.axvline(q025, color='#ff7f0e', linestyle=':',
               linewidth=1.5, alpha=0.7)
    ax.axvline(q975, color='#ff7f0e', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'$2\\sigma$: [{q025:.3f}, {q975:.3f}]')
    ax.axvline(-1.0, color='#9467bd', linestyle='-', linewidth=2.5,
               alpha=0.9, label=r'$\Lambda$CDM ($w = -1$)')
    
    textstr = '\n'.join([
        f'Equation of state $w$',
        f'Median: ${w_med:.3f} \\pm {w_err:.3f}$',
        f'$\\Delta w$ from $\\Lambda$CDM: {w_med - (-1.0):+.3f}',
        f'Tension with $w=-1$: '
        f'{abs(w_med - (-1.0))/w_err:.2f}$\\sigma$'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')
    
    ax.set_xlabel(r'Dark Energy Equation of State $w$', fontsize=14)
    ax.set_ylabel('Posterior Probability Density', fontsize=13)
    ax.set_title(
        r'Posterior Distribution of $w$ '
        r'(CHIME FRB $\times$ DESI Cross-Correlation)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    save_path = PLOTS_DIR / 'posterior_w.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_results_and_null_test(theta_bins, xi_mean, xi_mean_scr, cov_total,
                                best_fit_params, theory_args,
                                valid_bins, logger):
    """Generate data vs theory vs null-test comparison plot."""
    logger.info("[PLOT] Generating Data vs Theory vs Null-Test figure...")
    
    w, Om0, f_IGM, b_bgs, b_lrg, sigma_loc_arcmin = best_fit_params
    
    (z_grid, f_e_z, nzi_bgs_tuple, nzi_lrg_tuple,
     nz_grid_data, nz_w_grid, nz_Om0_grid, nz_f_IGM_grid,
     pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr,
     fixed, cfg, bpws, ells, eff_ells,
     leg_matrix_eff, *rest) = theory_args
    
    leg_matrix_blind = rest[0] if rest else leg_matrix_eff
    
    # Interpolate N(z) at best-fit cosmology
    w_clamped_nz = np.clip(w, nz_w_grid.min(), nz_w_grid.max())
    Om0_clamped_nz = np.clip(Om0, nz_Om0_grid.min(), nz_Om0_grid.max())
    f_IGM_clamped_nz = np.clip(f_IGM, nz_f_IGM_grid.min(), nz_f_IGM_grid.max())
    
    iw_nz = np.clip(np.searchsorted(nz_w_grid, w_clamped_nz) - 1,
                    0, len(nz_w_grid) - 2)
    iom_nz = np.clip(np.searchsorted(nz_Om0_grid, Om0_clamped_nz) - 1,
                     0, len(nz_Om0_grid) - 2)
    ifigm_nz = np.clip(np.searchsorted(nz_f_IGM_grid, f_IGM_clamped_nz) - 1,
                       0, len(nz_f_IGM_grid) - 2)
    
    w0, w1 = nz_w_grid[iw_nz], nz_w_grid[iw_nz + 1]
    Om0_0, Om0_1 = nz_Om0_grid[iom_nz], nz_Om0_grid[iom_nz + 1]
    f0, f1 = nz_f_IGM_grid[ifigm_nz], nz_f_IGM_grid[ifigm_nz + 1]
    
    tw = (w_clamped_nz - w0) / (w1 - w0) if w1 != w0 else 0.0
    tOm0 = (Om0_clamped_nz - Om0_0) / (Om0_1 - Om0_0) if Om0_1 != Om0_0 else 0.0
    tf = (f_IGM_clamped_nz - f0) / (f1 - f0) if f1 != f0 else 0.0
    
    n_frb_z_interp = np.zeros(len(z_grid))
    for iw_i in range(2):
        for iom_i in range(2):
            for if_i in range(2):
                weight = (((1 - tw) if iw_i == 0 else tw)
                          * ((1 - tOm0) if iom_i == 0 else tOm0)
                          * ((1 - tf) if if_i == 0 else tf))
                n_frb_z_interp += weight * nz_grid_data[
                    iw_nz + iw_i, iom_nz + iom_i, ifigm_nz + if_i, :
                ]
    
    norm_nfrb = trapezoid(n_frb_z_interp, z_grid)
    if norm_nfrb > 0:
        n_frb_z_interp /= norm_nfrb
    
    bgs_x, bgs_y = nzi_bgs_tuple
    lrg_x, lrg_y = nzi_lrg_tuple
    
    xi_th_full = compute_theory_xi(
        w, Om0, f_IGM, b_bgs, b_lrg, sigma_loc_arcmin,
        z_grid, f_e_z, (bgs_x, bgs_y), (lrg_x, lrg_y), n_frb_z_interp,
        pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr,
        fixed, cfg, bpws, ells, eff_ells,
        leg_matrix_eff, leg_matrix_blind
    )
    
    if xi_th_full is None:
        return
    
    err_bars = np.sqrt(np.diag(cov_total))
    theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2.0
    n_theta = len(theta_mid)
    
    xi_bgs_data = xi_mean[:n_theta]
    xi_lrg_data = xi_mean[n_theta:]
    xi_bgs_scr = xi_mean_scr[:n_theta]
    xi_lrg_scr = xi_mean_scr[n_theta:]
    xi_bgs_th = xi_th_full[:n_theta]
    xi_lrg_th = xi_th_full[n_theta:]
    err_bgs = err_bars[:n_theta]
    err_lrg = err_bars[n_theta:]
    
    theta_plot = theta_mid
    plt.style.use('default')
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # BGS panel
    ax1 = axes[0]
    ax1.errorbar(theta_plot, xi_bgs_data * theta_plot,
                 yerr=err_bgs * theta_plot, fmt='o', color='#1f77b4',
                 capsize=4, capthick=1.5, markersize=7,
                 label='Data (CHIME × DESI BGS)', zorder=5)
    ax1.errorbar(theta_plot, xi_bgs_scr * theta_plot,
                 yerr=err_bgs * theta_plot, fmt='D', color='dimgray',
                 alpha=0.9, capsize=3, markersize=6,
                 label='Null Test (DM Scrambled)', zorder=4)
    ax1.plot(theta_plot, xi_bgs_th * theta_plot, '-', color='#d62728',
             linewidth=2.0,
             label=f'Best-fit Model ($w={w:.2f}, f_{{IGM}}={f_IGM:.2f}$)',
             zorder=6)
    
    fitted_mask = valid_bins[:n_theta]
    if np.any(fitted_mask):
        ax1.axvspan(
            theta_plot[fitted_mask].min() * 0.9,
            theta_plot[fitted_mask].max() * 1.1,
            alpha=0.1, color='green', label='Fitted Scales'
        )
    
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.set_ylabel(r'$\theta \times \xi_{DM, BGS}(\theta)$', fontsize=13)
    ax1.set_title(r'Cross-Correlation: FRB DM $\times$ DESI BGS',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xscale('log')
    
    # LRG panel
    ax2 = axes[1]
    ax2.errorbar(theta_plot, xi_lrg_data * theta_plot,
                 yerr=err_lrg * theta_plot, fmt='o', color='#1f77b4',
                 capsize=4, capthick=1.5, markersize=7,
                 label='Data (CHIME × DESI LRG)', zorder=5)
    ax2.errorbar(theta_plot, xi_lrg_scr * theta_plot,
                 yerr=err_lrg * theta_plot, fmt='D', color='dimgray',
                 alpha=0.9, capsize=3, markersize=6,
                 label='Null Test (DM Scrambled)', zorder=4)
    ax2.plot(theta_plot, xi_lrg_th * theta_plot, '-', color='#d62728',
             linewidth=2.0, label='Best-fit Model', zorder=6)
    
    fitted_mask_lrg = valid_bins[n_theta:]
    if np.any(fitted_mask_lrg):
        ax2.axvspan(
            theta_plot[fitted_mask_lrg].min() * 0.9,
            theta_plot[fitted_mask_lrg].max() * 1.1,
            alpha=0.1, color='green', label='Fitted Scales'
        )
    
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_xlabel(r'Angular Separation $\theta$ [deg]', fontsize=13)
    ax2.set_ylabel(r'$\theta \times \xi_{DM, LRG}(\theta)$', fontsize=13)
    ax2.set_title(r'Cross-Correlation: FRB DM $\times$ DESI LRG',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xscale('log')
    
    plt.tight_layout()
    save_path = PLOTS_DIR / 'xi_data_theory_null.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class FwCCPipeline_v52_0:
    """Main FwCC pipeline orchestrator.
    
    Parameters
    ----------
    use_lrg : bool
        Include LRG tracer in joint analysis.
    use_anchors : bool
        Include anchor likelihood in Step 2.
    gauss_fig : bool
        Use Gaussian f_IGM prior for diagnostic runs.
    cache_suffix : str
        Suffix for cache files to allow multiple parallel runs.
    """
    
    def __init__(self, use_lrg=True, use_anchors=False,
                 gauss_fig=False, cache_suffix=""):
        self.cfg = copy.deepcopy(CONFIG)
        self.cfg['use_lrg'] = use_lrg
        self.fixed = copy.deepcopy(FIXED_PARAMS)
        self.logger = setup_logging()
        self.priors_s1 = copy.deepcopy(PRIORS_STEP1)
        self.priors_s2 = copy.deepcopy(PRIORS_STEP2)
        
        if gauss_fig:
            self.priors_s2['f_IGM'] = {
                'type': 'gauss', 'loc': 0.83, 'scale': 0.05,
                'min': 0.60, 'max': 1.00
            }
        
        self.use_anchors = use_anchors
        self.cache_suffix = cache_suffix
        self.nside = self.cfg['nside']
        self.npix = hp.nside2npix(self.nside)
        self.z_grid = make_z_grid()
    
    def setup_blinding(self):
        """Setup geometric blinding for confirmation-bias mitigation."""
        blind_file = OUTPUT_DIR / "blinding_offset.txt"
        
        if blind_file.exists():
            with open(blind_file, 'r') as f:
                self.blind_offset = float(f.read().strip())
        else:
            seed = int(time.time()) % (2**32)
            rng = np.random.default_rng(seed=seed)
            self.blind_offset = rng.uniform(-0.1, 0.1)
            with open(blind_file, 'w') as f:
                f.write(f"{self.blind_offset:.10f}\n")
        
        self.thetas_blind = self.thetas * (1.0 + self.blind_offset)
        self.leg_matrix_eff_blind = np.array(
            [eval_legendre(l, np.cos(self.thetas_blind))
             for l in self.eff_ells]
        )
        
        # For final analysis: unblind
        UNBLIND_FINAL = True
        if UNBLIND_FINAL:
            self.blind_offset = 0.0
            self.thetas_blind = self.thetas
            self.leg_matrix_eff_blind = self.leg_matrix_eff
            self.logger.info("[BLINDING] Final analysis is UNBLINDED.")
    
    def load_data(self):
        """Load FRB catalog and DESI maps, apply selection cuts."""
        self.logger.info("[INFO] Loading Data...")
        
        frb = pd.read_csv(CATALOG_FRB)
        for col in ['RA_deg', 'Dec_deg', 'z_spec', 'DM_excess',
                    'DM_obs_err', 'telescope']:
            if col in frb.columns:
                if col != 'telescope':
                    frb[col] = pd.to_numeric(frb[col], errors='coerce')
                else:
                    frb[col] = frb[col].astype(str)
        
        for col in ['is_anchor', 'use_xi']:
            if col in frb.columns:
                frb[col] = frb[col].astype(str).str.strip().str.lower().isin(
                    ['true', '1']
                )
            else:
                frb[col] = False
        
        self.anchors = frb[
            frb['is_anchor'] & frb['z_spec'].notna() & (frb['z_spec'] > 0.001)
        ].copy()
        if 'telescope' in self.anchors.columns:
            self.anchors = self.anchors[
                self.anchors['telescope'].isin(['ASKAP', 'DSA-110', 'CHIME-KKO'])
            ]
        
        self.anchors['DM_obs_err'] = np.sqrt(
            self.anchors['DM_obs_err']**2 + 20.0**2
        )
        
        self.frb_all = frb[frb['use_xi']].copy()
        self.frb_all['DM_obs_err'] = self.frb_all['DM_obs_err'].fillna(35.0)
        self.anchors['DM_obs_err'] = self.anchors['DM_obs_err'].fillna(35.0)
        self.frb_all = self.frb_all[self.frb_all['DM_excess'] > 10.0].copy()
        
        self.logger.info("=" * 60)
        self.logger.info("[SYSTEMATICS] Applying CATALOG-LEVEL DETRENDING")
        self.logger.info("=" * 60)
        
        dm_raw_for_nz = self.frb_all['DM_excess'].values.copy()
        dm_clean, self.detrend_coeffs = catalog_detrend(
            self.frb_all, weights=None, l_max=1, logger=self.logger
        )
        
        self.frb_all['DM_excess'] = dm_clean
        self.frb_all['DM_excess_raw_nz'] = dm_raw_for_nz
        
        self.logger.info(
            f"[INFO] {len(self.frb_all)} FRBs loaded & detrended. "
            f"{len(self.anchors)} Anchors."
        )
        
        self.logger.info("[INFO] Loading DESI Maps...")
        self.bgs_delta = hp.read_map(
            DESI_MAPS_DIR / "DESI_BGS_BRIGHT_z05_03_delta_g.fits"
        )
        self.bgs_mask = hp.read_map(
            DESI_MAPS_DIR / "DESI_BGS_BRIGHT_z05_03_mask.fits"
        ).astype(bool)
        self.lrg_delta = hp.read_map(
            DESI_MAPS_DIR / "DESI_LRG_z30_80_delta_g.fits"
        )
        self.lrg_mask = hp.read_map(
            DESI_MAPS_DIR / "DESI_LRG_z30_80_mask.fits"
        ).astype(bool)
        
        if hp.npix2nside(len(self.bgs_delta)) != self.nside:
            self.bgs_delta = hp.ud_grade(self.bgs_delta, nside_out=self.nside)
            self.bgs_mask = (hp.ud_grade(
                self.bgs_mask.astype(float), nside_out=self.nside
            ) > 0.5)
            self.lrg_delta = hp.ud_grade(self.lrg_delta, nside_out=self.nside)
            self.lrg_mask = (hp.ud_grade(
                self.lrg_mask.astype(float), nside_out=self.nside
            ) > 0.5)
        
        bgs_mask_f = self.bgs_mask.astype(np.float64)
        lrg_mask_f = self.lrg_mask.astype(np.float64)
        
        self.logger.info("[INFO] Using UNION mask")
        base_mask = ((bgs_mask_f + lrg_mask_f) > 0.5).astype(np.float64)
        
        theta, phi = hp.pix2ang(self.nside, np.arange(self.npix))
        dec_deg = 90.0 - np.degrees(theta)
        chime_mask = (dec_deg > -20.0).astype(np.float64)
        
        self.combined_mask = base_mask * chime_mask
        
        self.logger.info(
            f"[INFO] Keeping full catalog of {len(self.frb_all)} FRBs "
            f"for map making."
        )
        
        # Load DESI n(z)
        bgs_nz_data = np.loadtxt(DESI_NZ_DIR / "BGS_BRIGHT_NGC_nz.txt")
        lrg_nz_data = np.loadtxt(DESI_NZ_DIR / "LRG_NGC_nz.txt")
        
        self.bgs_nz_x = bgs_nz_data[:, 0]
        self.bgs_nz_y = (bgs_nz_data[:, 1]
                         / trapezoid(bgs_nz_data[:, 1], bgs_nz_data[:, 0]))
        self.lrg_nz_x = lrg_nz_data[:, 0]
        self.lrg_nz_y = (lrg_nz_data[:, 1]
                         / trapezoid(lrg_nz_data[:, 1], lrg_nz_data[:, 0]))
        
        self.nzi_bgs_func = interp1d(
            self.bgs_nz_x, self.bgs_nz_y,
            bounds_error=False, fill_value=0.0
        )
        self.nzi_lrg_func = interp1d(
            self.lrg_nz_x, self.lrg_nz_y,
            bounds_error=False, fill_value=0.0
        )
        
        # Setup ell binning
        ells = np.arange(2, self.cfg['lmax'] + 1)
        self.bpws = np.zeros(len(ells), dtype=int)
        bin_id, l_start = 0, 2
        for i, l in enumerate(ells):
            step = 10 if l < 300 else 50
            if l >= l_start + step:
                bin_id += 1
                l_start = l
            self.bpws[i] = bin_id
        
        self.nmt_bins = nmt.NmtBin(bpws=self.bpws, ells=ells)
        self.ells = ells
        self.eff_ells = self.nmt_bins.get_effective_ells()
        
        # Angular bins
        self.theta_bins = np.linspace(0.1, 2.0, 15)
        self.thetas = np.radians(
            (self.theta_bins[:-1] + self.theta_bins[1:]) / 2
        )
        self.leg_matrix_eff = np.array(
            [eval_legendre(l, np.cos(self.thetas)) for l in self.eff_ells]
        )
        
        self.n_data = 2 * len(self.thetas)
        
        self.f_e_z = compute_fe_z(
            self.z_grid, self.fixed['Y_p'],
            self.fixed['z_re_H'], self.fixed['dz_re_H'],
            self.fixed['z_re_He'], self.fixed['dz_re_He']
        )
        
        self.setup_blinding()
    
    def run_step1(self):
        """Step 1: Calibrate host-DM distribution from anchor FRBs."""
        self.logger.info("[STEP 1] Calibrating DM_host on Anchors")
        
        args = (self.anchors, self.z_grid, self.f_e_z,
                self.fixed, self.priors_s1)
        
        np.random.seed(self.cfg['mcmc_step1']['seed'])
        n_w = self.cfg['mcmc_step1']['n_walkers']
        
        p_center = np.array([100.0, 50.0, 0.0])
        p0_cand = np.array([
            p_center + 0.1 * np.random.randn(3) for _ in range(n_w)
        ])
        
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        
        with multiprocessing.Pool(n_cores) as pool:
            sampler = emcee.EnsembleSampler(
                n_w, 3, ln_prob_step1, args=args, pool=pool
            )
            
            state = p0_cand
            check_interval = self.cfg['mcmc_step1']['check_interval']
            max_steps = self.cfg['mcmc_step1']['max_steps']
            
            for i in range(0, max_steps, check_interval):
                state = sampler.run_mcmc(state, check_interval, progress=True)
                chains = sampler.get_chain(
                    discard=self.cfg['mcmc_step1']['burn_in'],
                    thin=10, flat=True
                )
                np.save(CHAINS_DIR / 'chains_step1.npy', chains)
                
                self.mu_host_base = np.median(chains[:, 0])
                self.sig_host = np.median(chains[:, 1])
                self.gamma_host = np.median(chains[:, 2])
                
                self.logger.info(
                    f"[RESULT] mu_host_base (f_IGM=0.83) = "
                    f"{self.mu_host_base:.2f}"
                )
                self.logger.info(
                    f"[RESULT] sig_host   = {self.sig_host:.2f}"
                )
                self.logger.info(
                    f"[RESULT] gamma_host = {self.gamma_host:.3f}"
                )
    
    def run_step2(self):
        """Step 2: Main cosmological parameter inference."""
        self.logger.info("[STEP 2] Measuring w & f_IGM")
        
        # Precompute P(k) grid
        w_grid_pk = np.linspace(-2.6, 1.1, 20)
        Om0_grid_pk = np.linspace(0.14, 0.51, 20)
        pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr = precompute_pk_grid(
            w_grid_pk, Om0_grid_pk, self.z_grid,
            self.fixed, self.cfg, self.logger
        )
        
        # Precompute N(z) grid
        w_grid_nz = np.linspace(-2.0, 0.5, 15)
        Om0_grid_nz = np.linspace(0.15, 0.50, 10)
        f_IGM_grid = np.linspace(0.60, 1.00, 5)
        
        frb_arrays = {
            'dm_obs': self.frb_all['DM_excess_raw_nz'].values.astype(np.float64),
            'dm_err': self.frb_all['DM_obs_err'].values.astype(np.float64),
        }
        
        nz_grid_data, nz_w_grid, nz_Om0_grid, nz_f_IGM_grid = precompute_nz_grid(
            w_grid_nz, Om0_grid_nz, f_IGM_grid, self.z_grid, self.f_e_z,
            self.fixed, frb_arrays,
            self.mu_host_base, self.sig_host, self.gamma_host,
            self.logger, self.cache_suffix
        )
        
        # Construct event-weighted DM map inputs
        dm_cosmic = self.frb_all['DM_excess'].values
        pix_frb = hp.ang2pix(
            self.nside,
            np.radians(90.0 - self.frb_all['Dec_deg'].values),
            np.radians(self.frb_all['RA_deg'].values)
        )
        
        # Mean event density (Eq. 9)
        N_valid_combined = int(np.sum(self.combined_mask > 0.5))
        lambda_bar = len(self.frb_all) / float(N_valid_combined)
        self.logger.info(
            f"[MAP MAKING] Event-weighted scheme: "
            f"lambda_bar = {lambda_bar:.4f}"
        )
        
        # Create jackknife regions
        regions, unique_regions = create_equal_area_jackknife(
            self.combined_mask, self.nside, self.cfg['nside_jk'],
            self.cfg['target_n_jk_regions'], self.logger
        )
        self.regions = regions
        self.unique_regions = unique_regions
        
        # Run jackknife in parallel
        worker_args = [
            (r, np.where(self.combined_mask > 0.5)[0], regions,
             self.combined_mask, self.npix, self.nside,
             pix_frb, dm_cosmic, self.n_data,
             self.bgs_delta, self.bgs_mask,
             self.lrg_delta, self.lrg_mask,
             self.bpws, self.ells, self.eff_ells,
             self.leg_matrix_eff, lambda_bar)
            for r in unique_regions
        ]
        
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        with multiprocessing.Pool(n_cores) as pool:
            xi_jk = np.array(list(pool.map(_jackknife_worker, worker_args)))
            xi_jk = xi_jk[np.any(xi_jk != 0, axis=1)]
            n_jk = len(xi_jk)
        
        # Scrambled-DM null test
        dm_scrambled = np.random.permutation(dm_cosmic)
        worker_args_scr = [
            (r, np.where(self.combined_mask > 0.5)[0], regions,
             self.combined_mask, self.npix, self.nside,
             pix_frb, dm_scrambled, self.n_data,
             self.bgs_delta, self.bgs_mask,
             self.lrg_delta, self.lrg_mask,
             self.bpws, self.ells, self.eff_ells,
             self.leg_matrix_eff, lambda_bar)
            for r in unique_regions
        ]
        
        with multiprocessing.Pool(n_cores) as pool:
            xi_scr = np.array(list(pool.map(_jackknife_worker, worker_args_scr)))
            xi_scr = xi_scr[np.any(xi_scr != 0, axis=1)]
        
        # Compute filling factor (diagnostic, not used in amplitude)
        n_frb_map_local = np.zeros(self.npix, dtype=np.float64)
        np.add.at(n_frb_map_local, pix_frb, 1.0)
        valid_combined_bool = self.combined_mask > 0.5
        N_pix_combined = int(np.sum(valid_combined_bool))
        
        occupied_bool = n_frb_map_local[valid_combined_bool] > 0
        N_occupied = int(np.sum(occupied_bool))
        
        xi_mean_raw = np.mean(xi_jk, axis=0)
        xi_mean = xi_mean_raw.copy()
        
        diff_raw = xi_jk - xi_mean_raw
        cov_jk_raw = (n_jk - 1) / n_jk * np.dot(diff_raw.T, diff_raw)
        cov_jk = cov_jk_raw.copy()
        
        xi_mean_scr_raw = np.mean(xi_scr, axis=0)
        xi_mean_scr = xi_mean_scr_raw.copy()
        
        # Diagnostic output
        print("\n" + "=" * 70)
        print("DEEP DIAGNOSTICS: DATA vs NULL vs THEORY")
        print("=" * 70)
        n_theta = len(self.thetas)
        theta_mid = np.degrees(self.thetas)
        print(f"\n{'Theta [deg]':<12} | {'xi_data (BGS)':<14} | "
              f"{'xi_null (BGS)':<14} | {'xi_data (LRG)':<14} | "
              f"{'xi_null (LRG)':<14}")
        print("-" * 75)
        for i in range(n_theta):
            print(f"{theta_mid[i]:<12.3f} | {xi_mean[i]:<14.5e} | "
                  f"{xi_mean_scr[i]:<14.5e} | "
                  f"{xi_mean[n_theta + i]:<14.5e} | "
                  f"{xi_mean_scr[n_theta + i]:<14.5e}")
        print("=" * 70)
        
        cov_total = cov_jk
        
        # Scale cuts: 0.37° < theta < 1.46°
        valid_bins = np.zeros(2 * n_theta, dtype=bool)
        scale_cut = (theta_mid > 0.37) & (theta_mid < 1.46)
        valid_bins[:n_theta] = scale_cut
        if self.cfg['use_lrg']:
            valid_bins[n_theta:] = scale_cut
        else:
            valid_bins[n_theta:] = False
        
        self.logger.info(f"[SCALE CUTS] Article range: 0.37° < theta < 1.46°")
        self.logger.info(
            f"[SCALE CUTS] Valid BGS bins: {np.sum(valid_bins[:n_theta])}"
        )
        self.logger.info(
            f"[SCALE CUTS] Valid LRG bins: {np.sum(valid_bins[n_theta:])}"
        )
        
        xi_obs_cut = xi_mean[valid_bins]
        xi_scr_cut = xi_mean_scr[valid_bins]
        cov_cut = cov_total[np.ix_(valid_bins, valid_bins)]
        
        # PCA compression
        reg_term = 1e-4 * np.trace(cov_cut) / len(cov_cut)
        cov_cut_reg = cov_cut + reg_term * np.eye(len(cov_cut))
        eigvals, eigvecs = np.linalg.eigh(cov_cut_reg)
        
        self.logger.info(f"[PCA DIAG] Eigenvalues (top 10): {eigvals[:10]}")
        self.logger.info(
            f"[PCA DIAG] Ratio max/min: {eigvals[0] / eigvals[-1]:.2e}"
        )
        
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        keep_mask = eigvals > 1e-3 * eigvals.max()
        if not np.any(keep_mask):
            keep_mask[0] = True
        
        max_allowed_modes = max(1, n_jk - 2)
        if np.sum(keep_mask) > max_allowed_modes:
            keep_mask[max_allowed_modes:] = False
        
        n_modes = np.sum(keep_mask)
        self.logger.info(
            f"[PCA] Retained {n_modes} modes "
            f"(max allowed: {max_allowed_modes})"
        )
        
        P = eigvecs[:, keep_mask].T
        xi_obs_comp = P @ xi_obs_cut
        xi_scr_comp = P @ xi_scr_cut
        cov_comp = P @ cov_cut_reg @ P.T
        cov_inv_comp = np.linalg.inv(cov_comp)
        
        # Null test chi-squared
        diff_scr = xi_obs_comp - xi_scr_comp
        chi2_null = diff_scr @ cov_inv_comp @ diff_scr
        self.logger.info(
            f"[NULL TESTS] Scrambled DM Chi2 = {chi2_null:.2f} "
            f"(Expected ~ {n_modes})"
        )
        
        if n_jk < n_modes + 10:
            raise ValueError("Degenerate covariance matrix.")
        
        # Amplitude test against Lambda-CDM
        self.logger.info("\n" + "=" * 70)
        self.logger.info("AMPLITUDE TEST (A_hat) vs Lambda-CDM")
        self.logger.info("=" * 70)
        
        w_lcdm, Om0_lcdm, f_IGM_lcdm = -1.0, 0.315, 0.83
        b_bgs_fid, b_lrg_fid = 1.35, 2.10
        sigma_loc_fid = 10.0
        
        w_clamped_nz = np.clip(w_lcdm, nz_w_grid.min(), nz_w_grid.max())
        Om0_clamped_nz = np.clip(Om0_lcdm, nz_Om0_grid.min(), nz_Om0_grid.max())
        f_IGM_clamped_nz = np.clip(
            f_IGM_lcdm, nz_f_IGM_grid.min(), nz_f_IGM_grid.max()
        )
        
        iw_nz = np.clip(np.searchsorted(nz_w_grid, w_clamped_nz) - 1,
                        0, len(nz_w_grid) - 2)
        iom_nz = np.clip(np.searchsorted(nz_Om0_grid, Om0_clamped_nz) - 1,
                         0, len(nz_Om0_grid) - 2)
        ifigm_nz = np.clip(np.searchsorted(nz_f_IGM_grid, f_IGM_clamped_nz) - 1,
                           0, len(nz_f_IGM_grid) - 2)
        
        w0, w1 = nz_w_grid[iw_nz], nz_w_grid[iw_nz + 1]
        Om0_0, Om0_1 = nz_Om0_grid[iom_nz], nz_Om0_grid[iom_nz + 1]
        f0, f1 = nz_f_IGM_grid[ifigm_nz], nz_f_IGM_grid[ifigm_nz + 1]
        
        tw = (w_clamped_nz - w0) / (w1 - w0) if w1 != w0 else 0.0
        tOm0 = (Om0_clamped_nz - Om0_0) / (Om0_1 - Om0_0) if Om0_1 != Om0_0 else 0.0
        tf = (f_IGM_clamped_nz - f0) / (f1 - f0) if f1 != f0 else 0.0
        
        n_frb_z_interp_lcdm = np.zeros(len(self.z_grid))
        for iw_i in range(2):
            for iom_i in range(2):
                for if_i in range(2):
                    weight = (((1 - tw) if iw_i == 0 else tw)
                              * ((1 - tOm0) if iom_i == 0 else tOm0)
                              * ((1 - tf) if if_i == 0 else tf))
                    n_frb_z_interp_lcdm += weight * nz_grid_data[
                        iw_nz + iw_i, iom_nz + iom_i, ifigm_nz + if_i, :
                    ]
        
        norm_nfrb = trapezoid(n_frb_z_interp_lcdm, self.z_grid)
        if norm_nfrb > 0:
            n_frb_z_interp_lcdm /= norm_nfrb
        
        xi_th_lcdm_full = compute_theory_xi(
            w_lcdm, Om0_lcdm, f_IGM_lcdm, b_bgs_fid, b_lrg_fid, sigma_loc_fid,
            self.z_grid, self.f_e_z,
            (self.bgs_nz_x, self.bgs_nz_y),
            (self.lrg_nz_x, self.lrg_nz_y),
            n_frb_z_interp_lcdm,
            pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr,
            self.fixed, self.cfg, self.bpws, self.ells, self.eff_ells,
            self.leg_matrix_eff, self.leg_matrix_eff_blind
        )
        
        A_hat = None
        chi2_lcdm = None
        
        if xi_th_lcdm_full is not None:
            xi_th_lcdm_valid = xi_th_lcdm_full[valid_bins]
            xi_th_lcdm_comp = P @ xi_th_lcdm_valid
            
            num = xi_obs_comp @ cov_inv_comp @ xi_th_lcdm_comp
            den = xi_th_lcdm_comp @ cov_inv_comp @ xi_th_lcdm_comp
            A_hat = num / den
            A_err = 1.0 / np.sqrt(den)
            chi2_lcdm = ((xi_obs_comp - xi_th_lcdm_comp)
                         @ cov_inv_comp
                         @ (xi_obs_comp - xi_th_lcdm_comp))
            
            self.logger.info("   Lambda-CDM Theory Vector computed.")
            self.logger.info(f"   A_hat (Data / Lambda-CDM) = "
                             f"{A_hat:.3f} +/- {A_err:.3f}")
            self.logger.info(
                f"   Chi2(Lambda-CDM) = {chi2_lcdm:.2f} (ndof = {n_modes})"
            )
        else:
            self.logger.warning(
                "   Failed to compute Lambda-CDM theory vector."
            )
        
        # Pack theory arguments for MCMC workers
        theory_args = (
            self.z_grid, self.f_e_z,
            (self.bgs_nz_x, self.bgs_nz_y),
            (self.lrg_nz_x, self.lrg_nz_y),
            nz_grid_data, nz_w_grid, nz_Om0_grid, nz_f_IGM_grid,
            pk_grid_data, pk_w_grid, pk_Om0_grid, pk_k_arr,
            self.fixed, self.cfg, self.bpws, self.ells, self.eff_ells,
            self.leg_matrix_eff, self.leg_matrix_eff_blind
        )
        
        small_args = (
            xi_obs_comp, cov_inv_comp, P, valid_bins, n_jk,
            self.priors_s2, [False],
            self.anchors, self.mu_host_base, self.sig_host, self.gamma_host,
            self.z_grid, self.f_e_z, self.fixed, self.use_anchors
        )
        
        # Run MCMC
        np.random.seed(self.cfg['mcmc_step2']['seed'])
        n_w = self.cfg['mcmc_step2']['n_walkers']
        p0_cand = []
        
        for _ in range(n_w):
            w_prior = self.priors_s2['w']
            if w_prior['type'] == 'gauss':
                w = np.random.normal(w_prior['loc'], w_prior['scale'])
                w = np.clip(w, -2.0, 0.5)
            else:
                w = np.random.uniform(
                    w_prior['min'] + 0.1, w_prior['max'] - 0.1
                )
            
            Om0 = np.random.normal(0.315, 0.01)
            
            if self.priors_s2['f_IGM']['type'] == 'gauss':
                f_IGM = np.random.normal(
                    self.priors_s2['f_IGM']['loc'],
                    self.priors_s2['f_IGM']['scale']
                )
                f_IGM = np.clip(f_IGM, 0.65, 0.95)
            else:
                f_IGM = np.random.uniform(
                    self.priors_s2['f_IGM']['min'] + 0.05,
                    self.priors_s2['f_IGM']['max'] - 0.05
                )
            
            b_bgs = np.random.normal(1.35, 0.10)
            b_lrg = np.random.normal(2.10, 0.15)
            sigma_loc = np.random.lognormal(np.log(10.0), 0.2)
            p0_cand.append([w, Om0, f_IGM, b_bgs, b_lrg, sigma_loc])
        
        moves = [
            (emcee.moves.DEMove(), 0.5),
            (emcee.moves.DESnookerMove(), 0.5),
        ]
        
        with multiprocessing.Pool(
            n_cores, initializer=_init_worker, initargs=(theory_args,)
        ) as pool:
            sampler = emcee.EnsembleSampler(
                n_w, 6, ln_prob_step2, args=small_args,
                pool=pool, moves=moves
            )
            
            state = p0_cand
            check_interval = self.cfg['mcmc_step2']['check_interval']
            max_steps = self.cfg['mcmc_step2']['max_steps']
            prev_tau_max = None
            
            for i in range(0, max_steps, check_interval):
                state = sampler.run_mcmc(state, check_interval, progress=True)
                
                try:
                    tau = sampler.get_autocorr_time(quiet=True)
                    rhat = compute_split_rhat(
                        sampler.get_chain(flat=False), n_w
                    )
                    tau_max = np.max(tau)
                    
                    tau_stable = False
                    if prev_tau_max is not None:
                        tau_stable = (
                            abs(tau_max - prev_tau_max)
                            / max(prev_tau_max, 1) < 0.1
                        )
                    prev_tau_max = tau_max
                    
                    self.logger.info(
                        f"[STEP 2 MCMC] iter={sampler.iteration} | "
                        f"Rhat={rhat:.4f} | Tau_max={tau_max:.1f} | "
                        f"Stable={tau_stable}"
                    )
                    
                    if (np.all(sampler.iteration > 50 * tau)
                            and rhat < 1.01 and tau_stable):
                        self.logger.info(
                            f"[STEP 2 MCMC] CONVERGED at iteration "
                            f"{sampler.iteration}! Stopping early."
                        )
                        break
                except emcee.autocorr.AutocorrError:
                    self.logger.info(
                        f"[STEP 2 MCMC] iter={sampler.iteration} | "
                        f"(Tau not yet converged...)"
                    )
                except Exception:
                    pass
        
        chains = sampler.get_chain(
            discard=self.cfg['mcmc_step2']['burn_in'],
            thin=20, flat=True
        )
        np.save(CHAINS_DIR / 'chains_step2.npy', chains)
        np.save(OUTPUT_DIR / 'covariance/cov_jk.npy', cov_jk)
        
        # Compute posterior summaries
        labels = ['w', 'Om0', 'f_IGM', 'b_BGS', 'b_LRG', 'sigma_loc_arcmin']
        results = {}
        best_fit_params = []
        
        w_chain = chains[:, 0]
        kde = gaussian_kde(w_chain, bw_method=0.15)
        w_grid = np.linspace(w_chain.min(), w_chain.max(), 500)
        kde_pdf = kde(w_grid)
        mode_w = w_grid[np.argmax(kde_pdf)]
        
        if HAS_ARVIZ:
            hdi_w = az.hdi(w_chain, hdi_prob=0.68)
        else:
            hdi_w = np.percentile(w_chain, [16, 84])
        
        for i, label in enumerate(labels):
            med = np.median(chains[:, i])
            err = (np.percentile(chains[:, i], 84)
                   - np.percentile(chains[:, i], 16)) / 2
            
            if label == 'w':
                results[label] = {
                    'median': float(med),
                    'error': float(err),
                    'mode': float(mode_w),
                    'hpd_lower': float(hdi_w[0]),
                    'hpd_upper': float(hdi_w[1]),
                }
            else:
                results[label] = {
                    'median': float(med),
                    'error': float(err),
                }
            
            best_fit_params.append(float(med))
            self.logger.info(
                f"[RESULT] {label:12s} = {med:.4f} +/- {err:.4f}"
            )
        
        results['detrend_coeffs'] = self.detrend_coeffs
        results['config'] = {
            'use_lrg': self.cfg['use_lrg'],
            'use_anchors': self.use_anchors,
            'n_modes': int(n_modes),
            'A_hat': float(A_hat) if A_hat is not None else None,
            'chi2_lcdm': float(chi2_lcdm) if chi2_lcdm is not None else None,
        }
        
        with open(OUTPUT_DIR / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate plots
        if HAS_CORNER:
            try:
                plot_ranges = [
                    (-2.0, 0.5), (0.15, 0.50), (0.60, 1.00),
                    (0.8, 2.0), (1.0, 3.5), (1.0, 30.0),
                ]
                fig = corner.corner(
                    chains, labels=labels, show_titles=True,
                    quantiles=[0.16, 0.5, 0.84], range=plot_ranges
                )
                fig.savefig(
                    PLOTS_DIR / 'corner_step2.png',
                    dpi=150, bbox_inches='tight'
                )
                plt.close(fig)
            except Exception as e:
                self.logger.warning(f"Corner plot failed: {e}")
        
        plot_posterior_w(chains, results, self.logger)
        plot_results_and_null_test(
            self.theta_bins, xi_mean, xi_mean_scr, cov_total,
            best_fit_params, theory_args, valid_bins, self.logger
        )
        
        return results
    
    def run(self):
        """Execute the full FwCC pipeline."""
        self.load_data()
        self.run_step1()
        results = self.run_step2()
        return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    pipeline = FwCCPipeline_v52_0(
        use_lrg=True, use_anchors=True, gauss_fig=False
    )
    pipeline.run()
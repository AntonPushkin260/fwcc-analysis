import numpy as np
import pandas as pd
import os
from astropy.cosmology import wCDM
from scipy.interpolate import interp1d, RegularGridInterpolator
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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = r"output"
CATALOG_FRB = r"catalog.csv"
CATALOG_GAL = r"sdss.csv"

H0 = 67.4
Om0 = 0.315
Obh2 = 0.0224
sigma8 = 0.811
ns = 0.965

THETA_BINS = np.linspace(0.1, 5.0, 15)
Z_BINS = np.arange(0.05, 0.8, 0.05)
N_MOCKS = 500
N_PROC = max(1, cpu_count() - 2)

DM_MODELS = ['NE2001', 'YMW16']
DM_MW_DEFAULT = 45.0

class FRBAnalysisPipeline:
    def __init__(self, config=None):
        self.config = config or {}
        self.data = {}
        self.mask = None
        self.nside = 64
        self.results = {}
        self.cov_matrix = None
        self.pk_interpolator = None
        
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
            os.makedirs(os.path.join(BASE_DIR, 'plots'))

    def load_data(self):
        df_frb = pd.read_csv(CATALOG_FRB)
        
        dm_mw_col = None
        for col in ['DM_MW_NE2001', 'DM_MW_YMW16', 'DM_MW']:
            if col in df_frb.columns:
                dm_mw_col = col
                break
        
        if dm_mw_col is not None:
            df_frb['DM_excess'] = df_frb['DM'] - df_frb[dm_mw_col]
            logger.info(f"Using {dm_mw_col} for MW DM correction")
        elif 'DM_MW_model' in self.config:
            from frb_models import get_dm_mw
            ra, dec = df_frb['ra'].values, df_frb['dec'].values
            dm_mw = get_dm_mw(ra, dec, model=self.config['DM_MW_model'])
            df_frb['DM_excess'] = df_frb['DM'] - dm_mw
        else:
            logger.warning(f"No DM MW model found, using default {DM_MW_DEFAULT} pc cm^-3")
            df_frb['DM_excess'] = df_frb['DM'] - DM_MW_DEFAULT
        
        if 'z_type' in df_frb.columns:
            df_frb = df_frb[df_frb['z_type'] == 'spec']
        
        self.data['frb'] = df_frb.reset_index(drop=True)
        logger.info(f"Loaded {len(self.data['frb'])} FRBs with spectroscopic redshifts")
        
        df_gal = pd.read_csv(CATALOG_GAL, low_memory=False)
        
        col_mapping = {}
        for col in df_gal.columns:
            if col.lower() in ['ra', 'ra_j2000', 'raj2000']:
                col_mapping[col] = 'ra'
            elif col.lower() in ['dec', 'dec_j2000', 'decj2000']:
                col_mapping[col] = 'dec'
            elif col.lower() in ['z', 'redshift', 'z_spec']:
                col_mapping[col] = 'z'
        
        df_gal = df_gal.rename(columns=col_mapping)
        
        required_cols = ['ra', 'dec', 'z']
        for col in required_cols:
            if col not in df_gal.columns:
                raise ValueError(f"Required column {col} not found in galaxy catalog")
        
        df_gal['z'] = pd.to_numeric(df_gal['z'], errors='coerce')
        df_gal = df_gal.dropna(subset=['ra', 'dec', 'z'])
        
        z_min, z_max = Z_BINS[0], Z_BINS[-1]
        df_gal = df_gal[(df_gal['z'] >= z_min) & (df_gal['z'] < z_max)]
        
        self.data['gal_bins'] = {}
        self.data['dz_distributions'] = {}
        self.data['gal_weights'] = {}
        
        for i in range(len(Z_BINS) - 1):
            zmin, zmax = Z_BINS[i], Z_BINS[i+1]
            mask_z = (df_gal['z'] >= zmin) & (df_gal['z'] < zmax)
            bin_data = df_gal[mask_z][['ra', 'dec', 'z']].reset_index(drop=True)
            self.data['gal_bins'][(zmin, zmax)] = bin_data
            
            if len(bin_data) > 0:
                z_hist, z_edges = np.histogram(bin_data['z'].values, bins=30, density=True)
                z_centers = (z_edges[:-1] + z_edges[1:]) / 2
                self.data['dz_distributions'][(zmin, zmax)] = (z_centers, z_hist)
                
                completeness = len(bin_data) / len(df_gal[mask_z]) if len(df_gal[mask_z]) > 0 else 1.0
                self.data['gal_weights'][(zmin, zmax)] = completeness
        
        logger.info(f"Created {len(self.data['gal_bins'])} galaxy redshift bins")

    def create_mask(self):
        all_ra, all_dec = [], []
        for bin_df in self.data['gal_bins'].values():
            if len(bin_df) > 0:
                all_ra.extend(bin_df['ra'].values)
                all_dec.extend(bin_df['dec'].values)
        
        if len(all_ra) == 0:
            logger.warning("No galaxies found for mask creation")
            self.mask = np.ones(hp.nside2npix(self.nside), dtype=bool)
            return self.mask
        
        all_ra, all_dec = np.array(all_ra), np.array(all_dec)
        
        npix = hp.nside2npix(self.nside)
        mask = np.zeros(npix, dtype=bool)
        theta = np.pi/2 - np.radians(all_dec)
        phi = np.radians(all_ra)
        indices = hp.ang2pix(self.nside, theta, phi)
        mask[np.unique(indices)] = True
        self.mask = mask
        logger.info(f"Survey mask created: {np.sum(mask)}/{npix} pixels")
        return mask

    def _get_rand_cat(self, n_rand, ra_data, dec_data, seed=42):
        np.random.seed(seed)
        valid_pix = np.where(self.mask)[0]
        
        if len(valid_pix) == 0:
            logger.warning("No valid pixels in mask, using full sky")
            valid_pix = np.arange(hp.nside2npix(self.nside))
        
        rra, rdec = [], []
        npix_per = int(np.ceil(n_rand / len(valid_pix)))
        
        for pix in valid_pix:
            theta_p = hp.pix2ang(self.nside, pix)[0]
            phi_p = np.random.uniform(0, 2*np.pi, npix_per)
            dec_p = np.degrees(np.pi/2 - theta_p)
            ra_p = np.degrees(phi_p)
            
            if len(dec_data) > 0:
                dec_min, dec_max = np.min(dec_data), np.max(dec_data)
                margin = (dec_max - dec_min) * 0.1
                msk = (dec_p >= dec_min - margin) & (dec_p <= dec_max + margin)
            else:
                msk = np.ones(len(dec_p), dtype=bool)
            
            rra.extend(ra_p[msk])
            rdec.extend(dec_p[msk])
        
        if len(rra) > n_rand:
            idx = np.random.choice(len(rra), n_rand, replace=False)
            rra = [rra[i] for i in idx]
            rdec = [rdec[i] for i in idx]
            
        return np.array(rra), np.array(rdec)

    def compute_xi(self, frb_df, gal_bins, theta_bins, rand_cats=None):
        n_bins = len(theta_bins) - 1
        xi_meas = np.zeros(n_bins)
        xi_err = np.zeros(n_bins)
        n_pairs = np.zeros(n_bins)
        
        ra1 = frb_df['ra'].values
        dec1 = frb_df['dec'].values
        dm_vals = frb_df['DM_excess'].values
        dm_errs = frb_df.get('DM_error', np.ones(len(frb_df)) * 10).values
        n_frb = len(ra1)
        
        if rand_cats is None:
            rand_cats = {}
            for k, v in gal_bins.items():
                if len(v) > 0:
                    rra, rdec = self._get_rand_cat(len(v)*10, v['ra'].values, v['dec'].values)
                    rand_cats[k] = pd.DataFrame({'ra': rra, 'dec': rdec})

        for i in range(n_bins):
            bin_edges = np.array([theta_bins[i], theta_bins[i+1]])
            xi_sum = 0.0
            xi_sq_sum = 0.0
            npairs_tot = 0
            
            for z_key, gal_df in gal_bins.items():
                if len(gal_df) == 0: continue
                
                ra2 = gal_df['ra'].values
                dec2 = gal_df['dec'].values
                n_gal = len(ra2)
                
                if z_key in rand_cats:
                    rra = rand_cats[z_key]['ra'].values
                    rdec = rand_cats[z_key]['dec'].values
                    n_rand = len(rra)
                    alpha = n_gal / n_rand if n_rand > 0 else 1.0
                else:
                    rra, rdec = None, None
                    alpha = 1.0
                
                try:
                    res_dd = DDtheta_mocks(1, n_threads=N_PROC, bin_edges=bin_edges, 
                                           RA1=ra1, DEC1=dec1, RA2=ra2, DEC2=dec2, verbose=False)
                    n_dd = res_dd['npairs'][0] if hasattr(res_dd['npairs'], '__len__') else res_dd['npairs']
                except Exception as e:
                    logger.warning(f"DDtheta failed for bin {z_key}: {e}")
                    continue
                
                if rra is not None and len(rra) > 0:
                    try:
                        res_dr = DDtheta_mocks(1, n_threads=N_PROC, bin_edges=bin_edges,
                                               RA1=ra1, DEC1=dec1, RA2=rra, DEC2=rdec, verbose=False)
                        n_dr = res_dr['npairs'][0] if hasattr(res_dr['npairs'], '__len__') else res_dr['npairs']
                        
                        delta_g = (n_dd / (alpha * n_dr)) - 1.0 if n_dr > 0 else 0.0
                    except:
                        delta_g = 0.0
                else:
                    theta_min_rad = np.radians(theta_bins[i])
                    theta_max_rad = np.radians(theta_bins[i+1])
                    sky_area = 2 * np.pi * (np.cos(theta_min_rad) - np.cos(theta_max_rad))
                    mean_density = n_gal / (4 * np.pi)
                    n_expected = mean_density * sky_area * n_frb
                    delta_g = (n_dd / n_expected) - 1.0 if n_expected > 0 else 0.0

                for frb_idx in range(n_frb):
                    dm_val = dm_vals[frb_idx]
                    xi_sum += dm_val * delta_g
                    xi_sq_sum += (dm_val * delta_g)**2
                
                npairs_tot += n_dd
            
            if npairs_tot > 0:
                xi_meas[i] = xi_sum / npairs_tot
                variance = (xi_sq_sum / npairs_tot) - xi_meas[i]**2
                if variance < 0:
                    variance = np.mean(dm_errs**2) * (delta_g**2 if 'delta_g' in locals() else 1.0)
                xi_err[i] = np.sqrt(variance) / np.sqrt(npairs_tot)
                n_pairs[i] = npairs_tot
            else:
                xi_meas[i] = np.nan
                xi_err[i] = np.nan
                n_pairs[i] = 0
            
        return xi_meas, xi_err, n_pairs

    def precompute_power_spectra(self, w_values, z_values, k_values):
        logger.info(f"Precomputing P(k,z) grid: {len(w_values)} w values, {len(z_values)} z values")
        pk_grid = np.zeros((len(w_values), len(z_values), len(k_values)))
        
        for iw, w in enumerate(w_values):
            try:
                pars = camb.CAMBparams()
                pars.set_cosmology(H0=H0, ombh2=Obh2, omch2=(Om0 - Obh2/(H0/100)**2)*(H0/100)**2, w=w)
                pars.InitPower.set_params(As=2.1e-9, ns=ns)
                pars.set_matter_power(redshifts=z_values, kmax=10.0)
                res = camb.get_results(pars)
                kh, _, pk = res.get_matter_power_spectrum(minkh=0.001, maxkh=10.0, npoints=len(k_values))
                
                for iz in range(len(z_values)):
                    interp_func = interp1d(kh, pk[:,iz], bounds_error=False, fill_value=0.0)
                    pk_grid[iw, iz, :] = interp_func(k_values)
                
                logger.info(f"  w={w:.2f} done")
            except Exception as e:
                logger.error(f"CAMB failed for w={w}: {e}")
                pk_grid[iw, :, :] = 0.0
        
        self.pk_interpolator = RegularGridInterpolator(
            (w_values, z_values, k_values), pk_grid, bounds_error=False, fill_value=0.0
        )
        self._pk_w_values = w_values
        self._pk_z_values = z_values
        self._pk_k_values = k_values
        return self.pk_interpolator

    def get_pk_fast(self, k, z, w):
        if self.pk_interpolator is None:
            k_vals = np.logspace(-3, 1, 100)
            z_vals = np.linspace(0.0, 2.0, 50)
            w_vals = np.linspace(-1.5, -0.5, 11)
            self.precompute_power_spectra(w_vals, z_vals, k_vals)
        
        k = np.atleast_1d(k)
        points = np.column_stack([np.full_like(k, w), np.full_like(k, z), k])
        return self.pk_interpolator(points)

    def get_model_xi(self, theta_bins, w, params):
        return compute_theory_xi_with_bins(theta_bins, self.data['dz_distributions'], 
                                           self.data['frb']['z'].values, w, params, 
                                           self.get_pk_fast)

    def generate_mock_catalog(self, frb_df, gal_bins, seed=0, cosmic_variance=True):
        np.random.seed(seed)
        
        frb_mock = frb_df.copy()
        
        if cosmic_variance:
            z_frb = frb_mock['z'].values
            delta_z = np.diff(Z_BINS)[0]
            n_bins_z = int(2.0 / delta_z)
            
            power_spectrum_amp = 0.1
            fluctuations = np.random.normal(0, power_spectrum_amp, n_bins_z)
            
            for i, z in enumerate(z_frb):
                bin_idx = min(int(z / delta_z), n_bins_z - 1)
                fluctuation = 1.0 + fluctuations[bin_idx]
                frb_mock.loc[i, 'DM_excess'] *= fluctuation
        
        dm_errors = frb_mock.get('DM_error', np.ones(len(frb_mock)) * 10).values
        noise = np.random.normal(0, dm_errors, len(frb_mock))
        frb_mock['DM_excess'] = frb_mock['DM_excess'].values + noise
        
        gal_bins_mock = {}
        for z_key, gal_df in gal_bins.items():
            gal_mock = gal_df.copy()
            
            if cosmic_variance and len(gal_mock) > 0:
                n_gal_expected = len(gal_mock)
                fluctuation = np.random.normal(1.0, 0.15)
                n_gal_new = int(n_gal_expected * fluctuation)
                n_gal_new = max(10, n_gal_new)
                
                if n_gal_new > len(gal_mock):
                    extra_idx = np.random.choice(len(gal_mock), n_gal_new - len(gal_mock), replace=True)
                    gal_extra = gal_mock.iloc[extra_idx].copy()
                    gal_extra['z'] += np.random.normal(0, 0.01, len(gal_extra))
                    gal_mock = pd.concat([gal_mock, gal_extra], ignore_index=True)
                elif n_gal_new < len(gal_mock):
                    gal_mock = gal_mock.sample(n=n_gal_new, random_state=seed)
            
            gal_bins_mock[z_key] = gal_mock.reset_index(drop=True)
        
        return frb_mock, gal_bins_mock

    def estimate_covariance(self, frb_df, gal_bins, theta_bins, n_mocks=100):
        logger.info(f"Estimating covariance from {n_mocks} mocks...")
        n_bins = len(theta_bins) - 1
        xi_mocks = np.zeros((n_mocks, n_bins))
        
        rand_cats = {}
        for k, v in gal_bins.items():
            if len(v) > 0:
                rra, rdec = self._get_rand_cat(len(v)*10, v['ra'].values, v['dec'].values)
                rand_cats[k] = pd.DataFrame({'ra': rra, 'dec': rdec})
        
        n_success = 0
        for m in range(n_mocks):
            try:
                frb_mock, gal_bins_mock = self.generate_mock_catalog(
                    frb_df, gal_bins, seed=m+1000, cosmic_variance=True
                )
                
                xi_mock, _, _ = self.compute_xi(frb_mock, gal_bins_mock, theta_bins, rand_cats)
                
                if not np.all(np.isnan(xi_mock)):
                    xi_mocks[n_success, :] = xi_mock
                    n_success += 1
                    
                    if (m + 1) % 50 == 0:
                        logger.info(f"  Mock {m+1}/{n_mocks} done ({n_success} successful)")
            except Exception as e:
                logger.warning(f"Mock {m} failed: {e}")
                continue
        
        if n_success < 10:
            logger.error(f"Only {n_success} successful mocks, covariance unreliable")
            return np.eye(n_bins) * 1e-4
        
        xi_mocks = xi_mocks[:n_success, :]
        logger.info(f"Using {n_success} successful mocks for covariance")
        
        cov_mat = np.cov(xi_mocks, rowvar=False)
        cov_mat = (cov_mat + cov_mat.T) / 2
        
        eigvals = np.linalg.eigvalsh(cov_mat)
        if np.any(eigvals < 0):
            logger.warning("Covariance matrix not positive definite, regularizing")
            cov_mat += np.eye(n_bins) * np.abs(np.min(eigvals)) * 1.1
        
        n_data = n_bins
        if n_mocks > n_data + 2:
            hartlap = (n_success - n_data - 2) / (n_success - 1)
            cov_mat = cov_mat / hartlap
            logger.info(f"Hartlap factor: {hartlap:.3f}")
        
        return cov_mat

    def run_mcmc(self, xi_obs, cov_mat, params):
        logger.info("Starting MCMC parameter estimation...")
        
        try:
            cov_inv = np.linalg.inv(cov_mat)
        except np.linalg.LinAlgError:
            logger.error("Covariance matrix singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(cov_mat)
        
        ndim = 2
        nwalkers = 50
        nsteps = 1000
        burn_in = 200
        
        p0 = np.array([-1.0, 0.84]) + 1e-4 * np.random.randn(nwalkers, ndim)
        
        def ln_prob(p):
            w, f_igm = p
            if not (-2.0 < w < 0.0 and 0.0 < f_igm < 1.0):
                return -np.inf
            
            p_mod = params.copy()
            p_mod['f_IGM'] = f_igm
            xi_mod = self.get_model_xi(THETA_BINS, w, p_mod)
            
            valid = ~np.isnan(xi_obs) & ~np.isnan(xi_mod)
            if np.sum(valid) < 3:
                return -np.inf
            
            diff = xi_obs[valid] - xi_mod[valid]
            cov_valid = cov_mat[np.ix_(valid, valid)]
            
            try:
                cov_inv_valid = np.linalg.inv(cov_valid)
            except:
                return -np.inf
            
            chi2 = np.dot(diff.T, np.dot(cov_inv_valid, diff))
            return -0.5 * chi2

        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob)
        sampler.run_mcmc(p0, nsteps, progress=True)
        
        try:
            tau = sampler.get_autocorr_time()
            logger.info(f"Autocorrelation times: w={tau[0]:.1f}, f_IGM={tau[1]:.1f}")
            if np.any(nsteps < 50 * tau):
                logger.warning("MCMC may not have converged")
        except:
            logger.warning("Could not compute autocorrelation time")
        
        samples = sampler.get_chain(discard=burn_in, flat=True)
        w_samp = samples[:, 0]
        f_samp = samples[:, 1]
        
        w_med = np.median(w_samp)
        w_lo, w_hi = np.percentile(w_samp, [16, 84])
        f_med = np.median(f_samp)
        f_lo, f_hi = np.percentile(f_samp, [16, 84])
        
        logger.info(f"w = {w_med:.3f} +{w_hi-w_med:.3f}/-{w_med-w_lo:.3f}")
        logger.info(f"f_IGM = {f_med:.3f} +{f_hi-f_med:.3f}/-{f_med-f_lo:.3f}")
        
        return {
            'w': w_med,
            'w_err': (w_hi - w_lo) / 2,
            'w_16': w_lo,
            'w_84': w_hi,
            'f_IGM': f_med,
            'f_IGM_err': (f_hi - f_lo) / 2,
            'samples': samples
        }

    def run(self):
        logger.info("Starting FRB-LSS cross-correlation analysis")
        
        self.load_data()
        self.create_mask()
        
        logger.info("Precomputing power spectra grid...")
        k_vals = np.logspace(-3, 1, 150)
        z_vals = np.linspace(0.0, 2.0, 60)
        w_vals = np.linspace(-1.5, -0.5, 15)
        self.precompute_power_spectra(w_vals, z_vals, k_vals)
        
        xi_list = []
        xi_err_list = []
        for model in DM_MODELS:
            logger.info(f"Computing xi for DM model: {model}")
            df_tmp = self.data['frb'].copy()
            
            if model == 'NE2001' and 'DM_MW_NE2001' in self.data['frb'].columns:
                df_tmp['DM_excess'] = self.data['frb']['DM'] - self.data['frb']['DM_MW_NE2001']
            elif model == 'YMW16' and 'DM_MW_YMW16' in self.data['frb'].columns:
                df_tmp['DM_excess'] = self.data['frb']['DM'] - self.data['frb']['DM_MW_YMW16']
            
            xi, err, npairs = self.compute_xi(df_tmp, self.data['gal_bins'], THETA_BINS)
            xi_list.append(xi)
            xi_err_list.append(err)
            logger.info(f"  Mean xi: {np.nanmean(xi):.3f}")
        
        xi_mean = np.mean(xi_list, axis=0)
        xi_sys_err = np.std(xi_list, axis=0)
        
        logger.info("Estimating covariance from mocks...")
        cov_mat = self.estimate_covariance(self.data['frb'], self.data['gal_bins'], THETA_BINS, n_mocks=N_MOCKS)
        
        diag_stat = np.sqrt(np.diag(cov_mat))
        logger.info(f"Covariance diagonal: min={np.min(diag_stat):.3e}, max={np.max(diag_stat):.3e}")
        
        for i in range(len(xi_sys_err)):
            if not np.isnan(xi_sys_err[i]):
                cov_mat[i, i] += xi_sys_err[i]**2
        
        self.cov_matrix = cov_mat
        
        params = {'galaxy_bias': 1.5, 'FRB_bias': 1.2, 'A_norm': 1.0, 'Omega_m': Om0, 'H0': H0}
        res = self.run_mcmc(xi_mean, cov_mat, params)
        
        self.results = res
        
        results_dict = {
            'w_best': float(res['w']),
            'w_err': float(res['w_err']),
            'w_16': float(res.get('w_16', res['w'] - res['w_err'])),
            'w_84': float(res.get('w_84', res['w'] + res['w_err'])),
            'f_IGM_best': float(res['f_IGM']),
            'f_IGM_err': float(res['f_IGM_err']),
            'n_frb': len(self.data['frb']),
            'n_gal_bins': len(self.data['gal_bins']),
            'n_mocks_used': N_MOCKS,
            'theta_bins': THETA_BINS.tolist(),
            'dm_models_marginalized': DM_MODELS,
            'cosmology': {'H0': H0, 'Om0': Om0, 'sigma8': sigma8},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(BASE_DIR, 'results.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info("Analysis complete. Results saved to results.json")
        
        if res['samples'] is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            samples = res['samples']
            
            axes[0].hist(samples[:, 0], bins=40, alpha=0.7, label='w')
            axes[0].axvline(res['w'], color='r', linestyle='--')
            axes[0].axvline(-1.0, color='k', linestyle=':', label='LCDM')
            axes[0].set_xlabel('w')
            axes[0].legend()
            
            axes[1].hist(samples[:, 1], bins=40, alpha=0.7, label='f_IGM')
            axes[1].axvline(res['f_IGM'], color='r', linestyle='--')
            axes[1].set_xlabel('f_IGM')
            axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'plots', 'posteriors.png'), dpi=300)
            plt.close()

def hubble(z, w, Om=Om0, H=H0):
    return H * np.sqrt(Om * (1+z)**3 + (1-Om) * (1+z)**(3*(1+w)))

def growth(z, w, Om=Om0):
    Om_z = Om * (1+z)**3 / (Om * (1+z)**3 + (1-Om)*(1+z)**(3*(1+w)))
    gamma = 0.55 + 0.05*(1+w)
    return Om_z**gamma / Om**gamma

def compute_theory_xi_with_bins(theta_bins, dz_distributions, frb_z_values, w, params, pk_func):
    k_vals = np.logspace(-3, 1, 150)
    
    if len(frb_z_values) > 0:
        frb_z_hist, frb_z_edges = np.histogram(frb_z_values, bins=30, density=True)
        frb_z_centers = (frb_z_edges[:-1] + frb_z_edges[1:]) / 2
        W_frb = frb_z_hist / np.max(frb_z_hist) if np.max(frb_z_hist) > 0 else np.ones_like(frb_z_centers) * 0.1
    else:
        frb_z_centers = np.linspace(0.1, 1.5, 20)
        W_frb = np.ones_like(frb_z_centers) * 0.1
    
    D_frb = growth(frb_z_centers, w)
    
    cosmo = wCDM(H0=params['H0'], Om0=params['Omega_m'], w0=w)
    
    theta_rad = np.radians((theta_bins[:-1] + theta_bins[1:])/2)
    xi = np.zeros_like(theta_rad)
    
    f_IGM = params.get('f_IGM', 0.84)
    b_g = params.get('galaxy_bias', 1.5)
    b_frb = params.get('FRB_bias', 1.2)
    A_norm = params.get('A_norm', 1.0)
    
    for z_key, (z_centers, z_hist) in dz_distributions.items():
        if len(z_hist) == 0 or np.max(z_hist) == 0:
            continue
            
        W_gal = z_hist / np.max(z_hist)
        D_gal = growth(z_centers, w)
        chi_gal = cosmo.comoving_distance(z_centers).value / 1000.0
        
        dz = z_centers[1] - z_centers[0] if len(z_centers) > 1 else 0.1
        
        for i, th in enumerate(theta_rad):
            int_z = 0.0
            for j, z in enumerate(z_centers):
                if z_hist[j] == 0:
                    continue
                    
                pk = pk_func(k_vals, z, w)
                j0_val = j0(np.outer(k_vals, np.array([chi_gal[j]])) * th)
                integrand = k_vals * pk * j0_val[:,0] * D_gal[j]**2 * W_gal[j]
                int_k = np.trapz(integrand, k_vals)
                int_z += int_k * dz
            
            xi[i] += b_g * b_frb * A_norm * int_z * 1e-4 * (f_IGM/0.84)
    
    return xi

if __name__ == "__main__":
    pipe = FRBAnalysisPipeline()
    pipe.run()

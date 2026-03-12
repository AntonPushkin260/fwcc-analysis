import numpy as np
from scipy.stats import norm
from .dm_model import dm_igm, dm_host_model
from .correlation import calculate_cross_correlation

def log_likelihood(params, frb_data, desi_maps, z_centers):
    w = params['w']
    mu_host = params['mu_host']
    alpha = params['alpha']
    f_IGM = params['f_IGM']
    model_params = {
        'Omega_m': 0.315,
        'Omega_b_h2': 0.0224,
        'H0': 67.4,
        'f_IGM': f_IGM,
        'chi_e': 1.0,
        'mu_host': mu_host,
        'sigma_host': 30.0,
        'alpha': alpha,
        'aperture_radius': 5.0
    }
    log_likelihood_sum = 0.0
    for _, frb in frb_data.iterrows():
        if frb['z_type'] == 'spec':
            z = frb['z']
            p_z = np.array([1.0])
            model_val = calculate_cross_correlation(frb, desi_maps, z_centers, w, model_params)
            sigma_total = np.sqrt(
                frb['DM_error']**2 + 
                (model_params['sigma_host'] / (1 + z))**2 +
                frb.get('LSS_noise', 15.0)**2 +
                frb.get('MW_noise', 10.0)**2
            )
            log_likelihood_sum += norm.logpdf(frb['DM_excess'], loc=model_val, scale=sigma_total)
        else:
            z_values = np.linspace(0.01, 2.5, 100)
            p_z = np.exp(-0.5 * ((z_values - frb['z_mean']) / frb['z_error'])**2)
            p_z /= np.sum(p_z)
            model_integral = 0.0
            for i, z in enumerate(z_values):
                frb_z = frb.copy()
                frb_z['z'] = z
                model_val = calculate_cross_correlation(frb_z, desi_maps, z_centers, w, model_params)
                model_integral += p_z[i] * model_val
            sigma_total = np.sqrt(
                frb['DM_error']**2 + 
                (model_params['sigma_host'] / (1 + frb['z_mean']))**2 +
                frb.get('LSS_noise', 15.0)**2 +
                frb.get('MW_noise', 10.0)**2
            )
            log_likelihood_sum += norm.logpdf(frb['DM_excess'], loc=model_integral, scale=sigma_total)
    return log_likelihood_sum

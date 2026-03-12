import numpy as np
import emcee
from scipy.optimize import minimize
from .likelihood import log_likelihood

def neg_log_likelihood(params, frb_data, desi_maps, z_centers):
    params_dict = {
        'w': params[0],
        'mu_host': params[1],
        'alpha': params[2],
        'f_IGM': params[3]
    }
    log_like = log_likelihood(params_dict, frb_data, desi_maps, z_centers)
    prior = 0.0
    if not (-2.0 < params_dict['w'] < 0.0):
        prior -= 1e10
    prior -= 0.5 * ((params_dict['f_IGM'] - 0.84) / 0.06)**2
    if params_dict['mu_host'] > 0:
        prior -= np.log(params_dict['mu_host'])
    prior -= 0.5 * ((params_dict['alpha'] - 1.0) / 0.5)**2
    return -(log_like + prior)

def run_mcmc(frb_data, desi_maps, z_centers, nwalkers=50, nsteps=100):
    initial_params = [-1.0, 50.0, 1.0, 0.84]
    result = minimize(
        neg_log_likelihood, 
        initial_params, 
        args=(frb_data, desi_maps, z_centers),
        method='L-BFGS-B',
        bounds=[(-2.0, 0.0), (0.0, 200.0), (0.1, 10.0), (0.5, 1.0)]
    )
    pos = result.x + 1e-4 * np.random.randn(nwalkers, len(initial_params))
    sampler = emcee.EnsembleSampler(nwalkers, len(initial_params), neg_log_likelihood, 
                                   args=(frb_data, desi_maps, z_centers))
    sampler.run_mcmc(pos, nsteps, progress=True)
    return sampler

def analyze_results(sampler, burn_in=200):
    samples = sampler.get_chain(discard=burn_in, flat=True)
    w_samples = samples[:, 0]
    mu_host_samples = samples[:, 1]
    alpha_samples = samples[:, 2]
    f_IGM_samples = samples[:, 3]
    w_median = np.median(w_samples)
    w_low = np.percentile(w_samples, 16)
    w_high = np.percentile(w_samples, 84)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.hist(w_samples, bins=30, density=True)
    plt.axvline(w_median, color='r', label=f'Median: {w_median:.2f}')
    plt.axvline(w_low, color='k', linestyle='--', label=f'16%: {w_low:.2f}')
    plt.axvline(w_high, color='k', linestyle='--', label=f'84%: {w_high:.2f}')
    plt.title('w Distribution')
    plt.xlabel('w')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.hist(f_IGM_samples, bins=30, density=True)
    plt.title('f_IGM Distribution')
    plt.xlabel('f_IGM')
    plt.subplot(2, 2, 3)
    plt.hist(mu_host_samples, bins=30, density=True)
    plt.title('mu_host Distribution')
    plt.xlabel('mu_host')
    plt.subplot(2, 2, 4)
    plt.hist(alpha_samples, bins=30, density=True)
    plt.title('alpha Distribution')
    plt.xlabel('alpha')
    plt.tight_layout()
    plt.savefig('figures/mcmc_results.png')
    print(f"w = {w_median:.2f} +{w_high-w_median:.2f}/-{w_median-w_low:.2f}")
    print(f"f_IGM = {np.mean(f_IGM_samples):.3f} ± {np.std(f_IGM_samples):.3f}")
    print(f"mu_host = {np.mean(mu_host_samples):.1f} ± {np.std(mu_host_samples):.1f}")
    print(f"alpha = {np.mean(alpha_samples):.2f} ± {np.std(alpha_samples):.2f}")
    return {
        'w': (w_median, w_low, w_high),
        'f_IGM': (np.mean(f_IGM_samples), np.std(f_IGM_samples)),
        'mu_host': (np.mean(mu_host_samples), np.std(mu_host_samples)),
        'alpha': (np.mean(alpha_samples), np.std(alpha_samples))
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from .dm_model import dm_igm, dm_host_model

class SKAForecaster:
    def __init__(self, true_w=-1.0, true_params=None):
        self.true_w = true_w
        self.true_params = true_params or {
            'mu_host': 50.0,
            'sigma_host': 30.0,
            'alpha': 1.0,
            'f_IGM': 0.84,
            'Omega_m': 0.315,
            'H0': 67.4
        }
        
    def generate_ska_catalog(self, n_frb=10**6, f_spec=0.2, sigma_sys=30.0, 
                            aperture_radius=5.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        ra = np.random.uniform(0, 360, n_frb)
        dec = np.arcsin(np.random.uniform(-1, 1, n_frb)) * 180 / np.pi
        z = np.random.gamma(2.5, 0.4, n_frb)
        z = z[z < 2.0]
        n_actual = len(z)
        ra = ra[:n_actual]
        dec = dec[:n_actual]
        n_spec = int(n_actual * f_spec)
        z_type = np.array(['photo'] * n_actual)
        spec_indices = np.random.choice(n_actual, n_spec, replace=False)
        z_type[spec_indices] = 'spec'
        dm_stat_error = np.random.normal(3, 1, n_actual)
        dm_stat_error = np.clip(dm_stat_error, 1, 10)
        dm_total_error = np.sqrt(dm_stat_error**2 + sigma_sys**2)
        dm_mw = np.random.exponential(50, n_actual)
        dm_excess = []
        for i in range(n_actual):
            dm_igm_val = dm_igm(
                z[i], self.true_w,
                f_IGM=self.true_params['f_IGM'],
                Omega_m=self.true_params['Omega_m'],
                H0=self.true_params['H0']
            )
            dm_host_mean, dm_host_std = dm_host_model(
                z[i],
                self.true_params['mu_host'],
                self.true_params['sigma_host']
            )
            dm_host_val = dm_host_mean + np.random.normal(0, dm_host_std)
            dm_total_val = dm_igm_val + dm_host_val
            noise = np.random.normal(0, dm_total_error[i])
            dm_excess.append(dm_total_val + noise)
        frb_data = pd.DataFrame({
            'ra': ra,
            'dec': dec,
            'z': z,
            'z_type': z_type,
            'z_mean': z,
            'z_error': np.where(z_type == 'spec', 0.001, np.random.uniform(0.05, 0.15, n_actual)),
            'DM_excess': dm_excess,
            'DM_error': dm_total_error,
            'DM_MW_NE2001': dm_mw,
            'LSS_noise': np.random.normal(10, 3, n_actual),
            'MW_noise': np.random.normal(5, 2, n_actual),
            'aperture_radius': aperture_radius
        })
        return frb_data
    
    def forecast_sigma_w(self, n_frb_list, f_spec_list, sigma_sys_list, 
                        aperture_radii=[2.0, 3.0, 5.0], n_realizations=10):
        results = []
        for n_frb in n_frb_list:
            for f_spec in f_spec_list:
                for sigma_sys in sigma_sys_list:
                    for aperture in aperture_radii:
                        sigma_w_values = []
                        for seed in range(n_realizations):
                            catalog = self.generate_ska_catalog(
                                n_frb=n_frb,
                                f_spec=f_spec,
                                sigma_sys=sigma_sys,
                                aperture_radius=aperture,
                                seed=seed
                            )
                            sigma_w = self._estimate_sigma_w_fisher(catalog, aperture)
                            sigma_w_values.append(sigma_w)
                        results.append({
                            'N_FRB': n_frb,
                            'f_spec': f_spec,
                            'sigma_sys': sigma_sys,
                            'aperture_radius': aperture,
                            'sigma_w_mean': np.mean(sigma_w_values),
                            'sigma_w_std': np.std(sigma_w_values),
                            'sigma_w_min': np.min(sigma_w_values),
                            'sigma_w_max': np.max(sigma_w_values)
                        })
        return pd.DataFrame(results)
    
    def _estimate_sigma_w_fisher(self, catalog, aperture_radius):
        spec_mask = catalog['z_type'] == 'spec'
        n_spec = np.sum(spec_mask)
        n_photo = np.sum(~spec_mask)
        n_eff = n_spec + 0.3 * n_photo
        mean_dm_error = np.mean(catalog['DM_error'])
        dDM_dw = 200.0
        F_ww = n_eff * (dDM_dw / mean_dm_error)**2
        sigma_w = 1.0 / np.sqrt(F_ww) if F_ww > 0 else 1.0
        return sigma_w
    
    def plot_forecast_comparison(self, forecast_df):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1 = axes[0, 0]
        for f_spec in forecast_df['f_spec'].unique():
            subset = forecast_df[forecast_df['f_spec'] == f_spec]
            ax1.errorbar(
                subset['N_FRB'], 
                subset['sigma_w_mean'],
                yerr=subset['sigma_w_std'],
                label=f'f_spec={f_spec:.2f}',
                marker='o'
            )
        ax1.set_xscale('log')
        ax1.set_xlabel('N_FRB')
        ax1.set_ylabel('σ(w)')
        ax1.set_title('Точность vs Количество FRB')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        for sigma_sys in forecast_df['sigma_sys'].unique():
            subset = forecast_df[forecast_df['sigma_sys'] == sigma_sys]
            ax2.errorbar(
                subset['f_spec'], 
                subset['sigma_w_mean'],
                yerr=subset['sigma_w_std'],
                label=f'σ_sys={sigma_sys} пк/см³',
                marker='s'
            )
        ax2.set_xlabel('Доля spec-z (f_spec)')
        ax2.set_ylabel('σ(w)')
        ax2.set_title('Точность vs Доля спектроскопических z')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]
        for n_frb in [10**5, 3*10**5, 10**6]:
            subset = forecast_df[forecast_df['N_FRB'] == n_frb]
            ax3.errorbar(
                subset['sigma_sys'], 
                subset['sigma_w_mean'],
                yerr=subset['sigma_w_std'],
                label=f'N={n_frb:.0e}',
                marker='^'
            )
        ax3.set_xlabel('Систематический пол σ_sys (пк/см³)')
        ax3.set_ylabel('σ(w)')
        ax3.set_title('Влияние систематики')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 1]
        pivot_data = forecast_df.pivot_table(
            values='sigma_w_mean',
            index='N_FRB',
            columns='f_spec'
        )
        im = ax4.imshow(pivot_data.values, aspect='auto', cmap='viridis_r')
        ax4.set_yticks(range(len(pivot_data.index)))
        ax4.set_yticklabels([f'{x:.0e}' for x in pivot_data.index])
        ax4.set_xticks(range(len(pivot_data.columns)))
        ax4.set_xticklabels([f'{x:.2f}' for x in pivot_data.columns], rotation=45)
        ax4.set_xlabel('Доля spec-z')
        ax4.set_ylabel('N_FRB')
        ax4.set_title('σ(w) в пространстве параметров')
        plt.colorbar(im, ax=ax4, label='σ(w)')
        plt.tight_layout()
        plt.savefig('figures/ska_forecasts.png')
        plt.show()
    
    def generate_scenario_report(self, scenario_name, n_frb, f_spec, sigma_sys, 
                                aperture=3.0):
        print(f"\n{'='*70}")
        print(f"СЦЕНАРИЙ: {scenario_name}")
        print(f"{'='*70}")
        print(f"N_FRB = {n_frb:.2e}")
        print(f"f_spec = {f_spec:.2f} ({n_frb*f_spec:.0f} FRB)")
        print(f"σ_sys = {sigma_sys} пк/см³")
        print(f"Апертура = {aperture}°")
        print(f"{'='*70}\n")
        catalog = self.generate_ska_catalog(
            n_frb=n_frb,
            f_spec=f_spec,
            sigma_sys=sigma_sys,
            aperture_radius=aperture,
            seed=42
        )
        print("СТАТИСТИКА КАТАЛОГА:")
        print(f"  Всего FRB: {len(catalog)}")
        print(f"  С spec-z: {np.sum(catalog['z_type']=='spec')} ({100*np.mean(catalog['z_type']=='spec'):.1f}%)")
        print(f"  С photo-z: {np.sum(catalog['z_type']=='photo')} ({100*np.mean(catalog['z_type']=='photo'):.1f}%)")
        print(f"  Медианное z: {np.median(catalog['z']):.3f}")
        print(f"  Среднее DM_error: {np.mean(catalog['DM_error']):.2f} пк/см³")
        print()
        sigma_w = self._estimate_sigma_w_fisher(catalog, aperture)
        print("ПРОГНОЗ ТОЧНОСТИ:")
        print(f"  σ(w) ≈ {sigma_w:.4f}")
        print(f"  Отношение к ΛCDM: {(sigma_w/0.01):.1f}× от целевой точности 0.01")
        print()
        print("ДОСТИЖЕНИЕ ЦЕЛЕЙ:")
        targets = [0.05, 0.02, 0.01, 0.005]
        for target in targets:
            if sigma_w <= target:
                print(f"  ✓ σ(w) ≤ {target:.3f} ДОСТИЖИМО")
            else:
                print(f"  ✗ σ(w) ≤ {target:.3f} НЕ ДОСТИЖИМО (требуется улучшение в {sigma_w/target:.1f}×)")
        print(f"\n{'='*70}\n")
        return catalog, sigma_w

def run_all_ska_scenarios():
    forecaster = SKAForecaster(true_w=-1.0)
    scenarios = {
        'conservative': {
            'n_frb': 10**5,
            'f_spec': 0.05,
            'sigma_sys': 40.0,
            'aperture': 5.0
        },
        'baseline': {
            'n_frb': 3*10**5,
            'f_spec': 0.10,
            'sigma_sys': 30.0,
            'aperture': 3.0
        },
        'optimistic': {
            'n_frb': 10**6,
            'f_spec': 0.20,
            'sigma_sys': 20.0,
            'aperture': 2.0
        }
    }
    all_results = []
    for name, params in scenarios.items():
        catalog, sigma_w = forecaster.generate_scenario_report(
            scenario_name=name.upper(),
            **params
        )
        all_results.append({
            'scenario': name,
            'sigma_w': sigma_w,
            'n_frb': params['n_frb'],
            'f_spec': params['f_spec']
        })
    print("\n" + "="*70)
    print("СВОДНАЯ ТАБЛИЦА ПРОГНОЗОВ")
    print("="*70)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    print("="*70 + "\n")
    return forecaster, results_df

import numpy as np
import pandas as pd
from scipy import stats
from .dm_model import dm_igm, dm_host_model
from .likelihood import log_likelihood

class RobustnessTests:
    def __init__(self, frb_data, desi_maps, z_centers, true_w=-1.0):
        self.frb_data = frb_data.copy()
        self.desi_maps = desi_maps
        self.z_centers = z_centers
        self.true_w = true_w
        self.results = {}
    
    def test_mw_model_variations(self, base_params):
        print("\n[ТЕСТ] Вариации MW-модели...")
        results = {}
        w_base = self._quick_estimate_w(base_params)
        results['baseline'] = w_base
        frb_modified = self.frb_data.copy()
        frb_modified['DM_MW_NE2001'] *= 1.2
        self.frb_data = frb_modified
        w_high = self._quick_estimate_w(base_params)
        results['MW_high_20pct'] = w_high
        frb_modified = self.frb_data.copy()
        frb_modified['DM_MW_NE2001'] *= 0.8
        self.frb_data = frb_modified
        w_low = self._quick_estimate_w(base_params)
        results['MW_low_20pct'] = w_low
        self.frb_data = self.frb_data.copy()
        delta_high = abs(w_high - w_base) / 0.15
        delta_low = abs(w_low - w_base) / 0.15
        print(f"  Базовое w: {w_base:.3f}")
        print(f"  MW +20%: {w_high:.3f} (Δ={delta_high:.2f}σ)")
        print(f"  MW -20%: {w_low:.3f} (Δ={delta_low:.2f}σ)")
        print(f"  Максимальное смещение: {max(delta_high, delta_low):.2f}σ")
        self.results['mw_model'] = results
        return results
    
    def test_host_distribution_variations(self, base_params):
        print("\n[ТЕСТ] Вариации распределения хозяев...")
        results = {}
        w_base = self._quick_estimate_w(base_params)
        results['baseline'] = w_base
        params_exp = base_params.copy()
        params_exp['mu_host'] = 40.0
        params_exp['sigma_host'] = 40.0
        w_exp = self._quick_estimate_w(params_exp)
        results['exponential_like'] = w_exp
        params_wide = base_params.copy()
        params_wide['sigma_host'] = 50.0
        w_wide = self._quick_estimate_w(params_wide)
        results['wide_distribution'] = w_wide
        print(f"  Базовое w: {w_base:.3f}")
        print(f"  Экспоненциальное: {w_exp:.3f}")
        print(f"  Широкий разброс: {w_wide:.3f}")
        self.results['host_distribution'] = results
        return results
    
    def test_outlier_removal(self, base_params, frac_remove=0.05):
        print(f"\n[ТЕСТ] Устойчивость к выбросам (удаляем {frac_remove*100:.0f}%...)")
        residuals = []
        for idx, frb in self.frb_data.iterrows():
            model_dm = dm_igm(
                frb['z'], self.true_w,
                f_IGM=base_params.get('f_IGM', 0.84)
            )
            residual = abs(frb['DM_excess'] - model_dm)
            residuals.append(residual)
        self.frb_data['residual'] = residuals
        n_remove = int(len(self.frb_data) * frac_remove)
        sorted_df = self.frb_data.sort_values('residual', ascending=False)
        frb_cleaned = sorted_df.iloc[n_remove:].copy()
        original_data = self.frb_data.copy()
        self.frb_data = frb_cleaned
        w_cleaned = self._quick_estimate_w(base_params)
        self.frb_data = original_data
        w_original = self._quick_estimate_w(base_params)
        delta = abs(w_cleaned - w_original) / 0.15
        print(f"  Исходное w: {w_original:.3f}")
        print(f"  После очистки: {w_cleaned:.3f}")
        print(f"  Смещение: {delta:.2f}σ")
        self.results['outliers'] = {
            'original': w_original,
            'cleaned': w_cleaned,
            'delta_sigma': delta
        }
        return self.results['outliers']
    
    def test_redshift_bins(self, base_params):
        print("\n[ТЕСТ] Разделение по бинам красного смещения...")
        z_median = np.median(self.frb_data['z'])
        low_z_mask = self.frb_data['z'] < z_median
        frb_low_z = self.frb_data[low_z_mask].copy()
        high_z_mask = self.frb_data['z'] >= z_median
        frb_high_z = self.frb_data[high_z_mask].copy()
        original_data = self.frb_data.copy()
        self.frb_data = frb_low_z
        w_low = self._quick_estimate_w(base_params)
        n_low = len(frb_low_z)
        self.frb_data = frb_high_z
        w_high = self._quick_estimate_w(base_params)
        n_high = len(frb_high_z)
        self.frb_data = original_data
        w_all = self._quick_estimate_w(base_params)
        print(f"  Низкие z (N={n_low}, median z={np.median(frb_low_z['z']):.2f}): w = {w_low:.3f}")
        print(f"  Высокие z (N={n_high}, median z={np.median(frb_high_z['z']):.2f}): w = {w_high:.3f}")
        print(f"  Все данные: w = {w_all:.3f}")
        print(f"  Разница: Δw = {w_high - w_low:.3f}")
        self.results['z_bins'] = {
            'low_z': w_low,
            'high_z': w_high,
            'all': w_all,
            'difference': w_high - w_low
        }
        self.frb_data = original_data
        return self.results['z_bins']
    
    def test_sky_coverage(self, base_params):
        print("\n[ТЕСТ] Ограничение по покрытию неба...")
        original_data = self.frb_data.copy()
        north_mask = self.frb_data['dec'] > 0
        frb_north = self.frb_data[north_mask].copy()
        south_mask = self.frb_data['dec'] < 0
        frb_south = self.frb_data[south_mask].copy()
        results = {}
        self.frb_data = frb_north
        w_north = self._quick_estimate_w(base_params)
        results['north'] = w_north
        self.frb_data = frb_south
        w_south = self._quick_estimate_w(base_params)
        results['south'] = w_south
        self.frb_data = original_data
        w_all = self._quick_estimate_w(base_params)
        results['all'] = w_all
        print(f"  Северное полушарие (N={len(frb_north)}): w = {w_north:.3f}")
        print(f"  Южное полушарие (N={len(frb_south)}): w = {w_south:.3f}")
        print(f"  Все данные: w = {w_all:.3f}")
        print(f"  Разница N-S: Δw = {w_north - w_south:.3f}")
        self.frb_data = original_data
        self.results['sky_coverage'] = results
        return results
    
    def _quick_estimate_w(self, params):
        def chi_squared(w):
            chi2 = 0.0
            for idx, frb in self.frb_data.iterrows():
                model_dm = dm_igm(
                    frb['z'], w,
                    f_IGM=params.get('f_IGM', 0.84),
                    Omega_m=params.get('Omega_m', 0.315)
                )
                dm_host, _ = dm_host_model(
                    frb['z'],
                    params.get('mu_host', 50.0),
                    params.get('sigma_host', 30.0)
                )
                residual = frb['DM_excess'] - (model_dm + dm_host)
                sigma = frb.get('DM_error', 10.0)
                chi2 += (residual / sigma)**2
            return chi2
        result = minimize_scalar(chi_squared, bounds=(-2.0, 0.0), method='bounded')
        return result.x if result.success else -1.0
    
    def run_all_tests(self, base_params):
        print("\n" + "="*70)
        print("ЗАПУСК ВСЕХ ТЕСТОВ УСТОЙЧИВОСТИ")
        print("="*70)
        self.test_mw_model_variations(base_params)
        self.test_host_distribution_variations(base_params)
        self.test_outlier_removal(base_params)
        self.test_redshift_bins(base_params)
        self.test_sky_coverage(base_params)
        print("\n" + "="*70)
        print("ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
        print("="*70)
        return self.results
    
    def print_summary(self):
        print("\n" + "="*70)
        print("СВОДКА ПО ТЕСТАМ УСТОЙЧИВОСТИ")
        print("="*70)
        for test_name, results in self.results.items():
            print(f"\n{test_name.upper()}:")
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        print("\n" + "="*70)

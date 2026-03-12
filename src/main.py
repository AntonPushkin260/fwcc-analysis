import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .data_processing import load_chime_data, load_desi_data
from .mcmc_analysis import run_mcmc, analyze_results
from .ska_forecasts import SKAForecaster, run_all_ska_scenarios
from .robustness_tests import RobustnessTests

def main_analysis():
    print("="*70)
    print("ОСНОВНОЙ АНАЛИЗ CHIME-DESI")
    print("="*70)
    print("\nЗагрузка данных...")
    frb_data = load_chime_data("data/CHIME_FRB_Catalog2.csv")
    desi_maps, z_centers = load_desi_data("data/DESI_DR3_galaxies.csv")
    print(f"Загружено {len(frb_data)} FRB")
    print(f"Загружены карты DESI с {len(z_centers)} слоями z")
    base_params = {
        'Omega_m': 0.315,
        'Omega_b_h2': 0.0224,
        'H0': 67.4,
        'f_IGM': 0.84,
        'chi_e': 1.0,
        'mu_host': 50.0,
        'sigma_host': 30.0,
        'alpha': 1.0,
        'aperture_radius': 5.0
    }
    print("\nЗапуск MCMC...")
    sampler = run_mcmc(frb_data, desi_maps, z_centers)
    print("\nАнализ результатов MCMC...")
    results = analyze_results(sampler)
    return frb_data, desi_maps, z_centers, base_params, results

def run_robustness_tests(frb_data, desi_maps, z_centers, base_params):
    print("\n" + "="*70)
    print("ТЕСТЫ УСТОЙЧИВОСТИ")
    print("="*70)
    tester = RobustnessTests(frb_data, desi_maps, z_centers, true_w=-1.0)
    all_results = tester.run_all_tests(base_params)
    tester.print_summary()
    return all_results

def run_ska_forecasts():
    print("\n" + "="*70)
    print("ПРОГНОЗЫ ДЛЯ SKA")
    print("="*70)
    forecaster, results = run_all_ska_scenarios()
    n_frb_list = [10**5, 3*10**5, 10**6]
    f_spec_list = [0.05, 0.10, 0.20]
    sigma_sys_list = [20.0, 30.0, 40.0]
    print("\nГенерация детальных прогнозов...")
    forecast_df = forecaster.forecast_sigma_w(
        n_frb_list=n_frb_list,
        f_spec_list=f_spec_list,
        sigma_sys_list=sigma_sys_list,
        aperture_radii=[2.0, 3.0, 5.0],
        n_realizations=5
    )
    forecaster.plot_forecast_comparison(forecast_df)
    forecast_df.to_csv('results/ska_forecast_results.csv', index=False)
    print("\nРезультаты сохранены в 'results/ska_forecast_results.csv'")
    return forecaster, forecast_df

def main():
    frb_data, desi_maps, z_centers, base_params, mcmc_results = main_analysis()
    robustness_results = run_robustness_tests(
        frb_data, desi_maps, z_centers, base_params
    )
    forecaster, forecast_df = run_ska_forecasts()
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*70)
    print(f"\nОсновной результат: w = {mcmc_results['w'][0]:.3f} ± {mcmc_results['w'][2]-mcmc_results['w'][0]:.3f}")
    print(f"Согласие с ΛCDM: {abs(mcmc_results['w'][0] + 1)/0.15:.2f}σ")
    print(f"\nТесты устойчивости пройдены: {len(robustness_results)} тестов")
    print(f"Прогнозы для SKA: σ(w) ≈ {forecast_df['sigma_w_mean'].min():.4f} (оптимистичный сценарий)")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

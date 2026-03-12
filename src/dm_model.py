import numpy as np
from scipy.integrate import quad
from scipy.constants import c, G, m_e, m_p

H0_PLANCK = 67.4
OMEGA_M_PLANCK = 0.315
OMEGA_B_H2 = 0.0224

def hubble_parameter(z, w, Omega_m=OMEGA_M_PLANCK, H0=H0_PLANCK):
    Omega_DE = 1 - Omega_m
    return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_DE * (1 + z)**(3 * (1 + w)))

def dm_igm(z, w, f_IGM=0.84, chi_e=1.0, Omega_m=OMEGA_M_PLANCK, 
           Omega_b_h2=OMEGA_B_H2, H0=H0_PLANCK):
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
        integral, _ = quad(integrand, 0, z)
    else:
        integral = np.array([quad(integrand, 0, zi)[0] for zi in z])
    return C * f_IGM * integral

def dm_host_model(z, mu_host=50.0, sigma_host=30.0):
    mean = mu_host / (1 + z)
    std = sigma_host / (1 + z)
    return mean, std

def dm_total(frb, w, params):
    z = frb.get('z', frb.get('z_mean', 0.5))
    dm_igm_val = dm_igm(
        z, w, 
        f_IGM=params.get('f_IGM', 0.84),
        chi_e=params.get('chi_e', 1.0),
        Omega_m=params.get('Omega_m', OMEGA_M_PLANCK),
        H0=params.get('H0', H0_PLANCK)
    )
    dm_host_mean, _ = dm_host_model(
        z, 
        params.get('mu_host', 50.0),
        params.get('sigma_host', 30.0)
    )
    dm_mw = frb.get('DM_MW_NE2001', 50.0)
    return dm_mw + dm_igm_val + dm_host_mean

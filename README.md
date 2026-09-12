# FwCC: FRB–w Cross-Correlation Pipeline

[![DOI](https://zenodo.org/badge/1293383709.svg)](https://doi.org/10.5281/zenodo.22725659)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-lightgrey.svg)]()
[![ApJ](https://img.shields.io/badge/ApJ-submitted%20Sep%202026-orange.svg)]()

> **Status:** Accompanying code for the manuscript *"FwCC: An FRB–Galaxy Cross-Correlation Pipeline and its Application to CHIME/FRB and DESI DR1"* by Anton Pushkin, submitted to **The Astrophysical Journal** (September 12, 2026). A preprint will be posted on **arXiv (astro-ph.CO)** within a few days. This software release is archived on Zenodo at [10.5281/zenodo.22725659](https://doi.org/10.5281/zenodo.22725659).

---

## Overview

**FwCC** (FRB–w Cross-Correlation) is an end-to-end analysis pipeline for measuring the angular cross-correlation between the extragalactic **Fast Radio Burst (FRB) dispersion measure (DM) field** and **foreground galaxy overdensity maps**, without assigning point redshifts to individual bursts. The pipeline is designed for cosmological inference of the dark energy equation-of-state parameter **w** and the ionised intergalactic baryon fraction **f_IGM**.

The current release (v2.0) applies the FwCC framework to **4,450 FRBs** from the CHIME/FRB Catalogue 2 and outriggers, cross-correlated with **DESI DR1** Bright Galaxy Survey (BGS) and Luminous Red Galaxy (LRG) tracers.

## Key Features

- **Event-weighted DM map construction** (Eq. 8–10 of the paper) that correctly handles sparse FRB sampling
- **Pseudo-C_ℓ estimation** via [`NaMaster`](https://github.com/LSSTDESC/NaMaster) with analytic mask mode-coupling correction
- **Cosmology-dependent N(z) precomputation** on a 3D grid over (w, Ω_m, f_IGM) with trilinear interpolation during MCMC
- **PCA compression** of spatial jackknife covariance to mitigate noise in poorly-constrained modes
- **Covariance-marginalised Sellentin–Heavens likelihood** (Sellentin & Heavens 2016)
- **Host-DM calibration** from 26 localized anchor FRBs with secure spectroscopic redshifts
- **Log-normal host-DM distribution** (Li et al. 2023) with redshift evolution
- **Geometric blinding** procedure to prevent confirmation bias
- **Conservative angular scale cuts** (0.37° < θ < 1.46°) and **DM-scrambling null tests**
- **Amplitude test** (Â) against the ΛCDM prediction
- **Scale-cut robustness tests** and multiple systematic checks

## Previous Version

This repository supersedes the earlier, simplified version of the pipeline described in:

> Pushkin, A. (2026). *"Resolving the Dark Energy Crisis with Fast Radio Bursts: A w-Measurement from CHIME and SDSS Cross-Correlation."* Zenodo. [doi:10.5281/zenodo.19200364](https://doi.org/10.5281/zenodo.19200364)

The basic formalism of the FwCC method was introduced there; the current release (v2.0) represents a complete rewrite with:
- Physically correct event-weighted map construction (filling-factor treatment)
- Article-matched angular scale cuts (0.37° < θ < 1.46°)
- Proper treatment of cosmology-dependent N(z)
- PCA-compressed jackknife covariance
- DESI DR1 (replacing SDSS) as the galaxy tracer
- Full Sellentin–Heavens likelihood
- Comprehensive mock-catalogue validation

The legacy code is preserved in the [`legacy/`](./legacy/) directory for reproducibility.

## Installation

### Requirements

- Python 3.9+
- [`numpy`](https://numpy.org/) ≥ 1.22
- [`scipy`](https://scipy.org/) ≥ 1.9
- [`pandas`](https://pandas.pydata.org/) ≥ 1.5
- [`matplotlib`](https://matplotlib.org/) ≥ 3.6
- [`healpy`](https://healpy.readthedocs.io/) ≥ 1.16
- [`astropy`](https://www.astropy.org/) ≥ 5.2
- [`emcee`](https://emcee.readthedocs.io/) ≥ 3.1
- [`NaMaster`](https://github.com/LSSTDESC/NaMaster) (pymaster) ≥ 0.1.9
- [`pyccl`](https://github.com/LSSTDESC/CCL) ≥ 2.7
- [`tqdm`](https://tqdm.github.io/) ≥ 4.64
- [`corner`](https://corner.readthedocs.io/) ≥ 2.2 *(optional, for corner plots)*
- [`arviz`](https://python.arviz.org/) ≥ 0.15 *(optional, for HDI computation)*

### Setup

```bash
git clone https://github.com/AntonPushkin260/fwcc-analysis.git
cd fwcc-analysis
pip install -r requirements.txt
```


## Usage

### Input Data
The pipeline requires the following publicly available data products in the repository root:

| File | Description | Source |
|------|-------------|--------|
| `final_catalog_v2.csv` | Combined FRB catalog (4,450 events) | Compiled from CHIME/FRB Cat. 2, Outriggers, DSA-110 |
| `DESI_BGS_BRIGHT_z05_03_delta_g.fits` | BGS overdensity map (HEALPix, N_side=64) | [DESI DR1](https://data.desi.lbl.gov/public/dr1/) |
| `DESI_BGS_BRIGHT_z05_03_mask.fits`    | BGS angular mask                         | [DESI DR1](https://data.desi.lbl.gov/public/dr1/) |
| `DESI_LRG_z30_80_delta_g.fits`        | LRG overdensity map (HEALPix, N_side=64) | [DESI DR1](https://data.desi.lbl.gov/public/dr1/) |
| `DESI_LRG_z30_80_mask.fits`           | LRG angular mask                         | [DESI DR1](https://data.desi.lbl.gov/public/dr1/) |
| `BGS_BRIGHT_NGC_nz.txt`               | BGS redshift distribution n(z)           | [DESI DR1](https://data.desi.lbl.gov/public/dr1/) |
| `LRG_NGC_nz.txt`                      | LRG redshift distribution n(z)           | [DESI DR1](https://data.desi.lbl.gov/public/dr1/) |


Users must obtain these data products from the respective collaboration archives:
 - CHIME/FRB Catalogue 2: https://chime-frb.ca/catalog
 - DESI Data Release 1: https://data.desi.lbl.gov/public/dr1/
 - DSA-110: Ravi et al. (2023), arXiv:2307.03344
   
### Running the Pipeline
```bash
python fwcc_pipeline.py
```
The pipeline executes a two-step inference:
1. **Step 1 — Host-DM calibration:** Constrains the host-galaxy DM distribution parameters (μ_host, σ_host, γ_host) from the 26 localized anchor FRBs using MCMC (`emcee`).
2. **Step 2 — Cosmological inference:** Jointly constrains (w, Ω_m, f_IGM, b_BGS, b_LRG, σ_loc) using the full 4,450-FRB sample cross-correlated with DESI DR1 tracers.

### Output
All outputs are written to output_FwCC/:
```
output_FwCC/
├── results.json           # Final parameter constraints and metadata
├── run.log                # Full pipeline log
├── blinding_offset.txt    # Geometric blinding parameter
├── plots/                 # Publication-quality figures
│   ├── posterior_w.png
│   ├── xi_data_theory_null.png
│   └── corner_step2.png
├── chains/                # MCMC chains (.npy)
│   ├── chains_step1.npy
│   └── chains_step2.npy
├── covariance/            # Jackknife covariance matrices
│   └── cov_jk.npy
└── matrices/              # Intermediate products
```



## Citation
If you use this code in your research, please cite the accompanying paper and the software release:
```
@article{Pushkin2026,
  author  = {Pushkin, Anton},
  title   = {{FwCC: An FRB--Galaxy Cross-Correlation Pipeline and its
             Application to CHIME/FRB and DESI DR1}},
  journal = {The Astrophysical Journal},
  year    = {2026},
  note    = {Submitted; arXiv preprint forthcoming}
}
```
**Software:**
```
@software{Pushkin2026FwCC_code,
  author    = {Pushkin, Anton},
  title     = {{FwCC: FRB-w Cross-Correlation Pipeline}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0.0},
  doi       = {10.5281/zenodo.22725659},
  url       = {https://github.com/AntonPushkin260/fwcc-analysis}
}
```
For the earlier preprint describing the basic FwCC formalism, see:
```
@misc{Pushkin2026v1,
  author    = {Pushkin, Anton},
  title     = {{Resolving the Dark Energy Crisis with Fast Radio Bursts:
               A w-Measurement from CHIME and SDSS Cross-Correlation}},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.19200364}
}
```

## Future Plans

This repository currently contains the complete analysis pipeline used for the accompanying ApJ submission. We are actively working on packaging FwCC as a proper Python library for easier installation and integration into other FRB cosmology workflows.

**Planned features:**
- **PyPI distribution:** `pip install fwcc` for one-command installation
- **Modular API:** Clean separation of map construction, pseudo-C_ℓ estimation, and MCMC inference
- **CLI interface:** Command-line tool for common analysis workflows
- **Extended tracer support:** Built-in loaders for DESI DR2, Euclid, and Rubin/LSST galaxy catalogues
- **Forward modelling:** Explicit circumgalactic medium (CGM) and halo gas profiles
- **Documentation:** Full Sphinx-based API documentation with Jupyter notebook tutorials
- **CI/CD:** Automated testing and continuous integration via GitHub Actions

**Timeline:** We aim to release v3.0.0 as a pip-installable package by Q4 2026.

## Contributing
This repository accompanies a submitted manuscript. After publication, bug reports, suggestions, and contributions will be welcome via GitHub Issues and Pull Requests.

## License
This project is licensed under the MIT License — see the [LICENSE](https://github.com/AntonPushkin260/fwcc-analysis/blob/main/LICENSE) file for details.


## Contact
Anton Pushkin,
Independent Researcher
 - Email: pushkin2601@mail.ru
 - [ORCID](https://orcid.org/0009-0006-3154-4168): 0009-0006-3154-4168
 - GitHub: github.com/AntonPushkin260

Last updated: September 12, 2026

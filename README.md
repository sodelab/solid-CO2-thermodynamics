# solid-CO2-thermodynamics
Interactive Jupyter workflow to compute and visualize the free‐energy phase diagram of solid CO2

This repository provides a fully documented, Jupyter‐driven workflow to compute the free‐energy phase diagram of CO₂ solid phases. The goal is to show how one can:

1. Optimize CO₂ crystal structures (in specified space groups) at various pressures.  
2. Compute phonon frequencies (and, if desired, phonon density of states or BZ integration) for each optimized structure.  
3. Evaluate free energy vs. temperature and volume/pressure for each phase.  
4. Assemble these free energies to determine phase stability ranges (T,P) and plot phase boundaries.  

> **Why CO₂?**  
> CO₂ has multiple solid phases (e.g., phases I, II, III, etc.) under different pressures and temperatures. By studying CO2 as a test case, students learn the full pipeline—from crystal structure relaxation to phonon-based free energies—while working with a small, well-characterized molecular solid.

---

## Contents

solid-CO2-thermodynamics/
├── README.md
├── LICENSE
├── .gitignore
├── notebooks/
│   ├── 00_setup_environment.ipynb
│   ├── 01_optimize_structures.ipynb
│   ├── 02_phonon_calculations.ipynb
│   ├── 03_free_energy_QHA.ipynb
│   └── 04_phase_diagram_plot.ipynb
├── scripts/
│   ├── optimize_structure.py
│   ├── phonon_run.py
│   ├── compute_free_energy.py
│   └── utils.py
├── data/
│   ├── raw_structures/         ← (e.g., CIF files for different phases/space groups)
│   ├── optimized_structures/   ← (output from optimizations, maybe POSCARs or Crystals)
│   ├── phonon_data/            ← (e.g., dynamical matrices/force constants)
│   └── free_energies/          ← (QHA results at each pressure)
├── docs/
│   ├── INSTALL.md              ← (conda/pip/virtualenv instructions, dependencies)
│   ├── USAGE.md                ← (how to run notebooks, run scripts, expected outputs)
│   └── THEORY.md               ← (short summary: QHA, why BZ sampling matters, approximations)
├── environment.yml             ← (Conda environment file, if you want reproducibility)
├── requirements.txt            ← (Python dependencies if you prefer pip)
└── tests/
    ├── test_optimize.py
    ├── test_phonon.py
    └── test_free_energy.py


- **[notebooks/](notebooks/)**  
  Interactive, step-by-step Jupyter notebooks for each major stage of the workflow.  
- **[scripts/](scripts/)**  
  Python scripts to run each stage (for batch or command-line usage).  
- **[data/](data/)**  
  - `raw_structures/`: Source CIFs for all CO₂ phases.  
  - `optimized_structures/`: Relaxed structures at target pressures.  
  - `phonon_data/`: Phonon outputs (frequencies, DOS).  
  - `free_energies/`: Computed free-energy tables.  
- **[docs/](docs/)**  
  - `INSTALL.md`: Installation instructions for Python packages, phonon codes, etc.  
  - `USAGE.md`: How to run notebooks & scripts.  
  - `THEORY.md`: Background on QHA and thermodynamic integration.  
- **[tests/](tests/)**  
  Unit tests for utility functions (e.g., free-energy integrator, structure readers).  
- **`environment.yml`** / **`requirements.txt`**  
  Exact Python environment specification (NumPy, SciPy, Phonopy, matplotlib, etc.).  

---

## Quick Start

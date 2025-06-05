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

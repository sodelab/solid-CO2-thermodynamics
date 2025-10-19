import numpy as np
from copy import deepcopy
import warnings
import logging

from typing import Any, Dict, Optional, Tuple, List

from ase import Atoms
from ase.io import write, read
from ase.spacegroup import crystal

import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from hiphive.structure_generation import (
    generate_mc_rattled_structures,
    generate_phonon_rattled_structures,
    generate_rattled_structures,
)
from hiphive import ClusterSpace, StructureContainer, ForceConstantPotential
from hiphive.utilities import prepare_structures

from trainstation import Optimizer

from utils_co2 import CO2TwoBodyRC_IMAGES

# Light logging via print
def log(msg: str):
    print(f"[QHA] {msg}")

# # ---- Save / load artifacts ----
def fcp_save(s_hash: str, fcp_array: np.ndarray):
    np.savez_compressed(f"fcp_{s_hash}.npz", fcp=fcp_array)
    log(f"Saved FCP → fcp_{s_hash}.npz")

def fcp_load(self, s_hash: str) -> Optional[np.ndarray]:
    p = f"fcp_{s_hash}.npz"
    if p.exists():
        data = np.load(p, allow_pickle=False)
        return data["fcp"]
    return None


def random_fcp(atoms, supercell, rattle_std, 
               cutoffs, n_structures, fit_method, ridge_alpha, 
               random_seed, s_hash, hiphive_config):
    """
    Generate FCP using random rattling.
    """
    import os

    prim_fit = deepcopy(atoms)
    prim_fit.set_constraint()
    prim_fit.calc = atoms.calc
    
    # Create supercell
    atoms_super = deepcopy(atoms).repeat(supercell)
    atoms_super.set_constraint()
    atoms_super.calc = atoms.calc
    
    # Validate cutoff
    rc = cutoffs[0]
    min_edge = np.min(np.linalg.norm(atoms_super.cell.array, axis=1))
    assert 2*rc < (min_edge - 1e-6), f"rc={rc} too large for edge={min_edge:.3f} Å"
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Generate structures
    structures = generate_rattled_structures(
        atoms_super, 
        n_structures=n_structures,
        rattle_std=rattle_std
    )
    
    # Calculate forces
    for at in structures:
        at.calc = CO2TwoBodyRC_IMAGES(rc=rc)
        #_ = at.get_forces()
        at.arrays['forces'] = at.get_forces()
        at.calc = None  # Clear calc to save memory

    
    # Setup hiphive
    cs = ClusterSpace(prim_fit, cutoffs)
    structures_prep = prepare_structures(structures, atoms_super)
    sc = StructureContainer(cs)
    for s in structures_prep:
        sc.add_structure(s)
    
    # Fit
    fit_kwargs = {'fit_method': fit_method, 'alpha': ridge_alpha}
    opt = Optimizer(sc.get_fit_data(), **fit_kwargs)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    
    # Save FCP if requested
    if hiphive_config.get('save_fcp', True):
        output_dirs = hiphive_config.get('output_dirs', {'fcps': './'})
        os.makedirs(output_dirs['fcps'], exist_ok=True)
        fcp.write(f"{output_dirs['fcps']}{s_hash}.fcp")
    
    # Convert to FC2 for phonopy
    unit_cell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell
    )
    ph = Phonopy(unit_cell, supercell_matrix=supercell)
    
    ase_sup = Atoms(
        cell=ph.supercell.cell,
        numbers=ph.supercell.numbers, 
        pbc=True,
        scaled_positions=ph.supercell.scaled_positions
    )
    
    fcs = fcp.get_force_constants(ase_sup)
    fc2 = fcs.get_fc_array(order=2)
    
    return fc2


def montecarlo_fcp(atoms, supercell, rattle_std, d_min, 
                   cutoffs, n_structures, fit_method, ridge_alpha, 
                   random_seed, s_hash, hiphive_config):
    """
    Generate FCP using Monte Carlo rattling.
    """
    import os

    prim_fit = deepcopy(atoms)
    prim_fit.set_constraint()
    prim_fit.calc = atoms.calc
    
    # Create supercell
    atoms_super = deepcopy(atoms).repeat(supercell)
    atoms_super.set_constraint()
    atoms_super.calc = atoms.calc

    # Validate cutoff
    rc = cutoffs[0]
    min_edge = np.min(np.linalg.norm(atoms_super.cell.array, axis=1))
    assert 2*rc < (min_edge - 1e-6), f"rc={rc} too large for edge={min_edge:.3f} Å"
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Generate structures
    structures = generate_mc_rattled_structures(
        atoms_super, 
        n_structures=n_structures,
        rattle_std=rattle_std, 
        d_min=d_min
    )
    
    # Calculate forces
    for at in structures:
        at.calc = CO2TwoBodyRC_IMAGES(rc=rc)
        #_ = at.get_forces()
        at.arrays['forces'] = at.get_forces()
        at.calc = None  # Clear calc to save memory
    
    
    # Setup hiphive
    cs = ClusterSpace(prim_fit, cutoffs)
    structures_prep = prepare_structures(structures, atoms_super)
    sc = StructureContainer(cs)
    for s in structures_prep:
        sc.add_structure(s)
    
    # Fit
    fit_kwargs = {'fit_method': fit_method, 'alpha': ridge_alpha}
    opt = Optimizer(sc.get_fit_data(), **fit_kwargs)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    
    # Save FCP if requested
    if hiphive_config.get('save_fcp', True):
        output_dirs = hiphive_config.get('output_dirs', {'fcps': './'})
        os.makedirs(output_dirs['fcps'], exist_ok=True)
        fcp.write(f"{output_dirs['fcps']}{s_hash}.fcp")
    
    # Convert to FC2 for phonopy
    unit_cell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell
    )
    ph = Phonopy(unit_cell, supercell_matrix=supercell)
    
    ase_sup = Atoms(
        cell=ph.supercell.cell,
        numbers=ph.supercell.numbers, 
        pbc=True,
        scaled_positions=ph.supercell.scaled_positions
    )
    
    fcs = fcp.get_force_constants(ase_sup)
    fc2 = fcs.get_fc_array(order=2)
    
    return fc2

def phonon_fcp(atoms, supercell, 
               cutoffs, n_structures, fit_method, ridge_alpha, 
               random_seed, s_hash, hiphive_config):
    """
    Generate FCP using phonon-based rattling with provided FC2.
    """
    import os
    
    # Check if FC2 is provided in config
    fc2 = hiphive_config.get('fc2')
    
    if fc2 is None:
        log(f"No FC2 provided for phonon rattling, generating fc2 from random_fcp.")
        # Fall back to random_fcp with the same parameters
        fc2 = random_fcp(
            atoms, supercell, 
            hiphive_config.get('rattle_std', 1e-3),  # Add default rattle_std for fallback
            cutoffs, n_structures, fit_method, ridge_alpha, 
            random_seed, s_hash, hiphive_config
        )
    else:
        # Continue with phonon rattling using the provided FC2
        log(f"Using phonon rattling with provided FC2 for hash {s_hash}")
        
    

    prim_fit = deepcopy(atoms)
    prim_fit.set_constraint()
    prim_fit.calc = atoms.calc
    
    # Create supercell
    atoms_super = deepcopy(atoms).repeat(supercell)
    atoms_super.set_constraint()
    atoms_super.calc = atoms.calc
    
    # Validate cutoff
    rc = cutoffs[0]
    min_edge = np.min(np.linalg.norm(atoms_super.cell.array, axis=1))
    assert 2*rc < (min_edge - 1e-6), f"rc={rc} too large for edge={min_edge:.3f} Å"
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Get phonon temperature and FC2 from config
    T = hiphive_config.get('phonon_temperature', 30.0)
    #fc2 = hiphive_config.get('fc2')  # FC2 should be provided in config

    # Generate phonon-rattled structures using the provided FC2
    structures_phonon_rattle = generate_phonon_rattled_structures(
        atoms_super, 
        fc2,
        n_structures, 
        T
    )
    
    # Calculate forces
    for at in structures_phonon_rattle:
        at.calc = CO2TwoBodyRC_IMAGES(rc=rc)
        #_ = at.get_forces()
        at.arrays['forces'] = at.get_forces()
        at.calc = None  # Clear calc to save memory

    
    # Setup hiphive
    cs = ClusterSpace(prim_fit, cutoffs)
    structures_prep = prepare_structures(structures_phonon_rattle, atoms_super)
    sc = StructureContainer(cs)
    for s in structures_prep:
        sc.add_structure(s)
    
    
    # Fit
    fit_kwargs = {'fit_method': fit_method, 'alpha': ridge_alpha}
    opt = Optimizer(sc.get_fit_data(), **fit_kwargs)
    opt.train()
    fcp = ForceConstantPotential(cs, opt.parameters)
    
    # Save FCP if requested
    if hiphive_config.get('save_fcp', True):
        output_dirs = hiphive_config.get('output_dirs', {'fcps': './'})
        os.makedirs(output_dirs['fcps'], exist_ok=True)
        fcp.write(f"{output_dirs['fcps']}{s_hash}.fcp")
    
    # Convert to FC2 for phonopy
    unit_cell = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        scaled_positions=atoms.get_scaled_positions(),
        cell=atoms.cell
    )
    ph = Phonopy(unit_cell, supercell_matrix=supercell)
    
    ase_sup = Atoms(
        cell=ph.supercell.cell,
        numbers=ph.supercell.numbers, 
        pbc=True,
        scaled_positions=ph.supercell.scaled_positions
    )
    
    fcs = fcp.get_force_constants(ase_sup)
    fc2 = fcs.get_fc_array(order=2)
    
    return fc2
    

def fc2_from_hiphive_config(atoms, hiphive_config, s_hash):
    """
    Modern interface: Generate FC2 from hiphive model using configuration dictionary.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Primitive structure
    hiphive_config : dict
        Complete hiphive configuration dictionary containing all parameters
    s_hash : str
        Structure hash for file naming
        
    Returns
    -------
    np.ndarray
        Force constants matrix (FC2)
    """
    # Extract parameters from config with defaults
    supercell = hiphive_config.get('supercell_size', (2, 2, 2))
    rattle_type = hiphive_config.get('rattle_type', 'mc').lower()
    rattle_std = hiphive_config.get('rattle_std', 1e-3)
    d_min = hiphive_config.get('d_min', 1.0)
    cutoffs = hiphive_config.get('cutoffs', [5.45, 4.00, 3.00])
    n_structures = hiphive_config.get('n_structures', 10)
    fit_method = hiphive_config.get('fit_method', 'ridge')
    ridge_alpha = hiphive_config.get('ridge_alpha', 1e-6)
    random_seed = hiphive_config.get('random_seed', 42)
    
    atoms.set_constraint()

    # Route to appropriate function based on rattle_type
    if rattle_type == "mc":
        return montecarlo_fcp(
            atoms, supercell, rattle_std, d_min, 
            cutoffs, n_structures, fit_method, ridge_alpha, 
            random_seed, s_hash, hiphive_config
        )
    elif rattle_type == "random":
        return random_fcp(
            atoms, supercell, rattle_std, 
            cutoffs, n_structures, fit_method, ridge_alpha, 
            random_seed, s_hash, hiphive_config
        )
    elif rattle_type == "phonon":
        return phonon_fcp(
            atoms, supercell, 
            cutoffs, n_structures, fit_method, ridge_alpha, 
            random_seed, s_hash, hiphive_config
        )
    else:
        raise ValueError(f"Unknown rattle_type: {rattle_type}. Choose from 'mc', 'random', 'phonon'")

def fc2_from_hiphive_model(atoms, super, RATTLE_TYPE, rattle_std, d_min, shash, hiphive_config=None):
    """
    Legacy interface: Generate FC2 from hiphive model (backward compatibility).
    
    This function maintains the old signature for backward compatibility but internally
    uses the new dictionary-based approach when hiphive_config is provided.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Primitive structure
    super : tuple
        Supercell dimensions
    RATTLE_TYPE : str
        Type of rattling ('mc', 'random', 'phonon')
    rattle_std : float
        Standard deviation for rattling
    d_min : float
        Minimum distance parameter
    shash : str
        Structure hash for file naming
    hiphive_config : dict, optional
        Complete hiphive configuration dictionary. If provided, parameters
        from this dict take precedence over individual arguments.
        
    Returns
    -------
    np.ndarray
        Force constants matrix (FC2)
    """
    
    # If hiphive_config is provided, use the modern approach
    if hiphive_config is not None:
        # Override config with any explicitly passed parameters
        config = hiphive_config.copy()
        config['supercell_size'] = super
        config['rattle_type'] = RATTLE_TYPE.lower()
        config['rattle_std'] = rattle_std
        config['d_min'] = d_min
        
        return fc2_from_hiphive_config(atoms, config, shash)
    
    # Legacy path: use default parameters when no config provided
    default_config = {
        'supercell_size': super,
        'rattle_type': RATTLE_TYPE.lower(),
        'rattle_std': rattle_std,
        'd_min': d_min,
        'cutoffs': [5.45, 4.00, 3.00],
        'n_structures': 10,
        'fit_method': 'ridge',
        'ridge_alpha': 1e-6,
        'random_seed': 42,
        'save_fcp': True,
        'output_dirs': {
            'fcps': './',
            'scph_trajs': './'
        }
    }
    
    return fc2_from_hiphive_config(atoms, default_config, shash)


# ---- Finite displacement fallback ----
def fc2_from_finite_displacements(unitcell_atoms: Atoms, super=(2,2,2), disp=0.01):
    ucell = phonopy.structure.atoms.PhonopyAtoms(symbols=unitcell_atoms.get_chemical_symbols(),
                                                 cell=unitcell_atoms.cell,
                                                 scaled_positions=unitcell_atoms.get_scaled_positions())
    ph = Phonopy(unitcell=ucell, supercell_matrix=super)
    ph.generate_displacements(distance=disp)

    forces = []
    for sc in ph.supercells_with_displacements:
        sc_ase = Atoms(numbers=sc.numbers, cell=sc.cell, pbc=True,
                       scaled_positions=sc.scaled_positions)
        sc_ase.calc = deepcopy(unitcell_atoms.calc)
        forces.append(sc_ase.get_forces())

    ph.forces = forces
    ph.produce_force_constants()
    return ph.force_constants

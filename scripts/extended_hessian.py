import numpy as np
import time
from energy import wrap_coordinates_by_carbon_fractional
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available, progress bars will be disabled")

try:
    from co2_potential import (
        p1b, p1b_gradient, p1b_hessian_rev, p1b_hessian_fwd,
        sapt, sapt_gradient, sapt_hessian_rev, sapt_hessian_fwd,
        p2b_4, p2b_gradient_4, p2b_hessian_4_rev, p2b_hessian_4_fwd,
        p2b_5, p2b_gradient_5, p2b_hessian_5_rev, p2b_hessian_5_fwd
    )
    ANALYTICAL_AVAILABLE = True
    print("Analytical gradients and Hessians available from co2_potential")
except ImportError:
    print("Warning: co2_potential analytical functions not available, using finite differences only")
    from co2_potential import p1b, sapt
    ANALYTICAL_AVAILABLE = False

# Add this to visualize and debug the Hessian matrix
import matplotlib.pyplot as plt
import numpy as np

def debug_hessian(hessian_data):
    # Plot the Hessian matrix as a heatmap
    print("Debugging Hessian matrix:")
    h = hessian_data
    plt.figure(figsize=(10, 10))
    plt.imshow(h, cmap='RdBu', interpolation='nearest')
    plt.colorbar()
    plt.title('Hessian Matrix')
    print("Hessian matrix shape:", h.shape)
    # Check for symmetry
    symmetry_diff = np.max(np.abs(h - h.T))
    print(f"Hessian symmetry check: max(|H - H^T|) = {symmetry_diff}")
    
    # Check eigenvalues
    eigvals = np.linalg.eigvalsh(h)
    print(f"Number of zero/negative eigenvalues: {sum(eigvals < 1e-10)}")
    print(f"Smallest eigenvalues: {eigvals[:6]}")
    
    # Check if acoustic sum rule is satisfied
    n_atoms = h.shape[0] // 3
    for i in range(3):  # For each direction x,y,z
        for j in range(n_atoms):
            row_sum = sum(h[3*j+i, 3*k:3*k+3] for k in range(n_atoms))
            print(f"Sum rule check atom {j}, dir {i}: {np.sum(row_sum):.6f}")


def enforce_asr_extended_hessian(extended_hessian, verbose=False):
    """
    Enforce Acoustic Sum Rule (ASR) explicitly on a dictionary-based extended Hessian.

    Parameters:
        extended_hessian (dict): 
            Keys are tuples (ix, iy, iz) representing lattice cell indices.
            Values are Hessian matrices (3n x 3n numpy arrays).
            The key (0,0,0) is the central cell Hessian after neighbor subtraction.

    Returns:
        dict: ASR-corrected extended Hessian dictionary.
    """

    # Make a copy to avoid mutating original data
    extended_hessian = {cell: hess.copy() for cell, hess in extended_hessian.items()}

    central_key = (0, 0, 0)
    central_hessian = extended_hessian[central_key]

    n3 = central_hessian.shape[0]
    natoms = n3 // 3

    # Reshape central Hessian to (natoms, 3, natoms, 3)
    central_hessian_rs = central_hessian.reshape(natoms, 3, natoms, 3)

    # Initialize sum over neighbor cells
    sum_neighbors = np.zeros_like(central_hessian_rs)

    for shift, neighbor_hessian in extended_hessian.items():
        if shift == central_key:
            continue  # Skip central cell
        neighbor_rs = neighbor_hessian.reshape(natoms, 3, natoms, 3)
        sum_neighbors += neighbor_rs

    # Enforce ASR explicitly on central Hessian
    for j in range(natoms):
        for a in range(3):
            # Sum over all neighbor interactions
            neighbor_sum = sum_neighbors[j, a, :, :].sum(axis=0)  # shape (3,)

            # Sum over off-diagonal terms within central cell
            central_offdiag_sum = np.sum(
                [central_hessian_rs[j, a, k, :] for k in range(natoms) if k != j],
                axis=0
            )

            # Set diagonal term explicitly to enforce ASR
            central_hessian_rs[j, a, j, :] = -(neighbor_sum + central_offdiag_sum)

    # Update central Hessian back into dictionary
    extended_hessian[central_key] = central_hessian_rs.reshape(n3, n3)

    # Optional: Verify ASR explicitly
    corrected_central = extended_hessian[central_key]
    if verbose:
        print("After ASR enforcement (dictionary structure):")
    for j in range(natoms):
        for a in range(3):
            row_sum = corrected_central[3*j + a, :].sum()
            print(f" atom {j}, dir {a}: row-sum = {row_sum:.3e}")

    return extended_hessian


def energy_extended(nfrags, crd, pbc, potential='sapt'):
    """
    Energy function that mirrors the C++ energy function structure.
    
    Parameters:
    -----------
    nfrags : int
        Number of fragments (molecules)
    crd : np.array
        Flattened coordinates array [nfrags * 9] (9 coords per CO2 molecule)
    pbc : np.array
        Periodic boundary conditions [a, b, c]
    potential : str
        Type of potential to use ('sapt', 'p2b_4', 'p2b_5')
        
    Returns:
    --------
    float : Total energy
    """
    nrg = 0.0
    
    # Select potential function
    if potential == 'sapt':
        p2b_func = sapt
    elif potential == 'p2b_4':
        p2b_func = p2b_4
    elif potential == 'p2b_5':
        p2b_func = p2b_5
    else:
        raise ValueError(f"Unknown potential: {potential}")
    
    # 1-body terms: p1b for each fragment
    for ifrag in range(nfrags):
        fragment_coords = crd[ifrag*9:(ifrag+1)*9]
        nrg += p1b(fragment_coords)
    
    # 2-body terms: interactions across periodic images
    for x in range(-1, 2):  # -1, 0, 1
        for y in range(-1, 2):
            for z in range(-1, 2):
                
                for ifrag in range(nfrags):
                    for jfrag in range(nfrags):
                        
                        # Extract fragment coordinates
                        mol_i = crd[ifrag*9:(ifrag+1)*9].copy()
                        mol_j = crd[jfrag*9:(jfrag+1)*9].copy()
                        
                        # Apply periodic shifts to second molecule
                        # Each molecule has 3 atoms × 3 coordinates
                        for p in range(3):  # 3 atoms per molecule
                            mol_j[p*3 + 0] += x * pbc[0]  # x coordinate
                            mol_j[p*3 + 1] += y * pbc[1]  # y coordinate  
                            mol_j[p*3 + 2] += z * pbc[2]  # z coordinate
                        
                        # Set interaction factor
                        factor = 0.5
                        if x == 0 and y == 0 and z == 0:
                            factor = 1.0
                            if ifrag >= jfrag:
                                continue
                        
                        # Combine molecules for dimer calculation
                        dimer_coords = np.concatenate([mol_i, mol_j])
                        nrg += p2b_func(dimer_coords) * factor
    
    return nrg

def convert_eigenvalues_to_frequencies(eigenvalues: np.ndarray, 
                                     hessian_units: str = "kcal/mol/angstrom2/amu") -> np.ndarray:
    """
    Convert eigenvalues to frequencies in cm^-1.
    
    Args:
        eigenvalues: Eigenvalues of the mass-weighted Hessian
        hessian_units: Units of the Hessian matrix
        
    Returns:
        Frequencies in cm^-1
    """
    # Define conversion factor based on Hessian units
    if hessian_units.lower() == "kcal/mol/angstrom2/amu":
        # 1 kcal/mol/Å²/amu to cm^-1: approximately 108.6
        conversion_factor = 108.6
    elif hessian_units.lower() == "ev/angstrom2/amu":
        # 1 eV/Å²/amu to cm^-1: approximately 521.47
        conversion_factor = 521.47
    else:
        raise ValueError(f"Unsupported Hessian units: {hessian_units}")
    
    # Convert eigenvalues to frequencies
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * conversion_factor
    
    return frequencies

def mass_weight_hessian(hessian, masses, nfrags):
    """
    Apply mass weighting to the Hessian matrix.
    
    Parameters:
    -----------
    hessian : np.array
        Unweighted Hessian matrix
    masses : np.array
        Atomic masses for each atom (3 copies for x,y,z coords)
    nfrags : int
        Number of molecular fragments
        
    Returns:
    --------
    np.array : Mass-weighted Hessian matrix
    """
    # For CO2: C=12.011, O=15.999 amu
    co2_masses = np.array([12.011, 15.999, 15.999])  # C, O, O
    
    # Create mass array for all coordinates
    mass_array = np.zeros(nfrags * 9)
    for ifrag in range(nfrags):
        for iatom in range(3):  # 3 atoms per CO2
            for icoord in range(3):  # x, y, z coordinates
                mass_array[ifrag*9 + iatom*3 + icoord] = co2_masses[iatom]
    
    # Mass weight the Hessian: H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
    mass_weighted_hessian = hessian.copy()
    # Convert nested loops to vectorized operations
    mass_factors = 1.0 / np.sqrt(np.outer(mass_array, mass_array))
    mass_weighted_hessian = hessian * mass_factors
    
    return mass_weighted_hessian


def mass_weight_extended_hessian(extended_hessian: dict, nfrags: int) -> dict:
    """
    Apply mass-weighting to all matrices in the extended Hessian dictionary.
    
    Parameters:
    -----------
    extended_hessian : dict
        Dictionary of extended Hessian matrices with shift tuples as keys
    nfrags : int
        Number of molecular fragments
        
    Returns:
    --------
    dict
        Dictionary of mass-weighted extended Hessian matrices
    """
    # For CO2: C=12.011, O=15.999 amu
    co2_masses = np.array([12.011, 15.999, 15.999])  # C, O, O
    
    # Create mass array for all coordinates
    mass_array = np.zeros(nfrags * 9)
    for ifrag in range(nfrags):
        for iatom in range(3):  # 3 atoms per CO2
            for icoord in range(3):  # x, y, z coordinates
                mass_array[ifrag*9 + iatom*3 + icoord] = co2_masses[iatom]
    
    # Create a new dictionary for mass-weighted extended Hessians
    mass_weighted_extended = {}
    
    # Loop through all shifts and apply mass-weighting to each Hessian
    for shift_key, hessian in extended_hessian.items():
        # Mass weight the Hessian: H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
        mw_hessian = hessian.copy()
        for i in range(len(mass_array)):
            for j in range(len(mass_array)):
                mw_hessian[i, j] /= np.sqrt(mass_array[i] * mass_array[j])
        
        # Store in the new dictionary
        mass_weighted_extended[shift_key] = mw_hessian
    
    return mass_weighted_extended

def compute_extended_hessian(structure, stepsize=0.005, method='mixed', potential='sapt', 
                           use_structure_energy=False, pressure=0.0, 
                           verbose=False, use_tqdm=True):
    """
    Compute extended Hessian matrix using analytical and/or finite differences.
    Mirrors the C++ xderivative function structure.
    
    Parameters:
    -----------
    structure : pymatgen.Structure
        Crystal structure
    stepsize : float
        Finite difference step size (default 0.005 like C++)
    method : str
        'analytical' - use analytical Hessians where available
        'finite_diff' - use finite differences
        'mixed' - use analytical for gamma point, finite diff for extended
    potential : str
        Type of potential to use ('sapt', 'p2b_4', 'p2b_5')
    use_structure_energy : bool
        If True, use structure-based energy, if False use flattened coordinate energy
    pressure : float
        Applied pressure (currently not implemented)
    verbose : bool
        If True, print detailed information during calculation
    use_tqdm : bool
        If True, show progress bars (requires tqdm library)
        
    Returns:
    --------
    tuple : (gamma_point_hessian, extended_hessian_dict)
        gamma_point_hessian : np.array
            Hessian matrix for gamma point (central cell)
        extended_hessian_dict : dict
            Extended Hessian elements for different lattice shifts
    """
    
    if verbose:
        print(f"Computing extended Hessian using method: {method}, potential: {potential}")
    start_time = time.time()
    
    # Get molecular structure and coordinates
    updated_cart_coords, molecule_assignment, molecules_grouped = wrap_coordinates_by_carbon_fractional(structure)
    
    nfrags = len(molecules_grouped)
    ncoord = nfrags * 9  # 9 coordinates per molecule
    
    # Flatten coordinates for energy function
    crd = np.zeros(ncoord)
    for i, group in enumerate(molecules_grouped):
        for j, atom_idx in enumerate(group):
            crd[i*9 + j*3:(i*9 + j*3 + 3)] = updated_cart_coords[atom_idx]
    
    # Periodic boundary conditions
    pbc = np.array([structure.lattice.a, structure.lattice.b, structure.lattice.c])
    
    # Initialize extended Hessian storage
    extended_hessian = {}
    
    if verbose:
        print(f"System info: {nfrags} molecules, {ncoord} coordinates")
        print(f"PBC: {pbc}")
    
    # === GAMMA POINT HESSIAN ===
    if (method == 'analytical' or method == 'mixed') and ANALYTICAL_AVAILABLE:
        gamma_hessian = compute_gamma_hessian_analytical(crd, nfrags, molecules_grouped, 
                                                      potential, verbose, use_tqdm)
    else:
        gamma_hessian = compute_gamma_hessian_finite_diff(crd, nfrags, pbc, stepsize, 
                                                       potential, verbose, use_tqdm)
    
    tmp_gamma_hessian = gamma_hessian.copy()

    if verbose:
        print(f"Gamma point Hessian computed using {'analytical' if method in ['analytical', 'mixed'] and ANALYTICAL_AVAILABLE else 'finite differences'}")
    
    # === EXTENDED HESSIAN FOR PERIODIC IMAGES ===
    if verbose:
        print(f"Computing extended Hessian for periodic images...")
    
    # Create shift combinations
    shifts = [(x-1, y-1, z-1) for x in range(3) for y in range(3) for z in range(3)]
    # Remove the central cell (0,0,0)
    shifts.remove((0,0,0))
    
    # Process shifts with or without progress bar
    if use_tqdm and TQDM_AVAILABLE:
        shift_iterator = tqdm(shifts, desc="Processing lattice shifts")
    else:
        shift_iterator = shifts
    
    for shift_key in shift_iterator:
        x, y, z = shift_key[0], shift_key[1], shift_key[2]  # Convert back to 0,1,2 format
        
        # Lattice shift vector
        lat = np.array([x, y, z], dtype=float)
        
        # Compute extended Hessian for this shift
        if method == 'analytical' and ANALYTICAL_AVAILABLE:
            shift_hessian = compute_shift_hessian_analytical(
                crd, nfrags, pbc, lat, potential, verbose, use_tqdm
            )
        else:
            shift_hessian = compute_shift_hessian_finite_diff(
                crd, nfrags, pbc, lat, stepsize, potential, verbose, use_tqdm
            )
        
        # Store extended Hessian for this shift
        extended_hessian[shift_key] = shift_hessian

        # Subtract from gamma point (as in C++ code)
        tmp_gamma_hessian -= shift_hessian
        
        if verbose:
            print(f"Completed shift {shift_key}")
    
    extended_hessian[(0, 0, 0)] = tmp_gamma_hessian.copy()  # Store gamma point Hessian at (0,0,0)
    
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Extended Hessian computation completed in {elapsed_time:.2f} seconds")
    
    return gamma_hessian, extended_hessian

def compute_gamma_hessian_analytical(crd, nfrags, molecules_grouped, potential='sapt', 
                                   verbose=False, use_tqdm=True):
    """
    Compute gamma point Hessian using analytical methods.
    
    Parameters:
    -----------
    crd : np.array
        Flattened coordinates array
    nfrags : int
        Number of fragments (molecules)
    molecules_grouped : list
        List of molecule atom indices
    potential : str
        Type of potential to use
    verbose : bool
        If True, print detailed information
    use_tqdm : bool
        If True, show progress bars
    """
    ncoord = nfrags * 9
    gamma_hessian = np.zeros((ncoord, ncoord))
    
    # Select functions based on potential
    if potential == 'sapt':
        p1b_hess_func = p1b_hessian_fwd
        p2b_hess_func = sapt_hessian_fwd
    elif potential == 'p2b_4':
        p1b_hess_func = p1b_hessian_fwd
        p2b_hess_func = p2b_hessian_4_fwd
    elif potential == 'p2b_5':
        p1b_hess_func = p1b_hessian_fwd
        p2b_hess_func = p2b_hessian_5_fwd
    else:
        raise ValueError(f"Unknown potential: {potential}")
    
    if verbose:
        print("Computing 1-body contributions to gamma point Hessian analytically...")
    
    # Create fragment iterator with or without progress bar
    if use_tqdm and TQDM_AVAILABLE:
        frag_iterator = tqdm(range(nfrags), desc="1-body Hessians")
    else:
        frag_iterator = range(nfrags)
    
    # 1-body contributions (monomer Hessians)
    for ifrag in frag_iterator:
        i_start, i_end = ifrag * 9, (ifrag + 1) * 9
        fragment_coords = crd[i_start:i_end]
        
        # Get analytical Hessian for this fragment
        fragment_hessian = p1b_hess_func(fragment_coords)
        
        # Add to gamma point Hessian
        gamma_hessian[i_start:i_end, i_start:i_end] += fragment_hessian
    
    if verbose:
        print("Computing 2-body contributions to gamma point Hessian analytically...")
    
    # Create dimer pair iterator with or without progress bar
    dimer_pairs = [(i, j) for i in range(nfrags) for j in range(i+1, nfrags)]
    if use_tqdm and TQDM_AVAILABLE:
        pair_iterator = tqdm(dimer_pairs, desc="2-body Hessians")
    else:
        pair_iterator = dimer_pairs
    
    # 2-body contributions (dimer Hessians within central cell)
    for ifrag, jfrag in pair_iterator:
        i_start, i_end = ifrag * 9, (ifrag + 1) * 9
        j_start, j_end = jfrag * 9, (jfrag + 1) * 9
        
        # Create dimer coordinates
        mol_i = crd[i_start:i_end]
        mol_j = crd[j_start:j_end]
        dimer_coords = np.concatenate([mol_i, mol_j])
        
        # Get analytical Hessian for this dimer
        dimer_hessian = p2b_hess_func(dimer_coords)
        
        # Map dimer Hessian to global Hessian
        # dimer_hessian is 18x18, need to map to appropriate blocks
        gamma_hessian[i_start:i_end, i_start:i_end] += dimer_hessian[:9, :9]      # i-i block
        gamma_hessian[i_start:i_end, j_start:j_end] += dimer_hessian[:9, 9:]      # i-j block
        gamma_hessian[j_start:j_end, i_start:i_end] += dimer_hessian[9:, :9]      # j-i block
        gamma_hessian[j_start:j_end, j_start:j_end] += dimer_hessian[9:, 9:]      # j-j block
    
    return gamma_hessian

def compute_gamma_hessian_finite_diff(crd, nfrags, pbc, stepsize, potential='sapt', 
                                    verbose=False, use_tqdm=True):
    """
    Compute gamma point Hessian using finite differences.
    
    Parameters:
    -----------
    crd : np.array
        Flattened coordinates array
    nfrags : int
        Number of fragments (molecules)
    pbc : np.array
        Periodic boundary conditions
    stepsize : float
        Finite difference step size
    potential : str
        Type of potential to use
    verbose : bool
        If True, print detailed information
    use_tqdm : bool
        If True, show progress bars
    """
    ncoord = nfrags * 9
    gamma_hessian = np.zeros((ncoord, ncoord))
    
    if verbose:
        print("Computing gamma point Hessian using finite differences...")
    
    # Create coordinate iterator with or without progress bar
    coord_pairs = [(n0, n1) for n0 in range(ncoord) for n1 in range(ncoord)]
    if use_tqdm and TQDM_AVAILABLE:
        pair_iterator = tqdm(coord_pairs, desc="Finite diff Hessian")
    else:
        pair_iterator = range(ncoord)
        if verbose:
            print(f"  Processing {ncoord}x{ncoord} = {ncoord**2} coordinate pairs...")
    
    if use_tqdm and TQDM_AVAILABLE:
        # Use tqdm for the outer loop
        for n0, n1 in pair_iterator:
            d0 = np.zeros(2)
            for i in range(2):
                d1 = np.zeros(2)
                for j in range(2):
                    refmol = crd.copy()
                    refmol[n0] += stepsize if i else -stepsize
                    refmol[n1] += stepsize if j else -stepsize
                    
                    d1[j] = energy_extended(nfrags, refmol, pbc, potential)
                
                d0[i] = (d1[1] - d1[0]) / (2 * stepsize)
            
            dd = (d0[1] - d0[0]) / (2 * stepsize)
            gamma_hessian[n0, n1] = dd
            gamma_hessian[n1, n0] = dd
    else:
        # When not using tqdm, use a nested loop but potentially with verbose output
        for n0 in range(ncoord):
            if verbose and n0 % max(1, ncoord//10) == 0:
                print(f"  Processing coordinate {n0}/{ncoord} ({100*n0/ncoord:.1f}%)...")
                
            for n1 in range(ncoord):
                d0 = np.zeros(2)
                for i in range(2):
                    d1 = np.zeros(2)
                    for j in range(2):
                        refmol = crd.copy()
                        refmol[n0] += stepsize if i else -stepsize
                        refmol[n1] += stepsize if j else -stepsize
                        
                        d1[j] = energy_extended(nfrags, refmol, pbc, potential)
                    
                    d0[i] = (d1[1] - d1[0]) / (2 * stepsize)
                
                dd = (d0[1] - d0[0]) / (2 * stepsize)
                gamma_hessian[n0, n1] = dd
    
    return gamma_hessian

def compute_shift_hessian_analytical(crd, nfrags, pbc, lat, potential='sapt',
                                   verbose=False, use_tqdm=True):
    """
    Compute Hessian for a specific lattice shift using analytical derivatives.
    
    Parameters:
    -----------
    crd : np.array
        Flattened coordinates array
    nfrags : int
        Number of fragments (molecules)
    pbc : np.array
        Periodic boundary conditions
    lat : np.array
        Lattice shift vector
    potential : str
        Type of potential to use ('sapt', 'p2b_4', 'p2b_5')
    verbose : bool
        If True, print detailed information
    use_tqdm : bool
        If True, show progress bars
        
    Returns:
    --------
    np.ndarray
        Shift Hessian matrix for the specified lattice shift
    """
    ncoord = nfrags * 9
    shift_hessian = np.zeros((ncoord, ncoord))
    
    # Select potential functions based on potential type
    if potential == 'sapt':
        p2b_hess_func = sapt_hessian_fwd
    elif potential == 'p2b_4':
        p2b_hess_func = p2b_hessian_4_fwd
    elif potential == 'p2b_5':
        p2b_hess_func = p2b_hessian_5_fwd
    else:
        raise ValueError(f"Unknown potential: {potential}")
    
    shift_str = f"({lat[0]}, {lat[1]}, {lat[2]})"
    if verbose:
        print(f"Computing shift Hessian for {shift_str} using analytical derivatives...")
    
    # Create fragment pair iterator with or without progress bar
    frag_pairs = [(i, j) for i in range(nfrags) for j in range(nfrags)]
    if use_tqdm and TQDM_AVAILABLE:
        frag_iterator = tqdm(frag_pairs, desc=f"Shift {shift_str} (analytical)")
    else:
        frag_iterator = frag_pairs
    
    for ifrag, jfrag in frag_iterator:
        # Extract fragment coordinates
        i_start, i_end = ifrag * 9, (ifrag + 1) * 9
        j_start, j_end = jfrag * 9, (jfrag + 1) * 9
        
        mol_i = crd[i_start:i_end].copy()
        mol_j = crd[j_start:j_end].copy()
        
        # Apply lattice shift to second molecule
        for p in range(9):
            coord_type = p % 3  # x, y, or z coordinate
            mol_j[p] += lat[coord_type] * pbc[coord_type]
        
        # Compute analytical Hessian for this dimer
        dimer_coords = np.concatenate([mol_i, mol_j])
        dimer_hessian = p2b_hess_func(dimer_coords)
        
        # Map the dimer Hessian blocks to the shift Hessian
        shift_hessian[i_start:i_end, i_start:i_end] += dimer_hessian[:9, :9]   # i-i block
        shift_hessian[i_start:i_end, j_start:j_end] += dimer_hessian[:9, 9:]   # i-j block
        shift_hessian[j_start:j_end, i_start:i_end] += dimer_hessian[9:, :9]   # j-i block
        shift_hessian[j_start:j_end, j_start:j_end] += dimer_hessian[9:, 9:]   # j-j block
    
    # Ensure the shift Hessian is symmetric (should already be from analytical functions,
    # but enforce it to be safe)
    shift_hessian = 0.5 * (shift_hessian + shift_hessian.T)
    
    return shift_hessian

def compute_shift_hessian_finite_diff(crd, nfrags, pbc, lat, stepsize, potential='sapt',
                                    verbose=False, use_tqdm=True):
    """
    Compute Hessian for a specific lattice shift using finite differences.
    
    Parameters:
    -----------
    crd : np.array
        Flattened coordinates array
    nfrags : int
        Number of fragments (molecules)
    pbc : np.array
        Periodic boundary conditions
    lat : np.array
        Lattice shift vector
    stepsize : float
        Finite difference step size
    potential : str
        Type of potential to use
    verbose : bool
        If True, print detailed information
    use_tqdm : bool
        If True, show progress bars
    """
    ncoord = nfrags * 9
    shift_hessian = np.zeros((ncoord, ncoord))
    
    # Select potential function
    if potential == 'sapt':
        p2b_func = sapt
    elif potential == 'p2b_4':
        p2b_func = p2b_4
    elif potential == 'p2b_5':
        p2b_func = p2b_5
    else:
        raise ValueError(f"Unknown potential: {potential}")
    
    shift_str = f"({lat[0]}, {lat[1]}, {lat[2]})"
    if verbose:
        print(f"Computing shift Hessian for {shift_str} using finite differences...")
    
    # Create fragment pair iterator with or without progress bar
    frag_pairs = [(i, j) for i in range(nfrags) for j in range(nfrags)]
    if use_tqdm and TQDM_AVAILABLE:
        frag_iterator = tqdm(frag_pairs, desc=f"Shift {shift_str}")
    else:
        frag_iterator = frag_pairs
    
    for ifrag, jfrag in frag_iterator:
        # Coordinate ranges for each molecule (9 coords each)
        i_start, i_end = ifrag * 9, (ifrag + 1) * 9
        j_start, j_end = jfrag * 9, (jfrag + 1) * 9
        
        for n0 in range(9):  # 9 coordinates per molecule
            for n1 in range(9):
                
                global_n0 = ifrag * 9 + n0
                global_n1 = jfrag * 9 + n1
                
                d0 = np.zeros(2)
                for i in range(2):
                    d1 = np.zeros(2)
                    for j in range(2):
                        
                        # Extract molecule coordinates
                        refmol1 = crd[i_start:i_end].copy()
                        refmol2 = crd[j_start:j_end].copy()
                        
                        # Perturb coordinates
                        refmol1[n0] += stepsize if i else -stepsize
                        refmol2[n1] += stepsize if j else -stepsize
                        
                        # Apply lattice shift to second molecule
                        # Shift each atom (3 coords per atom, 3 atoms per molecule)
                        for p in range(9):
                            coord_type = p % 3  # x, y, or z coordinate
                            refmol2[p] += lat[coord_type] * pbc[coord_type]
                        
                        # Compute dimer energy
                        refdim = np.concatenate([refmol1, refmol2])
                        d1[j] = p2b_func(refdim)
                    
                    d0[i] = (d1[1] - d1[0]) / (2 * stepsize)
                
                dd = (d0[1] - d0[0]) / (2 * stepsize)
                shift_hessian[global_n0, global_n1] = dd
                shift_hessian[global_n1, global_n0] = dd
    
    return shift_hessian

def compute_hessian_at_structure(structure, stepsize=0.005, method='mixed', potential='sapt',
                               verbose=False, use_tqdm=False):
    """
    Convenience function to compute Hessian for a given structure.
    
    Parameters:
    -----------
    structure : pymatgen.Structure
        Crystal structure
    stepsize : float
        Finite difference step size
    method : str
        'analytical' - use analytical Hessians where available
        'finite_diff' - use finite differences  
        'mixed' - use analytical for gamma point, finite diff for extended
    potential : str
        Type of potential to use ('sapt', 'p2b_4', 'p2b_5')
    verbose : bool
        If True, print detailed information during calculation
    use_tqdm : bool
        If True, show progress bars (requires tqdm library)
        
    Returns:
    --------
    dict : Results containing Hessian matrices and eigenvalues
    """
    
    gamma_hessian, extended_hessian = compute_extended_hessian(
        structure, stepsize, method, potential, verbose=verbose, use_tqdm=use_tqdm
    )
    
    # Get number of fragments
    _, _, molecules_grouped = wrap_coordinates_by_carbon_fractional(structure)
    nfrags = len(molecules_grouped)
    
    # Mass weight the Hessian
    mass_weighted_gamma = mass_weight_hessian(gamma_hessian, None, nfrags)

    # Mass weight the extended Hessian matrices
    mass_weighted_extended = mass_weight_extended_hessian(extended_hessian, nfrags)
    
    # Compute eigenvalues for stability analysis
    if verbose:
        print("Computing eigenvalues and eigenvectors...")

    eigenvals, eigenvects = np.linalg.eigh(mass_weighted_gamma)

    # Sum all mass-weighted extended Hessian matrices into one
    summed_mass_weighted_extended = np.zeros_like(mass_weighted_gamma)
    for hess in mass_weighted_extended.values():
        summed_mass_weighted_extended += hess

    # Compute eigenvalues and eigenvectors of the summed extended Hessian
    eigvals_ext, eigvecs_ext = np.linalg.eigh(summed_mass_weighted_extended)

    # Compare to mass_weighted_gamma
    if verbose:
        print("\nComparison of summed mass-weighted extended Hessian to mass_weighted_gamma:")
        diff = np.abs(summed_mass_weighted_extended - mass_weighted_gamma)
        print(f"  Max abs diff: {np.max(diff):.8e}")
        print(f"  RMS diff: {np.sqrt(np.mean(diff**2)):.8e}")
        print(f"  Max eigval diff: {np.max(np.abs(eigvals_ext - eigenvals)):.8e}")
    
    # Convert eigenvalues to frequencies
    frequencies = convert_eigenvalues_to_frequencies(eigenvals, "kcal/mol/angstrom2/amu")
    
    results = {
        'gamma_hessian': gamma_hessian,
        'mass_weighted_gamma': mass_weighted_gamma,
        'extended_hessian': extended_hessian,
        'mass_weighted_extended': mass_weighted_extended,
        'eigenvalues': eigenvals,
        'frequencies_cm1': frequencies,
        'eigenvectors': eigenvects,
        'n_negative_modes': np.sum(eigenvals < -1e-6),
        'n_imaginary_freqs': np.sum(frequencies < 0),
        'stepsize': stepsize,
        'method': method,
        'potential': potential
    }

    if verbose:
        print(f"\nHessian analysis (method: {method}, potential: {potential}):")
        print(f"  Matrix size: {gamma_hessian.shape[0]}x{gamma_hessian.shape[1]}")
        print(f"  Number of negative eigenvalues: {results['n_negative_modes']}")
        print(f"  Number of imaginary frequencies: {results['n_imaginary_freqs']}")
        print(f"  Smallest eigenvalue: {np.min(eigenvals):.8f}")
        print(f"  Largest eigenvalue: {np.max(eigenvals):.8f}")
        print(f"  Lowest frequency: {np.min(frequencies):.2f} cm^-1")
        print(f"  Highest frequency: {np.max(frequencies):.2f} cm^-1")
        
        # Print frequency summary
        positive_freqs = frequencies[frequencies > 1e-3]  # Remove near-zero frequencies
        if len(positive_freqs) > 0:
            print(f"  Frequency range (positive): {np.min(positive_freqs):.2f} to {np.max(positive_freqs):.2f} cm^-1")
        
        # Print some eigenvalues and frequencies
        print(f"\nFirst 10 eigenvalues:")
        for i in range(min(10, len(eigenvals))):
            print(f"  {i+1:2d}: {eigenvals[i]:12.8f} -> {frequencies[i]:8.2f} cm^-1")
        
        if len(eigenvals) > 10:
            print(f"\nLast 10 eigenvalues:")
            for i in range(max(0, len(eigenvals)-10), len(eigenvals)):
                print(f"  {i+1:2d}: {eigenvals[i]:12.8f} -> {frequencies[i]:8.2f} cm^-1")
        
        # Print Hessian matrix statistics
        print(f"\nHessian matrix statistics:")
        print(f"  Minimum element: {np.min(gamma_hessian):.8f}")
        print(f"  Maximum element: {np.max(gamma_hessian):.8f}")
        print(f"  Mean absolute element: {np.mean(np.abs(gamma_hessian)):.8f}")
        print(f"  Frobenius norm: {np.linalg.norm(gamma_hessian):.8f}")
        
        extended_hessian = enforce_asr_extended_hessian(extended_hessian, verbose)
        debug_hessian(gamma_hessian)

        # Check symmetry
        symmetry_error = np.max(np.abs(gamma_hessian - gamma_hessian.T))
        print(f"  Symmetry error: {symmetry_error:.2e}")
    
    return results

def compare_hessian_methods(structure, stepsize=0.005, potential='sapt'):
    """
    Compare analytical vs finite difference Hessian calculations.
    """
    if not ANALYTICAL_AVAILABLE:
        print("Analytical methods not available - cannot compare")
        return None
    
    print("Comparing analytical vs finite difference Hessian methods...")
    
    # Compute using both methods
    results_analytical = compute_hessian_at_structure(
        structure, stepsize, 'analytical', potential
    )
    results_finite_diff = compute_hessian_at_structure(
        structure, stepsize, 'finite_diff', potential
    )
    
    # Compare gamma point Hessians
    diff = results_analytical['gamma_hessian'] - results_finite_diff['gamma_hessian']
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"Gamma point Hessian comparison:")
    print(f"  Maximum absolute difference: {max_diff:.8f}")
    print(f"  RMS difference: {rms_diff:.8f}")
    
    # Compare eigenvalues
    eig_diff = results_analytical['eigenvalues'] - results_finite_diff['eigenvalues']
    max_eig_diff = np.max(np.abs(eig_diff))
    
    print(f"  Maximum eigenvalue difference: {max_eig_diff:.8f}")
    
    # Compare frequencies
    freq_diff = results_analytical['frequencies_cm1'] - results_finite_diff['frequencies_cm1']
    max_freq_diff = np.max(np.abs(freq_diff))
    
    print(f"  Maximum frequency difference: {max_freq_diff:.2f} cm^-1")
    
    return {
        'analytical': results_analytical,
        'finite_diff': results_finite_diff,
        'max_diff': max_diff,
        'rms_diff': rms_diff,
        'max_eig_diff': max_eig_diff,
        'max_freq_diff': max_freq_diff
    }


# Alternative: Save everything in a single .npz file
def save_hessian_all_in_one(results, filename="hessian_all", include_metadata=True, verbose=False):
    """
    Save all Hessian results in a single .npz file.
    
    Parameters:
        results (dict): Results from compute_hessian_at_structure
        filename (str): Base output filename (without extension)
        include_metadata (bool): Whether to save metadata
    
    Returns:
        str: Path to the saved file
    """
    import json
    import time
    import os
    
    # Ensure the base path exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Prepare data for saving
    save_data = {
        'gamma_hessian': results['gamma_hessian'],
        'eigenvalues': results['eigenvalues'],
        'eigenvectors': results['eigenvectors'],
        'mass_weighted_gamma': results['mass_weighted_gamma'],
        'frequencies': results['frequencies_cm1'],
    }
    # Add extended Hessian matrices
    for shift_key, shift_hessian in results['extended_hessian'].items():
        shift_str = f"extended_{shift_key[0]}_{shift_key[1]}_{shift_key[2]}"
        save_data[shift_str] = shift_hessian

    # Add mass-weighted extended Hessian matrices
    for shift_key, shift_hessian in results['mass_weighted_extended'].items():
        shift_str = f"mass_weighted_extended_{shift_key[0]}_{shift_key[1]}_{shift_key[2]}"
        save_data[shift_str] = shift_hessian
    
    # Save all data in compressed format
    full_filename = f"{filename}.npz"
    np.savez_compressed(full_filename, **save_data)
    
    # Save metadata separately if requested
    if include_metadata:
        metadata = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'method': results.get('method', 'unknown'),
            'potential': results.get('potential', 'unknown'),
            'stepsize': float(results.get('stepsize', 0.005)),
            'n_negative_modes': int(results.get('n_negative_modes', 0)),
            'n_imaginary_freqs': int(results.get('n_imaginary_freqs', 0)),
            'gamma_hessian_shape': list(results['gamma_hessian'].shape),
            'data_keys': list(save_data.keys())
        }
        metadata['extended_shifts'] = [[int(k[0]), int(k[1]), int(k[2])] 
                                for k in results['extended_hessian'].keys()]
        
        metadata['eigenvalue_stats'] = {
            'min': float(np.min(results['eigenvalues'])),
            'max': float(np.max(results['eigenvalues'])),
            'count': int(len(results['eigenvalues']))
        }
    
        # Save metadata to JSON file
        meta_filename = f"{filename}_metadata.json"
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=2)

    if verbose:
        print(f"Saved all Hessian data to {full_filename}")
    return full_filename

def load_hessian_all_in_one(filename, load_metadata=True, verbose=False):
    """
    Load all Hessian results from a single .npz file.
    
    Parameters:
        filename (str): Path to the .npz file
        load_metadata (bool): Whether to load metadata (if available)
    
    Returns:
        tuple: (results_dict, metadata) where metadata may be None
    """
    import os
    import json
    
    if not filename.endswith('.npz'):
        filename += '.npz'
    
    # Load data from npz file
    with np.load(filename) as data:
        results = {
            'gamma_hessian': data['gamma_hessian'],
            'eigenvalues': data['eigenvalues'],
            'eigenvectors': data['eigenvectors'],
            'mass_weighted_gamma': data['mass_weighted_gamma'],
            'frequencies_cm1': data['frequencies']

        }

        # Load extended Hessian matrices
        extended_hessian = {}
        mass_weighted_extended = {}
        
        for key in data.files:
            # Load regular extended Hessian matrices
            if key.startswith('extended_'):
                # Extract shift key from data key
                shift_str = key.replace('extended_', '')
                shift_parts = shift_str.split('_')
                if len(shift_parts) == 3:
                    try:
                        shift_key = tuple(int(p) for p in shift_parts)
                        extended_hessian[shift_key] = data[key]
                    except ValueError:
                        print(f"Warning: Could not parse shift key from {key}")
            
            # Load mass-weighted extended Hessian matrices
            elif key.startswith('mass_weighted_extended_'):
                shift_str = key.replace('mass_weighted_extended_', '')
                shift_parts = shift_str.split('_')
                if len(shift_parts) == 3:
                    try:
                        shift_key = tuple(int(p) for p in shift_parts)
                        mass_weighted_extended[shift_key] = data[key]
                    except ValueError:
                        print(f"Warning: Could not parse shift key from {key}")
        
        if extended_hessian:
            results['extended_hessian'] = extended_hessian
            if verbose:
                print(f"Loaded {len(extended_hessian)} extended Hessian matrices")
        
        if mass_weighted_extended:
            results['mass_weighted_extended'] = mass_weighted_extended
            if verbose:
                print(f"Loaded {len(mass_weighted_extended)} mass-weighted extended Hessian matrices")
    
    # Load metadata
    metadata = None
    if load_metadata:
        base_filename = filename.replace('.npz', '')
        meta_filename = f"{base_filename}_metadata.json"
        if os.path.exists(meta_filename):
            with open(meta_filename, 'r') as f:
                metadata = json.load(f)
    
    return results, metadata

def save_hessian_h5(results, filename="hessian_data", include_metadata=True, verbose=False):
    """
    Save all Hessian results in a single HDF5 file.
    
    Parameters:
        results (dict): Results from compute_hessian_at_structure
        filename (str): Base output filename (without extension)
        include_metadata (bool): Whether to save metadata as attributes
        verbose (bool): Whether to print verbose output
    
    Returns:
        str: Path to the saved file
    """
    import time
    import os
    import h5py
    
    # Ensure the base path exists
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Add .h5 extension if not present
    full_filename = f"{filename}.h5" if not filename.endswith('.h5') else filename
    
    with h5py.File(full_filename, 'w') as f:
        # Create main groups
        main_group = f.create_group('hessian_data')
        extended_group = main_group.create_group('extended_hessian')
        mass_weighted_group = main_group.create_group('mass_weighted_extended')
        
        # Save core arrays
        main_arrays = {
            'gamma_hessian': results['gamma_hessian'],
            'eigenvalues': results['eigenvalues'],
            'eigenvectors': results['eigenvectors'],
            'mass_weighted_gamma': results['mass_weighted_gamma'],
            'frequencies_cm1': results['frequencies_cm1']
        }
        
        for key, array in main_arrays.items():
            main_group.create_dataset(key, data=array, compression="gzip", compression_opts=9)
        
        # Save extended Hessian matrices
        for shift_key, shift_hessian in results['extended_hessian'].items():
            shift_str = f"{shift_key[0]}_{shift_key[1]}_{shift_key[2]}"
            extended_group.create_dataset(shift_str, data=shift_hessian, 
                                        compression="gzip", compression_opts=9)
        
        # Save mass-weighted extended Hessian matrices
        for shift_key, shift_hessian in results['mass_weighted_extended'].items():
            shift_str = f"{shift_key[0]}_{shift_key[1]}_{shift_key[2]}"
            mass_weighted_group.create_dataset(shift_str, data=shift_hessian, 
                                             compression="gzip", compression_opts=9)
        
        # Add metadata as attributes if requested
        if include_metadata:
            attrs = main_group.attrs
            attrs['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            attrs['method'] = results.get('method', 'unknown')
            attrs['potential'] = results.get('potential', 'unknown')
            attrs['stepsize'] = float(results.get('stepsize', 0.005))
            attrs['n_negative_modes'] = int(results.get('n_negative_modes', 0))
            attrs['n_imaginary_freqs'] = int(results.get('n_imaginary_freqs', 0))
            attrs['gamma_hessian_shape'] = str(list(results['gamma_hessian'].shape))
            
            # Save extended shifts
            shift_list = []
            for shift_key in results['extended_hessian'].keys():
                shift_list.append(f"{shift_key[0]},{shift_key[1]},{shift_key[2]}")
            
            attrs['extended_shifts'] = ';'.join(shift_list)
            attrs['eigenvalue_min'] = float(np.min(results['eigenvalues']))
            attrs['eigenvalue_max'] = float(np.max(results['eigenvalues']))
            attrs['eigenvalue_count'] = int(len(results['eigenvalues']))
    
    if verbose:
        print(f"Saved all Hessian data to {full_filename}")
    
    return full_filename

def load_hessian_h5(filename, load_metadata=True, verbose=False):
    """
    Load all Hessian results from an HDF5 file.
    
    Parameters:
        filename (str): Path to the HDF5 file
        load_metadata (bool): Whether to load metadata
        verbose (bool): Whether to print verbose output
    
    Returns:
        tuple: (results_dict, metadata) where metadata may be None
    """
    import h5py
    import os
    
    # Add .h5 extension if not present
    filename = f"{filename}.h5" if not filename.endswith('.h5') else filename
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"HDF5 file not found: {filename}")
    
    with h5py.File(filename, 'r') as f:
        # Check if the expected structure exists
        if 'hessian_data' not in f:
            raise ValueError(f"Invalid HDF5 file format: missing 'hessian_data' group")
        
        main_group = f['hessian_data']
        results = {}
        
        # Load core arrays
        for key in ['gamma_hessian', 'eigenvalues', 'eigenvectors', 'mass_weighted_gamma', 'frequencies_cm1']:
            if key in main_group:
                results[key] = main_group[key][()]
            elif key == 'frequencies_cm1' and 'frequencies' in main_group:
                # Handle possible name variations
                results[key] = main_group['frequencies'][()]
        
        # Load extended Hessian matrices
        if 'extended_hessian' in main_group:
            extended_group = main_group['extended_hessian']
            extended_hessian = {}
            
            for shift_str in extended_group:
                shift_parts = shift_str.split('_')
                if len(shift_parts) == 3:
                    try:
                        shift_key = tuple(int(p) for p in shift_parts)
                        extended_hessian[shift_key] = extended_group[shift_str][()]
                    except ValueError:
                        if verbose:
                            print(f"Warning: Could not parse shift key from {shift_str}")
            
            results['extended_hessian'] = extended_hessian
            if verbose:
                print(f"Loaded {len(extended_hessian)} extended Hessian matrices")
        
        # Load mass-weighted extended Hessian matrices
        if 'mass_weighted_extended' in main_group:
            mass_weighted_group = main_group['mass_weighted_extended']
            mass_weighted_extended = {}
            
            for shift_str in mass_weighted_group:
                shift_parts = shift_str.split('_')
                if len(shift_parts) == 3:
                    try:
                        shift_key = tuple(int(p) for p in shift_parts)
                        mass_weighted_extended[shift_key] = mass_weighted_group[shift_str][()]
                    except ValueError:
                        if verbose:
                            print(f"Warning: Could not parse shift key from {shift_str}")
            
            results['mass_weighted_extended'] = mass_weighted_extended
            if verbose:
                print(f"Loaded {len(mass_weighted_extended)} mass-weighted extended Hessian matrices")
        
        # Load metadata if requested
        metadata = None
        if load_metadata and hasattr(main_group, 'attrs'):
            attrs = main_group.attrs
            metadata = {key: attrs[key] for key in attrs}
            
            # Convert shape string back to list if needed
            if 'gamma_hessian_shape' in metadata and isinstance(metadata['gamma_hessian_shape'], str):
                try:
                    import ast
                    metadata['gamma_hessian_shape'] = ast.literal_eval(metadata['gamma_hessian_shape'])
                except (ValueError, SyntaxError):
                    if verbose:
                        print("Warning: Could not parse gamma_hessian_shape attribute")
            
            # Convert extended shifts string back to list
            if 'extended_shifts' in metadata:
                try:
                    shifts_str = metadata['extended_shifts']
                    shifts = []
                    for shift_str in shifts_str.split(';'):
                        x, y, z = map(int, shift_str.split(','))
                        shifts.append([x, y, z])
                    metadata['extended_shifts'] = shifts
                except (ValueError, AttributeError):
                    if verbose:
                        print("Warning: Could not parse extended_shifts attribute")
    
    if verbose:
        print(f"Successfully loaded data from {filename}")
        print(f"  Gamma Hessian shape: {results['gamma_hessian'].shape}")
        if 'extended_hessian' in results:
            print(f"  Extended Hessians: {len(results['extended_hessian'])}")
        if 'mass_weighted_extended' in results:
            print(f"  Mass-weighted Extended Hessians: {len(results['mass_weighted_extended'])}")
    
    return results, metadata

# Update the test function to demonstrate the new options
if __name__ == "__main__":
    from spacegroup import Pa3
    from symmetry import build_unit_cell
    import argparse
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Extended Hessian calculation and testing')
    parser.add_argument('--method', choices=['finite_diff', 'analytical', 'mixed'], 
                      default='finite_diff', help='Method for Hessian calculation')
    parser.add_argument('--potential', choices=['sapt', 'p2b_4', 'p2b_5'], 
                      default='sapt', help='Potential function to use')
    parser.add_argument('--stepsize', type=float, default=0.005, help='Finite difference step size')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--no-tqdm', dest='use_tqdm', action='store_false', 
                      help='Disable progress bars')
    parser.add_argument('--test-save-load', action='store_true', 
                      help='Test save/load functionality')
    parser.add_argument('--format', choices=['npz', 'h5', 'both'], default='both',
                      help='File format for saving Hessian data')
    
    args = parser.parse_args()
    
    # Test with Pa3 structure
    print("Computing extended Hessian...")
    
    pa3_structure = Pa3(a=5.48469436)
    pa3_structure.adjust_fractional_coords(bond_length=1.1609)
    structure = build_unit_cell(pa3_structure)
    
    # Check if method is available
    if args.method in ['analytical', 'mixed'] and not ANALYTICAL_AVAILABLE:
        print(f"Warning: {args.method} method requested but analytical functions not available.")
        print("Switching to finite_diff method.")
        args.method = 'finite_diff'
    
    # Check if potential is available
    if args.potential in ['p2b_4', 'p2b_5'] and not ANALYTICAL_AVAILABLE:
        print(f"Warning: {args.potential} potential requested but not available.")
        print("Switching to sapt potential.")
        args.potential = 'sapt'
    
    try:
        # Calculate Hessian
        results = compute_hessian_at_structure(
            structure, 
            stepsize=args.stepsize, 
            method=args.method, 
            potential=args.potential,
            verbose=args.verbose,
            use_tqdm=args.use_tqdm
        )
        
        # Save files based on requested format
        if args.format in ['npz', 'both']:
            npz_file = save_hessian_all_in_one(results, filename="test_hessian", 
                                             include_metadata=True, verbose=args.verbose)
            print(f"NPZ file saved: {npz_file}")

        if args.format in ['h5', 'both']:
            h5_file = save_hessian_h5(results, filename="test_hessian", 
                                    include_metadata=True, verbose=args.verbose)
            print(f"HDF5 file saved: {h5_file}")

        # Test data integrity if requested
        if args.test_save_load:
            print("\nTesting data integrity...")
            
            if args.format in ['npz', 'both']:
                npz_data, npz_meta = load_hessian_all_in_one("test_hessian", verbose=args.verbose)
                npz_match = np.allclose(results['gamma_hessian'], npz_data['gamma_hessian'])
                print(f"NPZ integrity check: {'PASSED' if npz_match else 'FAILED'}")
            
            if args.format in ['h5', 'both']:
                h5_data, h5_meta = load_hessian_h5("test_hessian", verbose=args.verbose)
                h5_match = np.allclose(results['gamma_hessian'], h5_data['gamma_hessian'])
                print(f"HDF5 integrity check: {'PASSED' if h5_match else 'FAILED'}")
            
    except Exception as e:
        print(f"Error with {args.method}/{args.potential}: {e}")
        import traceback
        traceback.print_exc()

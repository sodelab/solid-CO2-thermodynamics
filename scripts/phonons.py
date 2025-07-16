import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Union
import time
import os
import scipy.optimize
from scipy import constants
from scipy.interpolate import InterpolatedUnivariateSpline

# Physical constants in appropriate units
KB = constants.k  # Boltzmann constant in J/K
HBAR = constants.hbar  # Reduced Planck constant in J·s
AMU = constants.atomic_mass  # Atomic mass unit in kg
AVOGADRO = constants.N_A  # Avogadro's number
J_TO_KCAL = 1.0 / 4184.0  # Conversion from J to kcal
CM1_TO_KCALMOL = 0.0028591  # Conversion from cm^-1 to kcal/mol
EH_TO_KCALMOL = 627.5095  # Conversion from Hartree to kcal/mol
ANGSTROM3_TO_CM3 = 1e-24  # Conversion from Å³ to cm³

def write_extended_hessian_txt(extended_hessian, outfilename):
    """
    Write extended Hessians to a text file in Hartree/Å² units, flattened, with shift keys.
    """
    KCALMOL_TO_HARTREE = 1.0 / 627.509474  # 1 kcal/mol = 1/627.509474 Hartree

    # Ensure (0,0,0) is included (if not, add a zero matrix)
    all_shifts = set(extended_hessian.keys())
    n = next(iter(extended_hessian.values())).shape[0]
    if (0, 0, 0) not in all_shifts:
        extended_hessian[(0, 0, 0)] = np.zeros((n, n))

    # Sort shifts lexicographically
    for shift in sorted(extended_hessian.keys()):
        hess = extended_hessian[shift]
        # Convert to Hartree/Å²
        hess_hartree = hess * KCALMOL_TO_HARTREE
        # Flatten row-major
        flat = hess_hartree.flatten()
        # Write to file
        with open(outfilename, 'a') as f:
            f.write(f"{shift[0]} {shift[1]} {shift[2]}\n")
            f.write(''.join(f"{x:.8f}\n" for x in flat))

def calculate_gamma_phonons(hessian_data, structure, remove_acoustic=True):
    """
    Calculate phonon frequencies at the gamma point (q=0,0,0).
    
    Parameters:
    -----------
    hessian_data : dict
        Dictionary containing Hessian matrix and related data
    structure : Structure
        Pymatgen Structure object
    remove_acoustic : bool
        Whether to project out acoustic modes
        
    Returns:
    --------
    dict
        Dictionary containing frequencies and eigenvectors
    """
    print("Calculating gamma point phonons...")
    
    # Extract data from the hessian_data dictionary
    if 'mass_weighted_gamma' in hessian_data:
        hessian = hessian_data['mass_weighted_gamma']
    else:
        hessian = hessian_data['hessian']
        # Mass-weight the Hessian if not already done
        masses = []
        for site in structure:
            mass = site.specie.atomic_mass * AMU  # Convert to kg
            masses.extend([mass, mass, mass])  # One mass for each direction (x,y,z)
        
        masses = np.array(masses)
        mass_matrix = np.outer(1.0/np.sqrt(masses), 1.0/np.sqrt(masses))
        hessian = hessian * mass_matrix
    
    # Diagonalize the Hessian to get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    
    # Convert eigenvalues to frequencies (cm^-1)
    # For eigenvalues in kcal/mol/Å² units
    frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * 108.6
    
    # Sort by frequency
    idx = np.argsort(frequencies)
    frequencies = frequencies[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # If requested, project out acoustic modes (set first 3 to ~0)
    if remove_acoustic and len(frequencies) > 3:
        frequencies[:3] = 0.0
    
    return {
        'frequencies': frequencies,
        'eigenvectors': eigenvectors
    }

def calculate_phonon_dispersion(hessian_data, structure, path=None, symm_points=None, n_points=50, remove_acoustic=True):
    """
    Calculate phonon dispersion along a path in the Brillouin zone.
    
    Parameters:
    -----------
    hessian_data : dict
        Dictionary containing Hessian matrix and related data
    structure : Structure
        Pymatgen Structure object
    path : list
        List of high-symmetry point labels to visit (e.g., ['G', 'X', 'M', 'G'])
    symm_points : dict
        Dictionary mapping labels to k-point coordinates
    n_points : int
        Number of points along each path segment
    remove_acoustic : bool
        Whether to project out acoustic modes
        
    Returns:
    --------
    dict
        Dictionary containing dispersion data
    """
    print("Calculating phonon dispersion...")
    
    # Get high symmetry points if not provided
    if symm_points is None:
        symm_points = get_high_symmetry_points(structure.lattice)
    
    # Define default path if not provided
    if path is None:
        path = ['G', 'X', 'M', 'G', 'R', 'X']
    
    # Generate k-points along the path
    q_points = []
    q_labels = []
    q_positions = [0]
    distances = []
    
    current_distance = 0.0
    previous_point = None
    
    for i, label in enumerate(path):
        if label not in symm_points:
            raise ValueError(f"Unknown high symmetry point: {label}")
        
        point = symm_points[label]
        q_labels.append(label)
        
        if i < len(path) - 1:
            next_label = path[i+1]
            next_point = symm_points[next_label]
            
            # Generate points along this segment
            for j in range(n_points if i < len(path) - 2 else n_points + 1):
                t = j / n_points
                k_point = (1-t) * point + t * next_point
                q_points.append(k_point)
                
                # Calculate distance for the x-axis
                if previous_point is not None:
                    segment_length = np.linalg.norm(k_point - previous_point)
                    current_distance += segment_length
                
                distances.append(current_distance)
                previous_point = k_point
            
            q_positions.append(current_distance)
    
    # Convert lists to arrays
    q_points = np.array(q_points)
    distances = np.array(distances)
    
    # Extract Hessian data
    if 'mass_weighted_gamma' in hessian_data and 'mass_weighted_extended' in hessian_data:
        gamma_hessian = hessian_data['mass_weighted_gamma']
        extended_hessian = hessian_data['mass_weighted_extended']
    else:
        gamma_hessian = hessian_data['hessian']
        extended_hessian = hessian_data['extended_hessian']
        # Would need to mass-weight here if not already done
    
    # Calculate frequencies along the path
    frequencies = compute_phonon_dispersion(gamma_hessian, extended_hessian, q_points)
    
    # If requested, project out acoustic modes
    if remove_acoustic and frequencies.shape[1] > 3:
        # Set the first 3 modes to zero at Gamma point
        gamma_idx = np.where(np.all(q_points == symm_points['G'], axis=1))[0]
        if len(gamma_idx) > 0:
            for idx in gamma_idx:
                frequencies[idx, :3] = 0.0
    
    return {
        'q_points': q_points,
        'distances': distances,
        'frequencies': frequencies,
        'q_labels': q_labels,
        'q_positions': q_positions
    }

def calculate_phonon_dos(hessian_data, structure, mesh=(10, 10, 10), 
                        energy_range=(0, 300), energy_step=1.0, remove_acoustic=True):
    """
    Calculate phonon density of states (DOS).
    
    Parameters:
    -----------
    hessian_data : dict
        Dictionary containing Hessian matrix and related data
    structure : Structure
        Pymatgen Structure object
    mesh : tuple
        k-point mesh dimensions
    energy_range : tuple
        Min and max energy range in cm^-1
    energy_step : float
        Energy step size in cm^-1
    remove_acoustic : bool
        Whether to project out acoustic modes
        
    Returns:
    --------
    dict
        Dictionary containing DOS data
    """
    print(f"Calculating phonon DOS with {mesh[0]}×{mesh[1]}×{mesh[2]} mesh...")
    
    # Extract Hessian data
    if 'mass_weighted_gamma' in hessian_data and 'mass_weighted_extended' in hessian_data:
        gamma_hessian = hessian_data['mass_weighted_gamma']
        extended_hessian = hessian_data['mass_weighted_extended']
    else:
        gamma_hessian = hessian_data['hessian']
        extended_hessian = hessian_data['extended_hessian']
        # Would need to mass-weight here if not already done
    
    # Generate k-point mesh
    k_mesh = generate_kpoint_mesh(mesh[0])
    
    # Calculate frequencies on the mesh
    frequencies = compute_phonon_dispersion(gamma_hessian, extended_hessian, k_mesh)
    
    # If requested, exclude acoustic modes
    if remove_acoustic and frequencies.shape[1] > 3:
        # Remove the lowest 3 modes at each k-point
        # This is a simplification - for a proper treatment we would need to identify
        # the acoustic modes based on their dispersion behavior
        frequencies = frequencies[:, 3:]
    
    # Compute DOS
    energies, dos = compute_dos(
        frequencies, 
        bin_width=energy_step,
        freq_range=energy_range,
        include_negative=True
    )
    
    return {
        'energies': energies,
        'dos': dos,
        'mesh': mesh
    }

def get_high_symmetry_points(lattice, lattice_type='cubic'):
    """
    Get high symmetry points for a given lattice.
    
    Parameters:
    -----------
    lattice : Lattice
        Pymatgen Lattice object
    lattice_type : str
        Type of lattice ('cubic', 'fcc', 'bcc', 'tetragonal', 'orthorhombic', etc.)
        
    Returns:
    --------
    dict
        Dictionary mapping labels to k-point coordinates
    """
    # For now, we'll support several lattice types with simple definitions
    
    if lattice_type.lower() == 'cubic':
        return {
            'G': np.array([0.0, 0.0, 0.0]),  # Gamma
            'X': np.array([0.5, 0.0, 0.0]),  # X
            'M': np.array([0.5, 0.5, 0.0]),  # M
            'R': np.array([0.5, 0.5, 0.5])   # R
        }
    elif lattice_type.lower() == 'fcc':
        return {
            'G': np.array([0.0, 0.0, 0.0]),   # Gamma
            'X': np.array([0.5, 0.0, 0.5]),   # X
            'L': np.array([0.5, 0.5, 0.5]),   # L
            'W': np.array([0.5, 0.25, 0.75])  # W
        }
    elif lattice_type.lower() == 'bcc':
        return {
            'G': np.array([0.0, 0.0, 0.0]),   # Gamma
            'H': np.array([0.5, 0.5, 0.5]),   # H
            'N': np.array([0.0, 0.0, 0.5]),   # N
            'P': np.array([0.25, 0.25, 0.25]) # P
        }
    elif lattice_type.lower() == 'tetragonal':
        return {
            'G': np.array([0.0, 0.0, 0.0]),   # Gamma
            'X': np.array([0.5, 0.0, 0.0]),   # X
            'M': np.array([0.5, 0.5, 0.0]),   # M
            'Z': np.array([0.0, 0.0, 0.5]),   # Z
            'R': np.array([0.5, 0.5, 0.5]),   # R
            'A': np.array([0.5, 0.5, 0.0])    # A (sometimes labeled as M)
        }
    elif lattice_type.lower() == 'orthorhombic':
        return {
            'G': np.array([0.0, 0.0, 0.0]),   # Gamma
            'X': np.array([0.5, 0.0, 0.0]),   # X
            'Y': np.array([0.0, 0.5, 0.0]),   # Y
            'Z': np.array([0.0, 0.0, 0.5]),   # Z
            'S': np.array([0.5, 0.5, 0.0]),   # S
            'T': np.array([0.0, 0.5, 0.5]),   # T
            'U': np.array([0.5, 0.0, 0.5]),   # U
            'R': np.array([0.5, 0.5, 0.5])    # R
        }
    else:
        print(f"Warning: Lattice type '{lattice_type}' not recognized. Using cubic high-symmetry points.")
        return {
            'G': np.array([0.0, 0.0, 0.0]),
            'X': np.array([0.5, 0.0, 0.0]),
            'M': np.array([0.5, 0.5, 0.0]),
            'R': np.array([0.5, 0.5, 0.5])
        }


def compute_phonon_dispersion(gamma_hessian: np.ndarray, 
                             extended_hessian: Dict[Tuple[int, int, int], np.ndarray],
                             kpoints: np.ndarray) -> np.ndarray:
    """
    Compute phonon frequencies at specified k-points.
    
    Parameters:
    -----------
    gamma_hessian : np.ndarray
        Mass-weighted Hessian matrix at the gamma point
    extended_hessian : Dict[Tuple[int, int, int], np.ndarray]
        Extended Hessian matrices for different lattice shifts
    kpoints : np.ndarray
        k-points at which to compute the phonon frequencies, shape (n_kpoints, 3)
    
    Returns:
    --------
    np.ndarray
        Phonon frequencies at each k-point in cm^-1, shape (n_kpoints, n_modes)
    """

    n_dof = gamma_hessian.shape[0]  # Number of degrees of freedom
    n_kpoints = kpoints.shape[0]
    
    # Pre-allocate output array
    frequencies = np.zeros((n_kpoints, n_dof), dtype=np.complex128)
    
    # Loop over k-points
    for ik, k in enumerate(kpoints):
        # Start with gamma point Hessian
        dynamical_matrix = gamma_hessian.copy().astype(np.complex128)
        
        # Add contributions from extended Hessian with phase factors
        for shift, hessian in extended_hessian.items():
            # Calculate phase factor exp(i k·r)
            phase = np.exp(1j * 2 * np.pi * np.dot(k, shift))
            
            # Add contribution to dynamical matrix
            dynamical_matrix += hessian * phase
        
        # Ensure the dynamical matrix is Hermitian
        dynamical_matrix = 0.5 * (dynamical_matrix + dynamical_matrix.conj().T)
        
        # Diagonalize to get frequencies
        eigenvalues = np.linalg.eigvalsh(dynamical_matrix)
        
        # Convert eigenvalues to frequencies (cm^-1)
        # For CO2 potential in kcal/mol/Å² units, the conversion factor is ~108.6
        frequencies[ik] = np.sign(eigenvalues.real) * np.sqrt(np.abs(eigenvalues.real)) * 108.6
    
    return frequencies.real


def generate_kpoint_mesh(nk: int = 10) -> np.ndarray:
    """
    Generate a uniform k-point mesh in the Brillouin zone.
    
    Parameters:
    -----------
    nk : int
        Number of k-points along each direction
    
    Returns:
    --------
    np.ndarray
        k-points in the Brillouin zone, shape (nk^3, 3)
    """
    # Create a 3D grid of k-points from 0 to 1
    kx = np.linspace(0, 1, nk, endpoint=False)
    ky = np.linspace(0, 1, nk, endpoint=False)
    kz = np.linspace(0, 1, nk, endpoint=False)
    
    # Create the mesh grid
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz)
    
    # Reshape to get a list of k-points
    kpoints = np.vstack([kx_grid.flatten(), ky_grid.flatten(), kz_grid.flatten()]).T
    
    return kpoints


def generate_high_symmetry_path(n_points: int = 100) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Generate a path through high-symmetry points in the Brillouin zone.
    
    Parameters:
    -----------
    n_points : int
        Number of points along each segment of the path
    
    Returns:
    --------
    Tuple containing:
        - np.ndarray: k-points along the path
        - List[str]: Labels for high-symmetry points
        - List[int]: Positions of high-symmetry points in the path
    """
    # Define high-symmetry points for cubic systems
    high_symmetry_points = {
        'Γ': np.array([0.0, 0.0, 0.0]),
        'X': np.array([0.5, 0.0, 0.0]),
        'M': np.array([0.5, 0.5, 0.0]),
        'R': np.array([0.5, 0.5, 0.5])
    }
    
    # Define path: Γ -> X -> M -> Γ -> R
    path_names = ['Γ', 'X', 'M', 'Γ', 'R']
    
    # Generate k-points along the path
    k_points = []
    labels = []
    positions = [0]
    
    total_points = 0
    for i in range(len(path_names) - 1):
        start = high_symmetry_points[path_names[i]]
        end = high_symmetry_points[path_names[i + 1]]
        
        # Generate points along this segment
        segment = np.linspace(0, 1, n_points)
        segment_points = np.outer(1-segment, start) + np.outer(segment, end)
        
        # Add to overall path
        k_points.append(segment_points)
        total_points += n_points
        positions.append(total_points - 1)
    
    # Combine all segments
    k_points = np.vstack(k_points)
    
    return k_points, path_names, positions


def compute_dos(frequencies: np.ndarray, 
               bin_width: float = 1.0, 
               freq_range: Tuple[float, float] = (-200, 5000),
               include_negative: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phonon density of states.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Phonon frequencies at all k-points, shape (n_kpoints, n_modes)
    bin_width : float
        Width of frequency bins in cm^-1
    freq_range : Tuple[float, float]
        Range of frequencies to include in the DOS
    include_negative : bool
        Whether to include negative frequencies (default: True)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (frequency_bins, dos) where dos[i] is the density of states at frequency_bins[i]
    """
    # Count negative frequencies for warning
    n_negative = np.sum(frequencies < 0)
    if n_negative > 0:
        print(f"WARNING: {n_negative} negative frequencies detected ({n_negative/frequencies.size:.1%} of total)")
        print(f"Minimum frequency: {np.min(frequencies):.1f} cm^-1")
    
    # Flatten the frequencies array
    freq_flat = frequencies.flatten()
    
    # Handle negative frequencies based on flag
    if not include_negative:
        freq_flat = freq_flat[freq_flat >= 0]
        if freq_range[0] < 0:
            # Adjust lower bound if excluding negatives
            freq_range = (0, freq_range[1])
    
    # Create frequency bins
    freq_bins = np.arange(freq_range[0], freq_range[1] + bin_width, bin_width)
    freq_centers = freq_bins[:-1] + bin_width/2
    print(f"Computing DOS with {len(freq_centers)} bins from {freq_range[0]} to {freq_range[1]} cm^-1 with width {bin_width} cm^-1")
    
    # Compute histogram
    hist, _ = np.histogram(freq_flat, bins=freq_bins, density=True)
    
    return freq_centers, hist


def load_hessian(hessian_file: str) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], np.ndarray]]:
    """
    Load Hessian matrices from file.
    
    Parameters:
    -----------
    hessian_file : str
        Path to the .npz file containing Hessian matrices
    
    Returns:
    --------
    Tuple containing:
        - np.ndarray: Gamma point Hessian
        - Dict[Tuple[int, int, int], np.ndarray]: Extended Hessian
    """
    print(f"Loading Hessian from {hessian_file}...")
    
    if not os.path.exists(hessian_file):
        raise FileNotFoundError(f"Hessian file not found: {hessian_file}")
    
    data = np.load(hessian_file, allow_pickle=True)
    
    gamma_hessian = data['gamma_hessian']
    
    # Load extended Hessian
    extended_hessian = {}
    for key in data.keys():
        if key.startswith('ext_'):
            # Extract the shift from the key name
            shift_str = key[4:]  # Remove 'ext_' prefix
            shift = tuple(map(int, shift_str.split('_')))
            extended_hessian[shift] = data[key]
    
    print(f"Loaded Hessian matrices: gamma point and {len(extended_hessian)} extended shifts")
    return gamma_hessian, extended_hessian


def plot_phonon_dispersion(q_points, frequencies, labels, label_positions, output_file='phonon_dispersion.png'):
    """
    Plot phonon dispersion curves
    
    Parameters:
        q_points: List of q-points
        frequencies: Array of frequencies at each q-point
        labels: Labels for high-symmetry points
        label_positions: Positions of labels on the x-axis
        output_file: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot horizontal axis
    x = np.arange(len(q_points))
    
    # Plot each branch
    n_branches = frequencies.shape[1]
    for branch in range(n_branches):
        plt.plot(x, frequencies[:, branch], 'b-')
    
    # Add labels and vertical lines
    for label, pos in zip(labels, label_positions):
        plt.axvline(x=pos, color='k', linestyle='--', alpha=0.3)
    
    plt.xticks(label_positions, labels)
    plt.ylabel('Frequency (cm$^{-1}$)')
    plt.title('Phonon Dispersion')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Phonon dispersion plot saved to {output_file}")


def plot_dos(freq_bins, dos, output_file='phonon_dos.png'):
    """
    Plot phonon density of states
    
    Parameters:
        freq_bins: Frequency bins
        dos: Density of states values
        output_file: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot DOS, highlighting negative frequencies in red
    negative_mask = freq_bins < 0
    if np.any(negative_mask):
        plt.fill_between(freq_bins[negative_mask], dos[negative_mask], 
                        color='red', alpha=0.7, label='Imaginary modes')
    
    plt.fill_between(freq_bins[~negative_mask], dos[~negative_mask], 
                    color='blue', alpha=0.7, label='Real modes')
    
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Density of States')
    plt.title('Phonon Density of States')
    plt.grid(alpha=0.3)
    if np.any(negative_mask):
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Phonon DOS plot saved to {output_file}")


def compute_and_plot_phonons(hessian_file, nk_mesh=10, n_path_points=50, 
                            dos_bin_width=2.0, output_dir='.'):
    """
    Compute and plot phonon dispersion and DOS
    
    Parameters:
        hessian_file: Path to the Hessian file
        nk_mesh: Number of k-points along each direction for the mesh
        n_path_points: Number of points along each segment of the high-symmetry path
        dos_bin_width: Width of DOS bins in cm^-1
        output_dir: Directory to save output files
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Hessian
    results, metadata = load_hessian_all_in_one(hessian_file)
    #gamma_hessian = hessian_data['gamma_hessian']  # Assuming this is how to access the array
    #extended_hessian = hessian_data['extended_hessian']  # Assuming this key exists
    #gamma_hessian = results.get('mass_weighted_gamma')
    #extended_hessian = results.get('extended_hessian')
    gamma_hessian = results.get('mass_weighted_gamma')
    print(gamma_hessian)
    #gamma_hessian = results.get('mass_weighted_gamma', results['gamma_hessian'])
    extended_hessian = results.get('mass_weighted_extended', results['extended_hessian'])
    
    #write_extended_hessian_txt(results['extended_hessian'], "extended_hessian.txt")

    # Generate k-points for dispersion (high-symmetry path)
    print("Generating high-symmetry k-point path...")
    k_path, path_labels, label_positions = generate_high_symmetry_path(n_path_points)
    
    # Compute phonon frequencies along the path
    print("Computing phonon dispersion along high-symmetry path...")
    path_frequencies = compute_phonon_dispersion(gamma_hessian, extended_hessian, k_path)
    
    # Plot dispersion
    dispersion_file = os.path.join(output_dir, 'phonon_dispersion.png')
    plot_phonon_dispersion(k_path, path_frequencies, path_labels, label_positions, dispersion_file)
    
    # Generate k-points for DOS (uniform mesh)
    print(f"Generating {nk_mesh}x{nk_mesh}x{nk_mesh} k-point mesh for DOS...")
    k_mesh = generate_kpoint_mesh(nk_mesh)
    
    # Compute phonon frequencies on the mesh
    print("Computing phonon frequencies on mesh...")
    mesh_frequencies = compute_phonon_dispersion(gamma_hessian, extended_hessian, k_mesh)
    
    # Compute DOS
    print("Computing phonon density of states...")
    freq_bins, dos = compute_dos(mesh_frequencies, bin_width=dos_bin_width, 
                                freq_range=(-200, 5000), include_negative=True)
    
    # Plot DOS
    dos_file = os.path.join(output_dir, 'phonon_dos.png')
    plot_dos(freq_bins, dos, dos_file)
    
    # Save frequency data
    np.savez(os.path.join(output_dir, 'phonon_data.npz'),
             path_kpoints=k_path,
             path_frequencies=path_frequencies,
             mesh_frequencies=mesh_frequencies,
             dos_freq_bins=freq_bins,
             dos=dos)
    
    print("Phonon calculation complete!")
    
    return {
        'path_frequencies': path_frequencies,
        'mesh_frequencies': mesh_frequencies,
        'dos_freq_bins': freq_bins,
        'dos': dos
    }


def plot_combined_dispersion_and_dos(path_frequencies, freq_bins, dos, 
                                    labels, label_positions, output_file='phonon_combined.png'):
    """
    Create a combined plot with dispersion on the left and DOS on the right
    
    Parameters:
        path_frequencies: Frequencies along high-symmetry path
        freq_bins: Frequency bins for DOS
        dos: Density of states
        labels: Labels for high-symmetry points
        label_positions: Positions of high-symmetry points
        output_file: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                  gridspec_kw={'width_ratios': [2, 1]})
    
    # Plot dispersion on the left
    x = np.arange(len(path_frequencies))
    n_branches = path_frequencies.shape[1]
    
    for branch in range(n_branches):
        ax1.plot(x, path_frequencies[:, branch], 'b-')
    
    # Add labels and vertical lines
    for label, pos in zip(labels, label_positions):
        ax1.axvline(x=pos, color='k', linestyle='--', alpha=0.3)
    
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Frequency (cm$^{-1}$)')
    ax1.set_title('Phonon Dispersion')
    ax1.grid(alpha=0.3)
    
    # Plot DOS on the right
    negative_mask = freq_bins < 0
    if np.any(negative_mask):
        ax2.fill_betweenx(freq_bins[negative_mask], 0, dos[negative_mask], 
                        color='red', alpha=0.7, label='Imaginary modes')
    
    ax2.fill_betweenx(freq_bins[~negative_mask], 0, dos[~negative_mask], 
                    color='blue', alpha=0.7, label='Real modes')
    
    ax2.set_xlabel('DOS')
    ax2.set_title('Density of States')
    ax2.grid(alpha=0.3)
    if np.any(negative_mask):
        ax2.legend()
    
    # Align y-axes
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Combined phonon plot saved to {output_file}")


if __name__ == "__main__":
    import argparse
    from OLDextended_hessian import compute_hessian_at_structure, save_hessian_all_in_one, load_hessian_all_in_one
    from pymatgen.core import Structure
    
    parser = argparse.ArgumentParser(description="Compute and plot phonon properties")
    parser.add_argument("--structure", type=str, help="Path to structure file (POSCAR, CIF, etc.)")
    parser.add_argument("--hessian", type=str, help="Path to pre-computed Hessian file (.npz)")
    parser.add_argument("--output", type=str, default="phonon_output", help="Output directory")
    parser.add_argument("--nk", type=int, default=10, help="k-point mesh size")
    parser.add_argument("--path-points", type=int, default=50, help="Points per segment on the high-symmetry path")
    parser.add_argument("--bin-width", type=float, default=2.0, help="DOS bin width in cm^-1")
    parser.add_argument("--compute-hessian", action="store_true", help="Compute Hessian from structure")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine the Hessian file path
    hessian_file = args.hessian
    
    # If no Hessian provided or compute-hessian flag is set, calculate it
    if (not hessian_file or args.compute_hessian) and args.structure:
        print(f"Loading structure from {args.structure}...")
        structure = Structure.from_file(args.structure)
        
        hessian_file = os.path.join(args.output, "hessian.npz")
        print(f"Computing Hessian for the structure...")
        
        # Compute Hessian
        results = compute_hessian_at_structure(
            structure, 
            stepsize=0.005,
            method='mixed',
            potential='sapt'
        )
        
        # Save the Hessian
        save_hessian_all_in_one(results, hessian_file.replace('.npz', ''))
    else:
        print(f"Using provided Hessian file: {hessian_file}")
    
    if not os.path.exists(hessian_file):
        print("Error: Hessian file not found.")
        exit(1)
    
    # Compute and plot phonon properties
    results = compute_and_plot_phonons(
        hessian_file,
        nk_mesh=args.nk,
        n_path_points=args.path_points,
        dos_bin_width=args.bin_width,
        output_dir=args.output
    )
    
    # Create combined plot
    k_path, path_labels, label_positions = generate_high_symmetry_path(args.path_points)
    combined_file = os.path.join(args.output, 'phonon_combined.png')
    
    plot_combined_dispersion_and_dos(
        results['path_frequencies'],
        results['dos_freq_bins'],
        results['dos'],
        path_labels,
        label_positions,
        combined_file
    )
    
    print("All phonon calculations and plots completed successfully!")

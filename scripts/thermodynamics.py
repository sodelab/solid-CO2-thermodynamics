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


def compute_phonon_dispersion(gamma_hessian: np.ndarray, 
                             extended_hessian: Dict[Tuple[int, int, int], np.ndarray],
                             kpoints: np.ndarray,
                             verbose: bool = False,
                             debug: bool = False) -> np.ndarray:
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
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    
    Returns:
    --------
    np.ndarray
        Phonon frequencies at each k-point in cm^-1, shape (n_kpoints, n_modes)
    """
    n_dof = gamma_hessian.shape[0]  # Number of degrees of freedom
    n_kpoints = kpoints.shape[0]
    
    # Pre-allocate output array
    frequencies = np.zeros((n_kpoints, n_dof), dtype=np.complex128)
    
    if debug:
        print(f"DEBUG: gamma_hessian shape: {gamma_hessian.shape}")
        print(f"DEBUG: Extended hessian has {len(extended_hessian)} entries")
        print(f"DEBUG: First few eigenvalues of gamma hessian: {np.linalg.eigvalsh(gamma_hessian)[:5]}")
    
    if verbose or debug:
        print(f"Computing phonon frequencies for {n_kpoints} k-points...")
    
    # Loop over k-points
    for ik, k in enumerate(kpoints):
        # Start with gamma point Hessian
        dynamical_matrix = gamma_hessian.copy().astype(np.complex128)
        
        # Add contributions from extended Hessian with phase factors
        if verbose and ik % max(1, n_kpoints//10) == 0:
            print(f"  Calculating phonon frequencies for k-point {ik+1}/{n_kpoints} at k = {k}")
        
        # Debug for individual k-points
        if debug and ik < 3:  # Only show first 3 to avoid flooding output
            print(f"DEBUG: Processing k-point {ik}: {k}")
            
        for shift, hessian in extended_hessian.items():
            # Calculate phase factor exp(i k·r)
            phase = np.exp(1j * 2 * np.pi * np.dot(k, shift))
            
            # Add contribution to dynamical matrix
            dynamical_matrix += hessian * phase
            
            if debug and ik == 0 and shift == (1,0,0):  # Example of detailed debug
                print(f"DEBUG: Phase for shift {shift}: {phase}")
                print(f"DEBUG: First few elements of hessian: {hessian.flatten()[:5]}")
        
        # Ensure the dynamical matrix is Hermitian
        dynamical_matrix = 0.5 * (dynamical_matrix + dynamical_matrix.conj().T)
        
        # Diagonalize to get frequencies
        eigenvalues = np.linalg.eigvalsh(dynamical_matrix.real)
        
        # Convert eigenvalues to frequencies (cm^-1)
        # For CO2 potential in kcal/mol/Å² units, the conversion factor is ~108.6
        frequencies[ik] = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * 108.6
        
        if debug and ik < 3:
            print(f"DEBUG: First 5 frequencies at k={k}: {frequencies[ik,:5]}")
    
    if verbose or debug:
        print("Phonon frequency calculation completed.")
        print(f"  Frequency range: {np.min(frequencies.real):.1f} to {np.max(frequencies.real):.1f} cm^-1")
    
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


def compute_dos(frequencies: np.ndarray, 
               bin_width: float = 1.0, 
               freq_range: Tuple[float, float] = (-200, 5000),
               include_negative: bool = True,
               verbose: bool = False,
               debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (frequency_bins, dos) where dos[i] is the density of states at frequency_bins[i]
    """
    # Count negative frequencies for warning
    n_negative = np.sum(frequencies < 0)
    
    if debug:
        print(f"DEBUG: Frequencies shape: {frequencies.shape}")
        print(f"DEBUG: Frequency range: {np.min(frequencies):.4f} to {np.max(frequencies):.4f} cm^-1")
        print(f"DEBUG: Number of negative frequencies: {n_negative} ({n_negative/frequencies.size:.2%} of total)")
        
    if n_negative > 0 and (verbose or debug):
        print(f"WARNING: {n_negative} negative frequencies detected ({n_negative/frequencies.size:.1%} of total)")
        print(f"Minimum frequency: {np.min(frequencies):.1f} cm^-1")
    
    # Flatten the frequencies array
    freq_flat = frequencies.flatten()
    
    if debug:
        print(f"DEBUG: Total frequency points after flattening: {len(freq_flat)}")
    
    # Handle negative frequencies based on flag
    if not include_negative:
        n_before = len(freq_flat)
        freq_flat = freq_flat[freq_flat >= 0]
        if debug:
            print(f"DEBUG: Removed {n_before - len(freq_flat)} negative frequency points")
            
        if freq_range[0] < 0:
            # Adjust lower bound if excluding negatives
            freq_range = (0, freq_range[1])
            if debug:
                print(f"DEBUG: Adjusted frequency range to {freq_range}")
    
    # Create frequency bins
    freq_bins = np.arange(freq_range[0], freq_range[1] + bin_width, bin_width)
    freq_centers = freq_bins[:-1] + bin_width/2
    
    if debug:
        print(f"DEBUG: Created {len(freq_bins)} bin edges ({len(freq_centers)} bin centers)")
        print(f"DEBUG: Bin edges range from {freq_bins[0]:.1f} to {freq_bins[-1]:.1f}")
    
    # Compute histogram
    hist, edges = np.histogram(freq_flat, bins=freq_bins, density=True)
    
    if debug:
        print(f"DEBUG: Histogram sum: {np.sum(hist * np.diff(edges)):.6f} (should be close to 1.0)")
        print(f"DEBUG: Max DOS value: {np.max(hist):.6e} at {freq_centers[np.argmax(hist)]:.1f} cm^-1")
        
    if verbose or debug:
        print(f"DOS calculated with {len(freq_centers)} frequency bins")
        print(f"  Frequency range: {freq_range[0]} to {freq_range[1]} cm^-1")
        print(f"  Bin width: {bin_width} cm^-1")
    
    return freq_centers, hist

def calculate_thermodynamics_cpp(frequencies: np.ndarray, 
                        temperature: float, 
                        pressure: float = 0.0,
                        volume: float = None,
                        static_energy: float = 0.0,
                        bin_width: float = 1.0, 
                        freq_range: Tuple[float, float] = (0, 5000),
                        verbose: bool = False,
                        debug: bool = False) -> Dict:
    """Calculate thermodynamic properties using same approach as the C++ code"""
    
    # Constants - matching C++ constants
    Eh_J = 4.35974e-18   # Hartree to Joules
    Eh_cm1 = 219474.63   # cm^-1 to Hartree
    Eh_TO_KCAL = 627.5095  # Hartree to kcal/mol conversion
    
    if debug:
        print(f"DEBUG: calculate_thermodynamics_cpp called with T={temperature}K, P={pressure}GPa")
        print(f"DEBUG: Frequencies shape: {frequencies.shape}, Static energy: {static_energy}")
    
    # Compute DOS - use unnormalized DOS (matches C++ approach)
    freq_bins, dos_norm = compute_dos(frequencies, bin_width, freq_range=(-200, 5000), 
                                     verbose=verbose, debug=debug)
    
    # Make DOS unnormalized (total counts) - matches C++ behavior
    nmodes = frequencies.shape[1]  # Number of modes per k-point
    nkpoints = len(frequencies)    # Total number of k-points

    # Scale DOS to represent actual number of states without normalization
    dos = dos_norm * (nmodes * nkpoints)
    
    if debug:
        print(f"DEBUG: DOS scaling factor: {nmodes * nkpoints}")
        print(f"DEBUG: Scaled DOS max value: {np.max(dos):.6e}")

    # Initialize variables (like C++ code)
    N = 0.0
    beta = 1.0 / (temperature * KB / Eh_J)
    st = 0.0
    ut = 0.0
    zpe = 0.0
    
    if debug:
        print(f"DEBUG: beta = {beta:.6e}")
        print(f"DEBUG: Processing {len(freq_bins)} frequency bins")
    
    # Loop over frequency bins (replicating C++ approach)
    for i, omega in enumerate(freq_bins):
        # Skip negative frequencies
        if omega <= 0:
            continue
        
        # Accumulate total DOS
        N += dos[i]
        
        # Calculate dimensionless frequency
        bho = beta * omega / Eh_cm1
        
        # Zero-point energy
        zpe += 0.5 * (omega) / Eh_cm1 * dos[i]
        
        # Special handling for omega≈0 (like C++)
        if omega < 0.0001:
            ut += (1/beta) * dos[i]
            st += np.log(1 - np.exp(-beta * 0.0001 / Eh_cm1)) * dos[i]
            if debug and i < 5:
                print(f"DEBUG: Near-zero frequency special case: omega={omega:.6e}, dos={dos[i]:.6e}")
        else:
            ut += (omega / Eh_cm1 / (np.exp(bho) - 1)) * dos[i]
            st += np.log(1 - np.exp(-bho)) * dos[i]
            
        if debug and i < 5 or (debug and i % 1000 == 0):
            print(f"DEBUG: bin {i}: omega={omega:.2f}, bho={bho:.6e}, dos={dos[i]:.6e}")
            print(f"DEBUG: Current sums - zpe={zpe:.6e}, ut={ut:.6e}, st={st:.6e}")
    
    N = nkpoints
    if verbose or debug:
        print(f"Internal energy contribution: {ut/N/100*Eh_TO_KCAL * 4184} J/mol")
    
    # Calculate final thermodynamic properties - converting to desired units
    entropy = (-st * KB / N) * AVOGADRO
    zpe_energy = zpe / N * Eh_TO_KCAL  # kcal/mol
    internal_energy = (ut / N + zpe / N) * Eh_TO_KCAL  # kcal/mol
    free_energy = static_energy + zpe_energy - entropy * temperature / 4184.0  # kcal/mol
    
    if debug:
        print(f"DEBUG: Conversion factors:")
        print(f"DEBUG: KB = {KB}, AVOGADRO = {AVOGADRO}")
        print(f"DEBUG: Raw calculation values:")
        print(f"DEBUG: st = {st}, zpe = {zpe}, ut = {ut}")
        print(f"DEBUG: N = {N}")
    
    # Gibbs free energy if volume is provided
    gibbs_free_energy = None
    if volume is not None:
        # Convert pressure from GPa to kcal/(mol·Å³)
        pressure_kcal = pressure * 1e9 / (4184.0 * 1e30 / AVOGADRO)
        gibbs_free_energy = free_energy + pressure_kcal * volume
        
        if debug:
            print(f"DEBUG: pressure_kcal = {pressure_kcal:.6e} kcal/(mol·Å³)")
            print(f"DEBUG: PV term = {pressure_kcal * volume:.6f} kcal/mol")
    
    # Convert entropy to SI units
    entropy_si = entropy * 4184.0  # Convert to J/(mol·K)
    
    # Output results
    if verbose or debug:
        print("\nThermodynamic Properties:")
        print(f"Temperature: {temperature} K")
        print(f"Pressure: {pressure} GPa")
        print(f"Zero-point energy: {zpe_energy:.6f} kcal/mol")
        print(f"Entropy: {entropy:.6f} J/(mol·K)")
        print(f"Heat capacity (Cv): {0:.6f} cal/(mol·K)")
        print(f"Helmholtz free energy: {free_energy:.6f} kcal/mol")
        
        if gibbs_free_energy is not None:
            print(f"Gibbs free energy: {gibbs_free_energy:.6f} kcal/mol")

    return {
        'temperature': temperature,  # K
        'pressure': pressure,  # GPa
        'volume': volume,  # Å³
        'frequency_bins': freq_bins,  # cm^-1
        'dos': dos,  # arbitrary units
        'zero_point_energy': zpe_energy,  # kcal/mol
        'internal_energy': internal_energy,  # kcal/mol
        'entropy': entropy_si,  # J/(mol·K)
        'helmholtz_energy': free_energy,  # kcal/mol
        'gibbs_energy': gibbs_free_energy,  # kcal/mol if volume is provided
        'heat_capacity_v': 0,  # cal/(mol·K)
        'static_energy': static_energy  # kcal/mol
    }

def calculate_thermodynamics(frequencies: np.ndarray, 
                           temperature: float, 
                           pressure: float = 0.0,
                           volume: float = None,
                           static_energy: float = 0.0,
                           bin_width: float = 1.0, 
                           freq_range: Tuple[float, float] = (0, 5000),
                           verbose: bool = False,
                           debug: bool = False) -> Dict:
    """
    Calculate thermodynamic properties from phonon frequencies.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Phonon frequencies at all k-points
    temperature : float
        Temperature in Kelvin
    pressure : float
        Pressure in GPa
    volume : float
        Volume in Å³
    static_energy : float
        Static lattice energy in kcal/mol
    bin_width : float
        Width of frequency bins in cm^-1
    freq_range : Tuple[float, float]
        Range of frequencies to include in the DOS
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    
    Returns:
    --------
    Dict
        Dictionary containing thermodynamic properties
    """
    # Debugging output
    if debug:
        print(f"\nDEBUG: Calculating thermodynamics at T = {temperature:.1f}K")
        print(f"DEBUG: Input frequencies shape: {frequencies.shape}")
        print(f"DEBUG: Static energy: {static_energy:.6f} kcal/mol")
        print(f"DEBUG: Volume: {volume} Å³, Pressure: {pressure} GPa")
    
    if verbose or debug:
        print(f"\n=== Debugging thermodynamics at T = {temperature:.1f}K ===")
    
    # Compute DOS
    freq_bins, dos = compute_dos(frequencies, bin_width, freq_range=(-200, 5000), 
                                verbose=verbose, debug=debug)
    
    if verbose or debug:
        print(f"  DOS: {len(freq_bins)} bins, freq range: {np.min(freq_bins):.1f} to {np.max(freq_bins):.1f} cm^-1")
        print(f"  Negative frequencies: {np.sum(freq_bins < 0)} bins")
        print(f"  DOS integral (sum of all states): {np.sum(dos):.4f}")
    
    # Scale DOS
    nmodes = frequencies.shape[1]  # Number of modes
    dos *= nmodes
    
    if debug:
        print(f"DEBUG: DOS scaled by nmodes = {nmodes}")
        print(f"DEBUG: Scaled DOS integral: {np.sum(dos):.4f}")

    # Mask for positive, non-zero frequencies (avoid numerical issues)
    nonzero_mask = (freq_bins > 1e-6)
    if np.any(freq_bins < 0) and (verbose or debug):
        print(f"  Excluding {np.sum(freq_bins < 0)} negative frequency bins from thermodynamic calculations")
        
    if debug:
        print(f"DEBUG: Positive frequency bins: {np.sum(nonzero_mask)} of {len(freq_bins)}")

    # Calculate zero-point energy (this is temperature-independent)
    zero_point_energy = 0.5 * np.sum(freq_bins[nonzero_mask] * CM1_TO_KCALMOL * dos[nonzero_mask])
    
    if debug:
        print(f"DEBUG: CM1_TO_KCALMOL conversion factor: {CM1_TO_KCALMOL}")
        print(f"DEBUG: ZPE calculation = 0.5 * sum(freq * {CM1_TO_KCALMOL} * dos) = {zero_point_energy:.6f}")
    
    if verbose or debug:
        print(f"  Zero-point energy: {zero_point_energy:.6f} kcal/mol")
        print(f"  Static energy: {static_energy:.6f} kcal/mol")
    
    # Initialize thermodynamic quantities
    internal_energy = zero_point_energy  # Start with ZPE
    entropy = 0.0
    cv = 0.0
    
    # Handle the T→0 limit analytically
    if temperature < 1e-10:
        if debug:
            print(f"DEBUG: Near-zero temperature case (T={temperature})")
            print(f"DEBUG: Using analytical limit: E=ZPE, S=0, Cv=0")
            
        # At T=0, internal_energy = zero_point_energy, entropy = 0, cv = 0
        free_energy = static_energy + zero_point_energy
    else:
        # Convert temperature to appropriate units
        beta = 1.0 / (temperature * KB * J_TO_KCAL * AVOGADRO)  # 1/(kT) in mol/kcal
        
        if debug:
            print(f"DEBUG: Temperature = {temperature} K")
            print(f"DEBUG: KB = {KB}, J_TO_KCAL = {J_TO_KCAL}, AVOGADRO = {AVOGADRO}")
            print(f"DEBUG: beta = 1/(kT) = {beta:.6e} mol/kcal")
        
        # For non-zero temperatures, calculate thermodynamic quantities safely
        x = beta * freq_bins[nonzero_mask] * CM1_TO_KCALMOL
        
        if debug:
            print(f"DEBUG: x = beta * freq * CM1_TO_KCALMOL")
            print(f"DEBUG: x has {len(x)} values, ranging from {np.min(x):.2e} to {np.max(x):.2e}")
        
        # Handle large x values to prevent overflow
        small_x_mask = x < 200  # exp(200) is around the limit of float64
        
        if debug:
            print(f"DEBUG: x values < 200: {np.sum(small_x_mask)} of {len(x)}")
            if not np.any(small_x_mask):
                print(f"DEBUG: WARNING - No manageable x values! Check your input parameters.")

        # Calculate terms only for manageable x values
        if np.any(small_x_mask):
            x_small = x[small_x_mask]
            dos_small = dos[nonzero_mask][small_x_mask]
            ex = np.exp(x_small)
            
            if debug:
                print(f"DEBUG: Using {len(x_small)} frequencies with manageable exp(x) values")
                print(f"DEBUG: First few x values: {x_small[:5]}")
                print(f"DEBUG: First few exp(x) values: {ex[:5]}")
            
            # Internal energy contribution (beyond zero-point)
            internal_energy_term = np.sum((x_small / (ex - 1)) * dos_small)
            internal_energy += internal_energy_term
            
            if debug:
                print(f"DEBUG: Internal energy term calculation:")
                print(f"DEBUG: Sum of (x / (exp(x) - 1)) * dos = {internal_energy_term:.6e}")
                print(f"DEBUG: Total internal energy = ZPE + term = {zero_point_energy:.6f} + {internal_energy_term:.6e} = {internal_energy:.6f}")
            
            # Entropy contribution
            entropy_term = np.sum(-np.log(1 - 1/ex) * dos_small)
            entropy = entropy_term * KB * AVOGADRO * J_TO_KCAL
            
            if debug:
                print(f"DEBUG: Entropy term calculation:")
                print(f"DEBUG: Sum of -log(1 - 1/exp(x)) * dos = {entropy_term:.6e}")
                print(f"DEBUG: Entropy = term * KB * AVOGADRO * J_TO_KCAL = {entropy:.6f}")
            
            # Heat capacity term
            cv_term = np.sum(x_small**2 * ex / (ex - 1)**2 * dos_small)
            cv = cv_term * KB * AVOGADRO * J_TO_KCAL * 1000  # in cal/(mol·K)
            
            if debug:
                print(f"DEBUG: Heat capacity term calculation:")
                print(f"DEBUG: Sum of x^2 * exp(x) / (exp(x) - 1)^2 * dos = {cv_term:.6e}")
                print(f"DEBUG: Cv = term * KB * AVOGADRO * J_TO_KCAL * 1000 = {cv:.6f}")

            # Debug x values (dimensionless frequencies)
            if verbose or debug:
                print(f"  x value statistics:")
                print(f"    Min x: {np.min(x):.2e}, Max x: {np.max(x):.2e}, Mean x: {np.mean(x):.2e}")
                print(f"    x < 100: {np.sum(x < 100)}/{len(x)} points")
                print(f"    x in [100, 500): {np.sum((x >= 100) & (x < 500))}/{len(x)} points")
                print(f"    x >= 500: {np.sum(x >= 500)}/{len(x)} points")
        
        # Calculate Helmholtz free energy
        free_energy = static_energy + zero_point_energy - entropy * temperature
        
        if debug:
            print(f"DEBUG: Helmholtz free energy calculation:")
            print(f"DEBUG: F = E_static + ZPE - T*S")
            print(f"DEBUG: F = {static_energy:.6f} + {zero_point_energy:.6f} - {temperature} * {entropy:.6f} = {free_energy:.6f}")
    
    # Gibbs free energy if volume is provided
    gibbs_free_energy = None
    if volume is not None:
        # Convert pressure from GPa to kcal/(mol·Å³)
        pressure_kcal = pressure * 1e9 / (4184.0 * 1e30 / AVOGADRO)
        gibbs_free_energy = free_energy + pressure_kcal * volume
        
        if debug:
            print(f"DEBUG: Gibbs free energy calculation:")
            print(f"DEBUG: Pressure conversion: {pressure} GPa = {pressure_kcal:.6e} kcal/(mol·Å³)")
            print(f"DEBUG: G = F + PV = {free_energy:.6f} + {pressure_kcal:.6e} * {volume} = {gibbs_free_energy:.6f}")
    
    # Convert entropy to SI units
    entropy_si = entropy * 4184.0  # Convert to J/(mol·K)
    
    if debug:
        print(f"DEBUG: Entropy in SI units: {entropy} * 4184.0 = {entropy_si} J/(mol·K)")
    
    # Output results
    if verbose or debug:
        print("  Thermodynamic Properties:")
        print(f"  Temperature: {temperature} K")
        print(f"  Pressure: {pressure} GPa")
        print(f"  Zero-point energy: {zero_point_energy:.6f} kcal/mol")
        print(f"  Entropy: {entropy:.6f} J/(mol·K)")
        print(f"  Heat capacity (Cv): {cv:.6f} cal/(mol·K)")
        print(f"  Helmholtz free energy: {free_energy:.6f} kcal/mol")
        if gibbs_free_energy is not None:
            print(f"  Gibbs free energy: {gibbs_free_energy:.6f} kcal/mol")

    return {
        'temperature': temperature,  # K
        'pressure': pressure,  # GPa
        'volume': volume,  # Å³
        'frequency_bins': freq_bins,  # cm^-1
        'dos': dos,  # arbitrary units
        'zero_point_energy': zero_point_energy,  # kcal/mol
        'internal_energy': internal_energy,  # kcal/mol
        'entropy': entropy_si,  # J/(mol·K)
        'helmholtz_energy': free_energy,  # kcal/mol
        'gibbs_energy': gibbs_free_energy,  # kcal/mol if volume is provided
        'heat_capacity_v': cv,  # cal/(mol·K)
        'static_energy': static_energy  # kcal/mol
    }


def calculate_qha_properties(volumes: np.ndarray, 
                           free_energies: np.ndarray,
                           temperature: float,
                           pressure: float = 0.0,
                           verbose: bool = False,
                           debug: bool = False) -> Dict:
    """
    Calculate properties using the quasiharmonic approximation.
    
    Parameters:
    -----------
    volumes : np.ndarray
        Array of volumes in Å³
    free_energies : np.ndarray
        Helmholtz free energies at each volume in kcal/mol
    temperature : float
        Temperature in K
    pressure : float
        Pressure in GPa
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    
    Returns:
    --------
    Dict
        Dictionary of QHA properties
    """
    if debug:
        print(f"\nDEBUG: calculate_qha_properties called:")
        print(f"DEBUG: Temperature: {temperature} K, Pressure: {pressure} GPa")
        print(f"DEBUG: Volumes array: {volumes}")
        print(f"DEBUG: Free energies array: {free_energies}")
    
    # Convert pressure from GPa to kcal/(mol·Å³)
    pressure_kcal = pressure * 1e9 / (4184.0 * 1e30 / AVOGADRO)
    
    if debug:
        print(f"DEBUG: Pressure conversion: {pressure} GPa = {pressure_kcal:.6e} kcal/(mol·Å³)")
    
    # Calculate Gibbs free energy G = F + PV for each volume
    gibbs_energies = free_energies + pressure_kcal * volumes
    
    if debug:
        print(f"DEBUG: Gibbs energies = F + PV:")
        for v, f, g in zip(volumes, free_energies, gibbs_energies):
            print(f"DEBUG:   V={v:.2f} Å³: F={f:.6f}, G={g:.6f} kcal/mol")
    
    # Find the volume that minimizes G at this pressure
    # Use cubic spline interpolation for smoother minimum finding
    spline = InterpolatedUnivariateSpline(volumes, gibbs_energies, k=3)
    
    if debug:
        print(f"DEBUG: Created cubic spline interpolation for G(V)")
    
    # Find the minimum point
    volume_range = np.linspace(min(volumes), max(volumes), 1000)
    gibbs_interp = spline(volume_range)
    min_idx = np.argmin(gibbs_interp)
    equilibrium_volume = volume_range[min_idx]
    gibbs_energy = gibbs_interp[min_idx]
    
    if debug:
        print(f"DEBUG: Minimum search: sampling {len(volume_range)} points between {min(volumes):.2f} and {max(volumes):.2f}")
        print(f"DEBUG: Minimum found at index {min_idx}, V={equilibrium_volume:.4f}, G={gibbs_energy:.6f}")
    
    # Calculate isothermal bulk modulus: B_T = V * (d²F/dV²)
    d2fdv2 = spline.derivative(2)(equilibrium_volume)
    bulk_modulus = equilibrium_volume * d2fdv2 * (4184.0 * 1e30 / AVOGADRO) * 1e-9  # GPa
    
    if debug:
        print(f"DEBUG: Second derivative at minimum: {d2fdv2:.6e}")
        print(f"DEBUG: Bulk modulus calculation: V * d²F/dV² * conversion = {bulk_modulus:.4f} GPa")
    
    # Create interpolation for F(V) to calculate pressure
    f_spline = InterpolatedUnivariateSpline(volumes, free_energies, k=3)
    
    # Calculate pressure as -dF/dV at equilibrium volume
    pressure_calculated = -f_spline.derivative(1)(equilibrium_volume)
    pressure_gpa = pressure_calculated * (4184.0 * 1e30 / AVOGADRO) * 1e-9  # GPa
    
    if debug:
        print(f"DEBUG: Calculated pressure: -dF/dV = {pressure_calculated:.6e}")
        print(f"DEBUG: Converted to GPa: {pressure_gpa:.6f} GPa")
        print(f"DEBUG: Specified pressure: {pressure:.6f} GPa")
        print(f"DEBUG: Pressure difference: {pressure_gpa - pressure:.6f} GPa")
    
    if verbose or debug:
        print(f"  QHA properties at T={temperature} K, P={pressure} GPa:")
        print(f"    Equilibrium volume: {equilibrium_volume:.3f} Å³")
        print(f"    Gibbs free energy: {gibbs_energy:.6f} kcal/mol")
        print(f"    Bulk modulus: {bulk_modulus:.2f} GPa")
        print(f"    Calculated pressure: {pressure_gpa:.4f} GPa")
    
    return {
        'equilibrium_volume': equilibrium_volume,  # Å³
        'gibbs_energy': gibbs_energy,  # kcal/mol
        'bulk_modulus': bulk_modulus,  # GPa
        'calculated_pressure': pressure_gpa  # GPa
    }


def thermal_expansion_coefficient(volumes: np.ndarray, 
                                temperatures: np.ndarray,
                                verbose: bool = False,
                                debug: bool = False) -> np.ndarray:
    """
    Calculate thermal expansion coefficient from volume vs temperature data.
    
    Parameters:
    -----------
    volumes : np.ndarray
        Equilibrium volumes at different temperatures in Å³
    temperatures : np.ndarray
        Corresponding temperatures in K
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    
    Returns:
    --------
    np.ndarray
        Thermal expansion coefficients in K^-1
    """
    if debug:
        print(f"\nDEBUG: thermal_expansion_coefficient called")
        print(f"DEBUG: Temperatures: {temperatures}")
        print(f"DEBUG: Volumes: {volumes}")
    
    # Create a cubic spline interpolation of V(T)
    v_spline = InterpolatedUnivariateSpline(temperatures, volumes, k=3)
    
    if debug:
        print(f"DEBUG: Created cubic spline interpolation for V(T)")
        # Generate more points to show the spline curve
        t_fine = np.linspace(min(temperatures), max(temperatures), 100)
        v_fine = v_spline(t_fine)
        print(f"DEBUG: Spline evaluation at selected temperatures:")
        for i, t in enumerate(temperatures):
            print(f"DEBUG:   T={t:.1f}K: V_data={volumes[i]:.4f}, V_spline={v_spline(t):.4f}")
    
    # Calculate dV/dT
    dvdt = v_spline.derivative(1)(temperatures)
    
    if debug:
        print(f"DEBUG: First derivative (dV/dT) at each temperature:")
        for t, dv in zip(temperatures, dvdt):
            print(f"DEBUG:   T={t:.1f}K: dV/dT={dv:.6e}")
    
    # Calculate α = (1/V)(dV/dT)
    alpha = dvdt / volumes
    
    if debug:
        print(f"DEBUG: Thermal expansion coefficient (α = (1/V)(dV/dT)):")
        for t, v, dv, a in zip(temperatures, volumes, dvdt, alpha):
            print(f"DEBUG:   T={t:.1f}K: V={v:.4f}, dV/dT={dv:.6e}, α={a:.6e} K^-1")
    
    if verbose or debug:
        print("Thermal expansion coefficients:")
        for t, a in zip(temperatures, alpha):
            print(f"  T = {t:.1f} K: α = {a:.6e} K^-1")
    
    return alpha


def compute_qha(hessian_files: List[str], 
               volumes: List[float], 
               static_energies: List[float], 
               temperatures: List[float],
               pressures: List[float],
               nk: int = 10,
               verbose: bool = False,
               debug: bool = False) -> Dict:
    """
    Perform a full quasiharmonic approximation calculation.
    
    Parameters:
    -----------
    hessian_files : List[str]
        List of paths to Hessian files for different volumes
    volumes : List[float]
        List of volumes in Å³ corresponding to each Hessian file
    static_energies : List[float]
        List of static energies in kcal/mol for each volume
    temperatures : List[float]
        List of temperatures in K to evaluate
    pressures : List[float]
        List of pressures in GPa to evaluate
    nk : int
        Number of k-points along each direction
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    
    Returns:
    --------
    Dict
        Nested dictionary of QHA results
    """
    from extended_hessian import load_hessian_all_in_one
    
    if debug:
        print(f"\nDEBUG: compute_qha called")
        print(f"DEBUG: Number of volumes: {len(volumes)}")
        print(f"DEBUG: Number of hessian files: {len(hessian_files)}")
        print(f"DEBUG: Number of static energies: {len(static_energies)}")
        print(f"DEBUG: Temperatures: {temperatures}")
        print(f"DEBUG: Pressures: {pressures}")
        print(f"DEBUG: k-point mesh parameter: {nk}")
    
    # Generate k-point mesh
    kpoints = generate_kpoint_mesh(nk)
    if verbose or debug:
        print(f"Using {len(kpoints)} k-points for phonon dispersion")
    
    if debug:
        print(f"DEBUG: First few k-points: {kpoints[:3]}")
    
    # Calculate free energy for each volume and temperature
    free_energies = np.zeros((len(volumes), len(temperatures)))
    all_thermo_data = {}
    
    # Loop over volumes
    for i, (hessian_file, volume, static_energy) in enumerate(zip(hessian_files, volumes, static_energies)):
        if verbose or debug:
            print(f"\nProcessing volume {i+1}/{len(volumes)}: {volume:.2f} Å³")
        
        if debug:
            print(f"DEBUG: Loading Hessian file: {hessian_file}")
            print(f"DEBUG: Static energy: {static_energy}")
        
        # Load Hessian data
        results, metadata = load_hessian_all_in_one(hessian_file)
        
        if debug:
            print(f"DEBUG: Hessian loaded successfully")
            print(f"DEBUG: Metadata: {metadata}")
            print(f"DEBUG: Available results keys: {list(results.keys())}")
        
        # Extract Hessian matrices
        gamma_hessian = results.get('mass_weighted_gamma')
        extended_hessian = results.get('mass_weighted_extended')
        
        if debug:
            print(f"DEBUG: Gamma Hessian shape: {gamma_hessian.shape}")
            print(f"DEBUG: Extended Hessian contains {len(extended_hessian)} entries")
        
        # Compute phonon frequencies at k-points
        if verbose or debug:
            print(f"  Computing phonon frequencies...")
        frequencies = compute_phonon_dispersion(gamma_hessian, extended_hessian, kpoints, 
                                               verbose=verbose, debug=debug)

        # Store thermodynamic data for each temperature
        all_thermo_data[volume] = {}
        
        # Loop over temperatures
        for j, temperature in enumerate(temperatures):
            if temperature < 1e-6:  # Avoid division by zero
                temperature = 1e-6
                if debug:
                    print(f"DEBUG: Adjusting near-zero temperature to {temperature}")
                
            if verbose or debug:
                print(f"\nCalculating thermodynamic properties at T = {temperature:.1f} K")
            
            thermo_data = calculate_thermodynamics(
                frequencies, temperature, volume=volume, 
                static_energy=static_energy, verbose=verbose, debug=debug
            )

            # Store free energy for QHA
            free_energies[i, j] = thermo_data['helmholtz_energy']
            
            if debug:
                print(f"DEBUG: Helmholtz free energy at V={volume:.2f}, T={temperature:.1f}K: {thermo_data['helmholtz_energy']:.6f}")
            
            # Store thermodynamic data
            all_thermo_data[volume][temperature] = thermo_data
    
    if debug:
        print(f"\nDEBUG: Free energy matrix shape: {free_energies.shape}")
        print(f"DEBUG: Free energy summary:")
        for i, v in enumerate(volumes):
            for j, t in enumerate(temperatures):
                print(f"DEBUG:   V={v:.2f}, T={t:.1f}K: F={free_energies[i,j]:.6f}")
    
    # Perform QHA calculations for each temperature and pressure
    qha_results = {}

    # Convert lists to arrays
    volumes_array = np.array(volumes)
    temperatures_array = np.array(temperatures)
    
    # For each temperature and pressure, find equilibrium volume and properties
    for temperature in temperatures:
        qha_results[temperature] = {}
        
        # Get free energies at this temperature for all volumes
        t_idx = np.where(temperatures_array == temperature)[0][0]
        f_t = free_energies[:, t_idx]
        
        if debug:
            print(f"\nDEBUG: Processing temperature {temperature} K")
            print(f"DEBUG: Free energies at this temperature: {f_t}")
        
        for pressure in pressures:
            if verbose or debug:
                print(f"\nQHA calculation for T = {temperature:.1f} K, P = {pressure:.2f} GPa")
            
            # Calculate QHA properties
            qha_props = calculate_qha_properties(volumes_array, f_t, temperature, pressure, 
                                               verbose=verbose, debug=debug)
            qha_results[temperature][pressure] = qha_props
    
    # Calculate thermal expansion at ambient pressure
    if 0.0 in pressures:
        if debug:
            print(f"\nDEBUG: Calculating thermal expansion coefficients at P=0")
            
        p_idx = pressures.index(0.0)
        eq_volumes = [qha_results[t][0.0]['equilibrium_volume'] for t in temperatures]
        
        if debug:
            print(f"DEBUG: Equilibrium volumes at P=0: {eq_volumes}")
            
        alphas = thermal_expansion_coefficient(np.array(eq_volumes), temperatures_array, 
                                             verbose=verbose, debug=debug)
        
        # Add thermal expansion to results
        for i, t in enumerate(temperatures):
            qha_results[t][0.0]['thermal_expansion'] = alphas[i]
            
            if debug:
                print(f"DEBUG: Added thermal expansion coefficient {alphas[i]:.6e} K^-1 at T={t:.1f}K")
    
    if debug:
        print(f"\nDEBUG: QHA calculations complete")
    
    return {
        'volumes': volumes,
        'temperatures': temperatures,
        'pressures': pressures,
        'static_energies': static_energies,
        'free_energies': free_energies,
        'thermo_data': all_thermo_data,
        'qha_results': qha_results
    }
def plot_phonon_dos_individual(qha_results: Dict, temperature: float = 300.0, 
                             output_dir: str = '.', 
                             verbose: bool = False,
                             debug: bool = False):
    """
    Create separate phonon DOS plots for each volume, including negative frequencies.
    
    Parameters:
    -----------
    qha_results : Dict
        Results from compute_qha function
    temperature : float
        Temperature at which to plot the DOS (closest available will be used)
    output_dir : str
        Directory to save plots
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    """
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    
    if debug:
        print(f"\nDEBUG: plot_phonon_dos_individual called")
        print(f"DEBUG: Requested temperature: {temperature}K")
        print(f"DEBUG: Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    volumes = qha_results['volumes']
    temperatures = qha_results['temperatures']
    thermo_data = qha_results['thermo_data']
    
    # Find closest temperature to requested temperature
    closest_temp = min(temperatures, key=lambda x: abs(x - temperature))
    if verbose or debug:
        print(f"Using temperature {closest_temp} K (closest to requested {temperature} K)")
    
    if debug:
        print(f"DEBUG: Available temperatures: {temperatures}")
        print(f"DEBUG: Closest match: {closest_temp}K")
        print(f"DEBUG: Number of volumes to process: {len(volumes)}")
    
    # Create a separate plot for each volume
    for i, volume in enumerate(volumes):
        if debug:
            print(f"\nDEBUG: Creating plot for volume {i+1}/{len(volumes)}: {volume:.2f} Å³")
            
        plt.figure(figsize=(12, 7))
        
        # Get DOS data for this volume
        freq_bins = thermo_data[volume][closest_temp]['frequency_bins']
        dos = thermo_data[volume][closest_temp]['dos']
        
        if debug:
            print(f"DEBUG: Frequency bins: {len(freq_bins)} points from {np.min(freq_bins):.1f} to {np.max(freq_bins):.1f}")
            print(f"DEBUG: DOS max value: {np.max(dos):.6e} at {freq_bins[np.argmax(dos)]:.1f} cm^-1")
        
        # Calculate statistics for negative frequencies
        neg_mask = freq_bins < 0
        n_negative = np.sum(neg_mask)
        
        if debug:
            print(f"DEBUG: Number of negative frequency bins: {n_negative}")
            if n_negative > 0:
                print(f"DEBUG: Min negative frequency: {np.min(freq_bins[neg_mask]):.1f} cm^-1")
        
        # Plot DOS including negative frequencies
        plt.plot(freq_bins, dos, linewidth=1.5)
        
        # Highlight negative frequency region
        if n_negative > 0:
            plt.fill_between(freq_bins[neg_mask], dos[neg_mask], alpha=0.2, color='red',
                           label=f'{n_negative} negative frequency bins')
            
            if debug:
                print(f"DEBUG: Added highlight for negative frequency region")
        
        # Add vertical line at zero frequency
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7, label='Zero frequency')
        
        # Add annotations
        plt.xlabel('Frequency (cm$^{-1}$)')
        plt.ylabel('Density of States')
        plt.title(f'Phonon DOS at Volume = {volume:.2f} Å$^3$ (T = {closest_temp} K)')
        plt.grid(True)
        
        # Add legend if there are negative frequencies
        if n_negative > 0:
            plt.legend()
        
        # Save to file
        filename = f'phonon_dos_vol_{volume:.2f}'.replace('.', '_') + '.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        
        if verbose or debug:
            print(f"Saved DOS plot for volume {volume:.2f} Å³")

def plot_qha_results(qha_results: Dict, output_dir: str = '.', 
                   verbose: bool = False,
                   debug: bool = False):
    """
    Create plots from QHA results.
    
    Parameters:
    -----------
    qha_results : Dict
        Results from compute_qha function
    output_dir : str
        Directory to save plots
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    """
    import matplotlib.pyplot as plt
    import os
    
    if debug:
        print(f"\nDEBUG: plot_qha_results called")
        print(f"DEBUG: Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    volumes = qha_results['volumes']
    temperatures = qha_results['temperatures']
    pressures = qha_results['pressures']
    free_energies = qha_results['free_energies']
    qha_data = qha_results['qha_results']
    
    if debug:
        print(f"DEBUG: Volumes: {volumes}")
        print(f"DEBUG: Temperatures: {temperatures}")
        print(f"DEBUG: Pressures: {pressures}")
        print(f"DEBUG: Free energies matrix shape: {free_energies.shape}")
    
    # Plot 1: Free energy vs volume at different temperatures
    if debug:
        print(f"\nDEBUG: Creating Free Energy vs Volume plot")
        
    plt.figure(figsize=(10, 6))
    for i, temp in enumerate(temperatures):
        if debug:
            print(f"Plotting free energy for temperature {temp} K")
            print(f"free energies: {free_energies[:, i]}")
        plt.plot(volumes, free_energies[:, i], 'o-', label=f'{temp} K')
    
    plt.xlabel('Volume (Å$^3$)')
    plt.ylabel('Helmholtz Free Energy (kcal/mol)')
    plt.title('Free Energy vs Volume at Different Temperatures')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'free_energy_volume.png'), dpi=300)
    if not verbose:
        plt.close()
    
    if debug:
        print(f"DEBUG: Saved free_energy_volume.png")
    
    # Plot 2: Equilibrium volume vs temperature at different pressures
    if debug:
        print(f"\nDEBUG: Creating Volume vs Temperature plot")
        
    plt.figure(figsize=(10, 6))
    for p in pressures:
        if debug:
            print(f"Plotting equilibrium volume for pressure {p} GPa")
            eq_volumes = [qha_data[t][p]['equilibrium_volume'] for t in temperatures]
            print(f"equilibrium volumes: {eq_volumes}")
        eq_volumes = [qha_data[t][p]['equilibrium_volume'] for t in temperatures]
        plt.plot(temperatures, eq_volumes, 'o-', label=f'{p} GPa')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Equilibrium Volume (Å$^3$)')
    plt.title('Thermal Expansion: Volume vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_temperature.png'), dpi=300)
    if not verbose:
        plt.close()
    
    if debug:
        print(f"DEBUG: Saved volume_temperature.png")
    
    # Plot 3: Thermal expansion coefficient vs temperature (at P=0)
    if 0.0 in pressures:
        if debug:
            print(f"\nDEBUG: Creating Thermal Expansion Coefficient plot")
            
        p_idx = pressures.index(0.0)
        alphas = [qha_data[t][0.0].get('thermal_expansion', 0) for t in temperatures]
        
        if debug:
            print(f"DEBUG: Thermal expansion coefficients: {alphas}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(temperatures, alphas, 'o-')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Thermal Expansion Coefficient (K$^{-1}$)')
        plt.title('Thermal Expansion Coefficient vs Temperature')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'thermal_expansion.png'), dpi=300)
        if not verbose:
            plt.close()    
            
        if debug:
            print(f"DEBUG: Saved thermal_expansion.png")
    
    # Plot 4: Bulk modulus vs temperature at different pressures
    if debug:
        print(f"\nDEBUG: Creating Bulk Modulus plot")
        
    plt.figure(figsize=(10, 6))
    for p in pressures:
        bulk_moduli = [qha_data[t][p]['bulk_modulus'] for t in temperatures]
        
        if debug:
            print(f"DEBUG: Bulk moduli at P={p} GPa: {bulk_moduli}")
            
        plt.plot(temperatures, bulk_moduli, 'o-', label=f'{p} GPa')
    
    plt.xlabel('Temperature (K)')
    plt.ylabel('Bulk Modulus (GPa)')
    plt.title('Bulk Modulus vs Temperature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bulk_modulus.png'), dpi=300)
    if not verbose:
        plt.close()
    
    if debug:
        print(f"DEBUG: Saved bulk_modulus.png")
    
    # Plot 5: DOS at different volumes (at room temperature)
    if debug:
        print(f"\nDEBUG: Creating DOS at different volumes plot")
        
    room_temp = min(temperatures, key=lambda x: abs(x - 300))
    
    if debug:
        print(f"DEBUG: Using temperature {room_temp}K (closest to room temperature)")
        
    plt.figure(figsize=(10, 6))
    
    thermo_data = qha_results['thermo_data']
    for i, v in enumerate(volumes):
        if i % max(1, len(volumes)//5) == 0:  # Plot only a few volumes
            if debug:
                print(f"DEBUG: Adding DOS for volume {v:.2f} Å³")
                
            freq_bins = thermo_data[v][room_temp]['frequency_bins']
            dos = thermo_data[v][room_temp]['dos']
            plt.plot(freq_bins, dos, label=f'V = {v:.1f} Å$^3$')
    
    plt.xlabel('Frequency (cm$^{-1}$)')
    plt.ylabel('Density of States')
    plt.title(f'Phonon DOS at Different Volumes (T = {room_temp} K)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phonon_dos.png'), dpi=300)
    if not verbose:
        plt.close()
    
    if debug:
        print(f"DEBUG: Saved phonon_dos.png")
    
    # NEW PLOT: Gibbs free energy vs volume at different temperatures
    # Create plots for each pressure value
    for p in pressures:
        if debug:
            print(f"\nDEBUG: Creating Gibbs Free Energy vs Volume plot for P={p} GPa")
            
        plt.figure(figsize=(10, 6))
        
        # Convert pressure from GPa to kcal/(mol·Å³)
        pressure_kcal = p * 1e9 / (4184.0 * 1e30 / AVOGADRO)
        
        # For each temperature, plot G = F + PV
        for i, temp in enumerate(temperatures):
            gibbs_energies = free_energies[:, i] + pressure_kcal * np.asarray(volumes)
            
            # Plot Gibbs free energy curve
            plt.plot(volumes, gibbs_energies, 'o-', label=f'{temp} K')
            
            # Mark the equilibrium volume with an X
            eq_vol = qha_data[temp][p]['equilibrium_volume']
            eq_gibbs = qha_data[temp][p]['gibbs_energy']
            plt.plot(eq_vol, eq_gibbs, 'kx', markersize=10)
            
            if debug:
                print(f"DEBUG: T={temp}K: Equilibrium V={eq_vol:.3f} Å³, G={eq_gibbs:.6f} kcal/mol")
        
        plt.xlabel('Volume (Å$^3$)')
        plt.ylabel('Gibbs Free Energy (kcal/mol)')
        plt.title(f'Gibbs Free Energy vs Volume at P = {p} GPa')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gibbs_energy_volume_p{p}.png'.replace('.', '_')), dpi=300)
        if not verbose:
            plt.close()
        
        if debug:
            print(f"DEBUG: Saved gibbs_energy_volume_p{p}.png")
    
    if verbose or debug:
        print(f"All plots saved to {output_dir}/")


def run_qha_workflow(structure_template, volume_factors, 
                   temperatures=[0, 50, 100, 150, 200, 250, 300],
                   pressures=[0.0, 1.0, 2.0], nk=8,
                   output_dir='qha_results',
                   verbose=False,
                   debug=False,
                   structure_type='Pa3'):
    """
    Run a complete QHA workflow.
    
    Parameters:
    -----------
    structure_template : pymatgen.Structure
        Template structure
    volume_factors : List[float]
        List of volume scaling factors to use
    temperatures : List[float]
        List of temperatures in K
    pressures : List[float]
        List of pressures in GPa
    nk : int
        Number of k-points along each direction
    output_dir : str
        Directory to save results
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    """
    from extended_hessian import compute_hessian_at_structure
    from energy import compute_energy_from_cell
    import os
    from pymatgen.core.structure import Structure
    
    if debug:
        print(f"\nDEBUG: run_qha_workflow called")
        print(f"DEBUG: Output directory: {output_dir}")
        print(f"DEBUG: Volume factors: {volume_factors}")
        print(f"DEBUG: Temperatures: {temperatures}")
        print(f"DEBUG: Pressures: {pressures}")
        print(f"DEBUG: nk = {nk}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create structures with different volumes
    structures = []
    volumes = []
    static_energies = []
    hessian_files = []
    
    for factor in volume_factors:
        if debug:
            print(f"\nDEBUG: Processing volume factor {factor:.3f}")
            
        # Scale the structure
        scaled_structure = structure_template.copy()
        scaled_structure.scale_lattice(structure_template.volume * factor)
        
        if debug:
            print(f"DEBUG: Scaled structure volume: {scaled_structure.volume:.3f} Å³")
            print(f"DEBUG: Lattice parameters: a={scaled_structure.lattice.a:.4f}, b={scaled_structure.lattice.b:.4f}, c={scaled_structure.lattice.c:.4f}")

        # If the structure is a Pa3 structure, adjust fractional coordinates
        if structure_type == 'Pa3':
            pa3_structure = Pa3(a=scaled_structure.lattice.a)
            pa3_structure.adjust_fractional_coords(bond_length=1.1609)
            scaled_structure = build_unit_cell(pa3_structure)
        elif structure_type == 'Cmce':
            cmce_structure = Cmce(a=scaled_structure.lattice.a,
                                  b=scaled_structure.lattice.b, 
                                  c=scaled_structure.lattice.c)
            cmce_structure.adjust_fractional_coords(bond_length=1.1609,bond_angle=90.0)
            scaled_structure = build_unit_cell(cmce_structure)
        
        if debug:
            print(f"DEBUG: Created Pa3 structure with a={scaled_structure.lattice.a:.4f}")
            print(f"DEBUG: Final structure volume: {scaled_structure.volume:.3f} Å³")
        
        structures.append(scaled_structure)
        volumes.append(scaled_structure.volume)
        
        # Calculate static energy
        # This would normally come from your energy calculator
        # For now, just use a placeholder function
        #static_energy = calculate_static_energy(scaled_structure)
        static_energy = compute_energy_from_cell(scaled_structure, None)
        static_energies.append(static_energy)
        
        if debug:
            print(f"DEBUG: Calculated static energy: {static_energy:.6f} kcal/mol")
        
        # Generate a file path for this structure's Hessian
        hessian_file = os.path.join(output_dir, f"hessian_vol_{factor:.3f}.npz")
        hessian_files.append(hessian_file)
        
        if debug:
            print(f"DEBUG: Hessian file path: {hessian_file}")
        
        # Check if the Hessian file already exists
        if not os.path.exists(hessian_file):
            if verbose or debug:
                print(f"\nCalculating Hessian for volume factor {factor:.3f}")
            
            if debug:
                print(f"DEBUG: Hessian file does not exist, calculating now")
                
            # Calculate Hessian
            results = compute_hessian_at_structure(
                scaled_structure, 
                stepsize=0.005,
                method='finite_diff',
                potential='sapt',
                verbose=verbose
            )
            
            if debug:
                print(f"DEBUG: Hessian calculation complete")
                print(f"DEBUG: Results keys: {list(results.keys())}")
            
            # Save the Hessian
            from extended_hessian import save_hessian_all_in_one
            save_hessian_all_in_one(results, hessian_file.replace('.npz', ''))
            
            if debug:
                print(f"DEBUG: Saved Hessian to {hessian_file}")
            
        elif verbose or debug:
            print(f"\nUsing existing Hessian file for volume factor {factor:.3f}")
    
    if verbose or debug:
        print(f"Static energies: {static_energies}")

    # Run the QHA calculation
    qha_results = compute_qha(
        hessian_files,
        volumes,
        static_energies,
        temperatures,
        pressures,
        nk=nk,
        verbose=verbose,
        debug=debug
    )
    
    # Plot results
    plot_qha_results(qha_results, output_dir, verbose=verbose, debug=debug)

    # Plot individual DOS plots with all frequencies
    if debug:
        plot_phonon_dos_individual(qha_results, temperature=300.0, output_dir=output_dir, 
                                 verbose=verbose, debug=debug)
    
    # Save results
    results_file = os.path.join(output_dir, 'qha_results.npz')
    np.savez_compressed(results_file, **qha_results)
    
    if debug:
        print(f"\nDEBUG: Results saved to {results_file}")
    
    return qha_results

def run_qha_workflow_sg(spacegroup,
                        init_params,
                        volume_factors, 
                        temperatures=[0, 50, 100, 150, 200, 250, 300],
                        pressures=[0.0], 
                        nk=8,
                        output_dir='qha_results',
                        verbose=False,
                        debug=False):
    """
    Run a complete QHA workflow.
    
    Parameters:
    -----------
    structure_template : pymatgen.Structure
        Template structure
    volume_factors : List[float]
        List of volume scaling factors to use
    temperatures : List[float]
        List of temperatures in K
    pressures : List[float]
        List of pressures in GPa
    nk : int
        Number of k-points along each direction
    output_dir : str
        Directory to save results
    verbose : bool, optional
        Whether to print detailed information
    debug : bool, optional
        Whether to print extensive debug information
    """
    from extended_hessian import compute_hessian_at_structure
    import os
    from pymatgen.core.structure import Structure
    
    if debug:
        print(f"\nDEBUG: run_qha_workflow called")
        print(f"DEBUG: Output directory: {output_dir}")
        print(f"DEBUG: Volume factors: {volume_factors}")
        print(f"DEBUG: Temperatures: {temperatures}")
        print(f"DEBUG: Pressures: {pressures}")
        print(f"DEBUG: nk = {nk}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create initial structure from spacegroup and parameters
    if spacegroup == 'Pa3':
        # Create a Pa3 structure with given parameters
        a_opt = init_params.get('a')
        bond_length_opt = init_params.get('bond_length')
        pa3_structure = Pa3(a=a_opt)
        pa3_structure.adjust_fractional_coords(bond_length=bond_length_opt)
        structure_template = build_unit_cell(pa3_structure)
        if debug:
            print(f"DEBUG: Created Pa3 structure with a={structure_template.lattice.a:.4f} b={structure_template.lattice.b:.4f} c={structure_template.lattice.c:.4f}")

    elif spacegroup == 'Cmce':
        # Create a Cmce structure with given parameters
        a_opt = init_params.get('a')
        b_opt = init_params.get('b')
        c_opt = init_params.get('c')
        bond_length_opt = init_params.get('bond_length')
        bond_angle_opt = init_params.get('bond_angle')
        cmce_structure = Cmce(a=a_opt, b=b_opt, c=c_opt)
        cmce_structure.adjust_fractional_coords(bond_length=bond_length_opt, bond_angle=bond_angle_opt)
        structure_template = build_unit_cell(cmce_structure)
        if debug:
            print(f"DEBUG: Created Cmce structure with a={structure_template.lattice.a:.4f}, b={structure_template.lattice.b:.4f}, c={structure_template.lattice.c:.4f}")
    elif spacegroup == 'P42mnm':
        # Create a P42mnm structure with given parameters
        a_opt = init_params.get('a')
        b_opt = init_params.get('b')
        c_opt = init_params.get('c')
        bond_length_opt = init_params.get('bond_length')
        p42mnm_structure = P42mnm(a=a_opt, c=c_opt)
        p42mnm_structure.adjust_fractional_coords(bond_length=bond_length_opt)
        structure_template = build_unit_cell(p42mnm_structure)
        if debug:
            print(f"DEBUG: Created P42mnm structure with a={structure_template.lattice.a:.4f}, b={structure_template.lattice.b:.4f}, c={structure_template.lattice.c:.4f}")
    

    structures = []
    volumes = []
    static_energies = []
    hessian_files = []
    
    for factor in volume_factors:
        if debug:
            print(f"\nDEBUG: Processing volume factor {factor:.3f}")
            
        # Scale the structure
        scaled_structure = structure_template.copy()
        scaled_structure.scale_lattice(structure_template.volume * factor)
        
        if debug:
            print(f"DEBUG: Scaled structure volume: {scaled_structure.volume:.3f} Å³")
            print(f"DEBUG: Lattice parameters: a={scaled_structure.lattice.a:.4f}, b={scaled_structure.lattice.b:.4f}, c={scaled_structure.lattice.c:.4f}")

        if spacegroup == 'Pa3':
            pa3_structure = Pa3(a=scaled_structure.lattice.a)
            pa3_structure.adjust_fractional_coords(bond_length=bond_length_opt)
            scaled_structure = build_unit_cell(pa3_structure)

        elif spacegroup == 'Cmce':
            cmce_structure = Cmce(a=scaled_structure.lattice.a,
                                  b=scaled_structure.lattice.b, 
                                  c=scaled_structure.lattice.c)
            cmce_structure.adjust_fractional_coords(bond_length=bond_length_opt,bond_angle=bond_angle_opt)
            scaled_structure = build_unit_cell(cmce_structure)

        elif spacegroup == 'P42mnm':
            p42mnm_structure = P42mnm(a=scaled_structure.lattice.a, 
                                      c=scaled_structure.lattice.c)
            p42mnm_structure.adjust_fractional_coords(bond_length=bond_length_opt)
            scaled_structure = build_unit_cell(p42mnm_structure)

        #elif spacegroup == 'R3c':

            # p42mnm_structure = R3c(a=scaled_structure.lattice.a, 
            #                           c=scaled_structure.lattice.c)
            # p42mnm_structure.adjust_fractional_coords(bond_length=bond_length_opt)
            # scaled_structure = build_unit_cell(p42mnm_structure)
        
        
        # structures.append(scaled_structure)
        volumes.append(scaled_structure.volume)
        
        # Calculate static energy
        # This would normally come from your energy calculator
        # For now, just use a placeholder function
        static_energy = compute_energy_from_cell(scaled_structure, None)
        static_energies.append(static_energy)
        
        if debug:
            print(f"DEBUG: Calculated static energy: {static_energy:.6f} kcal/mol")
        
        # Generate a file path for this structure's Hessian
        hessian_file = os.path.join(output_dir, f"hessian_vol_{factor:.3f}.npz")
        hessian_files.append(hessian_file)
        
        if debug:
            print(f"DEBUG: Hessian file path: {hessian_file}")
        
        # Check if the Hessian file already exists
        if not os.path.exists(hessian_file):
            if verbose or debug:
                print(f"\nCalculating Hessian for volume factor {factor:.3f}")
            
            if debug:
                print(f"DEBUG: Hessian file does not exist, calculating now")
                
            # Calculate Hessian
            results = compute_hessian_at_structure(
                scaled_structure, 
                stepsize=0.005,
                method='mixed',
                potential='sapt',
                verbose=verbose
            )
            
            if debug:
                print(f"DEBUG: Hessian calculation complete")
                print(f"DEBUG: Results keys: {list(results.keys())}")
            
            # Save the Hessian
            from extended_hessian import save_hessian_all_in_one
            save_hessian_all_in_one(results, hessian_file.replace('.npz', ''))
            
            if debug:
                print(f"DEBUG: Saved Hessian to {hessian_file}")
            
        elif verbose or debug:
            print(f"\nUsing existing Hessian file for volume factor {factor:.3f}")
    
    if verbose or debug:
        print(f"Static energies: {static_energies}")

    # Run the QHA calculation
    qha_results = compute_qha(
        hessian_files,
        volumes,
        static_energies,
        temperatures,
        pressures,
        nk=nk,
        verbose=verbose,
        debug=debug
    )
    
    # Plot results
    plot_qha_results(qha_results, output_dir, verbose=verbose, debug=debug)

    # Plot individual DOS plots with all frequencies
    if verbose or debug:
        plot_phonon_dos_individual(qha_results, temperature=300.0, output_dir=output_dir, 
                                 verbose=verbose, debug=debug)
    
    # Save results
    results_file = os.path.join(output_dir, 'qha_results.npz')
    np.savez_compressed(results_file, **qha_results)
    
    if debug:
        print(f"\nDEBUG: Results saved to {results_file}")
    
    return qha_results


def calculate_static_energy(structure):
    """
    Calculate the static energy of a structure.
    
    This is a placeholder function - in practice, you would use your
    actual energy calculation function here.
    
    Parameters:
    -----------
    structure : pymatgen.Structure
        Structure to calculate energy for
        
    Returns:
    --------
    float
        Static energy in kcal/mol
    """
    from energy import wrap_coordinates_by_carbon_fractional
    from co2_potential import p1b, sapt
    import numpy as np
    
    # Get coordinates and molecule grouping
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
    
    # Calculate energy
    energy = 0.0
    
    # 1-body terms: p1b for each fragment
    for ifrag in range(nfrags):
        fragment_coords = crd[ifrag*9:(ifrag+1)*9]
        energy += p1b(fragment_coords)
    
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
                        for p in range(3):  # 3 atoms per molecule
                            mol_j[p*3 + 0] += x * pbc[0]
                            mol_j[p*3 + 1] += y * pbc[1]
                            mol_j[p*3 + 2] += z * pbc[2]
                        
                        # Set interaction factor
                        factor = 0.5
                        if x == 0 and y == 0 and z == 0:
                            factor = 1.0
                            if ifrag >= jfrag:
                                continue
                        
                        # Combine molecules for dimer calculation
                        dimer_coords = np.concatenate([mol_i, mol_j])
                        energy += sapt(dimer_coords) * factor
    
    return energy


if __name__ == "__main__":
    import argparse
    from spacegroup import Pa3, Cmce, P42mnm, R3c
    from symmetry import build_unit_cell
    
    parser = argparse.ArgumentParser(description="Quasiharmonic Approximation Calculations")
    parser.add_argument("--mode", choices=['single', 'qha'], default='single',
                      help="Mode of operation: single temperature or full QHA")
    parser.add_argument("--hessian", help="Path to Hessian file (for single mode)")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in K (for single mode)")
    parser.add_argument("--pressure", type=float, default=0.0, help="Pressure in GPa (for single mode)")
    parser.add_argument("--nk", type=int, default=8, help="Number of k-points along each direction")
    parser.add_argument("--output-dir", default="qha_results", help="Output directory")
    parser.add_argument("--vol-min", type=float, default=0.95, help="Minimum volume factor (for QHA mode)")
    parser.add_argument("--vol-max", type=float, default=1.05, help="Maximum volume factor (for QHA mode)")
    parser.add_argument("--vol-steps", type=int, default=5, help="Number of volume steps (for QHA mode)")
    parser.add_argument("--t-min", type=float, default=0.0, help="Minimum temperature in K (for QHA mode)")
    parser.add_argument("--t-max", type=float, default=300.0, help="Maximum temperature in K (for QHA mode)")
    parser.add_argument("--t-steps", type=int, default=7, help="Number of temperature steps (for QHA mode)")
    parser.add_argument("--p-values", type=float, nargs='+', default=[0.0, 1.0, 2.0],
                      help="Pressure values in GPa (for QHA mode)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output (implies verbose)")

    args = parser.parse_args()
    
    # If debug is enabled, automatically enable verbose too
    if args.debug:
        args.verbose = True
        print("DEBUG mode enabled (includes all verbose output)")
    elif args.verbose:
        print("Verbose mode enabled")
    
    if args.mode == 'single':
        # Process a single Hessian file at a specific temperature
        from extended_hessian import load_hessian_extended, process_hessian_for_thermodynamics
        
        if not args.hessian:
            print("Error: --hessian argument required for single mode")
            exit(1)
        
        # Generate k-point mesh
        kpoints = generate_kpoint_mesh(args.nk)
        if args.verbose:
            print(f"Using {len(kpoints)} k-points for phonon dispersion")
        
        # Load the Hessian data
        results, metadata = load_hessian_extended(args.hessian)
        
        # Extract needed data
        gamma_hessian = results.get('mass_weighted_hessian', results['gamma_hessian'])
        extended_hessian = results.get('extended_hessian', {})
        
        # Compute phonon frequencies at k-points
        if args.verbose:
            print(f"Computing phonon frequencies...")
        frequencies = compute_phonon_dispersion(gamma_hessian, extended_hessian, kpoints, 
                                              verbose=args.verbose, debug=args.debug)

        # Calculate thermodynamic properties
        if args.verbose:
            print(f"Calculating thermodynamic properties at T = {args.temperature} K, P = {args.pressure} GPa")
        thermo_props = calculate_thermodynamics(
            frequencies, args.temperature, pressure=args.pressure, 
            verbose=args.verbose, debug=args.debug
        )

        # Output results
        print("\nThermodynamic Properties:")
        print(f"Temperature: {args.temperature} K")
        print(f"Pressure: {args.pressure} GPa")
        print(f"Zero-point energy: {thermo_props['zero_point_energy']:.6f} kcal/mol")
        print(f"Entropy: {thermo_props['entropy']:.6f} J/(mol·K)")
        print(f"Heat capacity (Cv): {thermo_props['heat_capacity_v']:.6f} cal/(mol·K)")
        print(f"Helmholtz free energy: {thermo_props['helmholtz_energy']:.6f} kcal/mol")
        
        # Plot DOS
        import matplotlib.pyplot as plt
        
        # Define a simple DOS plotting function
        def plot_dos(freq_bins, dos, save_path=None):
            plt.figure(figsize=(10, 6))
            plt.plot(freq_bins, dos)
            plt.xlabel('Frequency (cm$^{-1}$)')
            plt.ylabel('Density of States')
            plt.title('Phonon Density of States')
            plt.grid(True)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300)
                if args.verbose:
                    print(f"DOS plot saved to {save_path}")
            plt.close()
            
        plot_dos(thermo_props['frequency_bins'], thermo_props['dos'], 
                save_path=f"{args.hessian.replace('.npz', '')}_dos.png")
        
    else:  # QHA mode
        # Create a Pa3 structure as an example
        pa3_structure = Pa3(a=5.4848)
        pa3_structure.adjust_fractional_coords(bond_length=1.1609)
        structure = build_unit_cell(pa3_structure)
        
        # Create volume factors
        volume_factors = np.linspace(args.vol_min, args.vol_max, args.vol_steps)
        
        # Create temperature range
        temperatures = np.linspace(args.t_min, args.t_max, args.t_steps)
        
        # Run QHA workflow
        # qha_results = run_qha_workflow(
        #     structure,
        #     volume_factors,
        #     temperatures=temperatures,
        #     pressures=args.p_values,
        #     nk=args.nk,
        #     output_dir=args.output_dir,
        #     verbose=args.verbose,
        #     debug=args.debug

        # )

        # spacegroup = 'Pa3'
        # init_params={'a': 5.4848, 'bond_length': 1.1609}

        # spacegroup = 'Cmce'
        # init_params={'a': 5.0617, 'b': 4.8746, 'c': 6.7887, 'bond_length': 1.1613, 'bond_angle': 45.07}

        spacegroup = 'P42mnm'
        init_params={'a':  4.248, 'c': 4.631, 'bond_length': 1.161}

        qha_results = run_qha_workflow_sg(
            spacegroup,
            init_params,
            volume_factors,
            temperatures=temperatures,
            pressures=args.p_values,
            nk=args.nk,
            output_dir=args.output_dir,
            verbose=args.verbose,
            debug=args.debug
        )
        
        # Print summary of QHA results
        print("\nQHA Results Summary:")
        print(f"Volume range: {min(qha_results['volumes']):.2f} - {max(qha_results['volumes']):.2f} Å³")
        print(f"Temperature range: {min(temperatures):.1f} - {max(temperatures):.1f} K")
        print(f"Pressure values: {args.p_values} GPa")

        # Print equilibrium volumes and free energies at lowest pressure and room temperature (300K) if available
        room_temp = min(temperatures, key=lambda x: abs(x - 300))
        lowest_pressure = min(args.p_values)
        eq_volume = qha_results['qha_results'][room_temp][lowest_pressure]['equilibrium_volume']
        free_energy = qha_results['qha_results'][room_temp][lowest_pressure]['gibbs_energy']
        print(f"\nEquilibrium properties at {room_temp} K and {lowest_pressure} GPa:")
        print(f"  Equilibrium volume: {eq_volume:.2f} Å³")
        print(f"  Free energy: {free_energy:.4f} kcal/mol")


        
        # # Print thermal expansion at room temperature (300K) if available
        # room_temp = min(temperatures, key=lambda x: abs(x - 300))
        # if 0.0 in args.p_values:
        #     thermal_exp = qha_results['qha_results'][room_temp][0.0].get('thermal_expansion', 0)
        #     print(f"Thermal expansion coefficient at {room_temp}K: {thermal_exp:.6e} K⁻¹")

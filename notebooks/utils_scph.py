# scph_worker.py
import time
import os
import numpy as np
from hiphive import ClusterSpace, ForceConstantPotential
from hiphive.self_consistent_phonons import self_consistent_harmonic_model
from ase.optimize import BFGS
from utils_co2 import CO2TwoBodyRC_IMAGES

def scale_to_volume(atoms, target_V):
    a = atoms.copy()
    s = (target_V / a.get_volume())**(1/3)
    a.set_cell(a.cell * s, scale_atoms=True)
    return a

def process_temperature(T, V, geom, supercell_shape, cutoffs, 
                       n_structures=100, n_iterations=30, mix_schedule=None, 
                       use_standardize=True, alpha_schedule=None, 
                       imag_schedule=None, beta_blend=0.3,
                       prev_params=None, rc=5.45):
    """Process a single (T,V) combination for SCPH."""
    print(f"Starting T={T:.1f} K, V={V:.1f}", flush=True)
    start_time = time.perf_counter()
    
    # Set default schedules if not provided
    if mix_schedule is None:
        mix_schedule = [1e-1]*5 + [8e-2]*5 + [3e-2]*5 + [8e-3]*(n_iterations-15)

    if alpha_schedule is None:
        alpha_schedule = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3] if use_standardize else [1e-2, 3e-3, 1e-3]
    
    if imag_schedule is None:
        imag_schedule = [0.1]*5 + [0.03]*5 + [0.01]*(n_iterations-10)
    
    # Prepare geometry at this volume
    geom_v = scale_to_volume(geom, V)
    geom_v.calc = CO2TwoBodyRC_IMAGES(rc=rc)
    geom_v.set_constraint()
    
    # Run BFGS if needed (or skip if already relaxed)
    BFGS(geom_v, logfile=None).run(fmax=1e-3, steps=100)
    
    # Setup for SCPH
    supercell = geom_v.repeat(supercell_shape)
    supercell.calc = CO2TwoBodyRC_IMAGES(rc=rc)
    cs = ClusterSpace(geom_v, cutoffs)
    
    # Initialize parameters trajectory
    params_traj = []
    x_prev = prev_params  # Start from previous temperature's parameters if available
    
    # SCPH iterations
    for it in range(n_iterations):
        # Get current alpha and imag values from schedules
        mix = mix_schedule[min(it, len(mix_schedule)-1)]
        alpha = alpha_schedule[min(it, len(alpha_schedule)-1)]
        imag = imag_schedule[min(it, len(imag_schedule)-1)]
        
        try:
            # Run a single SCPH iteration
            out = self_consistent_harmonic_model(
                supercell, 
                supercell.calc,  # Use the real calculator 
                cs, 
                max(T, 1e-3),
                mix, 
                1,  # Run 1 SCPH step per outer-loop iter
                n_structures,
                imag_freq_factor=imag,
                parameters_start=x_prev,
                QM_statistics=False,
                fit_kwargs={"fit_method":"ridge", "standardize":use_standardize, "alpha":alpha},

            )
            x_fit = out[-1]
            
            # Apply proximal damping of parameter updates
            if x_prev is not None:
                x_new = (1.0 - beta_blend) * x_prev + beta_blend * x_fit
            else:
                x_new = x_fit
                
            params_traj.append(x_new)
            x_prev = x_new
            
            print(f"  Iter {it}: |params| = {np.linalg.norm(x_new):.6f}")
            
        except Exception as e:
            print(f"Error at iter {it} for T={T:.1f} K, V={V:.1f}: {str(e)}")
            if len(params_traj) == 0:
                raise  # Re-raise if we haven't made any progress
            break  # Otherwise use last good parameters
    
    # Create FCP from final parameters
    if len(params_traj) > 0:
        fcp_scph = ForceConstantPotential(cs, params_traj[-1])
        
        # Save results
        os.makedirs('fcps_mp/', exist_ok=True)
        os.makedirs('scph_trajs_mp/', exist_ok=True)
        fcp_scph.write(f'fcps_mp/scph_T{T}_V{V:.1f}.fcp')
        np.savetxt(f'scph_trajs_mp/scph_parameters_T{T}_V{V:.1f}', np.array(params_traj))
    else:
        print(f"No valid parameters found for T={T:.1f} K, V={V:.1f}")
        
    elapsed = time.perf_counter() - start_time
    print(f"Finished T={T:.1f} K, V={V:.1f} in {elapsed/60:.2f} min", flush=True)
    
    # Return the final parameters for possible reuse in next temperature
    return (T, V, params_traj[-1] if len(params_traj) > 0 else None)

def process_volume(V, geom, supercell_shape, cutoffs, temperatures,
                   n_structures=100, n_iterations=30, mix_schedule=None, 
                   use_standardize=True, alpha_schedule=None, 
                   imag_schedule=None, beta_blend=0.3, rng_seed=42, rc=5.45):
    """Process all temperatures for a single volume, reusing parameters between temperatures."""
    print(f"Starting volume V={V:.1f} Å³", flush=True)
    start_time = time.perf_counter()
    
    # Set default schedules if not provided
    if mix_schedule is None:
        mix_schedule = [3e-1]*5 + [9e-2]*5 + [8e-2]*5 + [5e-2]*(n_iterations-15)
    
    if alpha_schedule is None:
        alpha_schedule = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3] if use_standardize else [1e-2, 3e-3, 1e-3]
    
    if imag_schedule is None:
        imag_schedule = [0.1]*5 + [0.03]*5 + [0.01]*(n_iterations-10)
    
    # Prepare geometry at this volume
    geom_v = scale_to_volume(geom, V)
    geom_v.calc = geom.calc
    #geom_v.calc = CO2TwoBodyRC_IMAGES(rc=rc)
    geom_v.set_constraint()
    
    # Run BFGS if needed (or skip if already relaxed)
    from ase.optimize import BFGS
    BFGS(geom_v, logfile=None).run(fmax=1e-3, steps=100)
    
    # Setup for SCPH
    supercell = geom_v.repeat(supercell_shape)
    supercell.calc = geom_v.calc
    #supercell.calc = CO2TwoBodyRC_IMAGES(rc=rc)
    cs = ClusterSpace(geom_v, cutoffs)
    
    # Process all temperatures in sequence
    prev_params = None
    results = []
    
    for T in sorted(temperatures):
        print(f"  Volume {V:.1f} - Processing T={T:.1f} K...", flush=True)
        temp_start = time.perf_counter()
        
        # Initialize parameters trajectory
        params_traj = []
        x_prev = prev_params  # Use previous temperature's parameters if available
        
        # SCPH iterations
        for it in range(n_iterations):
            # Get current alpha and imag values from schedules
            mix = mix_schedule[min(it, len(mix_schedule)-1)]
            alpha = alpha_schedule[min(it, len(alpha_schedule)-1)]
            imag = imag_schedule[min(it, len(imag_schedule)-1)]
            
            try:
                # Run a single SCPH iteration
                QM_stats = True if T < 10 else False

                out = self_consistent_harmonic_model(
                    supercell, 
                    supercell.calc,  # Use the real calculator 
                    cs, 
                    max(T, 1e0),
                    mix, 
                    1,  # Run 1 SCPH step per outer-loop iter
                    n_structures,
                    imag_freq_factor=imag,
                    parameters_start=x_prev,
                    QM_statistics=QM_stats,
                    fit_kwargs={"fit_method":"ridge", "standardize":use_standardize, "alpha":alpha},
                    #random_state=rng_seed + it if rng_seed is not None else None
                )
                x_fit = out[-1]
                
                # Apply proximal damping of parameter updates
                if x_prev is not None:
                    x_new = (1.0 - beta_blend) * x_prev + beta_blend * x_fit
                else:
                    x_new = x_fit
                    
                params_traj.append(x_new)
                x_prev = x_new
                
                print(f"    Iter {it}: |params| = {np.linalg.norm(x_new):.6f}", flush=True)
                
            except Exception as e:
                print(f"Error at iter {it} for T={T:.1f} K, V={V:.1f}: {str(e)}", flush=True)
                if len(params_traj) == 0:
                    raise  # Re-raise if we haven't made any progress
                break  # Otherwise use last good parameters
        
        # Create FCP from final parameters
        if len(params_traj) > 0:
            fcp_scph = ForceConstantPotential(cs, params_traj[-1])
            
            # Save results
            os.makedirs('fcps_mp/', exist_ok=True)
            os.makedirs('scph_trajs_mp/', exist_ok=True)
            fcp_scph.write(f'fcps_mp/scph_T{T}_V{V:.1f}.fcp')
            np.savetxt(f'scph_trajs_mp/scph_parameters_T{T}_V{V:.1f}', np.array(params_traj))
            
            # Save parameters for next temperature
            prev_params = params_traj[-1]
        else:
            print(f"No valid parameters found for T={T:.1f} K, V={V:.1f}", flush=True)
        
        temp_elapsed = time.perf_counter() - temp_start
        print(f"  Finished T={T:.1f} K in {temp_elapsed/60:.2f} min", flush=True)
        
        # Add to results
        results.append((T, V, prev_params))
    
    elapsed = time.perf_counter() - start_time
    print(f"Volume {V:.1f} completed in {elapsed/60:.2f} min", flush=True)
    
    return (V, results)

def process_volume_enhanced(V, geom, supercell_shape, cutoffs, temperatures,
                          n_structures=100, n_iterations=30, mix_schedule=None, 
                          use_standardize=True, alpha_schedule=None, 
                          imag_schedule=None, beta_blend=0.3, orig_params=None, rng_seed=42, 
                          rc=5.45, n_jobs=4, conv_rel_dx=1e-3, conv_patience=3):
    """Process all temperatures for a single volume with enhanced SCPH."""
    print(f"Starting volume V={V:.1f} Å³", flush=True)
    start_time = time.perf_counter()
    
    # Set default schedules if not provided
    if mix_schedule is None:
        mix_schedule = [3e-1]*5 + [9e-2]*5 + [8e-2]*5 + [5e-2]*(n_iterations-15)
    
    if alpha_schedule is None:
        alpha_schedule = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3] if use_standardize else [1e-2, 3e-3, 1e-3]
    
    if imag_schedule is None:
        imag_schedule = [0.1]*5 + [0.03]*5 + [0.01]*(n_iterations-10)
    
    # Prepare geometry at this volume
    geom_v = scale_to_volume(geom, V)
    geom_v.calc = CO2TwoBodyRC_IMAGES(rc=rc)
    geom_v.set_constraint()
    
    # Run BFGS if needed (or skip if already relaxed)
    from ase.optimize import BFGS
    BFGS(geom_v, logfile=None).run(fmax=1e-3, steps=100)
    
    # Setup for SCPH
    supercell = geom_v.repeat(supercell_shape)
    supercell.calc = CO2TwoBodyRC_IMAGES(rc=rc)
    cs = ClusterSpace(geom_v, cutoffs)
    
    # Process all temperatures in sequence
    if orig_params is None:
        prev_params = None
    else:
        prev_params = orig_params

    results = []
    
    for T in sorted(temperatures):
        print(f"  Volume {V:.1f} - Processing T={T:.1f} K...", flush=True)
        temp_start = time.perf_counter()
        
        # Determine if we should use quantum statistics
        QM_stats = True if T < 10 else False
        
        # Select current iteration's parameters based on schedule
        # Parameters schedules are applied inside the enhanced function
        

        try:
            # Run enhanced SCPH with convergence criteria
            params_traj, metrics, converged = self_consistent_harmonic_model_enhanced(
                supercell, 
                supercell.calc,
                cs, 
                max(T, 1e0),  # Use minimum effective T=1K
                mix_schedule[0],  # Initial mixing (will be adjusted based on schedule)
                n_iterations,
                n_structures,
                QM_statistics=QM_stats,
                parameters_start=prev_params,
                imag_freq_factor=imag_schedule[0],  # Initial value
                fit_kwargs={"fit_method":"ridge", "standardize":use_standardize, 
                           "alpha":alpha_schedule[0]},  # Initial alpha
                n_jobs=n_jobs,
                verbose=1,
                conv_rel_dx=conv_rel_dx,
                conv_patience=conv_patience,
                beta_blend=beta_blend,
                return_metrics=True,
                random_seed=rng_seed
            )
            
            # Create FCP from final parameters
            fcp_scph = ForceConstantPotential(cs, params_traj[-1])
            
            # Save results
            os.makedirs('fcps_mp/', exist_ok=True)
            os.makedirs('scph_trajs_mp/', exist_ok=True)
            fcp_scph.write(f'fcps_mp/scph_T{T}_V{V:.1f}.fcp')
            np.savetxt(f'scph_trajs_mp/scph_parameters_T{T}_V{V:.1f}', np.array(params_traj))
            
            # Save convergence metrics
            np.save(f'scph_trajs_mp/metrics_T{T}_V{V:.1f}.npy', metrics)
            
            # Save parameters for next temperature
            prev_params = params_traj[-1]
            
            # Print convergence information
            if converged:
                print(f"    Converged in {len(params_traj)-1} iterations", flush=True)
            else:
                print(f"    Completed {len(params_traj)-1} iterations without full convergence", flush=True)
                
        except Exception as e:
            print(f"Error for T={T:.1f} K, V={V:.1f}: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            if prev_params is None:
                print(f"  No previous parameters available, skipping remaining temperatures", flush=True)
                break
            else:
                print(f"  Using previous temperature's parameters and continuing", flush=True)
        
        temp_elapsed = time.perf_counter() - temp_start
        print(f"  Finished T={T:.1f} K in {temp_elapsed/60:.2f} min", flush=True)
        
        # Add to results
        results.append((T, V, prev_params))
    
    elapsed = time.perf_counter() - start_time
    print(f"Volume {V:.1f} completed in {elapsed/60:.2f} min", flush=True)
    
    return (V, results)

# Add to utils_scph.py or create a new utility file
import multiprocessing
from functools import partial
from hiphive.utilities import prepare_structures

def prepare_structure_parallel(structure, atoms_ideal, calc, check_permutation=False):
    """Process a single structure for parallelization."""
    try:
        # This function calculates forces on a single structure
        return prepare_structures([structure], atoms_ideal, calc, check_permutation)[0]
    except Exception as e:
        print(f"Error in structure preparation: {e}")
        return None

def prepare_structures_parallel(structures, atoms_ideal, calc, check_permutation=False, n_jobs=None):
    """Parallel version of prepare_structures."""
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count() // 2  # Use half available cores by default
    
    # Create a partial function with fixed arguments
    worker_func = partial(prepare_structure_parallel, 
                         atoms_ideal=atoms_ideal, 
                         calc=calc,
                         check_permutation=check_permutation)
    
    # Use multiprocessing pool to parallelize
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(worker_func, structures)
    
    # Filter out None results (from errors)
    return [r for r in results if r is not None]


    # %load /Users/osode/opt/miniconda3/envs/phonopy/lib/python3.12/site-packages/hiphive/self_consistent_phonons/self_consistent_harmonic_model.py
import numpy as np
from hiphive.force_constant_model import ForceConstantModel
from hiphive import StructureContainer
from hiphive.utilities import prepare_structures
from hiphive.structure_generation import generate_rattled_structures, \
                                         generate_phonon_rattled_structures
from trainstation import Optimizer


def self_consistent_harmonic_model(atoms_ideal, calc, cs, T, alpha,
                                   n_iterations, n_structures,
                                   QM_statistics=False, parameters_start=None,
                                   imag_freq_factor=1.0, fit_kwargs={}):
    """
    Constructs a set of self-consistent second-order force constants
    that provides the closest match to the potential energy surface at
    a the specified temperature.

    Parameters
    ----------
    atoms_ideal : ase.Atoms
        ideal structure
    calc : ASE calculator object
        `calculator
        <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_
        to be used as reference potential
    cs : ClusterSpace
        clusterspace onto which to project the reference potential
    T : float
        temperature in K
    alpha : float
        stepsize in optimization algorithm
    n_iterations : int
        number of iterations in poor mans
    n_structures : int
        number of structures to use when fitting
    QM_statistics : bool
        if the amplitude of the quantum harmonic oscillator shoule be used
        instead of the classical amplitude in the phonon rattler
    parameters_start : numpy.ndarray
        parameters from which to start the optimization
    image_freq_factor: float
        If a squared frequency, w2, is negative then it is set to
        w2 = imag_freq_factor * np.abs(w2)
    fit_kwargs : dict
        kwargs to be used in the fitting process (via Optimizer)

    Returns
    -------
    list(numpy.ndarray)
        sequence of parameter vectors generated while iterating to
        self-consistency
    """

    if not 0 < alpha <= 1:
        raise ValueError('alpha must be between 0.0 and 1.0')

    if max(cs.cutoffs.orders) != 2:
        raise ValueError('ClusterSpace must be second order')

    # initialize things
    sc = StructureContainer(cs)
    fcm = ForceConstantModel(atoms_ideal, cs)

    # generate initial model
    if parameters_start is None:
        print('Creating initial model')
        rattled_structures = generate_rattled_structures(atoms_ideal, n_structures, 0.03)
        rattled_structures = prepare_structures(rattled_structures, atoms_ideal, calc, False)
        for structure in rattled_structures:
            sc.add_structure(structure)
        opt = Optimizer(sc.get_fit_data(), train_size=1.0, **fit_kwargs)
        opt.train()
        parameters_start = opt.parameters
        sc.delete_all_structures()

    # run poor mans self consistent
    parameters_old = parameters_start.copy()
    parameters_traj = [parameters_old]

    for i in range(n_iterations):
        # generate structures with old model
        print('Iteration {}'.format(i))
        fcm.parameters = parameters_old
        fc2 = fcm.get_force_constants().get_fc_array(order=2, format='ase')
        phonon_rattled_structures = generate_phonon_rattled_structures(
            atoms_ideal, fc2, n_structures, T, QM_statistics=QM_statistics,
            imag_freq_factor=imag_freq_factor)
        phonon_rattled_structures = prepare_structures(
            phonon_rattled_structures, atoms_ideal, calc, False)

        # fit new model
        for structure in phonon_rattled_structures:
            sc.add_structure(structure)
        opt = Optimizer(sc.get_fit_data(), train_size=1.0, **fit_kwargs)
        opt.train()
        sc.delete_all_structures()

        # update parameters
        parameters_new = alpha * opt.parameters + (1-alpha) * parameters_old

        # print iteration summary
        disps = [atoms.get_array('displacements') for atoms in phonon_rattled_structures]
        disp_ave = np.mean(np.abs(disps))
        disp_max = np.max(np.abs(disps))
        x_new_norm = np.linalg.norm(parameters_new)
        delta_x_norm = np.linalg.norm(parameters_old-parameters_new)
        print('    |x_new| = {:.5f}, |delta x| = {:.8f}, disp_ave = {:.5f}, disp_max = {:.5f}, '
              'rmse = {:.5f}'.format(x_new_norm, delta_x_norm, disp_ave, disp_max, opt.rmse_train))
        parameters_traj.append(parameters_new)
        parameters_old = parameters_new

    return parameters_traj

def self_consistent_harmonic_model_enhanced(
    atoms_ideal, calc, cs, T, alpha,
    n_iterations, n_structures,
    QM_statistics=False, parameters_start=None,
    imag_freq_factor=1.0, fit_kwargs={},
    # --- Enhanced options ---
    n_jobs=None,                 # Parallel structure processing
    verbose=1,                   # 0=silent, 1=summary, 2=details, 3=debug
    conv_rel_dx=1e-3,            # Relative parameter change threshold
    conv_rmse=1e-4,              # RMSE improvement threshold
    conv_patience=3,             # Required consecutive convergent iterations
    max_disp=None,               # Maximum displacement cap (Å)
    return_metrics=False,        # Return per-iteration metrics for analysis
    log_fn=print,                # Custom logging function
    # --- Advanced options ---
    filter_params_by_order=None, # Only update params of specific order
    T_min_effective=1.0,         # Minimum effective temperature for sampling
    beta_blend=0.3,              # Parameter blending for smoother updates
    random_seed=None             # Random seed for reproducibility
):
    """
    Enhanced parallel self-consistent harmonic model with convergence criteria
    and detailed metrics tracking.
    
    Parameters
    ----------
    atoms_ideal : ase.Atoms
        Ideal structure
    calc : ASE calculator object
        Calculator for reference potential
    cs : ClusterSpace
        Cluster space onto which to project the potential
    T : float
        Temperature in K
    alpha : float
        Parameter mixing ratio (0-1)
    n_iterations : int
        Maximum number of iterations
    n_structures : int
        Number of structures to use when fitting
    QM_statistics : bool
        Use quantum statistics for displacement amplitudes
    parameters_start : numpy.ndarray
        Initial parameters for optimization
    imag_freq_factor : float
        Handling of imaginary frequencies
    fit_kwargs : dict
        Parameters for Optimizer
    n_jobs : int
        Number of parallel processes for structure preparation
    verbose : int
        Verbosity level (0-3)
    conv_rel_dx : float
        Convergence threshold for relative parameter change
    conv_rmse : float
        Convergence threshold for RMSE improvement
    conv_patience : int
        Required consecutive iterations meeting convergence criteria
    max_disp : float
        Maximum displacement cap (Angstroms)
    return_metrics : bool
        Return detailed metrics dictionary
    log_fn : callable
        Function for logging output
    filter_params_by_order : int or list
        Only update parameters of specific order(s)
    T_min_effective : float
        Minimum effective temperature for sampling
    beta_blend : float
        Blending factor for parameter updates (0-1)
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    list(numpy.ndarray) or tuple
        Sequence of parameter vectors, with optional metrics if return_metrics=True
    """
    import numpy as np
    from math import isfinite
    import time
    from hiphive.force_constant_model import ForceConstantModel
    from hiphive import StructureContainer
    from hiphive.structure_generation import generate_rattled_structures, \
                                            generate_phonon_rattled_structures
    from trainstation import Optimizer

    # Validation
    if not 0 < alpha <= 1:
        raise ValueError('alpha must be between 0.0 and 1.0')
    
    if max(cs.cutoffs.orders) != 2 and not filter_params_by_order:
        raise ValueError('ClusterSpace must be second order or you must specify filter_params_by_order')
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Use effective temperature for sampling
    T_eff = max(float(T), T_min_effective)
    
    # Helper function for logging based on verbosity
    def _log(msg, level=1):
        if verbose >= level and log_fn is not None:
            log_fn(msg)

    # Initialize containers and model
    sc = StructureContainer(cs)
    fcm = ForceConstantModel(atoms_ideal, cs)
    
    # Determine number of parallel jobs
    if n_jobs is None:
        import multiprocessing
        n_jobs = max(1, multiprocessing.cpu_count() // 2)
    
    # Generate initial model if no starting parameters
    if parameters_start is None:
        _log('Creating initial model', 1)
        rattled_structures = generate_rattled_structures(atoms_ideal, n_structures, 0.03)
        
        # Use parallel structure preparation
        if n_jobs > 1:
            rattled_structures = prepare_structures_parallel(
                rattled_structures, atoms_ideal, calc, False, n_jobs=n_jobs)
        else:
            rattled_structures = prepare_structures(
                rattled_structures, atoms_ideal, calc, False)
        
        try:
            for structure in rattled_structures:
                sc.add_structure(structure)
            opt = Optimizer(sc.get_fit_data(), train_size=1.0, **fit_kwargs)
            opt.train()
            parameters_start = opt.parameters
        finally:
            sc.delete_all_structures()

    # Prepare for SCPH iterations
    parameters_old = parameters_start.copy()
    parameters_traj = [parameters_old]
    metrics = []
    
    # Track convergence
    prev_rmse = None
    converged = False
    patience_counter = 0
    
    # Determine parameter filter indices if needed
    filter_indices = None
    if filter_params_by_order is not None:
        filter_orders = filter_params_by_order
        if not isinstance(filter_orders, (list, tuple)):
            filter_orders = [filter_orders]
        
        _log(f"Filtering parameters to update only orders: {filter_orders}", 2)
        filter_indices = []
        for order in filter_orders:
            for i, orbit in enumerate(cs.orbit_list):
                if orbit.order == order:
                    filter_indices.append(i)
    
    # Add timing info to metrics
    total_start_time = time.perf_counter()

    # Main SCPH loop
    for i in range(n_iterations):
        iter_start = time.perf_counter()
        _log(f'Iteration {i}', 1)
        
        # Generate structures with current parameters
        fcm.parameters = parameters_old
        fc2 = fcm.get_force_constants().get_fc_array(order=2, format='ase')
        
        # Generate phonon-rattled structures
        gen_start = time.perf_counter()
        phonon_rattled_structures = generate_phonon_rattled_structures(
            atoms_ideal, fc2, n_structures, T_eff, 
            QM_statistics=QM_statistics, 
            imag_freq_factor=imag_freq_factor)
        gen_time = time.perf_counter() - gen_start
        
        # Apply displacement cap if specified
        if max_disp is not None:
            for s in phonon_rattled_structures:
                disps = s.get_array('displacements')
                disp_norms = np.linalg.norm(disps, axis=1)
                max_norm = disp_norms.max()
                if max_norm > max_disp:
                    scale = max_disp / max_norm
                    s.set_array('displacements', disps * scale)
                    s.set_positions(atoms_ideal.get_positions() + disps * scale)
        
        # Prepare structures (calculate forces) - parallel version
        prep_start = time.perf_counter()
        if n_jobs > 1:
            phonon_rattled_structures = prepare_structures_parallel(
                phonon_rattled_structures, atoms_ideal, calc, False, n_jobs=n_jobs)
        else:
            phonon_rattled_structures = prepare_structures(
                phonon_rattled_structures, atoms_ideal, calc, False)
        prep_time = time.perf_counter() - prep_start
        
        # Skip iteration if no valid structures (error handling)
        if not phonon_rattled_structures:
            _log("Warning: No valid structures generated. Skipping iteration.", 1)
            continue
            
        # Fit new model
        fit_start = time.perf_counter()
        try:
            for structure in phonon_rattled_structures:
                sc.add_structure(structure)
            
            # Training
            opt = Optimizer(sc.get_fit_data(), train_size=1.0, **fit_kwargs)
            opt.train()
            
            # Get parameters from optimizer
            opt_parameters = opt.parameters
            
            # Apply mixing (optionally with filtering)
            if filter_indices is not None:
                parameters_mixed = parameters_old.copy()
                parameters_mixed[filter_indices] = (alpha * opt_parameters[filter_indices] + 
                                                  (1-alpha) * parameters_old[filter_indices])
            else:
                parameters_mixed = alpha * opt_parameters + (1-alpha) * parameters_old
                
            # Apply additional blending for smoother updates
            if beta_blend < 1.0 and i > 0:
                parameters_new = (1.0 - beta_blend) * parameters_old + beta_blend * parameters_mixed
            else:
                parameters_new = parameters_mixed
        
        finally:
            sc.delete_all_structures()
        fit_time = time.perf_counter() - fit_start
            
        # Calculate metrics
        disps = [atoms.get_array('displacements') for atoms in phonon_rattled_structures]
        disp_ave = np.mean(np.abs(disps)) if disps else 0.0
        disp_max = np.max(np.abs(disps)) if disps else 0.0
        
        x_new_norm = np.linalg.norm(parameters_new)
        delta_x_norm = np.linalg.norm(parameters_old-parameters_new)
        rel_delta_x = delta_x_norm / max(x_new_norm, 1e-12)
        
        rmse = opt.rmse_train
        delta_rmse = abs(rmse - prev_rmse) if prev_rmse is not None else None
        prev_rmse = rmse
        
        # Store metrics with timing information
        iter_end = time.perf_counter()
        iter_time = iter_end - iter_start
        
        iter_metrics = {
            'iteration': i,
            'x_norm': float(x_new_norm),
            'delta_x': float(delta_x_norm),
            'rel_delta_x': float(rel_delta_x),
            'disp_ave': float(disp_ave),
            'disp_max': float(disp_max),
            'rmse': float(rmse),
            'delta_rmse': None if delta_rmse is None else float(delta_rmse),
            'time_total': float(iter_time),
            'time_generate': float(gen_time),
            'time_prepare': float(prep_time),  # Parallelized part
            'time_fit': float(fit_time),
            'time_other': float(iter_time - gen_time - prep_time - fit_time)
        }
        metrics.append(iter_metrics)
        
        # Check for convergence
        is_converged_dx = rel_delta_x < conv_rel_dx
        is_converged_rmse = delta_rmse is not None and delta_rmse < conv_rmse
        
        if is_converged_dx and is_converged_rmse:
            patience_counter += 1
            iter_metrics['converged'] = True
            if patience_counter >= conv_patience:
                converged = True
                _log(f"Converged after {i+1} iterations (patience={conv_patience})", 1)
                _log(f"  rel_dx={rel_delta_x:.3e}, delta_rmse={delta_rmse:.3e}", 2)
                break
        else:
            patience_counter = 0
            iter_metrics['converged'] = False
            
        # Detailed logging
        if verbose >= 1:
            _log('    |x_new| = {:.5f}, |delta x| = {:.8f} (rel={:.2e}), disp_ave = {:.5f}, '
                'disp_max = {:.5f}, rmse = {:.5f}{}'.format(
                x_new_norm, delta_x_norm, rel_delta_x, disp_ave, disp_max, rmse,
                "" if delta_rmse is None else f", delta_rmse = {delta_rmse:.5f}"
            ), 1)

            _log(f"    Time: {iter_time:.2f}s total [gen:{gen_time:.2f}s, "
                 f"prep:{prep_time:.2f}s ({n_structures}/{prep_time:.1f}={n_structures/max(1e-6,prep_time):.1f} struct/s), "
                 f"fit:{fit_time:.2f}s]", 2)
            
            if verbose >= 3:
                # Debug: parameter histograms or other detailed info
                _log(f"    Parameter stats: min={parameters_new.min():.3e}, "
                    f"max={parameters_new.max():.3e}, "
                    f"mean={parameters_new.mean():.3e}", 3)
        
        # Store parameters and prepare for next iteration
        parameters_traj.append(parameters_new)
        parameters_old = parameters_new

    # Report if didn't converge
    if not converged and verbose >= 1:
        _log(f"Reached maximum iterations ({n_iterations}) without convergence", 1)
    
    # Final timing information
    total_time = time.perf_counter() - total_start_time
    _log(f"SCPH completed in {total_time:.2f}s ({total_time/60:.2f}min)", 1)

    if len(metrics) > 0:
        avg_iter_time = sum(m['time_total'] for m in metrics) / len(metrics)
        avg_prep_time = sum(m['time_prepare'] for m in metrics) / len(metrics)
        _log(f"Average iteration time: {avg_iter_time:.2f}s (prep: {avg_prep_time:.2f}s, {100*avg_prep_time/max(1e-6,avg_iter_time):.1f}%)", 1)
    

    # Return results
    if return_metrics:
        return parameters_traj, metrics, converged
    else:
        return parameters_traj

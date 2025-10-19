
import os, json, hashlib, pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from pymatgen.core import Lattice, Structure as PMGStructure

from utils_hiphive import fc2_from_hiphive_config
#from utils_hiphive import (
#     fc2_from_hiphive_model,
#     fc2_from_hiphive_config,
#     fc2_from_finite_displacements )

from utils_co2 import ( CO2TwoBodyRC_IMAGES )

# Light logging via print
def log(msg: str):
    print(f"[QHA] {msg}")

# ---------------------------
# Structure standardization & hashing
# ---------------------------
def spglib_standardize_pmg_structure(pmg_struct, symprec: float = 1e-5):
    """
    Return a standardized conventional cell using spglib via pymatgen's SpacegroupAnalyzer.
    This reduces duplicates for hashing.
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(pmg_struct, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        return conv
    except Exception as e:
        log(f"Standardization failed, using original structure. Reason: {e}")
        return pmg_struct

def round_lattice_angles(a):
    # a, b, c in Angstrom; alpha, beta, gamma in degrees
    # Rounding: abc 0.01; angles 0.1
    return (round(a[0], 2), round(a[1], 2), round(a[2], 2),
            round(a[3], 1), round(a[4], 1), round(a[5], 1))

def frac_array_with_precision(frac, prec=3):
    return np.round(np.asarray(frac, float), prec)

def structure_fingerprint(pmg_struct, symprec: float, supercell: Tuple[int,int,int], fd_disp: float,
                          include_mesh: Optional[Tuple[int,int,int]]=None) -> Dict[str, Any]:
    """
    Build a canonical dict capturing requested invariants for hashing.
    - Standardized conventional cell
    - Space group number
    - Lattice (a,b,c,alpha,beta,gamma) rounded as specified
    - Species order and fractional coordinates rounded to 0.001
    - Supercell size and FD displacement amplitude
    - If provided, include mesh and symprec (for QHA dataset identity)
    """
    std = spglib_standardize_pmg_structure(pmg_struct, symprec)
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(std, symprec=symprec)
        sgnum = sga.get_space_group_number()
    except Exception:
        sgnum = None

    lat = std.lattice
    a, b, c = lat.a, lat.b, lat.c
    alpha, beta, gamma = lat.alpha, lat.beta, lat.gamma
    lat_tuple = round_lattice_angles((a,b,c,alpha,beta,gamma))

    species = [sp.symbol for sp in std.species]
    frac = frac_array_with_precision(std.frac_coords, 3).tolist()

    fp = {
        "space_group_number": sgnum,
        "lattice_abc_abg": lat_tuple,
        "species": species,
        "frac_coords": frac,
        "supercell": tuple(int(x) for x in supercell),
        "fd_disp": float(fd_disp),
        "symprec": float(symprec),
    }
    if include_mesh is not None:
        fp["mesh"] = tuple(int(x) for x in include_mesh)
    return fp

def hash_fingerprint(fp: Dict[str, Any]) -> str:
    s = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:10]

# ---------------------------
# Storage manager
# ---------------------------
@dataclass
class StorageConfig:
    base_dir: str = "qha_results"

class QHAStorage:
    """
    Manages data storage for a QHA workflow for a specific crystal phase.
    """
    def __init__(self, space_group: str, config: StorageConfig = StorageConfig()):
        self.space_group = str(space_group).lower()
        self.base_dir = Path(config.base_dir) / self.space_group
        self.hessian_dir = self.base_dir / "hessians"
        self.thermal_dir = self.base_dir / "thermal"
        self.results_dir = self.base_dir / "results"
        self.plots_dir = self.base_dir / "plots"
        self.meta_dir = self.base_dir / "meta"
        self._setup()

    def _setup(self):
        for d in [self.base_dir, self.hessian_dir, self.thermal_dir, self.results_dir, self.plots_dir, self.meta_dir]:
            d.mkdir(parents=True, exist_ok=True)
        log(f"Storage ready at: {self.base_dir.resolve()}")

    # ---- Paths keyed by structure hash ----
    def paths_for_hash(self, s_hash: str) -> Dict[str, Path]:
        return {
            "fc2": self.hessian_dir / f"fc2_{s_hash}.npz",
            "thermal": self.thermal_dir / f"thermal_{s_hash}.npz",
            "manifest": self.meta_dir / f"manifest_{s_hash}.json",
        }

    def get_results_path(self, pressure_gpa: float) -> Path:
        return self.results_dir / f"qha_results_{pressure_gpa:.1f}gpa.pkl"

    # ---- Save / load artifacts ----
    def save_fc2(self, s_hash: str, fc2_array: np.ndarray):
        p = self.paths_for_hash(s_hash)["fc2"]
        np.savez_compressed(p, fc2=fc2_array)
        log(f"Saved FC2 → {p}")

    def load_fc2(self, s_hash: str) -> Optional[np.ndarray]:
        p = self.paths_for_hash(s_hash)["fc2"]
        if p.exists():
            data = np.load(p, allow_pickle=False)
            return data["fc2"]
        return None

    def save_thermal(self, s_hash: str, temperatures: np.ndarray, free_energy_kJmol: np.ndarray,
                     cv_JmolK: np.ndarray, s_JmolK: np.ndarray, electronic_energy_eV: float):
        p = self.paths_for_hash(s_hash)["thermal"]
        np.savez_compressed(
            p,
            temperatures=np.asarray(temperatures, float),
            free_energy_kJmol=np.asarray(free_energy_kJmol, float),
            cv_JmolK=np.asarray(cv_JmolK, float),
            s_JmolK=np.asarray(s_JmolK, float),
            E_elec_eV=float(electronic_energy_eV),
        )
        log(f"Saved thermal arrays → {p}")

    def load_thermal(self, s_hash: str):
        p = self.paths_for_hash(s_hash)["thermal"]
        if p.exists():
            data = np.load(p, allow_pickle=False)
            return {
                "temperatures": data["temperatures"],
                "free_energy_kJmol": data["free_energy_kJmol"],
                "cv_JmolK": data["cv_JmolK"],
                "s_JmolK": data["s_JmolK"],
                "E_elec_eV": float(data["E_elec_eV"]),
            }
        return None

    def save_manifest(self, s_hash: str, manifest: Dict[str, Any]):
        p = self.paths_for_hash(s_hash)["manifest"]
        with open(p, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        log(f"Saved manifest → {p}")

    def load_manifest(self, s_hash: str) -> Optional[Dict[str, Any]]:
        p = self.paths_for_hash(s_hash)["manifest"]
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
        return None

    def save_qha_results(self, results_dict: Dict[str, Any], pressure_gpa: float):
        path = self.get_results_path(pressure_gpa)
        with open(path, "wb") as f:
            pickle.dump(results_dict, f)
        log(f"Saved QHA results → {path}")

    def load_qha_results(self, pressure_gpa: float) -> Optional[Dict[str, Any]]:
        path = self.get_results_path(pressure_gpa)
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None


def scale_to_volume(atoms, target_V):
    a = atoms.copy()
    s = (target_V / a.get_volume())**(1/3)
    a.set_cell(a.cell * s, scale_atoms=True)
    return a

# ---------------------------
# Per-volume runner (hash-first cache; manifest saved for provenance)
# ---------------------------
def run_volume_point(
    pmg_structure,
    ase_atoms,
    storage: QHAStorage,
    symprec: float,
    supercell: Tuple[int,int,int],
    fd_disp: float,
    mesh: Tuple[int,int,int],
    temps: Tuple[int,int,int],
    eos: str = "vinet",
    use_hiphive: bool = False,
    asr_enforce: bool = True,
    relax_internals: bool = True,
    fmax: float = 1e-3,
    steps: int = 200,
    force: bool = False,
    rng_seed: Optional[int] = 7,
    hiphive_fcp: Any = None,
    hiphive_config: Dict[str, Any] = None,
):
    """
    Compute/cached for a single volume:
      - FC2 (npz)
      - Thermal properties arrays (npz)
      - Manifest (json; not used to gate cache)
    Returns a dict for this volume.
    """
    if rng_seed is not None:
        np.random.seed(int(rng_seed))

    fp = structure_fingerprint(
        pmg_structure, symprec=symprec, supercell=supercell, fd_disp=fd_disp, include_mesh=mesh
    )
    fp["asr_enforce"] = bool(asr_enforce)
    fp["eos"] = eos
    tmin, tmax, tstep = temps
    fp["temp_min"], fp["temp_max"], fp["temp_step"] = int(tmin), int(tmax), int(tstep)

    s_hash = hash_fingerprint(fp)

    if (not force):
        thermal = storage.load_thermal(s_hash)
        if thermal is not None:
            return {"hash": s_hash, "thermal": thermal, "manifest": storage.load_manifest(s_hash)}

    if hiphive_config is None:
        hiphive_config = {
            'use_hiphive': use_hiphive,
            'cutoffs': [5.45, 4.00, 3.00],
            'n_structures': 10,
            'scph': {'enabled': False}
        }

    # Recompute path
    from copy import deepcopy
    from ase.optimize import BFGS
    atoms = ase_atoms.copy()
    if relax_internals:
        atoms.calc = deepcopy(ase_atoms.calc)
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    

    # Static energy (eV)
    E_elec = float(atoms.get_potential_energy())

    # Build Phonopy object and FC2 (either HIPHIVE or finite displacement)
    import phonopy
    from phonopy import Phonopy
    u = phonopy.structure.atoms.PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions(),
    )
    ph = Phonopy(u, supercell_matrix=supercell, primitive_matrix=None, symprec=symprec)

    if use_hiphive:
        #fc2 = fc2_from_hiphive_model(atoms, super=supercell, RATTLE_TYPE="mc", rattle_std=1e-3, d_min=1.00, shash=s_hash)

        atoms.set_constraint()
        # Use the modern dictionary-based interface
        # Ensure supercell_size is set in config
        hiphive_config['supercell_size'] = supercell
        
        fc2 = fc2_from_hiphive_config(atoms, hiphive_config, s_hash)
    else:
        #fc2_from_finite_displacements
        ph.generate_displacements(distance=fd_disp)
        forces = []
        from ase import Atoms
        for sc in ph.supercells_with_displacements:
            sc_ase = Atoms(numbers=sc.numbers, cell=sc.cell, pbc=True, scaled_positions=sc.scaled_positions)
            sc_ase.calc = atoms.calc  # same calculator
            forces.append(sc_ase.get_forces())
        ph.forces = forces
        ph.produce_force_constants()
        fc2 = ph.force_constants

    ph.force_constants = fc2
    if asr_enforce:
        ph.symmetrize_force_constants()

    # Mesh + thermal properties
    ph.run_mesh(mesh, is_gamma_center=False)
    tmin, tmax, tstep = temps
    ph.run_thermal_properties(t_min=tmin, t_max=tmax, t_step=tstep)
    tp = ph.get_thermal_properties_dict()

    if True:
        # Path: Γ–X–M–Γ–R (simple cubic-like)
        def get_path(q_start, q_stop, N):
            return np.array([q_start + (q_stop-q_start) * i / (N - 1) for i in range(N)])
        
        Nq = 21
        G = np.array([0, 0, 0]); X = np.array([0.5, 0, 0]); M = np.array([0.5, 0.5, 0]); R = np.array([0.5, 0.5, 0.5])
        path = [get_path(G,X,Nq), get_path(X,M,Nq), get_path(M,G,Nq), get_path(G,R,Nq)]
    
        ph.run_band_structure(path)
        qnorms = ph.band_structure.distances
        freqs = ph.band_structure.frequencies
        
        kpts = [0.0, qnorms[0][-1], qnorms[1][-1], qnorms[2][-1], qnorms[3][-1]]
        kpts_labels = ['$\\Gamma$', 'X', 'M', '$\\Gamma$', 'R']
    
        #print(V)
        import matplotlib.pyplot as plt
        plt.figure()
        for q, freq in zip(qnorms, freqs):
            plt.plot(q, freq, linewidth=2.0)
        for x in kpts[1:-1]:
            plt.axvline(x=x, linewidth=0.9)
        plt.xlabel('Wave vector')
        plt.ylabel('Frequency (THz)')
        plt.xticks(kpts, kpts_labels)
        plt.xlim([0.0, qnorms[-1][-1]])
        plt.ylim([0.0, 5.0])
        plt.tight_layout()

    # Save artifacts
    storage.save_fc2(s_hash, fc2)
    storage.save_thermal(
        s_hash,
        temperatures=np.asarray(tp["temperatures"], float),
        free_energy_kJmol=np.asarray(tp["free_energy"], float),
        cv_JmolK=np.asarray(tp["heat_capacity"], float),
        s_JmolK=np.asarray(tp["entropy"], float),
        electronic_energy_eV=E_elec,
    )
    storage.save_manifest(s_hash, fp)

    # Enhanced return dictionary with hiphive info
    result = {
        "hash": s_hash,
        "thermal": storage.load_thermal(s_hash),
        "manifest": fp,
        "relaxed_atoms": atoms,
        "V": float(ase_atoms.get_volume()),
        "fc2": fc2,
        "hiphive_config": hiphive_config,  # Store config used
        "hiphive_fcp_path": f"{s_hash}.fcp" if hiphive_config.get('save_fcp', True) else None,
    }

    # Add SCPH results if enabled
    if hiphive_config.get('scph', {}).get('enabled', False):
        result["scph_results"] = run_scph_for_volume(atoms, hiphive_config, s_hash)

    return result


def compute_volume_phonon_properties(
    volume: float,
    ase_atoms,
    space_group: str,
    temps: Tuple[int,int,int],
    eos: str = "vinet",
    use_hiphive: bool = False,
    opt_config: Dict[str, Any] = None,
    phonon_config: Dict[str, Any] = None,
    hiphive_config: Dict[str, Any] = None,
    force = False,       # set True to recompute ignoring cache
    rng_seed = 11,
    verbose: int = 0,
    log_fn = None,
):
    """
    Compute/cached for a single volume:
      - FC2 (npz)
      - Thermal properties arrays (npz)
      - Manifest (json; not used to gate cache)
    Returns a dict for this volume.
    """

    if rng_seed is not None:
        np.random.seed(int(rng_seed))


    # Helper function for logging based on verbosity
    def _log(msg, level=1):
        if verbose >= level and log_fn is not None:
            log_fn(msg)

    # Scale to target volume
    geom = scale_to_volume(ase_atoms, volume)
    geom.calc = ase_atoms.calc

    # Build PMG structure for fingerprinting
    pmg = PMGStructure(Lattice(geom.cell.array), geom.get_chemical_symbols(), geom.get_positions(),
                        coords_are_cartesian=True, to_unit_cell=False)


    # Parse phonon config
    if phonon_config is None:
        phonon_config = {
            'supercell': (2,2,2),
            'fd_disp': 1e-3,
            'symprec': 1e-5,
            'mesh': (11,11,11),
            'asr_enforce': True,
        }
    supercell = phonon_config.get('supercell', (2,2,2))
    fd_disp = phonon_config.get('fd_disp', 1e-3)
    symprec = phonon_config.get('symprec', 1e-5)
    mesh = phonon_config.get('mesh', (11,11,11))
    asr_enforce = phonon_config.get('asr_enforce', True)

    # Parse optimization config
    if opt_config is None:
        opt_config = {
            'relax_internals': True,
            'fmax': 1e-3,
            'steps': 200,
        }
    relax_internals = opt_config.get('relax_internals', True)
    fmax = opt_config.get('fmax', 1e-3)
    steps = opt_config.get('steps', 200)
    

    fp = structure_fingerprint(
        pmg, symprec=symprec, supercell=supercell, fd_disp=fd_disp, include_mesh=mesh
    )
    fp["asr_enforce"] = bool(asr_enforce)
    fp["eos"] = eos
    tmin, tmax, tstep = temps
    fp["temp_min"], fp["temp_max"], fp["temp_step"] = int(tmin), int(tmax), int(tstep)

    s_hash = hash_fingerprint(fp)

    storage = QHAStorage(space_group)

    if (not force):
        thermal = storage.load_thermal(s_hash)
        if thermal is not None:
            return volume, {"hash": s_hash, 
                            "thermal": thermal, 
                            "manifest": storage.load_manifest(s_hash),
                            "volume": volume,
                            "opt_config": opt_config,
                            "phonon_config": phonon_config,
                            "hiphive_config": hiphive_config}
        
    if hiphive_config is None:
        hiphive_config = {
            'use_hiphive': use_hiphive,
            'cutoffs': [5.45, 4.00, 3.00],
            'n_structures': 10,
            'scph': {'enabled': False}
        }

    # Recompute path
    from copy import deepcopy
    from ase.optimize import BFGS
    atoms = geom.copy()
    if relax_internals:
        atoms.calc = deepcopy(geom.calc)
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    
    # Static energy (eV)
    E_elec = float(atoms.get_potential_energy())

    # Build Phonopy object and FC2 (either HIPHIVE or finite displacement)
    import phonopy
    from phonopy import Phonopy
    u = phonopy.structure.atoms.PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions(),
    )
    ph = Phonopy(u, supercell_matrix=supercell, primitive_matrix=None, symprec=symprec)

    if use_hiphive:

        atoms.set_constraint()
        # Use the modern dictionary-based interface
        # Ensure supercell_size is set in config
        hiphive_config['supercell_size'] = supercell
        
        fc2 = fc2_from_hiphive_config(atoms, hiphive_config, s_hash)
    else:
        #fc2_from_finite_displacements
        ph.generate_displacements(distance=fd_disp)
        forces = []
        from ase import Atoms
        for sc in ph.supercells_with_displacements:
            sc_ase = Atoms(numbers=sc.numbers, cell=sc.cell, pbc=True, scaled_positions=sc.scaled_positions)
            sc_ase.calc = atoms.calc  # same calculator
            forces.append(sc_ase.get_forces())
        ph.forces = forces
        ph.produce_force_constants()
        fc2 = ph.force_constants

    ph.force_constants = fc2
    if asr_enforce:
        ph.symmetrize_force_constants()

    # Mesh + thermal properties
    ph.run_mesh(mesh, is_gamma_center=False)
    tmin, tmax, tstep = temps
    ph.run_thermal_properties(t_min=tmin, t_max=tmax, t_step=tstep)
    tp = ph.get_thermal_properties_dict()

    if verbose > 2:
        # Path: Γ–X–M–Γ–R (simple cubic-like)
        def get_path(q_start, q_stop, N):
            return np.array([q_start + (q_stop-q_start) * i / (N - 1) for i in range(N)])
        
        Nq = 21
        G = np.array([0, 0, 0]); X = np.array([0.5, 0, 0]); M = np.array([0.5, 0.5, 0]); R = np.array([0.5, 0.5, 0.5])
        path = [get_path(G,X,Nq), get_path(X,M,Nq), get_path(M,G,Nq), get_path(G,R,Nq)]
    
        ph.run_band_structure(path)
        qnorms = ph.band_structure.distances
        freqs = ph.band_structure.frequencies
        
        kpts = [0.0, qnorms[0][-1], qnorms[1][-1], qnorms[2][-1], qnorms[3][-1]]
        kpts_labels = ['$\\Gamma$', 'X', 'M', '$\\Gamma$', 'R']

        import matplotlib.pyplot as plt
        plt.figure()
        for q, freq in zip(qnorms, freqs):
            plt.plot(q, freq, linewidth=2.0)
        for x in kpts[1:-1]:
            plt.axvline(x=x, linewidth=0.9)
        plt.xlabel('Wave vector')
        plt.ylabel('Frequency (THz)')
        plt.xticks(kpts, kpts_labels)
        plt.xlim([0.0, qnorms[-1][-1]])
        plt.ylim([0.0, 5.0])
        plt.tight_layout()

    # Save artifacts
    storage.save_fc2(s_hash, fc2)
    storage.save_thermal(
        s_hash,
        temperatures=np.asarray(tp["temperatures"], float),
        free_energy_kJmol=np.asarray(tp["free_energy"], float),
        cv_JmolK=np.asarray(tp["heat_capacity"], float),
        s_JmolK=np.asarray(tp["entropy"], float),
        electronic_energy_eV=E_elec,
    )
    storage.save_manifest(s_hash, fp)

    # Enhanced return dictionary with hiphive info
    result = {
        "hash": s_hash,
        "thermal": storage.load_thermal(s_hash),
        "manifest": fp,
        "relaxed_atoms": atoms,
        "volume": volume,
        "fc2": fc2,
        "opt_config": opt_config,
        "phonon_config": phonon_config,
        "hiphive_config": hiphive_config,  # Store config used
        "hiphive_fcp_path": f"{s_hash}.fcp" if hiphive_config.get('save_fcp', True) else None,
    }

    # Add SCPH results if enabled
    if hiphive_config.get('scph', {}).get('enabled', False):
        result["scph_results"] = run_scph_for_volume(atoms, hiphive_config, s_hash)

    return volume, result

def run_scph_for_volume(atoms, hiphive_config, volume_hash):
    """
    Run SCPH calculations for a given volume point.
    
    Parameters
    ----------
    atoms : ase.Atoms
        Relaxed structure at this volume
    hiphive_config : dict
        Configuration dictionary with SCPH parameters
    volume_hash : str
        Hash identifier for this volume
        
    Returns
    -------
    dict : SCPH results including trajectories and final FCPs
    """
    import os
    import numpy as np
    from copy import deepcopy
    from hiphive import ClusterSpace, ForceConstantPotential
    from hiphive.self_consistent_phonons import self_consistent_harmonic_model
    from hiphive.calculators import ForceConstantCalculator
    
    scph_config = hiphive_config['scph']
    
    # Setup directories
    for dir_name in hiphive_config['output_dirs'].values():
        os.makedirs(dir_name, exist_ok=True)
    
    # Prepare structures
    psym = deepcopy(atoms)
    psym.set_constraint()
    
    supercell_size = hiphive_config.get('supercell_size', (2, 2, 2))
    supercell_sym = psym.repeat(supercell_size)
    
    # Setup calculator
    rc = hiphive_config['cutoffs'][0]  # Use first cutoff as rc
    from utils_co2 import CO2TwoBodyRC_IMAGES
    base_calc = CO2TwoBodyRC_IMAGES(rc=rc)
    supercell_sym.calc = base_calc
    
    # Load seed FCP
    fcp = ForceConstantPotential.read(f"{volume_hash}.fcp")
    
    # Setup harmonic cluster space for SCPH
    cs = ClusterSpace(psym, scph_config['harmonic_cutoffs'])
    
    results = {
        'temperatures': [],
        'trajectories': {},
        'fcp_paths': {},
        'convergence_info': {}
    }
    
    prev_params = None
    
    for T in scph_config['temperatures']:
        T_eff = max(float(T), 1.0)  # Avoid T=0K
        results['temperatures'].append(T_eff)
        
        try:
            parameters_traj = self_consistent_harmonic_model(
                supercell_sym,
                base_calc,
                cs,
                T_eff,
                alpha=scph_config['alpha'],
                n_iterations=scph_config['n_iterations'],
                n_structures=scph_config['n_structures'],
                imag_freq_factor=scph_config['imag_freq_factor'],
                parameters_start=prev_params,
                QM_statistics=scph_config['QM_statistics'],
                fit_kwargs=scph_config['fit_kwargs']
            )
            
            # Save results
            fcp_scph = ForceConstantPotential(cs, parameters_traj[-1])
            V = atoms.get_volume()
            
            fcp_path = f"{hiphive_config['output_dirs']['fcps']}scph_T{T_eff}_V{V:.1f}.fcp"
            traj_path = f"{hiphive_config['output_dirs']['scph_trajs']}scph_parameters_T{T_eff}_V{V:.1f}.txt"
            
            fcp_scph.write(fcp_path)
            np.savetxt(traj_path, np.array(parameters_traj))
            
            results['trajectories'][T_eff] = traj_path
            results['fcp_paths'][T_eff] = fcp_path
            results['convergence_info'][T_eff] = {
                'n_iterations': len(parameters_traj),
                'final_params_norm': np.linalg.norm(parameters_traj[-1]),
                'converged': len(parameters_traj) < scph_config['n_iterations']
            }
            
            prev_params = parameters_traj[-1]
            
        except Exception as e:
            print(f"SCPH failed at T={T_eff}: {e}")
            results['convergence_info'][T_eff] = {'error': str(e)}
            break
    
    return results

# ---------------------------
# QHA assembly (across volumes)
# ---------------------------
def assemble_qha_inputs(volume_points: List[Dict[str, Any]]):
    """
    Combine per-volume thermal packs into QHA-ready arrays.
    Returns: volumes (list placeholder), temperatures (1d), E_elec (nV,), Fvib (nT,nV), Cv (nT,nV), Svib (nT,nV)
    Assumes all thermal packs share the same temperature grid.
    """
    nV = len(volume_points)
    assert nV > 0
    temps = np.asarray(volume_points[0]["thermal"]["temperatures"], float)
    nT = temps.size

    E = np.zeros(nV, float)
    F = np.zeros((nT, nV), float)
    Cv = np.zeros((nT, nV), float)
    S = np.zeros((nT, nV), float)

    volumes = []
    for i, vpt in enumerate(volume_points):
        th = vpt["thermal"]
        E[i] = float(th["E_elec_eV"])
        F[:, i] = th["free_energy_kJmol"]
        Cv[:, i] = th["cv_JmolK"]
        S[:, i] = th["s_JmolK"]
        volumes.append(0.0)  # caller fills actual volumes
    return volumes, temps, E, F, Cv, S

# ---------------------------
# CSV helper
# ---------------------------
def qha_to_csv(out_csv_path: Path, volumes: List[float], temperatures: np.ndarray,
               E_elec_eV: np.ndarray, Fvib_kJmol: np.ndarray, Cv_JmolK: np.ndarray, Svib_JmolK: np.ndarray):
    """
    Flatten to a tidy CSV: columns: V, T, E_elec_eV, Fvib_kJmol, Cv_JmolK, Svib_JmolK
    """
    import pandas as pd
    rows = []
    for j, V in enumerate(volumes):
        for i, T in enumerate(temperatures):
            rows.append({
                "V_A3": float(V),
                "T_K": float(T),
                "E_elec_eV": float(E_elec_eV[j]),
                "Fvib_kJmol": float(Fvib_kJmol[i, j]),
                "Cv_JmolK": float(Cv_JmolK[i, j]),
                "Svib_JmolK": float(Svib_JmolK[i, j]),
            })
    df = pd.DataFrame(rows)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df

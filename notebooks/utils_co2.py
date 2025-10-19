
"""
utils_co2.py
----------------
Clean, documented helpers for CO₂ molecular crystals (ASE + pymatgen interop).

What's included
- Molecular grouping via minimum image convention (fractional): `wrap_coordinates_by_carbon_fractional`
- Periodic image helpers: `image_shifts_for_rc`, `halfspace`
- "Sticky" reference mapping utilities: `build_reference_mapping`, `coords_with_mapping`
- Energetics from explicit images (1B + 2B): `compute_energy_from_cell_rc_images`
- Analytical forces (from p1b_gradient/sapt_gradient) with explicit images: `compute_gradients_from_cell_rc_images`

Notes
- Functions accept a **pymatgen Structure** (`structure`), not ASE Atoms.
- Energies use your provided `co2_potential` (p1b/sapt); gradients use corresponding *_gradient calls.
- The explicit-image summations avoid double counting by summing the positive half-space only.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
import warnings

import numpy as np

from ase import units
from ase.calculators.calculator import Calculator, all_changes

from pymatgen.core import Lattice, Structure as PMGStructure

from co2_potential import (
    p1b,
    p1b_gradient,
    sapt,
    sapt_gradient,
)

# -----------------------------------------------------------------------------
# Molecular grouping
# -----------------------------------------------------------------------------

def wrap_coordinates_by_carbon_fractional(structure):
    """
    Group atoms into CO₂ molecules (nearest 2 O around each C) using fractional
    minimum-image convention (MIC). Oxygen positions are wrapped to the MIC with
    respect to the assigned carbon (in fractional space), then returned as
    Cartesian coordinates.

    Parameters
    ----------
    structure : pymatgen.core.Structure
        Periodic structure containing only C and O atoms for CO₂.

    Returns
    -------
    updated_cart : (N, 3) np.ndarray
        Cartesian coordinates with O atoms wrapped to the MIC around each C.
    mol_assign : List[int | None]
        Molecule id for each atom index (None if unassigned).
    molecules : List[List[int]]
        Triples [C_idx, O1_idx, O2_idx] with original atom indices.
    """
    frac = np.asarray(structure.frac_coords, dtype=float)
    species = structure.species

    carbon_idx = [i for i, sp in enumerate(species) if sp.symbol == "C"]
    oxygen_idx = [i for i, sp in enumerate(species) if sp.symbol == "O"]

    if len(oxygen_idx) < 2 * len(carbon_idx):
        warnings.warn(
            f"[wrap] Fewer oxygens ({len(oxygen_idx)}) than 2×carbons ({2*len(carbon_idx)}). "
            "Proceeding but grouping may be incomplete."
        )

    updated_frac = frac.copy()
    mol_assign: List[int | None] = [None] * len(species)
    molecules: List[List[int]] = []
    O_avail = set(oxygen_idx)

    def min_image_frac(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return fractional shift dF (wrapped into [-0.5,0.5)) and MIC distance in Å."""
        dF = p2 - p1
        dF -= np.round(dF)
        dC = structure.lattice.get_cartesian_coords(dF)
        return dF, float(np.linalg.norm(dC))

    mol_id = 0
    for ci in carbon_idx:
        cF = updated_frac[ci]
        # rank all available oxygens by MIC distance to this carbon
        cand = []
        for oi in O_avail:
            dF, dist = min_image_frac(cF, updated_frac[oi])
            cand.append((dist, oi, dF))
        if len(cand) < 2:
            warnings.warn("[wrap] Not enough O available to assign 2 per C; skipping remainder.")
            break
        cand.sort(key=lambda t: t[0])

        chosen = []
        for k in range(2):
            _, oi, dF = cand[k]
            updated_frac[oi] = cF + dF  # place O at MIC w.r.t. C
            chosen.append(oi)
            O_avail.remove(oi)

        mol_assign[ci] = mol_id
        for oi in chosen:
            mol_assign[oi] = mol_id
        molecules.append([ci] + chosen)
        mol_id += 1

    updated_cart = structure.lattice.get_cartesian_coords(updated_frac)
    return updated_cart, mol_assign, molecules


# -----------------------------------------------------------------------------
# Periodic image helpers
# -----------------------------------------------------------------------------

def image_shifts_for_rc(lattice, rc: float) -> List[Tuple[int, int, int]]:
    """
    Integer translation vectors (i, j, k) whose *anchor-to-anchor* separation
    can fall within a spherical cutoff `rc` (Å). Uses lattice a, b, c lengths.
    """
    if rc <= 0:
        return [(0, 0, 0)]
    na = int(np.ceil(rc / lattice.a)) if lattice.a > 1e-12 else 0
    nb = int(np.ceil(rc / lattice.b)) if lattice.b > 1e-12 else 0
    nc = int(np.ceil(rc / lattice.c)) if lattice.c > 1e-12 else 0
    return [(i, j, k)
            for i in range(-na, na + 1)
            for j in range(-nb, nb + 1)
            for k in range(-nc, nc + 1)]


def halfspace(T: Tuple[int, int, int]) -> bool:
    """
    Select positive half-space to avoid double counting (exclude 0,0,0):
    keep T if (k>0) or (k==0 and j>0) or (k==0 and j==0 and i>0).
    """
    i, j, k = T
    return (k > 0) or (k == 0 and j > 0) or (k == 0 and j == 0 and i > 0)


# -----------------------------------------------------------------------------
# Sticky CO₂ reference mapping (optional utilities)
# -----------------------------------------------------------------------------

def build_reference_mapping(structure, warn_long_co: float = 1.6):
    """
    Assign each carbon to two nearest oxygens, keeping track of fractional
    cell translations to construct a consistent reference mapping across PBCs.

    Returns
    -------
    groups : List[List[int]]
        Triples of indices (C, O1, O2).
    shifts : Dict[int, np.ndarray]
        For each oxygen index, the integer translation vector t such that
        O_frac - t is near the carbon in the reference mapping.
    """
    frac = np.asarray(structure.frac_coords, float)
    spp = structure.species
    carb = [i for i, sp in enumerate(spp) if sp.symbol == "C"]
    oxy = [i for i, sp in enumerate(spp) if sp.symbol == "O"]

    if 2 * len(carb) > len(oxy):
        warnings.warn("[sticky-map] #O < 2×#C; mapping may be incomplete.")

    L = structure.lattice
    shifts: Dict[int, np.ndarray] = {}
    groups: List[List[int]] = []
    oxy_free = set(oxy)

    def nearest_oxygens_to_C(ci: int) -> List[Tuple[float, int, np.ndarray]]:
        cF = frac[ci]
        cand = []
        for oi in oxy_free:
            dF = frac[oi] - cF
            t = np.round(dF)
            dF -= t
            dC = L.matrix @ dF
            cand.append((float(np.linalg.norm(dC)), oi, t.astype(int)))
        cand.sort(key=lambda x: x[0])
        return cand[:2]

    for ci in carb:
        two = nearest_oxygens_to_C(ci)
        if len(two) < 2:
            raise RuntimeError("Not enough O left to assign to a C.")
        g = [ci]
        for _, oi, t in two:
            shifts[oi] = t
            g.append(oi)
            oxy_free.remove(oi)
        groups.append(g)

    cart = coords_with_mapping(structure, groups, shifts)
    for (ci, o1, o2) in groups:
        d1 = float(np.linalg.norm(cart[o1] - cart[ci]))
        d2 = float(np.linalg.norm(cart[o2] - cart[ci]))
        if d1 > warn_long_co or d2 > warn_long_co:
            warnings.warn(f"[sticky-map] long C–O ({d1:.2f}, {d2:.2f}) Å at reference")

    return groups, shifts


def coords_with_mapping(structure, groups: Sequence[Sequence[int]], shifts: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Apply stored integer shifts to oxygen fractional coordinates and return
    the corresponding Cartesian coordinates.
    """
    frac = np.asarray(structure.frac_coords, float)
    for _, o1, o2 in groups:
        for oi in (o1, o2):
            t = np.asarray(shifts[oi], int)
            frac[oi] = frac[oi] - t
    return structure.lattice.get_cartesian_coords(frac)

def compute_energy_from_cell_rc_images_mapped(structure, rc: float,
                                              groups: Sequence[Sequence[int]],
                                              shifts: Dict[int, np.ndarray]) -> float:
    """
    Same as compute_energy_from_cell_rc_images, but uses a persistent (groups, shifts)
    mapping so periodic images don't change between calls.
    """
    cart = coords_with_mapping(structure, groups, shifts)
    lat = structure.lattice

    # 1-body
    E = 0.0
    for g in groups:
        E += p1b(np.concatenate([cart[i] for i in g]))

    # 2-body central cell
    n = len(groups)
    anchors = np.asarray([cart[g[0]] for g in groups])
    shifts_ijk = image_shifts_for_rc(lat, rc)

    for i in range(n):
        mi = np.concatenate([cart[idx] for idx in groups[i]])
        ci = anchors[i]
        for j in range(i + 1, n):
            rvec = anchors[j] - ci
            if np.linalg.norm(rvec) <= rc + 1e-12:
                mj0 = np.concatenate([cart[idx] for idx in groups[j]])  # absolute coords
                E += sapt(np.concatenate([mi, mj0]))

    # 2-body images (positive half-space)
    for T in shifts_ijk:
        if T == (0, 0, 0) or not halfspace(T):
            continue
        RT = lat.get_cartesian_coords(np.array(T, float))
        for i in range(n):
            mi = np.concatenate([cart[idx] for idx in groups[i]])
            ci = anchors[i]
            for j in range(n):
                rvec = (anchors[j] + RT) - ci
                if np.linalg.norm(rvec) <= rc + 1e-12:
                    mj = (np.concatenate([cart[idx] for idx in groups[j]]).reshape(3, 3) + RT).ravel()
                    E += sapt(np.concatenate([mi, mj]))
    return float(E)


def compute_gradients_from_cell_rc_images_mapped(structure, rc: float,
                                                 groups: Sequence[Sequence[int]],
                                                 shifts: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Same as compute_gradients_from_cell_rc_images, but uses persistent (groups, shifts)
    to keep periodic images fixed across evaluations.
    """
    cart = coords_with_mapping(structure, groups, shifts)
    lat = structure.lattice

    n_atoms = len(structure)
    G = np.zeros((n_atoms, 3), float)

    # 1-body
    for g in groups:
        g9 = np.asarray(p1b_gradient(np.concatenate([cart[i] for i in g]))).reshape(3, 3)
        G[g[0]] += g9[0]
        G[g[1]] += g9[1]
        G[g[2]] += g9[2]

    # 2-body central cell
    n = len(groups)
    anchors = np.asarray([cart[g[0]] for g in groups])
    shifts_ijk = image_shifts_for_rc(lat, rc)

    for i in range(n):
        mi = np.concatenate([cart[idx] for idx in groups[i]])
        ci = anchors[i]
        for j in range(i + 1, n):
            if np.linalg.norm(anchors[j] - ci) <= rc + 1e-12:
                mj0 = np.concatenate([cart[idx] for idx in groups[j]])
                g18 = np.asarray(sapt_gradient(np.concatenate([mi, mj0]))).reshape(6, 3)
                gi, gj = groups[i], groups[j]
                G[gi[0]] += g18[0]; G[gi[1]] += g18[1]; G[gi[2]] += g18[2]
                G[gj[0]] += g18[3]; G[gj[1]] += g18[4]; G[gj[2]] += g18[5]

    # 2-body images (positive half-space)
    for T in shifts_ijk:
        if T == (0, 0, 0) or not halfspace(T):
            continue
        RT = lat.get_cartesian_coords(np.array(T, float))
        for i in range(n):
            mi = np.concatenate([cart[idx] for idx in groups[i]])
            ci = anchors[i]
            for j in range(n):
                if np.linalg.norm((anchors[j] + RT) - ci) <= rc + 1e-12:
                    mj = (np.concatenate([cart[idx] for idx in groups[j]]).reshape(3, 3) + RT).ravel()
                    g18 = np.asarray(sapt_gradient(np.concatenate([mi, mj]))).reshape(6, 3)
                    gi, gj = groups[i], groups[j]
                    G[gi[0]] += g18[0]; G[gi[1]] += g18[1]; G[gi[2]] += g18[2]
                    G[gj[0]] += g18[3]; G[gj[1]] += g18[4]; G[gj[2]] += g18[5]

    return G


# -----------------------------------------------------------------------------
# Energetics with explicit images
# -----------------------------------------------------------------------------

def compute_energy_from_cell_rc_images(structure, rc: float) -> float:
    """
    Explicit-image energy sum (kcal/mol). Correct for any rc. Avoids double
    counting using the positive half-space.
    """
    cart, _, groups = wrap_coordinates_by_carbon_fractional(structure)
    lat = structure.lattice

    # 1-body
    E = 0.0
    for g in groups:
        E += p1b(np.concatenate([cart[i] for i in g]))

    # 2-body central cell
    n = len(groups)
    anchors = np.asarray([cart[g[0]] for g in groups])
    shifts = image_shifts_for_rc(lat, rc)

    for i in range(n):
        mi = np.concatenate([cart[idx] for idx in groups[i]])
        ci = anchors[i]
        for j in range(i + 1, n):
            rvec = anchors[j] - ci
            if np.linalg.norm(rvec) <= rc + 1e-12:
                mj0 = np.concatenate([cart[idx] for idx in groups[j]])  # absolute coords
                E += sapt(np.concatenate([mi, mj0]))

    # 2-body images (positive half-space)
    for T in shifts:
        if T == (0, 0, 0) or not halfspace(T):
            continue
        RT = lat.get_cartesian_coords(np.array(T, float))
        for i in range(n):
            mi = np.concatenate([cart[idx] for idx in groups[i]])
            ci = anchors[i]
            for j in range(n):
                rvec = (anchors[j] + RT) - ci
                if np.linalg.norm(rvec) <= rc + 1e-12:
                    mj = (np.concatenate([cart[idx] for idx in groups[j]]).reshape(3, 3) + RT).ravel()
                    E += sapt(np.concatenate([mi, mj]))
    return float(E)


def compute_gradients_from_cell_rc_images(structure, rc: float) -> np.ndarray:
    """
    Cartesian gradients dE/dR (kcal/mol/Å) for each atom (N, 3). Uses
    analytical gradients for 1-body and 2-body terms; explicit-image sum
    with positive half-space to avoid double counting.
    """
    cart, _, groups = wrap_coordinates_by_carbon_fractional(structure)
    lat = structure.lattice
    A = lat.matrix

    n_atoms = len(structure)
    G = np.zeros((n_atoms, 3), float)

    # 1-body
    for g in groups:
        g9 = np.asarray(p1b_gradient(np.concatenate([cart[i] for i in g]))).reshape(3, 3)
        G[g[0]] += g9[0]
        G[g[1]] += g9[1]
        G[g[2]] += g9[2]

    # 2-body central cell
    n = len(groups)
    anchors = np.asarray([cart[g[0]] for g in groups])
    shifts = image_shifts_for_rc(lat, rc)

    for i in range(n):
        mi = np.concatenate([cart[idx] for idx in groups[i]])
        ci = anchors[i]
        for j in range(i + 1, n):
            if np.linalg.norm(anchors[j] - ci) <= rc + 1e-12:
                mj0 = np.concatenate([cart[idx] for idx in groups[j]])
                g18 = np.asarray(sapt_gradient(np.concatenate([mi, mj0]))).reshape(6, 3)
                gi, gj = groups[i], groups[j]
                G[gi[0]] += g18[0]; G[gi[1]] += g18[1]; G[gi[2]] += g18[2]
                G[gj[0]] += g18[3]; G[gj[1]] += g18[4]; G[gj[2]] += g18[5]

    # 2-body images (positive half-space)
    for T in shifts:
        if T == (0, 0, 0) or not halfspace(T):
            continue
        RT = A.T @ np.array(T, float)
        for i in range(n):
            mi = np.concatenate([cart[idx] for idx in groups[i]])
            ci = anchors[i]
            for j in range(n):
                if np.linalg.norm((anchors[j] + RT) - ci) <= rc + 1e-12:
                    mj = (np.concatenate([cart[idx] for idx in groups[j]]).reshape(3, 3) + RT).ravel()
                    g18 = np.asarray(sapt_gradient(np.concatenate([mi, mj]))).reshape(6, 3)
                    gi, gj = groups[i], groups[j]
                    G[gi[0]] += g18[0]; G[gi[1]] += g18[1]; G[gi[2]] += g18[2]
                    G[gj[0]] += g18[3]; G[gj[1]] += g18[4]; G[gj[2]] += g18[5]

    return G


def compute_direct_stress_fd(structure, rc: float, delta: float = 1e-3) -> np.ndarray:
    """
    Cauchy stress via symmetric finite-difference of strain at fixed fractional coords.
    Returns a 3x3 tensor in eV/Å^3 (ASE sign convention: compression negative).

    Parameters
    ----------
    structure : PMGStructure
        Reference structure. Fractional coordinates are held fixed.
    rc : float
        Real-space cutoff (Å) used inside compute_energy_from_cell_rc_images.
    delta : float
        Strain step for symmetric FD (unitless).

    Notes
    -----
    - Energy is evaluated with compute_energy_from_cell_rc_images (kcal/mol),
      then converted to eV before FD.
    - Volume used is the *unstrained* reference volume of `structure`.
    """
    # base cell and volume
    A = np.asarray(structure.lattice.matrix, float)   # 3x3
    V0 = float(structure.volume)

    # convenience (species & frac coords stay the same under pure strain)
    species = list(structure.species)
    frac = np.asarray(structure.frac_coords, float)

    s = np.zeros((3, 3), float)
    I = np.eye(3)

    for i in range(3):
        for j in range(i, 3):
            eps = np.zeros((3, 3), float)
            eps[i, j] = eps[j, i] = delta

            # + strain
            A_plus = A @ (I + eps)
            s_plus = PMGStructure(
                Lattice(A_plus), species, frac,
                coords_are_cartesian=False, to_unit_cell=False
            )
            E_plus_kcal = compute_energy_from_cell_rc_images(s_plus, rc)
            E_plus_eV = E_plus_kcal * (units.kcal / units._Nav)

            # - strain
            A_minus = A @ (I - eps)
            s_minus = PMGStructure(
                Lattice(A_minus), species, frac,
                coords_are_cartesian=False, to_unit_cell=False
            )
            E_minus_kcal = compute_energy_from_cell_rc_images(s_minus, rc)
            E_minus_eV = E_minus_kcal * (units.kcal / units._Nav)

            # symmetric FD derivative w.r.t. strain -> stress (eV/Å^3)
            sij = (E_plus_eV - E_minus_eV) / (2.0 * delta * V0)
            s[i, j] = s[j, i] = sij

    return s


def voigt6_from_tensor(sig: np.ndarray) -> np.ndarray:
    """3x3 symmetric tensor -> Voigt-6 (xx, yy, zz, yz, xz, xy)."""
    return np.array([sig[0,0], sig[1,1], sig[2,2], sig[1,2], sig[0,2], sig[0,1]], float)

def ase_atoms_to_pmg(atoms):
    return PMGStructure(Lattice(atoms.cell.array),
                        atoms.get_chemical_symbols(),
                        atoms.get_positions(),
                        coords_are_cartesian=True,
                        to_unit_cell=False)

class CO2TwoBodyRC_IMAGES(Calculator):
    """
    ASE calculator for CO₂ with explicit-image interactions.
    Provides energy, forces, and stress (via direct FD).
    """
    implemented_properties = ['energy', 'forces', 'stress']
    implemented_changes = all_changes

    def __init__(self, rc, **params):
        super().__init__(**params)
        self.rc = float(rc)

        # Sticky mapping cache
        self._groups: List[List[int]] | None = None
        self._shifts: Dict[int, np.ndarray] | None = None
        self._map_key = None
        self.warn_long_co = 1.8  # Å; sanity for when to rebuild mapping

    def _structure_key(self, s: PMGStructure):
        """Hashable key that changes if natoms/species/lattice change."""
        spp = tuple(sp.symbol for sp in s.species)
        # round lattice to avoid tiny float noise
        lat_key = tuple(np.round(s.lattice.matrix.ravel(), 8))
        return (len(s), spp, lat_key)

    def _ensure_mapping(self, s: PMGStructure):
        """
        Build or validate the (groups, shifts) sticky mapping.
        Rebuild if key changes or if mapped C–O bonds look unphysically long.
        """
        key = self._structure_key(s)
        rebuild = (self._map_key != key) or (self._groups is None) or (self._shifts is None)

        if not rebuild:
            # Validate current mapping: check C–O distances under current lattice/frac
            try:
                cart = coords_with_mapping(s, self._groups, self._shifts)
                for (ci, o1, o2) in self._groups:
                    d1 = float(np.linalg.norm(cart[o1] - cart[ci]))
                    d2 = float(np.linalg.norm(cart[o2] - cart[ci]))
                    if (d1 > self.warn_long_co) or (d2 > self.warn_long_co):
                        rebuild = True
                        break
            except Exception:
                rebuild = True

        if rebuild:
            self._groups, self._shifts = build_reference_mapping(s, warn_long_co=self.warn_long_co)
            self._map_key = key

    
    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pmg = ase_atoms_to_pmg(atoms)
        self._ensure_mapping(pmg)

        # --- Energy ---
        E_kcal = compute_energy_from_cell_rc_images_mapped(pmg, self.rc, self._groups, self._shifts)
        self.results['energy'] = float(E_kcal * (units.kcal / units._Nav))  # eV

        # --- Forces ---
        if ('forces' in properties) or ('stress' in properties):
            G = compute_gradients_from_cell_rc_images_mapped(pmg, self.rc, self._groups, self._shifts)
            F = (-G) * (units.kcal / units._Nav)  # eV/Å
            self.results['forces'] = np.ascontiguousarray(F, float)

        # --- Stress ---
        if 'stress' in properties:
            sig = compute_direct_stress_fd(pmg, self.rc, delta=1e-3)
            self.results['stress'] = voigt6_from_tensor(sig)

        return

__all__ = [
    "wrap_coordinates_by_carbon_fractional",
    "image_shifts_for_rc",
    "halfspace",
    "build_reference_mapping",
    "coords_with_mapping",
    "compute_energy_from_cell_rc_images_mapped",
    "compute_gradients_from_cell_rc_images_mapped",
    "compute_energy_from_cell_rc_images",
    "compute_gradients_from_cell_rc_images",
    "compute_direct_stress_fd",
    "voigt6_from_tensor",
    "CO2TwoBodyRC_IMAGES",
]

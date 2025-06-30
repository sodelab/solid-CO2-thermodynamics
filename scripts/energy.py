
import numpy as np
import math
from scipy.optimize import minimize
from symmetry import build_unit_cell
from spacegroup import Pa3, Cmce, P42mnm, R3c, Pna21
from co2_potential import p1b, p2b, sapt  # Assuming these are defined in co2_potential.py

import numpy as np

def apply_minimum_image_to_molecule2(updated_cart_coords, lattice, molecule_indices):
    """
    Given a global array of Cartesian coordinates, a lattice object, and a list of atom indices
    for a molecule (with the first atom assumed to be the reference carbon), this function 
    adjusts the coordinates for those atoms using the minimum image convention relative to the
    carbon atom, and returns the modified global array.
    
    Parameters:
        updated_cart_coords (np.array): Global Cartesian coordinates of atoms (N x 3).
        lattice: A lattice object with attribute .matrix (a 3x3 numpy array).
        molecule_indices (list): List of indices for atoms in the molecule.
        
    Returns:
        new_cart_coords (np.array): Global Cartesian coordinates with the molecule's atoms adjusted.
    """
    # Create a copy of the global coordinates so the original is not modified.
    new_cart_coords = updated_cart_coords.copy()
    
    # Use the carbon atom (first atom in the group) as the reference.
    ref = updated_cart_coords[molecule_indices[0]]
    
    # For each atom in the molecule, compute the minimum image displacement relative to the carbon.
    for idx in molecule_indices:
        pos = updated_cart_coords[idx]
        # Compute displacement relative to the reference carbon.
        d_cart = pos - ref
        # Convert the Cartesian displacement to fractional coordinates.
        d_frac = np.linalg.solve(lattice.matrix.T, d_cart)
        # Apply the minimum image convention in fractional space.
        d_frac_min = d_frac - np.round(d_frac)
        # Convert the wrapped displacement back to Cartesian coordinates.
        d_cart_min = lattice.matrix.T.dot(d_frac_min)
        # Update the coordinate for this atom.
        new_coord = ref + d_cart_min
        new_cart_coords[idx] = new_coord
        
    return new_cart_coords

def apply_minimum_image_to_molecule(updated_cart_coords, lattice, molecule_indices):
    """
    Given a list of updated cartesian coordinates, a lattice object, and a list of atom indices
    for a molecule (with the first atom assumed to be the reference carbon), this function shifts
    all atom coordinates so that they are expressed in the minimum image convention relative to
    the carbon atom.
    
    Parameters:
        updated_cart_coords (array): Global Cartesian coordinates of atoms.
        lattice: A lattice object with attributes .matrix (a 3x3 numpy array).
        molecule_indices (list): List of indices for atoms in a molecule.
        
    Returns:
        new_coords (list): List of adjusted Cartesian coordinates for the molecule.
    """
    # Use the carbon atom (first atom in the group) as the reference.
    ref = updated_cart_coords[molecule_indices[0]]
    new_coords = []
    # For each atom in the molecule:
    for idx in molecule_indices:
        pos = updated_cart_coords[idx]
        # Compute the displacement vector relative to the carbon.
        d_cart = pos - ref
        # Convert this Cartesian displacement to fractional coordinates.
        d_frac = np.linalg.solve(lattice.matrix.T, d_cart)
        # Apply the minimum image convention.
        d_frac_min = d_frac - np.round(d_frac)
        # Convert the adjusted fractional displacement back to Cartesian.
        d_cart_min = lattice.matrix.T.dot(d_frac_min)
        # The new coordinate is the carbon position plus the minimal-image displacement.
        new_coord = ref + d_cart_min
        new_coords.append(new_coord)
    return new_coords

def apply_minimum_image_to_all_molecules(updated_cart_coords, lattice, molecules_grouped):
    """
    Adjusts the Cartesian coordinates for all molecules using the minimum image convention,
    pegging each molecule's positions relative to its reference carbon atom.
    
    Parameters:
        updated_cart_coords (np.array): Global Cartesian coordinates (N x 3) for all atoms.
        lattice: A lattice object with attribute `matrix` (3x3 NumPy array).
        molecules_grouped (list): List of molecule groups, where each group is a list of atom indices 
                                  [carbon_index, oxygen_index1, oxygen_index2].
    
    Returns:
        new_cart_coords (np.array): Updated Cartesian coordinates for all atoms.
    """
    # Create a copy so we don't modify the original coordinates.
    new_cart_coords = updated_cart_coords.copy()
    
    # Iterate over each molecule group.
    for group in molecules_grouped:
        # Use the pre-existing function to adjust one molecule relative to its carbon.
        adjusted_coords = apply_minimum_image_to_molecule(updated_cart_coords, lattice, group)
        # Update the coordinates for atoms in the current molecule group.
        for idx, new_coord in zip(group, adjusted_coords):
            new_cart_coords[idx] = new_coord
    return new_cart_coords

def wrap_coordinates_by_carbon_fractional(structure):
    """
    Alternative wrapping approach using fractional coordinates:
    1. Work with fractional coordinates directly
    2. For each carbon, find the two closest oxygens using minimum image convention
    3. Assign oxygens to carbons and wrap them properly in fractional space
    4. Convert to Cartesian only after grouping
    
    Parameters:
        structure: A pymatgen Structure object
        
    Returns:
        updated_cart_coords: Updated cartesian coordinates
        molecule_assignment: List assigning each atom to a molecule
        molecules_grouped: List of lists, each containing [C_idx, O1_idx, O2_idx]
    """
    # Get fractional coordinates
    frac_coords = np.array(structure.frac_coords)
    species = structure.species
    
    # Identify carbon and oxygen indices
    carbon_indices = [i for i, sp in enumerate(species) if sp.symbol == "C"]
    oxygen_indices = [i for i, sp in enumerate(species) if sp.symbol == "O"]
    
    # Create a copy of fractional coordinates to update
    updated_frac_coords = np.array(frac_coords)
    
    # Create structures to track molecule assignments
    molecule_assignment = [None] * len(species)
    molecules_grouped = []
    oxygen_available = set(oxygen_indices)
    mol_id = 0
    
    # Helper function for minimum image distance calculation in fractional space
    def min_image_distance_fractional(pos1, pos2):
        """Calculate the minimum image distance between two points in fractional space"""
        d_frac = pos2 - pos1
        # Apply minimum image convention in fractional space
        d_frac_min = d_frac - np.round(d_frac)
        
        # Convert to Cartesian for true distance calculation
        d_cart_min = structure.lattice.matrix.dot(d_frac_min)
        return d_frac_min, np.linalg.norm(d_cart_min)
    
    # Process each carbon atom
    for ci in carbon_indices:
        c_frac_pos = updated_frac_coords[ci]
        
        # Find distances to all available oxygens using minimum image convention
        oxygen_distances = []
        for oi in oxygen_available:
            o_frac_pos = updated_frac_coords[oi]
            min_disp_frac, dist = min_image_distance_fractional(c_frac_pos, o_frac_pos)
            oxygen_distances.append((dist, oi, min_disp_frac))
        
        # Sort by distance and select the two closest oxygens
        oxygen_distances.sort()
        #print(f"Carbon at index {ci} has distances to oxygens: {[d[0] for d in oxygen_distances]}")
        
        if len(oxygen_distances) < 2:
            raise ValueError(f"Not enough available oxygens for carbon at index {ci}")
        
        # Get the two closest oxygens
        chosen_oxygens = []
        for idx in range(2):  # Get closest two oxygens
            dist, oi, min_disp_frac = oxygen_distances[idx]
            # Move oxygen to minimum image position relative to carbon in fractional space
            updated_frac_coords[oi] = c_frac_pos + min_disp_frac
            chosen_oxygens.append(oi)
            oxygen_available.remove(oi)
            
            # For debugging - print bond lengths if they're suspicious
            if dist > 1.5:  # CO bond is typically ~1.16Å
                print(f"Warning: Long C-O distance ({dist:.4f} Å) for C{ci}-O{oi}")
        
        # Assign molecule ID
        molecule_assignment[ci] = mol_id
        for oi in chosen_oxygens:
            molecule_assignment[oi] = mol_id
        
        # Add to grouped molecules
        molecules_grouped.append([ci] + chosen_oxygens)
        mol_id += 1
        
    # Verify all oxygens are assigned
    unassigned = [i for i in oxygen_indices if molecule_assignment[i] is None]
    if unassigned:
        print(f"Warning: {len(unassigned)} oxygen atoms remain unassigned")
    
    # Convert to Cartesian coordinates at the end
    updated_cart_coords = structure.lattice.get_cartesian_coords(updated_frac_coords)
    
    # Print molecule distances for verification
    # for idx, group in enumerate(molecules_grouped):
    #     c_idx = group[0]
    #     o1_idx = group[1]
    #     o2_idx = group[2]
        
    #     c_pos = updated_cart_coords[c_idx]
    #     o1_pos = updated_cart_coords[o1_idx]
    #     o2_pos = updated_cart_coords[o2_idx]

    #     #print(c_pos)
    #     #print(o1_pos)
    #     #print(o2_pos)
        
    #     co1 = np.linalg.norm(o1_pos - c_pos)
    #     co2 = np.linalg.norm(o2_pos - c_pos)
    #     oo = np.linalg.norm(o1_pos - o2_pos)
        
    #     print(f"Molecule {idx}:")
    #     print(f"  C-O1 distance: {co1:.6f} Å")
    #     print(f"  C-O2 distance: {co2:.6f} Å")
    #     print(f"  O1-O2 distance: {oo:.6f} Å")
    
    return updated_cart_coords, molecule_assignment, molecules_grouped

def wrap_coordinates(structure):
    """
    1. Get cartesian coordinates.
    2. For each oxygen atom, move it to minimum image with nearest carbon (Cartesian only).
    3. Group carbon dioxide molecules.
    """
    L = structure.lattice.matrix
    invL = np.linalg.inv(L)
    cart_coords = structure.lattice.get_cartesian_coords(np.array(structure.frac_coords))
    species = structure.species

    carbon_indices = [i for i, sp in enumerate(species) if sp.symbol == "C"]
    oxygen_indices = [i for i, sp in enumerate(species) if sp.symbol == "O"]

    updated_cart_coords = np.array(cart_coords)

    # Step 2: For each oxygen, move to minimum image with nearest carbon (Cartesian only)
    for oi in oxygen_indices:
        o_pos = updated_cart_coords[oi]
        min_dist = np.inf
        nearest_c = None
        nearest_disp = None
        for ci in carbon_indices:
            c_pos = updated_cart_coords[ci]
            d_cart = o_pos - c_pos
            coeffs = np.dot(invL, d_cart)
            coeffs_wrapped = coeffs - np.round(coeffs)
            d_cart_min = np.dot(L, coeffs_wrapped)
            dist = np.linalg.norm(d_cart_min)
            #print(oi, ci, dist)
            if dist < min_dist:
                min_dist = dist
                nearest_c = ci
                nearest_disp = d_cart_min
        updated_cart_coords[oi] = updated_cart_coords[nearest_c] + nearest_disp

    # Step 3: Group molecules
    molecule_assignment = [None] * len(species)
    molecules_grouped = []
    oxygen_assigned = set()
    mol_id = 0

    for ci in carbon_indices:
        c_pos = updated_cart_coords[ci]
        dists = []
        for oi in oxygen_indices:
            if oi in oxygen_assigned:
                continue
            o_pos = updated_cart_coords[oi]
            dist = np.linalg.norm(o_pos - c_pos)
            dists.append((dist, oi))
        if len(dists) < 2:
            raise ValueError("Not enough unassigned oxygen atoms to form a CO2 molecule with carbon at index {}".format(ci))
        dists.sort(key=lambda x: x[0])
        chosen_oxygens = [dists[0][1], dists[1][1]]
        oxygen_assigned.update(chosen_oxygens)
        molecule_assignment[ci] = mol_id
        for oi in chosen_oxygens:
            molecule_assignment[oi] = mol_id
        molecules_grouped.append([ci] + chosen_oxygens)
        mol_id += 1

    updated_cart_coords = apply_minimum_image_to_all_molecules(updated_cart_coords, structure.lattice, molecules_grouped)

    for idx, group in enumerate(molecules_grouped):
        c_idx, o1_idx, o2_idx = group
        c_pos = updated_cart_coords[c_idx]
        o1_pos = updated_cart_coords[o1_idx]
        o2_pos = updated_cart_coords[o2_idx]
        co1 = np.linalg.norm(o1_pos - c_pos)
        co2 = np.linalg.norm(o2_pos - c_pos)
        oo = np.linalg.norm(o1_pos - o2_pos)
        print(f"Molecule {idx}:")
        print(f"  C-O1 distance: {co1:.6f} Å")
        print(f"  C-O2 distance: {co2:.6f} Å")
        print(f"  O1-O2 distance: {oo:.6f} Å")
        if idx == 100:

            print("DEBUG")
            c_idx, o1_idx, o2_idx = group
            print(np.array(structure.frac_coords)[c_idx])
            print(np.array(structure.frac_coords)[o1_idx])
            print(np.array(structure.frac_coords)[o2_idx])

            c_frac = np.array(structure.frac_coords)[c_idx]
            o1_frac = np.array(structure.frac_coords)[o1_idx]
            o2_frac = np.array(structure.frac_coords)[o2_idx]

            # Convert the wrapped displacement back to Cartesian coordinates.
            c_cart = structure.lattice.matrix.T.dot(c_frac)
            o1_cart = structure.lattice.matrix.T.dot(o1_frac)
            o2_cart = structure.lattice.matrix.T.dot(o2_frac)

            co1_f = np.linalg.norm(o1_cart - c_cart)
            co2_f = np.linalg.norm(o2_cart - c_cart)
            oo_f = np.linalg.norm(o1_cart - o2_cart)

            print(f"Molecule {idx}:")
            print(f"  C-O1 distance: {co1_f:.6f} Å")
            print(f"  C-O2 distance: {co2_f:.6f} Å")
            print(f"  O1-O2 distance: {oo_f:.6f} Å")


            c_pos = updated_cart_coords[c_idx]
            o1_pos = updated_cart_coords[o1_idx]
            o2_pos = updated_cart_coords[o2_idx]
            co1 = np.linalg.norm(o1_pos - c_pos)
            co2 = np.linalg.norm(o2_pos - c_pos)
            oo = np.linalg.norm(o1_pos - o2_pos)
            print(f"Molecule {idx}:")
            print(f"  C-O1 distance: {co1:.6f} Å")
            print(f"  C-O2 distance: {co2:.6f} Å")
            print(f"  O1-O2 distance: {oo:.6f} Å")
            print(o1_pos-c_pos)
            print(o2_pos-c_pos)
            print(o1_pos-o2_pos)
            print(f"Lattice matrix:\n{structure.lattice.matrix}")
            print()

            new_cart_coords = apply_minimum_image_to_molecule2(updated_cart_coords, structure.lattice, [c_idx, o1_idx, o2_idx])
            c_pos = new_cart_coords[c_idx]
            o1_pos = new_cart_coords[o1_idx]
            o2_pos = new_cart_coords[o2_idx]
            co1 = np.linalg.norm(o1_pos - c_pos)
            co2 = np.linalg.norm(o2_pos - c_pos)
            oo = np.linalg.norm(o1_pos - o2_pos)
            print(f"  C-O1 distance: {co1:.6f} Å")
            print(f"  C-O2 distance: {co2:.6f} Å")
            print(f"  O1-O2 distance: {oo:.6f} Å")
            #exit(0)

    #updated_cart_coords = apply_minimum_image_to_all_molecules(updated_cart_coords, structure.lattice, molecules_grouped)

        # Second pass: check for unassigned oxygens and underfilled carbons
    unassigned_oxygens = [oi for oi in oxygen_indices if molecule_assignment[oi] is None]
    underfilled_carbons = [i for i, group in enumerate(molecules_grouped) if len(group) < 3]

    for ci in underfilled_carbons:
        c_idx = molecules_grouped[ci][0]
        c_pos = updated_cart_coords[c_idx]
        # Find nearest unassigned oxygen
        if not unassigned_oxygens:
            break
        dists = [(np.linalg.norm(updated_cart_coords[oi] - c_pos), oi) for oi in unassigned_oxygens]
        dists.sort(key=lambda x: x[0])
        nearest_oi = dists[0][1]
        molecules_grouped[ci].append(nearest_oi)
        molecule_assignment[nearest_oi] = ci
        unassigned_oxygens.remove(nearest_oi)

    return updated_cart_coords, molecule_assignment, molecules_grouped

def compute_energy_from_cell(structure, params):
    # Wrap coordinates and group atoms into CO2 molecules.
    updated_cart_coords, molecule_assignment, molecules_grouped = wrap_coordinates_by_carbon_fractional(structure)
    total_energy = 0.0
    #print(updated_cart_coords)

    #print(updated_cart_coords)
    # Compute monomer energy contributions.
    for group in molecules_grouped:
        # Each group is [C_index, O1_index, O2_index]. Extract their cartesian coordinates,
        # flatten into a 9-element array and pass to p1b.
        mol_coords = np.concatenate([updated_cart_coords[i] for i in group])
        #print(mol_coords)
        #print(p1b(mol_coords))
        total_energy += p1b(mol_coords)

    # Compute dimer energies for monomer pairs within the unit cell.
    n_molecules = len(molecules_grouped)
    for i in range(n_molecules):
        mol_i = np.concatenate([updated_cart_coords[idx] for idx in molecules_grouped[i]])
        for j in range(i + 1, n_molecules):
            mol_j = np.concatenate([updated_cart_coords[idx] for idx in molecules_grouped[j]])
            total_energy += sapt(np.concatenate([mol_i, mol_j]))

    # Compute dimer energies for interactions where one molecule is in the central cell and
    # the other is an image from a neighboring cell. Use a half weight to avoid double counting.
    #shifts = [-3, -2, -1, 0, 1, 2, 3]
    shifts = [-1, 0, 1]
    #shifts = [0]
    # Loop over all translations except the zero shift.
    for T in [(nx, ny, nz) for nx in shifts for ny in shifts for nz in shifts if (nx, ny, nz) != (0, 0, 0)]:
        # Displacement vector in cartesian coordinates.
        disp = structure.lattice.matrix.T.dot(np.array(T))
        #print(disp)
        #exit(0)
        for i in range(n_molecules):
            mol_i = np.concatenate([updated_cart_coords[idx] for idx in molecules_grouped[i]])
            for j in range(n_molecules):
                mol_j = np.concatenate([updated_cart_coords[idx] for idx in molecules_grouped[j]])
                # Shift molecule j by the displacement. Reshape to add disp to each atomic coordinate.
                mol_j_image = (mol_j.reshape(-1, 3) + disp).flatten()
                nrg=0.5 * sapt(np.concatenate([mol_i, mol_j_image]))
                #if math.fabs(nrg) > 1e-1:
                #    print(f"Warning: Large energy contribution from image {i} and {j} interaction: {nrg:.4f} kcal/mol")
                total_energy += 0.5 * sapt(np.concatenate([mol_i, mol_j_image]))

    return total_energy

def export_structure_to_xyz(structure, filename="structure.xyz"):
    """
    Exports a pymatgen Structure to an XYZ file for visualization.
    
    Parameters:
        structure (Structure): A pymatgen Structure object.
        filename (str): Name of the output XYZ file.
    """
    updated_cart_coords, molecule_assignment, molecules_grouped = wrap_coordinates_by_carbon_fractional(structure)
    #updated_cart_coords = structure.lattice.get_cartesian_coords(structure.frac_coords)
    #print(molecules_grouped)
    with open(filename, "w") as f:
        num_atoms = len(structure)
        # Write the number of atoms and a comment line.
        f.write(f"{num_atoms}\n")
        f.write("XYZ file generated from pymatgen Structure\n")
        # Write each atomic site: species and Cartesian coordinates.
        for i,site in enumerate(structure):
            x, y, z = updated_cart_coords[i]
            f.write(f"{site.species_string} {x:.8f} {y:.8f} {z:.8f}\n")
    print(f"Structure exported to {filename}") 

if __name__ == "__main__":
    import numpy as np
    from spacegroup import Pa3, Cmce, P42mnm
    from co2_potential import p1b, p2b  # Assuming these are defined in co2_potential.py

    # TEST 1: Pa-3 Structure
    # Create a Pa3 structure with a cubic lattice constant.
    pa3_structure = Pa3(a=5.5)
    # For Pa-3, assume the optimization variable for bond length is given in scaled form.
    # For testing, we simply use a value that after conversion gives a reasonable bond length.
    # Here we pass the actual bond length (in angstroms) to adjust_fractional_coords.
    scaled_bond_length = 1.16  # this is just an example conversion.
    pa3_structure.adjust_fractional_coords(bond_length=scaled_bond_length)
    structure_pa3 = build_unit_cell(pa3_structure)
    #wrap_coordinates(structure_pa3)
    
    energy_params_pa3 = {
        "pressure": 0.0,
        "symmetry": "pa-3",
    }

    #energy_pa3 = compute_energy_from_cell(structure_pa3, energy_params_pa3)
    #print("Pa-3 structure energy: {:.4f}".format(energy_pa3))

    # TEST 2: Cmce Structure
    # Create a Cmce structure with orthorhombic lattice parameters.
    cmce_structure = Cmce(a=5.3, b=4.8, c=6.6)
    # For Cmce, provide both bond length and bond angle.
    bond_length_cmce = 1.16  # angstrom, for example
    bond_angle_cmce = 45     # degrees, for example
    cmce_structure.adjust_fractional_coords(bond_length=bond_length_cmce, bond_angle=bond_angle_cmce)
    structure_cmce = build_unit_cell(cmce_structure)
    #wrap_coordinates(structure_cmce)
    
    energy_params_cmce = {
        "pressure": 2.0,
        "symmetry": "cmce",
    }
    
    #energy_cmce = compute_energy_from_cell(structure_cmce, energy_params_cmce)
    #print("Cmce structure energy: {:.4f}".format(energy_cmce))

    # TEST 3: P42/mnm Structure
    # Create a P42/mnm structure with tetragonal lattice parameters.
    p42mnm_structure = P42mnm(a=5.0, c=6.5)
    # For P42/mnm, provide both bond length and bond angle.
    bond_length_p42mnm = 1.16  # angstrom, for example
    p42mnm_structure.adjust_fractional_coords(bond_length=bond_length_p42mnm)
    structure_p42mnm = build_unit_cell(p42mnm_structure)
    #wrap_coordinates(structure_p42mnm)
    
    energy_params_p42mnm = {
        "pressure": 0.0,
        "symmetry": "p42/mnm",
    }
    #energy_p42mnm = compute_energy_from_cell(structure_p42mnm, energy_params_p42mnm)
    #print("P42/mnm structure energy: {:.4f}".format(energy_p42mnm))
   

    # TEST 3: R-3c Structure
    # Create a R-3c structure with tetragonal lattice parameters.
    #r3c_structure = R3c(a=9.78, c=12.16)
    r3c_structure = R3c()
    # For R-3c, provide both bond length and bond angle.
    #b1 = 1.16  # angstrom, for example
    #b2 = 1.16
    #phi = 45
    #theta = 60
    #r3c_structure.adjust_fractional_coords(bond_length1=b1, 
    #                                       bond_length2=b2, 
    #                                       bond_angle_phi=phi, 
    #                                       bond_angle_theta=theta)    
    structure_r3c = build_unit_cell(r3c_structure)
    #wrap_coordinates(structure_p42mnm)
    
    energy_params_r3c = {
        "pressure": .0,
        "symmetry": "r-3c",
    }
    
    energy_r3c = compute_energy_from_cell(structure_r3c, energy_params_r3c)
    print("R-3c structure energy: {:.4f}".format(energy_r3c))
    export_structure_to_xyz(structure_r3c, filename="r3c_structure-adjusted2.xyz")

    # TEST 4: Pna2_1 Structure
    # Create a Pna2_1 structure with orthorhombic lattice parameters.
    #pna21_structure = Pna21(a=3.34, b=5.29, c=9.73)
    # For Cmce, provide both bond length and bond angle.
    #bond_length_pna21 = 1.16  # angstrom, for example
    #bond_angle_phi = -41.29     # degrees, for example
    #bond_angle_theta = 157.2
    #pna21_structure.adjust_fractional_coords(bond_length=bond_length_pna21, bond_angle_phi=bond_angle_phi, bond_angle_theta=bond_angle_theta)
    #structure_pna21 = build_unit_cell(pna21_structure)
    #wrap_coordinates(structure_pna21)
    
    energy_params_pna21 = {
        "pressure": 0.0,
        "symmetry": "pna2_1",
    }
    
    #energy_pna21 = compute_energy_from_cell(structure_pna21, energy_params_pna21)
    #print("Pna2_1 structure energy: {:.4f}".format(energy_pna21))
    #export_structure_to_xyz(structure_pna21, filename="pna21_structure.xyz")

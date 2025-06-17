from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from spacegroup import Pa3, Cmce, P42mnm, R3c, Pna21
import numpy as np


def build_unit_cell_pmg(geom_params, spacegroup="Pa-3", frac_coords=None, species=None):
    """
    Build a unit cell using pymatgen's from_spacegroup method.
    
    Parameters:
        geom_params (dict): Contains lattice and other parameters. You can provide either:
            - "lattice_const": for a cubic cell (a single number), or
            - "lattice_params": a dict with keys "a", "b", "c", "alpha", "beta", "gamma"
              for a more general cell.
        spacegroup (str or int): The space group (e.g., "Pa-3", "Cmce", etc.).
        frac_coords (list, optional): Fractional coordinates for the asymmetric unit. 
            If not provided, defaults for the selected space group are used.
        species (list, optional): List of species corresponding to the above coordinates.
            Again, if not provided, defaults for the selected space group are used.
        
    Returns:
        structure: A pymatgen Structure object representing the full unit cell.
    """
    
    # A dictionary of default asymmetric-unit settings for different space groups.
    # Each entry contains the default species and fractional coordinates based on the symmetry convention.
    default_asymmetric_units = {
        "Pa-3": {
            "species": ["C", "O"],
            "coords": [
                [0.00, 0.00, 0.00],   # Carbon at a special position
                [0.12, 0.12, 0.12]    # Oxygen defined for the Pa-3 symmetry; adjust as needed.
            ],
            "lattice": Lattice.cubic()
        },
        "Cmce": {
            "species": ["C", "O"],
            "coords": [
                [0.00, 0.00, 0.00],   # Carbon is at (0, 0, 0)
                [0.00, 0.20, 0.90]    # Oxygen: note that for Cmce the convention is often (0, y, z)
            ],
            "lattice": Lattice.orthorhombic()
        },
        # Add other space groups here as 
        # Add other space groups here as needed.
    }
        # Build lattice based on provided parameters.
    if "lattice_params" in geom_params:
        lp = geom_params["lattice_params"]
        a = lp.get("a")
        b = lp.get("b", a)
        c = lp.get("c", a)
        alpha = lp.get("alpha", 90)
        beta  = lp.get("beta", 90)
        gamma = lp.get("gamma", 90)
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    else:
        # Fallback: assume a cubic lattice if only a lattice constant is provided.
        lattice_const = geom_params.get("lattice_const", 1.0)
        lattice = Lattice.cubic(lattice_const)
    # If fractional coordinates (or species) have not been provided,
    # try to set them based on the space group default.
    if frac_coords is None or species is None:
        if spacegroup in default_asymmetric_units:
            species = default_asymmetric_units[spacegroup]["species"]
            frac_coords = default_asymmetric_units[spacegroup]["coords"]
            lattice = default_asymmetric_units[spacegroup]["lattice"]
        else:
            # Fallback to a generic default if the chosen spacegroup is not in the dictionary.
            species = ["C", "O"]
            frac_coords = [
                [0.00, 0.00, 0.00],
                [0.00, 0.12, 0.12]
            ]
    
    # Build the structure from the space group.
    structure = Structure.from_spacegroup(spacegroup, lattice, species, frac_coords)
    
    # Convert the fractional coordinates to Cartesian coordinates.
    cart_coords = structure.cart_coords  # Automatically computes fractional -> Cartesian conversion.
    
    # Print Cartesian coordinates for each atomic site.
    print("Cartesian coordinates for each atomic site:")
    for site, coord in zip(structure, cart_coords):
        print(f"{site.species_string}: {coord}")
    
    return structure

def build_unit_cell_from_optimization(x, symmetry):
    """
    Build a unit cell based on the optimized variables and symmetry.
    
    Parameters:
        x (np.array): Optimization variables.
            • For Pa-3: [scaled_bond_length, lattice_const]
            • For Cmce: [bond_length, bond_angle, a, b, c]
            • For P42/mnm: [bond_length, bond_angle, a, c]
        symmetry (str): Crystal structure type.
        
    Returns:
        tuple: (lattice, spacegroup) where lattice is the built lattice object.
    """
    symmetry = symmetry.lower()
    if symmetry == "pa-3":
        # For Pa-3 the first variable is the scaled bond length; convert it.
        bond_length = x[0] * np.sqrt(3)
        lattice_const = x[1]
        structure = Pa3(a=lattice_const)
        structure.adjust_fractional_coords(bond_length)
        return structure.build_lattice(), structure.spacegroup
    elif symmetry == "cmce":
        bond_length = x[0]
        bond_angle = x[1]
        a, b, c = x[2], x[3], x[4]
        structure = Cmce(a=a, b=b, c=c)
        structure.adjust_fractional_coords(bond_length, bond_angle)
        return structure.build_lattice(), structure.spacegroup
    elif symmetry == "p42/mnm":
        bond_length = x[0]
        a, c = x[1], x[2]
        structure = P42mnm(a=a, c=c)
        structure.adjust_fractional_coords(bond_length)
        return structure.build_lattice(), structure.spacegroup
    elif symmetry == "r-3c":
        bond_length1 = x[0]
        bond_length2 = x[1]
        bond_angle_phi = x[2]
        bond_angle_theta = x[3]
        a, c = x[4], x[5]
        structure = R3c(a=a, c=c)
        structure.adjust_fractional_coords(bond_length1, 
                                           bond_length2, 
                                           bond_angle_phi, 
                                           bond_angle_theta)
        return structure.build_lattice(), structure.spacegroup
    elif symmetry == "pna2_1":
        bond_length = x[0]
        bond_angle_phi = x[1]
        bond_angle_theta = x[2]
        a, b, c = x[3], x[4], x[5]
        structure = Pna21(a=a, b=b, c=c)
        structure.adjust_fractional_coords(bond_length, bond_angle_phi, bond_angle_theta)
        return structure.build_lattice(), structure.spacegroup
    else:
        raise ValueError("Unsupported symmetry: " + symmetry)

def build_unit_cell(spacegroup_class, bond_length=None, bond_angle=None):
    """
    Build a unit cell using a SpaceGroup class.

    Parameters:
        spacegroup_class (SpaceGroup): An instance of a SpaceGroup subclass (e.g., Pa3, Cmce).
        bond_length (float, optional): Bond length to adjust fractional coordinates, if needed.

    Returns:
        structure: A pymatgen Structure object representing the full unit cell.
    """
    # If a bond_length (and bond_angle if needed) is provided, adjust the fractional coordinates.
    if bond_length is not None:
        if bond_angle is not None:
            spacegroup_class.adjust_fractional_coords(bond_length, bond_angle)
        else:
            spacegroup_class.adjust_fractional_coords(bond_length)

    # Build the lattice and structure
    lattice = spacegroup_class.build_lattice()
    species = spacegroup_class.species
    coords = spacegroup_class.coords
    sg = spacegroup_class.spacegroup

    # Create the structure
    structure = Structure.from_spacegroup(sg, lattice, species, coords)

    return structure

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
    
    return updated_cart_coords, molecule_assignment, molecules_grouped

if __name__ == "__main__":
   # Example 1: Using cubic (Pa-3) symmetry
    print("Testing Pa-3 space group:")
    pa3 = Pa3()
    structure_pa3 = build_unit_cell(pa3, bond_length=1.16)
    print("\nCubic (Pa-3) cell structure:")
    print(structure_pa3)

    # Example 2: Using orthorhombic (Cmce) symmetry
    print("\n\nTesting Cmce space group:")
    cmce = Cmce()
    structure_cmce = build_unit_cell(cmce, bond_length=1.16, bond_angle=45)
    print("\nOrthorhombic (Cmce) cell structure:")
    print(structure_cmce)

    # Example 3: Using tetragonal (P42/mnm) symmetry
    print("Testing P42/mnm space group:")
    p42mnm = P42mnm()
    structure_p42mnm = build_unit_cell(p42mnm)
    print("\n\nTetragonal (P42/mnm) cell structure:")
    print(structure_p42mnm)

    # Example 4: Using hexagonal (R-3c) symmetry
    print(f"\n\nTesting R-3c space group:")
    r3c = R3c()
    structure_r3c = build_unit_cell(r3c)
    print("\nHexagonal (R-3c) cell structure:")
    print(structure_r3c)

    #print(structure_r3c.lattice)
    #print("Initial fractional coordinates:", r3c.coords)
    #r3c.adjust_fractional_coords(bond_length1=1.16,bond_length2=1.16, bond_angle_phi=45, bond_angle_theta=60)  # Bond length = 1.16 Å, angle = 30°
    #print("Adjusted fractional coordinates:", r3c.coords)
    #structure_r3c = build_unit_cell(r3c)
    #print("\nHexagonal (R-3c) cell structure:")
    #print(structure_r3c)
    #print(structure_r3c.get_neighbor_list(r=1.6))

    # exit(0)

    # from pymatgen.io.cif import CifWriter

    # # Assuming 'structure' is your pymatgen Structure object.
    # cw = CifWriter(structure_r3c)
    # cw.write_file("CO2_hexagonal.cif")

    # # Test Pna2_1 space group
    # pna21 = Pna21()
    # print(f"\nTesting {pna21.spacegroup} space group:")
    # structure_pna21 = build_unit_cell(pna21)
    # print("\nOrthorhombic (Pna2_1) cell structure:")
    # print(structure_pna21)
    # print("Initial fractional coordinates:", pna21.coords)
    # #r3c.adjust_fractional_coords(bond_length1=1.16,bond_length2=1.16, bond_angle_phi=45, bond_angle_theta=60)  # Bond length = 1.16 Å, angle = 30°
    # #print("Adjusted fractional coordinates:", r3c.coords)
    # #structure_r3c = build_unit_cell(r3c)
    # print("\nOrthorhombic (Pna2_1) cell structure:")
    # print(structure_pna21)
    # structure_pna21.get_space_group_info()
    # print("Space group info:", structure_pna21.get_space_group_info())
    # #print(structure.get_neighbor_list()

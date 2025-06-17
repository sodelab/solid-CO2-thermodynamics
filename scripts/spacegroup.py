import math
import numpy as np
from pymatgen.core.lattice import Lattice
from pprint import pprint
#from pymatgen.core.structure import Structure

class SpaceGroup:
    def __init__(self, spacegroup, species, coords, lattice_params, angles=(90, 90, 90)):
        """
        Base class for space groups.

        Parameters:
            species (list): List of atomic species.
            coords (list): Fractional coordinates for the asymmetric unit.
            lattice_params (dict): Lattice parameters (a, b, c).
            angles (tuple): Lattice angles (alpha, beta, gamma).
        """
        self.species = species
        self.coords = coords
        self.lattice_params = lattice_params
        self.angles = angles
        self.spacegroup = spacegroup  # Placeholder for space group information
       
    def build_lattice(self):
        """Constructs a lattice using the lattice parameters and angles."""
        a = self.lattice_params.get("a")
        b = self.lattice_params.get("b", a)
        c = self.lattice_params.get("c", a)
        alpha, beta, gamma = self.angles
        return Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    
    def update_lattice_params(self, **kwargs):
        """
        Updates the lattice parameters dynamically.

        Parameters:
            kwargs: Dictionary of lattice parameters to update (e.g., a=4.5, b=4.8, c=6.0).
        """
        for key, value in kwargs.items():
            if key in self.lattice_params:
                self.lattice_params[key] = value
        #print(f"Updated lattice parameters for {self.spacegroup}: {self.lattice_params}")

    def get_coordinate_data(self):
        """
        Returns important data about the coordinates,
        such as computed bond lengths and angles.
        
        Each space group may have its own implementation.
        """
        raise NotImplementedError("Subclasses must implement get_coordinate_data.")

class Pa3(SpaceGroup):
    def __init__(self, a=5.5, bond_length=None):
        super().__init__(
            species=["C", "O"],
            coords=[
                [0.00, 0.00, 0.00],  # Carbon
                [0.12, 0.12, 0.12],  # Oxygen
            ],
            lattice_params={"a": a},  # Default cubic lattice parameter
            spacegroup = "Pa-3"
        )
        if bond_length is not None:
            self.adjust_fractional_coords(bond_length)
            
    def build_lattice(self):
        a = self.lattice_params.get("a")
        return Lattice.cubic(a)
    
    def adjust_fractional_coords(self, bond_length):
        """
        Adjusts the fractional coordinates of the oxygen atom to achieve the desired bond length.
        For Pa-3 the adjustments are done symmetrically along x, y, and z.
        """
        lattice = self.build_lattice()
        c_frac = self.coords[0]
        o_frac = self.coords[1]
        # Convert to Cartesian coordinates.
        c_cart = lattice.get_cartesian_coords(c_frac)
        o_cart = lattice.get_cartesian_coords(o_frac)
        # Calculate the current bond length.
        current_bond_length = math.sqrt(sum((c_cart[i] - o_cart[i])**2 for i in range(3)))
        scaling_factor = bond_length / current_bond_length
        # Update oxygen fractional coordinates.
        self.coords[1] = [ c_frac[i] + scaling_factor*(o_frac[i] - c_frac[i]) for i in range(3) ]

class Cmce(SpaceGroup):
    def __init__(self, a=5.3, b=4.8, c=6.6, bond_length=None, bond_angle=None):
        super().__init__(
            species=["C", "O"],
            coords=[
                [0.00, 0.00, 0.00],  # Carbon
                [0.00, 0.50, 0.90],  # Oxygen
            ],
            lattice_params={"a": a, "b": b, "c": c},  # Orthorhombic lattice
            spacegroup = "Cmce"
        )
        if bond_length is not None and bond_angle is not None:
            self.adjust_fractional_coords(bond_length, bond_angle)

    def build_lattice(self):
        a = self.lattice_params.get("a")
        b = self.lattice_params.get("b")
        c = self.lattice_params.get("c")
        return Lattice.orthorhombic(a, b, c)
    
    def adjust_fractional_coords(self, bond_length, bond_angle):
        """
        Adjusts the oxygen fractional coordinates based on the desired bond length and bond angle.
        The angle is measured relative to the z-axis; only the y and z coordinates are adjusted.
        """
        lattice = self.build_lattice()
        c_frac = self.coords[0]
        o_frac = self.coords[1]
        angle_rad = math.radians(bond_angle)
        z_disp = bond_length * math.cos(angle_rad)
        y_disp = bond_length * math.sin(angle_rad)
        z_frac_disp = z_disp / lattice.c
        y_frac_disp = y_disp / lattice.b
        # Adjust oxygen fractional coordinates (only y and z).
        self.coords[1][1] = c_frac[1] + y_frac_disp
        self.coords[1][2] = c_frac[2] + z_frac_disp

class P42mnm(SpaceGroup):
    def __init__(self, a=5.3, c=6.6, bond_length=None, bond_angle=None):
        super().__init__(
            species=["C", "O"],
            coords=[
                [0.50, 0.50, 0.50],  # Carbon
                [0.32, 0.32, 0.50],  # Oxygen
            ],
            lattice_params={"a": a, "c": c},  # Tetragonal lattice
            spacegroup = "P42/mnm"
        )
        if bond_length is not None and bond_angle is not None:
            self.adjust_fractional_coords(bond_length, bond_angle)
    
    def build_lattice(self):
        a = self.lattice_params.get("a")
        c = self.lattice_params.get("c")
        return Lattice.tetragonal(a, c)
    
    def adjust_fractional_coords(self, bond_length):
        """
        Adjusts the oxygen fractional coordinates based on the desired bond length and bond angle.
        The angle is measured relative to the x-axis in the x-y plane.
        """
        lattice = self.build_lattice()
        c_frac = self.coords[0]
        o_frac = self.coords[1]
        # Convert to Cartesian coordinates.
        c_cart = lattice.get_cartesian_coords(c_frac)
        o_cart = lattice.get_cartesian_coords(o_frac)
        # Calculate the current bond length.
        current_bond_length = math.sqrt(sum((c_cart[i] - o_cart[i])**2 for i in range(3)))
        scaling_factor = bond_length / current_bond_length
        # Update oxygen fractional coordinates.
        self.coords[1][0] = c_frac[0] + scaling_factor*(o_frac[0] - c_frac[0])
        self.coords[1][1] = c_frac[1] + scaling_factor*(o_frac[1] - c_frac[1])
    
class R3c(SpaceGroup):
    def __init__(self, a=8.628, c=10.604,
                 bond_length1=None, 
                 bond_length2=None,
                 bond_angle_phi=None,  
                bond_angle_theta=None):
        super().__init__(
            species=["C", "O", "C", "O"],
            coords=[
                [0, 0, 0],  # Carbon
                [0.0, 0.0, 0.108],  # Oxygen
                [0.0, 0.250, 0.250],  # Carbon
                [0.142, 0.320, 0.208]  # Oxygen

                #[0, 0, 0],  # Carbon
                #[0.0, 0.0, 0.108],  # Oxygen
                #[0.291, 0.00, 0.250],  # Carbon
                #[0.291, -0.127, 0.287]  # Oxygen

                #[0.667, 0.333, 0.333],  # Carbon
                #[0.667, 0.333, 0.237],  # Oxygen
                #[0.081, 0.415, 0.417],  # Carbon
                #[0.478, 0.018, 0.121]  # Oxygen
            ],
            lattice_params={"a": a, "c": c},  # Trigonal lattice
            #lattice_angles={"alpha": 90, "gamma": 120},  # Trigonal lattice
            spacegroup = "R-3c"
        )
        if (bond_length1 is not None and 
            bond_length2 is not None and 
            bond_angle_phi is not None and 
            bond_angle_theta is not None):
            self.adjust_fractional_coords(bond_length1, bond_length2, bond_angle_phi, bond_angle_theta)
    
    def build_lattice(self):
        a = self.lattice_params.get("a")
        c = self.lattice_params.get("c")
        return Lattice.hexagonal(a, c)
    
    def adjust_fractional_coords2(self, bond_length1, bond_length2, bond_angle_phi, bond_angle_theta):
        """
        Adjusts the oxygen fractional coordinates based on the desired bond lengths and angles.
        
        For the first oxygen coordinate (o_frac1 relative to c_frac1):
          - Modify only the z-direction. New z = c_frac1_z + (bond_length1 / lattice.c)
        
        For the second oxygen coordinate (o_frac2 relative to c_frac2):
          - Compute a Cartesian offset using spherical coordinates:
            offset_x = bond_length2 * sin(theta) * cos(phi)
            offset_y = bond_length2 * sin(theta) * sin(phi)
            offset_z = bond_length2 * cos(theta)
          - Convert the Cartesian offset to fractional coordinates and then add it to c_frac2.
        """
        lattice = self.build_lattice()
        
        # --- First oxygen (o_frac1) adjustment ---
        c_frac1 = self.coords[0]
        o_frac1 = self.coords[1]
        # Scale only along z-direction.
        new_o1_z = c_frac1[2] + bond_length1 / lattice.c
        # Keep x and y of the carbon reference.
        self.coords[1] = [c_frac1[0], c_frac1[1], new_o1_z]
        
        # --- Second oxygen (o_frac2) adjustment ---
        # Step 1: Get the Cartesian coordinate of the reference atom.
        c_frac2 = self.coords[2]
        o_frac2 = self.coords[3]
        c_cart = lattice.get_cartesian_coords(c_frac2)
        
        # Step 2: Convert angles to radians.
        phi_rad = math.radians(bond_angle_phi)
        theta_rad = math.radians(bond_angle_theta-30)
        
        # Step 3: Compute the offset in Cartesian coordinates.
        offset_cart = np.array([
            bond_length2 * math.sin(theta_rad) * math.sin(phi_rad),
            bond_length2 * math.sin(theta_rad) * math.cos(phi_rad),
            bond_length2 * math.cos(theta_rad)
            
            
        ])
        
        #print("Offset (Cartesian):", offset_cart)
        #print("Magnitude:", np.linalg.norm(offset_cart))
        # Step 4: Add offset to the reference Cartesian coordinate.
        o_cart = c_cart + offset_cart
        
        # Step 5: Convert back to fractional coordinates.
        o_frac2 = lattice.get_fractional_coords(o_cart)

        self.coords[3] = o_frac2

        """"
        c_frac2 = self.coords[2]
        o_frac2 = self.coords[3]
        # Convert the input angles from degrees to radians.
        phi_rad = math.radians(bond_angle_phi)
        theta_rad = math.radians(bond_angle_theta)
        # Build the offset vector in Cartesian coordinates.
        offset_cart = [
            bond_length2 * math.sin(theta_rad) * math.cos(phi_rad),
            bond_length2 * math.sin(theta_rad) * math.sin(phi_rad),
            bond_length2 * math.cos(theta_rad)
        ]
        # Convert the Cartesian offset to fractional coordinates.
        frac_offset = lattice.get_fractional_coords(offset_cart)
        #frac_offset = np.linalg.solve(lattice.matrix.T, offset_cart)
        # Update oxygen 2's fractional coordinate.
        self.coords[3] = [c_frac2[i] + frac_offset[i] for i in range(3)]

        """

    def adjust_fractional_coords(self, bond_length1, x, y, z):

        lattice = self.build_lattice()
        
        # --- First oxygen (o_frac1) adjustment ---
        c_frac1 = self.coords[0]
        o_frac1 = self.coords[1]
        # Scale only along z-direction.
        new_o1_z = c_frac1[2] + bond_length1 / lattice.c
        # Keep x and y of the carbon reference.
        self.coords[1] = [c_frac1[0], c_frac1[1], new_o1_z]
        
        # --- Second oxygen (o_frac2) adjustment ---
        # Step 1: Get the Cartesian coordinate of the reference atom.
        c_frac2 = self.coords[2]

        # Update oxygen position by adding the displacements directly in fractional space
        self.coords[3] = [
            c_frac2[0] + x,  # Add x displacement
            c_frac2[1] + y,  # Add y displacement
            c_frac2[2] + z   # Add z displacement
        ]



class Pna21(SpaceGroup):
    def __init__(self, a=3.34, b=5.29, c=9.73, bond_length=None, bond_angle_phi=None, bond_angle_theta=None):
        super().__init__(
            species=["C", "O", "O"],
            coords=[
                [0.118, 0.103, 0.626],  # Carbon
                [0.266, 0.026, 0.526],  # Oxygen
                [0.969, 0.179, 0.727],  # Oxygen

                #[0.118, 0.626, 0.103],  # Carbon
                #[0.266, 0.526, 0.026],  # Oxygen
                #[0.969, 0.727, 0.179],  # Oxygen

            ],
            lattice_params={"a": a, "b": b, "c": c},  # Orthorhombic lattice
            spacegroup = "Pna2_1"
        )
        if bond_length is not None and bond_angle_phi is not None and bond_angle_theta is not None:
            self.adjust_fractional_coords(bond_length, bond_angle_phi, bond_angle_theta)
    
    def build_lattice(self):
        a = self.lattice_params.get("a")
        b = self.lattice_params.get("b")
        c = self.lattice_params.get("c")
        return Lattice.orthorhombic(a, b, c)
    
    def adjust_fractional_coords(self, bond_length, bond_angle_phi, bond_angle_theta):
        """
        Adjusts the oxygen fractional coordinates based on the desired bond length and bond angle.
        The angle is measured relative to the x-axis in the x-y plane.
        """
        lattice = self.build_lattice()
        c_frac = self.coords[0]
        o_frac1 = self.coords[1]
        o_frac2 = self.coords[2]
        # Convert angles to radians.
        phi_rad = math.radians(bond_angle_phi)
        theta_rad = math.radians(bond_angle_theta)

        offset_cart = [
            bond_length * math.sin(theta_rad) * math.cos(phi_rad),
            bond_length * math.sin(theta_rad) * math.sin(phi_rad),
            bond_length * math.cos(theta_rad),
        ]
        frac_offset = lattice.get_fractional_coords(offset_cart)
        self.coords[1] = [c_frac[i] + frac_offset[i] for i in range(3)]
        self.coords[2] = [c_frac[i] - frac_offset[i] for i in range(3)]

    def get_coordinate_data(self):
        lattice = self.build_lattice()
        c_cart = lattice.get_cartesian_coords(self.coords[0])
        o_cart = lattice.get_cartesian_coords(self.coords[1])
        bond_vector = np.array(o_cart) - np.array(c_cart)
        bond_length = np.linalg.norm(bond_vector)
        if bond_length != 0:
            bond_angle_theta = math.degrees(math.acos(bond_vector[2] / bond_length))
            bond_angle_phi = math.degrees(math.atan2(bond_vector[1], bond_vector[0]))
        else:
            bond_angle = 0.0
            bond_phi = 0.0
        return {"bond_length": bond_length, "bond_angle_phi": bond_angle_phi, "bond_angle_theta": bond_angle_theta}

        
if __name__ == "__main__":
    # Test Pa-3 space group
    pa3 = Pa3()
    print(f"\nTesting {pa3.spacegroup} space group:")
    print("Initial lattice parameters:", pa3.lattice_params)
    print("Lattice:")
    print(pa3.build_lattice())  # Use initial lattice parameters
    print("Angles:", pa3.angles)

    print("\nLattice with updated parameter (a=4.111):")
    pa3_updated = Pa3(a=4.111)  # Create a new instance with updated lattice parameter
    print(pa3_updated.build_lattice())
    print("Updated lattice parameters:", pa3_updated.lattice_params)
    print("Angles:", pa3_updated.angles)

    # Test Cmce space group
    cmce = Cmce()
    print(f"\nTesting {cmce.spacegroup} space group:")
    print("Initial lattice parameters:", cmce.lattice_params)
    print("Lattice:")
    print(cmce.build_lattice())  # Use initial lattice parameters
    print("Angles:", cmce.angles)

    print("\nLattice with updated parameters (a=5.0, b=4.5, c=6.0):")
    cmce_updated = Cmce(a=5.0, b=4.5, c=6.0)  # Create a new instance with updated lattice parameters
    print(cmce_updated.build_lattice())
    print("Updated lattice parameters:", cmce_updated.lattice_params)
    print("Angles:", cmce_updated.angles)

    # Test P42/mnm space group
    p42mnm = P42mnm()
    print(f"\nTesting {p42mnm.spacegroup} space group:")
    print("Initial lattice parameters:", p42mnm.lattice_params)
    print("Lattice:")
    print(p42mnm.build_lattice())  # Use initial lattice parameters
    print("Angles:", p42mnm.angles)

    print("\nLattice with updated parameters (a=4.5, c=6.2):")
    p42mnm_updated = P42mnm(a=4.5, c=6.2)  # Create a new instance with updated lattice parameters
    print(p42mnm_updated.build_lattice())
    print("Updated lattice parameters:", p42mnm_updated.lattice_params)
    print("Angles:", p42mnm_updated.angles)

    # Test Pa-3 space group
    pa3 = Pa3()
    print(f"\nTesting {pa3.spacegroup} space group:")
    print("Initial lattice parameters:", pa3.lattice_params)
    print("Lattice:")
    print(pa3.build_lattice())
    pa3.update_lattice_params(a=4.5)  # Update lattice parameter a
    print("Updated lattice parameters:", pa3.lattice_params)
    print("Lattice:")
    print(pa3.build_lattice())

    # Test Cmce space group
    cmce = Cmce()
    print(f"\nTesting {cmce.spacegroup} space group:")
    print("Initial lattice parameters:", cmce.lattice_params)
    print("Lattice:")
    print(cmce.build_lattice())
    cmce.update_lattice_params(a=5.0, b=4.5, c=6.0)  # Update lattice parameters
    print("Updated lattice parameters:", cmce.lattice_params)
    print("Lattice:")
    print(cmce.build_lattice())

    # Test P42/mnm space group
    p42mnm = P42mnm()
    print(f"\nTesting {p42mnm.spacegroup} space group:")
    print("Initial lattice parameters:", p42mnm.lattice_params)
    print("Lattice:")
    print(p42mnm.build_lattice())
    p42mnm.update_lattice_params(a=4.5, c=6.2)  # Update lattice parameters
    print("Updated lattice parameters:", p42mnm.lattice_params)
    print("Lattice:")
    print(p42mnm.build_lattice())
    
    # Test P42/mnm space group
    r3c = R3c()
    print(f"\nTesting {r3c.spacegroup} space group:")
    print("Initial lattice parameters:", r3c.lattice_params)
    print("Lattice:")
    print(r3c.build_lattice())

    # Test Pna2_1 space group
    pna21 = Pna21()
    print(f"\nTesting {pna21.spacegroup} space group:")
    print("Initial lattice parameters:", pna21.lattice_params)
    print("Lattice:")
    print(pna21.build_lattice())
    print(pna21.get_coordinate_data)
    data = pna21.get_coordinate_data()
    print(f"\nCoordinate Data for {pna21.spacegroup}:")
    pprint(data)

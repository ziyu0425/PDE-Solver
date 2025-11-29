# pde_schema.py
"""
PDE Parameter Schema
--------------------
Data structures for representing parsed PDE information from natural language.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal


@dataclass
class PDEParameters:
    """
    Structured representation of a PDE problem extracted from natural language.
    """
    # PDE Type
    pde_type: Literal["heat", "wave", "advection", "poisson", "elasticity", "other"] = "heat"
    
    # Dimension
    dim: Literal[1, 2, 3] = 1
    
    # Domain geometry
    domain_size: Dict[str, float] = field(default_factory=dict)  # e.g., {"length": 2.0} for 1D, {"Lx": 1.0, "Ly": 1.0} for 2D, {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2} for 3D
    geometry_type: Optional[str] = None  # "box", "cylinder", "sphere", "cube", "column" (will be normalized)
    geometry_params: Dict[str, float] = field(default_factory=dict)  # e.g., {"cylinder_radius": 0.5}, {"sphere_radius": 1.0}, {"r_inner": 0.1, "r_outer": 1.0}
    coordinate_system: Optional[str] = None  # "cartesian", "cylindrical", "spherical" (inferred from geometry_type)
    
    # Spatial discretization
    nx: Optional[int] = None
    ny: Optional[int] = None
    nz: Optional[int] = None
    
    # Material/Physical parameters
    diffusivity: Optional[float] = None  # for heat equation
    wave_speed: Optional[float] = None   # for wave equation
    young_modulus: Optional[float] = None  # E (Pa) for elasticity
    poisson_ratio: Optional[float] = None  # nu for elasticity
    density: Optional[float] = None  # rho (kg/m³) for elasticity (for dynamic problems)
    material_params: Dict[str, float] = field(default_factory=dict)  # generic material parameters
    
    # Composite material parameters (for heat equation with high-conductivity core)
    core_radius: Optional[float] = None  # Radius of high-conductivity core (for cylindrical geometries, typically equals r1 for hollow cylinders)
    core_diffusivity: Optional[float] = None  # Diffusivity of core material (should be higher than base diffusivity)
    
    # Boundary conditions
    bc_type: Literal["dirichlet", "neumann", "robin", "mixed"] = "dirichlet"
    bc_values: Dict[str, Any] = field(default_factory=dict)  # e.g., {"T_left": 20.0, "T_right": 0.0} for 1D
    
    # Initial condition
    initial_type: Optional[str] = None  # "constant", "zero", "cosine", "sine"
    initial_value: Optional[float] = None  # for constant type
    initial_function: Optional[str] = None  # for cosine/sine: e.g., "cos(x)" or "sin(x)"
    initial_amplitude: Optional[float] = None  # amplitude for cosine/sine (default: 1.0)
    initial_wavenumber: Optional[float] = None  # wavenumber for cosine/sine (default: 1.0)
    
    # Source term
    source_type: Optional[str] = None  # "none" or "constant"
    source_value: Optional[float] = None  # value for constant source
    
    # Steady-state mode
    steady: Optional[bool] = None  # if True, solve steady-state instead of transient
    
    # Time discretization
    dt: Optional[float] = None
    num_steps: Optional[int] = None
    total_time: Optional[float] = None  # alternative to num_steps
    
    # Additional metadata
    field_name: str = "temperature"
    unit: str = "°C"
    notes: List[str] = field(default_factory=list)  # any notes or assumptions made
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                result[key] = value
            elif isinstance(value, (dict, list)):
                result[key] = value
            else:
                result[key] = str(value)
        return result


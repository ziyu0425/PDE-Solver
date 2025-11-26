# pde_parser_agent.py
"""
PDE Parser Agent
----------------
Parses natural language descriptions of PDE problems into structured PDEParameters.
"""

import json
import re
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from pde_schema import PDEParameters


def normalize_key(key: str) -> str:
    """
    Normalize JSON keys to snake_case to match PDEParameters dataclass fields.
    
    Examples:
    - "PDE Type" -> "pde_type"
    - "domain_size" -> "domain_size"
    - "bcValues" -> "bc_values"
    - "Initial Value" -> "initial_value"
    """
    # Convert to lowercase and replace spaces/other separators with underscores
    key = re.sub(r'[-\s]+', '_', key.lower().strip())
    # Remove any non-alphanumeric/underscore characters except underscores
    key = re.sub(r'[^a-z0-9_]', '', key)
    # Remove leading/trailing underscores
    key = key.strip('_')
    
    # Handle common variations and aliases
    key_mappings = {
        'pde_type': 'pde_type',
        'pde': 'pde_type',
        'type': 'pde_type',
        'dimension': 'dim',
        'dim': 'dim',
        'domain_size': 'domain_size',
        'domain': 'domain_size',
        'geometry': 'domain_size',
        'boundary_conditions': 'bc_values',
        'bc_values': 'bc_values',
        'bc': 'bc_values',
        'boundary_values': 'bc_values',
        'boundary': 'bc_values',
        'boundary_type': 'bc_type',
        'bc_type': 'bc_type',
        'initial_condition': 'initial_value',
        'initial_value': 'initial_value',
        'initial': 'initial_value',
        'ic': 'initial_value',
        'time_step': 'dt',
        'dt': 'dt',
        'delta_t': 'dt',
        'timestep': 'dt',
        'num_steps': 'num_steps',
        'number_of_steps': 'num_steps',
        'steps': 'num_steps',
        'total_time': 'total_time',
        'time': 'total_time',
        'field_name': 'field_name',
        'field': 'field_name',
        'unit': 'unit',
        'units': 'unit',
        'source_type': 'source_type',
        'source': 'source_type',
        'heat_source_type': 'source_type',
        'source_value': 'source_value',
        'heat_source_value': 'source_value',
        'source_strength': 'source_value',
        'steady': 'steady',
        'steady_state': 'steady',
        'equilibrium': 'steady',
        'length': 'domain_size',  # Special: length should go into domain_size for 1D
        'l': 'domain_size',
        'lx': 'domain_size',
        'ly': 'domain_size',
        'lz': 'domain_size',
    }
    
    return key_mappings.get(key, key)


def normalize_json_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively normalize all keys in a dictionary to snake_case.
    Also unwraps nested values that are unnecessarily wrapped.
    """
    normalized = {}
    for key, value in data.items():
        new_key = normalize_key(key)
        
        # Recursively normalize nested dictionaries
        if isinstance(value, dict):
            normalized[new_key] = normalize_json_keys(value)
        elif isinstance(value, list):
            # Normalize keys in list of dicts
            normalized[new_key] = [
                normalize_json_keys(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            normalized[new_key] = value
    
    # Post-process to unwrap nested structures
    return unwrap_nested_structures(normalized)


def unwrap_nested_structures(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unwrap nested structures where values are wrapped in dictionaries unnecessarily.
    Examples:
    - {"initial_value": {"initial_value": 10}} -> {"initial_value": 10}
    - {"bc_values": {"bc_values": {...}}} -> {"bc_values": {...}}
    """
    unwrapped = {}
    
    for key, value in data.items():
        if key == 'initial_value' and isinstance(value, dict):
            # If initial_value is a dict, try to unwrap it
            if 'initial_value' in value:
                # Unwrap: {"initial_value": 10} -> 10
                unwrapped_value = value['initial_value']
                if isinstance(unwrapped_value, (int, float)):
                    unwrapped[key] = unwrapped_value
                else:
                    unwrapped[key] = unwrapped_value
            else:
                # If it's a dict but doesn't have 'initial_value' key, check if it's a single primitive value
                if len(value) == 1:
                    single_value = list(value.values())[0]
                    if isinstance(single_value, (int, float, str)):
                        unwrapped[key] = single_value
                    else:
                        unwrapped[key] = value
                else:
                    unwrapped[key] = value
        elif key == 'bc_values' and isinstance(value, dict):
            # If bc_values contains nested 'bc_values', unwrap it
            if 'bc_values' in value:
                # Unwrap: {"bc_type": "dirichlet", "bc_values": {...}} -> {...}
                unwrapped[key] = value['bc_values']
            else:
                unwrapped[key] = value
        else:
            unwrapped[key] = value
    
    return unwrapped


PDE_PARSER_SYSTEM_PROMPT = """
You are an expert PDE (Partial Differential Equation) parser that extracts structured information from natural language descriptions of physical problems.

Your task is to parse natural language and extract the following structured information.

IMPORTANT: If the user query references a previous simulation (e.g., "change X", "same simulation", "modify Y"), 
use the context information provided. Only extract the CHANGES from the user's request, and infer the rest from context.
When in doubt, use reasonable defaults but explain your choices.

--------------------------------------------------
FOLLOW-UP QUESTIONS & CONTEXT
--------------------------------------------------
CRITICAL: When previous simulation parameters are provided in the context:
This means the user wants to MODIFY the previous simulation, NOT create a new one.

RULES FOR FOLLOW-UPS:
1. PRESERVE THE PROBLEM STRUCTURE: 
   - DO NOT change pde_type, dim, domain_size unless explicitly mentioned
   - If previous was 2D, keep it 2D unless user says "1D" or "3D"
   - If previous had domain_size, keep it unless user mentions new dimensions

2. EXTRACT ONLY CHANGES:
   - Return ONLY the parameters explicitly mentioned in the user's request
   - For ALL other parameters, return null/None/empty (they will be preserved)
   - DO NOT include domain_size, dim, pde_type unless they are mentioned

3. EXAMPLES:
   - User: "add a heat source of 10" (previous: 2D problem)
     → Return ONLY: {"source_type": "constant", "source_value": 10.0}
     → DO NOT return pde_type, dim, domain_size (preserve 2D)
   
   - User: "Change the left boundary to 50°C" (previous: 1D problem)
     → Return ONLY: {"bc_values": {"t_left": 50}}
     → Preserve dimension, domain, etc.
   
   - User: "Different initial temperature of 100°C"
     → Return ONLY: {"initial_value": 100}
     → Preserve everything else
   
   - User: "Same simulation" or "Run again"
     → Return empty JSON {} or minimal structure with only required fields

4. If NO previous parameters are provided (new query), parse everything from scratch with defaults

5. The system will automatically merge your extracted changes with previous parameters

--------------------------------------------------
STRUCTURED INFORMATION TO EXTRACT
--------------------------------------------------

1. **PDE Type** (CRITICAL - MUST be detected correctly):
   - "heat": Heat/diffusion equation (u_t - k*Δu = 0)
     * Keywords: "heat", "temperature", "thermal", "diffusion", "conduction"
   - "elasticity": Linear elasticity / solid mechanics
     * Keywords: "elasticity", "elastic", "stress", "strain", "displacement", "deformation", 
                "Young's modulus", "Poisson's ratio", "mechanical", "solid mechanics",
                "plate", "beam", "rod" (when combined with stress/strain/modulus)
     * If you see "Young's modulus", "E =", "Poisson's ratio", "nu =", "stress", "strain" → MUST set pde_type="elasticity"
   - "wave": Wave equation (u_tt - c²*Δu = 0)
   - "advection": Advection equation
   - "poisson": Poisson equation (steady state)
   - "other": Any other PDE type
   
   CRITICAL: If the user mentions "elasticity", "stress", "strain", "Young's modulus", or similar mechanics terms, 
   you MUST set pde_type="elasticity", NOT "heat"!

2. **Dimension**: Determine the spatial dimension (1, 2, or 3)

3. **Domain Geometry**: Extract domain size:
   - 1D: {"length": L} or {"L": L}
   - 2D: {"Lx": Lx, "Ly": Ly} or {"width": W, "height": H}
   - 3D: {"Lx": Lx, "Ly": Ly, "Lz": Lz} or {"width": W, "height": H, "depth": D}
   - CRITICAL FOR 3D: When multiple dimensions are mentioned (e.g., "1m*0.2m*0.2"), extract ALL THREE as {"Lx": first, "Ly": second, "Lz": third}

4. **Spatial Discretization**: Extract or infer grid resolution:
   - nx, ny, nz (number of grid points in each direction)

5. **Material/Physical Parameters**:
   - For heat: diffusivity (thermal diffusivity κ)
   - For wave: wave_speed (wave speed c)
   - For elasticity:
     * young_modulus: E (Pa or GPa) - Young's modulus, e.g., 210e9 Pa for steel
     * poisson_ratio: nu - Poisson's ratio, e.g., 0.3 for steel
     * density: rho (kg/m³) - Material density (for dynamic problems)
     * material_params: additional parameters:
       - "area" or "cross_sectional_area" (1D): cross-sectional area (m²)
       - "body_force" or "body_force_x" (1D): body force per unit length (N/m)
       - "body_fx", "body_fy", "body_fz" (2D/3D): body force components (N/m³)
       - "quantity": "stress" or "strain" - output field type
       - "plane_stress" (2D): bool - True for plane stress, False for plane strain
   - Other parameters in material_params dict

6. **Boundary Conditions**:
   - bc_type: "dirichlet", "neumann", "robin", or "mixed"
   - bc_values: dictionary of boundary values (use lowercase keys: "t_left", "t_right", "t_boundary")
     * 1D: {"t_left": value, "t_right": value} (required: both left and right must differ for visible evolution)
     * 2D/3D: {"t_boundary": value}
   - IMPORTANT: For 1D problems, ensure t_left ≠ t_right to create a gradient for time evolution

7. **Initial Condition** (CRITICAL - Parse in this order):
   - initial_type: "constant" (default), "zero", "cosine", or "sine"
     * "constant": u(x,0) = initial_value (uniform constant value)
     * "zero": u(x,0) = 0 (zero everywhere)
     * "cosine": u(x,0) = initial_amplitude * cos(initial_wavenumber * x)
     * "sine": u(x,0) = initial_amplitude * sin(initial_wavenumber * x)
   - initial_value: uniform initial value (for constant type)
   - initial_amplitude: amplitude for cosine/sine (default: 1.0)
   - initial_wavenumber: wavenumber for cosine/sine (default: 1.0)
   - CRITICAL PARSING ORDER (check in this sequence, FIRST match wins):
     1. Check for "cosine" or "cos" ANYWHERE in initial condition description → initial_type="cosine":
        - Examples: "cosine function", "cosine initial", "initial temperature is a cosine function", "cos initial condition"
        - If "cosine" or "cos" is found, ALWAYS set initial_type="cosine", even if "initial temperature" or "initial value" is also mentioned
        - MUST extract amplitude if mentioned:
          * "amplitude X" or "amplitude of X" or "amplitude=X" → initial_amplitude=X
          * "initial temperature is a cosine function with an amplitude of 10" → initial_type="cosine", initial_amplitude=10.0
          * "cosine with amplitude 10" → initial_type="cosine", initial_amplitude=10.0
        - If amplitude not mentioned, use default 1.0
        - Extract wavenumber if mentioned (e.g., "wavenumber k", "frequency k"), otherwise use default 1.0
     2. Check for "sine" or "sin" ANYWHERE in initial condition description → initial_type="sine" (same amplitude/wavenumber extraction as cosine)
     3. Check for "zero initial" or "initial condition 0" → initial_type="zero", initial_value=0
     4. Check for "initial temperature X" or "initial value X" (when NO cosine/sine/zero) → initial_type="constant", initial_value=X
     5. If NO initial condition mentioned → initial_type="constant", initial_value=0.0 (default)

8. **Source Term (IMPORTANT - Extract if mentioned)**:
   - source_type: "none" (default if not mentioned) or "constant" (if source is mentioned)
   - source_value: float value for constant source (extract the numerical value)
   - CRITICAL: If the user mentions ANY of these phrases, extract source_type="constant" and source_value:
     * "heat source of X" → source_type="constant", source_value=X
     * "heat source X" → source_type="constant", source_value=X
     * "source of X" → source_type="constant", source_value=X
     * "internal heat generation of X" → source_type="constant", source_value=X
     * "heating of X" → source_type="constant", source_value=X
     * "with a source X" → source_type="constant", source_value=X
   - If NO source is mentioned → source_type="none", source_value=0.0 (default)
   - Examples:
     * "simulate heat transfer with a heat source of 50" → source_type="constant", source_value=50.0
     * "1D rod with internal heating of 100" → source_type="constant", source_value=100.0
     * "heat equation" (no mention of source) → source_type="none", source_value=0.0

9. **Steady-State Mode** (for heat equation):
   - steady: bool, if True solve steady-state instead of transient
   - Examples: "steady-state", "steady state", "equilibrium", "find the equilibrium solution" → steady=True
   - Default: steady=False (transient mode)
   - Note: Elasticity problems are always static (steady)

10. **Elasticity-Specific Parameters** (when pde_type="elasticity"):
   - Extract young_modulus (E) if mentioned:
     * "Young's modulus of X Pa" → young_modulus=X
     * "E = X GPa" → young_modulus=X*1e9
     * "steel" → young_modulus=210e9 (typical steel value)
     * "aluminum" → young_modulus=70e9
     * Default: young_modulus=210e9 (steel)
   - Extract poisson_ratio (nu) if mentioned:
     * "Poisson's ratio of X" → poisson_ratio=X
     * "nu = X" → poisson_ratio=X
     * Default: poisson_ratio=0.3
   - Extract body forces:
     * "body force of X" (1D) → material_params["body_force"]=X
     * "body force (X, Y)" (2D) → material_params["body_fx"]=X, material_params["body_fy"]=Y
     * "gravity" or "with gravity" → apply gravitational body force:
       - ALWAYS set material_params["gravity"] = True (flag to indicate gravity should be applied)
       - For 2D problems: Gravity acts in the y-direction (downward, negative y)
         * material_params["body_fy"] = -9.81 * density (default density ≈ 7800 kg/m³ for steel)
         * Note: In 2D plane stress/strain, gravity acts in-plane in the y-direction.
           This models a vertical plate or beam. For a horizontal plate, gravity would act out-of-plane (z), 
           which is not captured in 2D - use 3D instead.
       - For 3D problems: material_params["body_fz"] = -9.81 * density (gravity acts in z-direction)
       - The dispatcher will automatically compute body_fy/body_fz from density and gravity if the flag is set
     * If no body forces mentioned, default to 0.0 (no loads) - solution will be zero with clamped boundaries
   - Extract output quantity:
     * "stress" → material_params["quantity"]="stress"
     * "strain" → material_params["quantity"]="strain"
     * Default: "stress"
   - For 2D elasticity:
     * "plane stress" → material_params["plane_stress"]=True
     * "plane strain" → material_params["plane_stress"]=False
     * Default: plane_stress=True

11. **Time Discretization** (for time-dependent problems, when steady=False):
   - dt: time step
   - num_steps: number of time steps
   - total_time: total simulation time (if num_steps not specified)
   - Note: Only used for heat/wave equations, not for elasticity (static problems)

12. **Metadata**:
   - field_name: name of the field (e.g., "temperature", "displacement", "stress", "strain")
   - unit: unit of the field (e.g., "°C", "m", "Pa", "-")
   - notes: list of assumptions or inferences made

IMPORTANT GUIDELINES:

- If information is not specified, make REASONABLE inferences based on context

CRITICAL REQUIREMENTS:

1. **Domain Size (MANDATORY)**: 
   - For 1D: MUST extract "length" or "L" from the description (e.g., "2 meter rod" → {"length": 2.0})
     * CRITICAL: If user says "X thick" or "X thickness", this is a material property (thickness), NOT the length
     * Examples:
       - "30nm thick conductor" → length should use DEFAULT (2.0 m), thickness info is material property
       - "2 meter rod" → {"length": 2.0}
       - "conductor 30nm thick" → length should use DEFAULT (2.0 m), do NOT use 30nm as length
     * If only thickness is mentioned (e.g., "30nm thick"), use default length (2.0 m) and note thickness in notes
   - For 2D: MUST extract as {"Lx": Lx, "Ly": Ly} (e.g., "1m x 1m plate" → {"Lx": 1.0, "Ly": 1.0})
     * CRITICAL: Extract BOTH dimensions separately when "x" or "*" separator is present
     * Patterns to recognize:
       - "1m x 1m plate" → {"Lx": 1.0, "Ly": 1.0}
       - "1m x 0.5m" → {"Lx": 1.0, "Ly": 0.5}
       - "1m*1m" → {"Lx": 1.0, "Ly": 1.0}
       - "width 1m height 0.5m" → {"Lx": 1.0, "Ly": 0.5}
     * Recognize "x", "*", "by", "×" as separators between dimensions
     * CRITICAL: If you see "1m x 1m", extract BOTH numbers:
       - First number (1m) → Lx = 1.0
       - Second number (1m) → Ly = 1.0
     * If only one dimension mentioned without separator (e.g., just "1m"), use it for both Lx and Ly
     * If "width" and "height" mentioned, map to Lx and Ly respectively
     * NEVER return {"domain_size": value} for 2D - ALWAYS return {"Lx": value1, "Ly": value2}
   - For 3D: MUST extract as {"Lx": Lx, "Ly": Ly, "Lz": Lz}
     * CRITICAL: Extract ALL THREE dimensions separately when multiple dimensions are mentioned
     * Patterns to recognize:
       - "1m x 0.2m x 0.2m" → {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2}
       - "1m*0.2m*0.2" → {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2}
       - "1m x 0.2m *0.2m" → {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2}
       - "width 1m height 0.2m depth 0.2m" → {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2}
     * Recognize "x", "*", "by", "×" as separators between dimensions
     * CRITICAL: If you see multiple numbers with units (like "1m*0.2m*0.2"), extract EACH as a separate dimension:
       - First number → Lx, Second number → Ly, Third number → Lz
     * If only one dimension mentioned, use it for all three
     * If "width", "height", "depth" mentioned, map to Lx, Ly, Lz respectively
     * NEVER return {"domain_size": value} for 3D - ALWAYS return {"Lx": value1, "Ly": value2, "Lz": value3}
   - If not mentioned, infer reasonable defaults: 1D → {"length": 2.0}, 2D → {"Lx": 1.0, "Ly": 1.0}, 3D → {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0}
   - CRITICAL: "thick" or "thickness" refers to material cross-section, NOT the domain length:
     * "30nm thick conductor" → length = 2.0 m (default), NOT 30nm
     * "2m long, 30nm thick" → length = 2.0 m, thickness is material property
     * Only use explicit length mentions like "2 meter", "2m long", "length 2m" as the domain length
   - CRITICAL: domain_size MUST be a dictionary with keys like "length", "Lx", "Ly", "Lz", NOT a nested structure like {"domain_size": value}

2. **Boundary Conditions**:
   - For 1D: MUST include both "t_left" and "t_right" in bc_values
   - Ensure t_left ≠ t_right to create a gradient (e.g., t_left=20, t_right=0)
   - Use lowercase keys in bc_values: "t_left", "t_right", "t_boundary"

3. **Initial Conditions**:
   - initial_value should be a single number (float/int), not a dictionary
   - Should differ from at least one boundary for visible evolution

4. **Time Discretization**:
   - For nanoscale problems (< 1 micrometer): Calculate t_char ≈ L²/(2×κ), use dt << t_char
   - For other scales: Use reasonable defaults (dt=0.01s, num_steps=50)

5. **Defaults if not specified**:
   * nx/ny/nz: 30-50 for 1D/2D, 10-20 for 3D
   * diffusivity (heat): 1e-5 m²/s for metals, 1e-6 for other materials
   * young_modulus (elasticity): 210e9 Pa (steel) if not specified
   * poisson_ratio (elasticity): 0.3 (steel) if not specified
   * dt: Calculate based on characteristic time or use 0.01s (heat only)
   * num_steps: 50-100 for transient problems (heat only)
   * source_type: "none" (if no source mentioned, heat only)
   * source_value: 0.0 (if no source mentioned, heat only)
   * steady: false (transient mode by default, heat only)
   * quantity (elasticity): "stress" if not specified

CRITICAL: Return your response as a JSON object with keys in snake_case format:
- Use lowercase letters only
- Separate words with underscores
- Examples: "pde_type" (not "PDE Type" or "pdeType"), "bc_values" (not "BC Values"), "initial_value" (not "Initial Value")

MANDATORY FIELDS - Always include these in your JSON response:
- initial_type: MUST be "constant", "zero", "cosine", or "sine" (default: "constant" if not mentioned, heat only)
- initial_value: MUST be a float (default: 0.0 if not mentioned, used only when initial_type="constant", heat only)
- initial_amplitude: MUST be a float (default: 1.0 if not mentioned, used when initial_type="cosine" or "sine", heat only)
- initial_wavenumber: MUST be a float (default: 1.0 if not mentioned, used when initial_type="cosine" or "sine", heat only)
- source_type: MUST be "none" or "constant" (default: "none" if not mentioned, heat only)
- source_value: MUST be a float (default: 0.0 if not mentioned, heat only)
- steady: MUST be a boolean (default: false if not mentioned, heat only)
- young_modulus: MUST be a float (default: 210e9 Pa if not mentioned, elasticity only)
- poisson_ratio: MUST be a float (default: 0.3 if not mentioned, elasticity only)

If source is mentioned in the description, you MUST extract:
  - source_type = "constant"
  - source_value = the numerical value mentioned

If source is NOT mentioned, you MUST include:
  - source_type = "none"
  - source_value = 0.0

CRITICAL: You MUST ALWAYS return a valid JSON object, never natural language text.
- If the input is unclear, incomplete, or interrupted, use reasonable defaults
- If the input is empty or meaningless, return a minimal valid JSON with defaults
- NEVER ask questions or request clarification in your response
- NEVER return explanatory text - ONLY return the JSON object
- If you cannot extract information, use the default values specified above

Example response format (ALWAYS return ONLY JSON, nothing else):

Example 1 - Heat 1D:
{
  "pde_type": "heat",
  "dim": 1,
  "domain_size": {"length": 2.0},
  "bc_values": {"t_left": 20.0, "t_right": 0.0},
  "initial_type": "constant",
  "initial_value": 0.0,
  "initial_amplitude": 1.0,
  "initial_wavenumber": 1.0,
  "source_type": "none",
  "source_value": 0.0,
  "steady": false,
  "field_name": "temperature",
  "unit": "°C",
  "notes": []
}

Example 1c - Heat 1D with cosine initial condition:
Input: "initial temperature is a cosine function with an amplitude of 10"
{
  "pde_type": "heat",
  "dim": 1,
  "domain_size": {"length": 2.0},
  "bc_values": {"t_left": 0.0, "t_right": 0.0},
  "initial_type": "cosine",
  "initial_value": 0.0,
  "initial_amplitude": 10.0,
  "initial_wavenumber": 1.0,
  "source_type": "none",
  "source_value": 0.0,
  "steady": false,
  "field_name": "temperature",
  "unit": "°C",
  "notes": []
}

Note: Even though the input says "initial temperature", because it ALSO says "cosine function", 
initial_type MUST be "cosine" (not "constant"), and the "10" is the amplitude (not initial_value).

Example 1b - Heat 1D with "30nm thick" (thickness is NOT length):
{
  "pde_type": "heat",
  "dim": 1,
  "domain_size": {"length": 2.0},
  "bc_values": {"t_left": 20.0, "t_right": 0.0},
  "initial_type": "constant",
  "initial_value": 0.0,
  "initial_amplitude": 1.0,
  "initial_wavenumber": 1.0,
  "source_type": "none",
  "source_value": 0.0,
  "steady": false,
  "field_name": "temperature",
  "unit": "°C",
  "notes": ["Thickness mentioned (30nm) - using default length (2.0 m)"]
}

Example 2 - Elasticity 2D with dimensions "1m x 1m plate":
{
  "pde_type": "elasticity",
  "dim": 2,
  "domain_size": {"Lx": 1.0, "Ly": 1.0},
  "bc_values": {},
  "young_modulus": 210e9,
  "poisson_ratio": 0.3,
  "material_params": {},
  "field_name": "von_mises_stress",
  "unit": "Pa",
  "notes": []
}

Example 3 - Elasticity 3D with dimensions "1m x 0.2m x 0.2m":
{
  "pde_type": "elasticity",
  "dim": 3,
  "domain_size": {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2},
  "bc_values": {},
  "young_modulus": 210e9,
  "poisson_ratio": 0.3,
  "material_params": {"gravity": true, "quantity": "strain"},
  "field_name": "von_mises_strain",
  "unit": "-",
  "notes": []
}

Example 4 - Elasticity 3D with dimensions "1m*0.2m*0.2" (asterisk separators):
{
  "pde_type": "elasticity",
  "dim": 3,
  "domain_size": {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2},
  "bc_values": {},
  "young_modulus": 210e9,
  "poisson_ratio": 0.3,
  "material_params": {"gravity": true},
  "field_name": "von_mises_stress",
  "unit": "Pa",
  "notes": []
}

CRITICAL RULES FOR DOMAIN_SIZE EXTRACTION:

FOR 2D PROBLEMS:
1. domain_size MUST ALWAYS be {"Lx": value, "Ly": value}
2. NEVER return {"domain_size": value} for 2D problems - this is WRONG
3. When user mentions "1m x 1m" or "1m*1m", extract BOTH numbers:
   - First number (1m) → Lx = 1.0
   - Second number (1m) → Ly = 1.0
4. Example: "1m x 1m plate" → {"Lx": 1.0, "Ly": 1.0}, NOT {"domain_size": 1.0}

FOR 3D PROBLEMS:
1. domain_size MUST ALWAYS be {"Lx": value, "Ly": value, "Lz": value}
2. NEVER return {"domain_size": value} for 3D problems - this is WRONG
3. When user mentions multiple dimensions like "1m*0.2m*0.2" or "1m x 0.2m x 0.2m":
   - Extract the FIRST number → Lx
   - Extract the SECOND number → Ly
   - Extract the THIRD number → Lz
4. Example: "1m*0.2m*0.2" → {"Lx": 1.0, "Ly": 0.2, "Lz": 0.2}, NOT {"domain_size": 0.2}

DO NOT include any text before or after the JSON. DO NOT explain your reasoning. DO NOT ask for clarification.
Return ONLY the JSON object.
"""


class PDEParserAgent:
    """Agent that parses natural language PDE descriptions into structured parameters."""
    
    def __init__(self, llm=None, model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        Initialize the PDE Parser Agent.
        
        Args:
            llm: Optional LangChain LLM instance. If None, creates a new ChatOpenAI instance.
            model_name: Model name for ChatOpenAI if llm is None.
            temperature: Temperature for LLM if llm is None.
        """
        if llm is None:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        else:
            self.llm = llm
        
        self.parser = JsonOutputParser()
    
    async def parse(self, description: str) -> PDEParameters:
        """
        Parse natural language description into PDEParameters.
        
        Args:
            description: Natural language description of the PDE problem.
            
        Returns:
            PDEParameters object with extracted information.
        """
        # Add format instructions for JSON output
        format_instructions = self.parser.get_format_instructions()
        full_prompt = f"{PDE_PARSER_SYSTEM_PROMPT}\n\n{format_instructions}"
        
        messages = [
            SystemMessage(content=full_prompt),
            HumanMessage(content=f"Parse the following PDE problem description:\n\n{description}"),
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Parse JSON response
        try:
            parsed_json = self.parser.parse(response.content)
        except Exception as e:
            # If parsing fails, try to extract JSON from response
            content = response.content.strip()
            
            # Detect PDE type from user description even if JSON parsing fails
            description_lower = description.lower()
            detected_pde_type = "heat"  # default
            if any(keyword in description_lower for keyword in ["elasticity", "elastic", "stress", "strain", "young", "poisson", "displacement", "deformation"]):
                detected_pde_type = "elasticity"
            elif any(keyword in description_lower for keyword in ["wave"]):
                detected_pde_type = "wave"
            
            # Check if response is natural language asking for clarification
            if not content.startswith('{') and not '{' in content:
                # Response is likely natural language, return default JSON with detected PDE type
                print(f"Warning: Parser returned natural language instead of JSON. Using defaults.")
                print(f"Response was: {content[:100]}...")
                print(f"Detected PDE type from description: {detected_pde_type}")
                
                if detected_pde_type == "elasticity":
                    parsed_json = {
                        "pde_type": "elasticity",
                        "dim": 2,
                        "domain_size": {"Lx": 1.0, "Ly": 1.0},
                        "bc_values": {},
                        "young_modulus": 210e9,
                        "poisson_ratio": 0.3,
                        "material_params": {},
                        "field_name": "displacement",
                        "unit": "m",
                        "notes": ["Used defaults due to unclear input - detected as elasticity"]
                    }
                else:
                    parsed_json = {
                        "pde_type": detected_pde_type,
                        "dim": 1,
                        "domain_size": {"length": 2.0},
                        "bc_values": {"t_left": 20.0, "t_right": 0.0},
                        "initial_value": 0.0,
                        "source_type": "none",
                        "source_value": 0.0,
                        "steady": False,
                        "field_name": "temperature",
                        "unit": "°C",
                        "notes": ["Used defaults due to unclear input"]
                    }
            else:
                # Try to find JSON object in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    try:
                        parsed_json = json.loads(json_str)
                    except json.JSONDecodeError:
                        # If JSON parsing still fails, use defaults with detected PDE type
                        print(f"Warning: Failed to parse JSON. Using defaults.")
                        description_lower = description.lower()
                        detected_pde_type = "heat"  # default
                        if any(keyword in description_lower for keyword in ["elasticity", "elastic", "stress", "strain", "young", "poisson", "displacement", "deformation"]):
                            detected_pde_type = "elasticity"
                        
                        if detected_pde_type == "elasticity":
                            parsed_json = {
                                "pde_type": "elasticity",
                                "dim": 2,
                                "domain_size": {"Lx": 1.0, "Ly": 1.0},
                                "bc_values": {},
                                "young_modulus": 210e9,
                                "poisson_ratio": 0.3,
                                "material_params": {},
                                "field_name": "displacement",
                                "unit": "m",
                                "notes": ["Used defaults due to JSON parsing error - detected as elasticity"]
                            }
                        else:
                            parsed_json = {
                                "pde_type": detected_pde_type,
                                "dim": 1,
                                "domain_size": {"length": 2.0},
                                "bc_values": {"t_left": 20.0, "t_right": 0.0},
                                "initial_value": 0.0,
                                "source_type": "none",
                                "source_value": 0.0,
                                "steady": False,
                                "field_name": "temperature",
                                "unit": "°C",
                                "notes": ["Used defaults due to JSON parsing error"]
                            }
                else:
                    # No JSON found, use defaults with detected PDE type
                    print(f"Warning: No JSON found in response. Using defaults.")
                    description_lower = description.lower()
                    detected_pde_type = "heat"  # default
                    if any(keyword in description_lower for keyword in ["elasticity", "elastic", "stress", "strain", "young", "poisson", "displacement", "deformation"]):
                        detected_pde_type = "elasticity"
                    
                    if detected_pde_type == "elasticity":
                        parsed_json = {
                            "pde_type": "elasticity",
                            "dim": 2,
                            "domain_size": {"Lx": 1.0, "Ly": 1.0},
                            "bc_values": {},
                            "young_modulus": 210e9,
                            "poisson_ratio": 0.3,
                            "material_params": {},
                            "field_name": "displacement",
                            "unit": "m",
                            "notes": ["Used defaults - no valid JSON in response - detected as elasticity"]
                        }
                    else:
                        parsed_json = {
                            "pde_type": detected_pde_type,
                            "dim": 1,
                            "domain_size": {"length": 2.0},
                            "bc_values": {"t_left": 20.0, "t_right": 0.0},
                            "initial_value": 0.0,
                            "source_type": "none",
                            "source_value": 0.0,
                            "steady": False,
                            "field_name": "temperature",
                            "unit": "°C",
                            "notes": ["Used defaults - no valid JSON in response"]
                        }
        
        # Normalize keys to snake_case
        normalized_json = normalize_json_keys(parsed_json)
        
        # Create PDEParameters object
        try:
            return PDEParameters(**normalized_json)
        except Exception as e:
            # If there are still issues, try to filter out unknown keys
            # Get valid field names from PDEParameters
            valid_fields = {f.name for f in PDEParameters.__dataclass_fields__.values()}
            filtered_json = {k: v for k, v in normalized_json.items() if k in valid_fields}
            try:
                return PDEParameters(**filtered_json)
            except Exception as e2:
                raise ValueError(
                    f"Failed to create PDEParameters from parsed JSON. "
                    f"Original error: {e}, After filtering: {e2}. "
                    f"Normalized JSON keys: {list(normalized_json.keys())}"
                )
    
    def parse_sync(self, description: str) -> PDEParameters:
        """
        Synchronous version of parse (uses asyncio.run internally).
        
        Args:
            description: Natural language description of the PDE problem.
            
        Returns:
            PDEParameters object with extracted information.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.parse(description))


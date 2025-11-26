# dispatcher_agent.py
"""
Dispatcher Agent
----------------
Takes structured PDE parameters and dispatches to appropriate MCP tools for solving.
"""

import json
import os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from pde_schema import PDEParameters


DISPATCHER_SYSTEM_PROMPT = """
You are a dispatcher agent that takes structured PDE parameters and calls the appropriate MCP tools to solve finite element problems.

Your task is to:
1. Receive structured PDE parameters (PDEParameters object)
2. Determine which MCP tool to call based on pde_type and dim
3. Map the parameters to the tool's arguments correctly
4. Call the solver tool (solve_heat_1D, solve_heat_2D, or solve_heat_3D)
5. Call the visualization tool (plot_time_series_field_from_file) with the result

AVAILABLE TOOLS:

1. solve_heat_1D: For 1D heat equation
   - Arguments: length, nx, diffusivity, T_left, T_right, T_initial, dt, num_steps, data_dir
   - Returns: SolveResult with data_file path

2. solve_heat_2D: For 2D heat equation
   - Arguments: Lx, Ly, nx, ny, diffusivity, T_boundary, T_initial, dt, num_steps, data_dir
   - Returns: SolveResult with data_file path

3. solve_heat_3D: For 3D heat equation
   - Arguments: Lx, Ly, Lz, nx, ny, nz, diffusivity, T_boundary, T_initial, dt, num_steps, data_dir
   - Returns: SolveResult with data_file path

4. plot_time_series_field_from_file: For visualization
   - Arguments: data_file (from solver result), field_name (optional), unit (optional), output_dir, filename
   - Returns: PlotResult with html_path

MAPPING RULES:

For 1D problems (dim=1, pde_type="heat"):
  - Call solve_heat_1D
  - Map: domain_size["length"] → length
  - Map: bc_values["T_left"] → T_left
  - Map: bc_values["T_right"] → T_right
  - Map: initial_value → T_initial
  - Map: diffusivity → diffusivity
  - Calculate dt and num_steps from time discretization

For 2D problems (dim=2, pde_type="heat"):
  - Call solve_heat_2D
  - Map: domain_size["Lx"] → Lx, domain_size["Ly"] → Ly
  - Map: bc_values.get("T_boundary", ...) → T_boundary
  - Map: initial_value → T_initial

For 3D problems (dim=3, pde_type="heat"):
  - Call solve_heat_3D
  - Map: domain_size["Lx"] → Lx, domain_size["Ly"] → Ly, domain_size["Lz"] → Lz
  - Map: bc_values.get("T_boundary", ...) → T_boundary
  - Map: initial_value → T_initial

CRITICAL: After calling the solver, ALWAYS call plot_time_series_field_from_file to generate visualization.

Return a summary of:
1. Which tools were called and with what parameters
2. The paths to the generated data file and visualization HTML file
3. Any errors or warnings encountered
"""


class DispatcherAgent:
    """Agent that dispatches structured PDE parameters to MCP tools."""
    
    def __init__(self, mcp_client: MultiServerMCPClient, llm=None, model_name: str = "gpt-4o", temperature: float = 0.0):
        """
        Initialize the Dispatcher Agent.
        
        Args:
            mcp_client: MultiServerMCPClient instance with FEniCS tools loaded.
            llm: Optional LangChain LLM instance. If None, creates a new ChatOpenAI instance.
            model_name: Model name for ChatOpenAI if llm is None.
            temperature: Temperature for LLM if llm is None.
        """
        self.mcp_client = mcp_client
        if llm is None:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        else:
            self.llm = llm
    
    async def dispatch(self, pde_params: PDEParameters) -> Dict[str, Any]:
        """
        Dispatch PDE parameters to appropriate MCP tools and solve.
        
        Args:
            pde_params: PDEParameters object with parsed PDE information.
            
        Returns:
            Dictionary with:
            - "solver_result": Result from solver tool
            - "plot_result": Result from plot tool
            - "summary": Summary of the simulation
        """
        # Get available tools
        tools = await self.mcp_client.get_tools()
        tool_map = {tool.name: tool for tool in tools}
        
        # Determine which solver to use based on PDE type
        if pde_params.pde_type == "heat":
            solver_name = f"solve_heat_{pde_params.dim}D"
            if solver_name not in tool_map:
                raise ValueError(f"Solver tool {solver_name} not available")
            solver_tool = tool_map[solver_name]
            
            # Build solver arguments based on dimension
            if pde_params.dim == 1:
                solver_args = self._build_1d_args(pde_params)
            elif pde_params.dim == 2:
                solver_args = self._build_2d_args(pde_params)
            elif pde_params.dim == 3:
                solver_args = self._build_3d_args(pde_params)
            else:
                raise ValueError(f"Unsupported dimension: {pde_params.dim}")
                
        elif pde_params.pde_type == "elasticity":
            solver_name = f"solve_elasticity_{pde_params.dim}D_static"
            if solver_name not in tool_map:
                raise ValueError(f"Solver tool {solver_name} not available")
            solver_tool = tool_map[solver_name]
            
            # Build elasticity solver arguments based on dimension
            if pde_params.dim == 1:
                solver_args = self._build_elasticity_1d_args(pde_params)
            elif pde_params.dim == 2:
                solver_args = self._build_elasticity_2d_args(pde_params)
            elif pde_params.dim == 3:
                solver_args = self._build_elasticity_3d_args(pde_params)
            else:
                raise ValueError(f"Unsupported dimension: {pde_params.dim}")
        else:
            raise ValueError(f"Currently only 'heat' and 'elasticity' PDE types are supported, got: {pde_params.pde_type}")
        
        # Call solver
        try:
            solver_result = await solver_tool.ainvoke(solver_args)
            
            # Check if the result contains an error
            if isinstance(solver_result, dict) and "error" in solver_result:
                raise ValueError(f"Solver returned an error: {solver_result['error']}")
            
            # Handle JSON string responses (MCP tools may return JSON strings)
            if isinstance(solver_result, str):
                try:
                    # Try to parse as JSON directly
                    solver_result = json.loads(solver_result.strip())
                except json.JSONDecodeError:
                    # If it's not valid JSON, try to extract JSON object from the string
                    start_idx = solver_result.find('{')
                    end_idx = solver_result.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = solver_result[start_idx:end_idx]
                        solver_result = json.loads(json_str)
                    else:
                        raise ValueError(f"Could not parse solver result as JSON: {solver_result[:200]}")
            
            data_file = self._extract_value(solver_result, "data_file")

            if not data_file:
                # More detailed error message
                error_msg = f"Solver did not return a data_file. "
                error_msg += f"Result type: {type(solver_result)}, "
                if isinstance(solver_result, dict):
                    error_msg += f"Keys: {list(solver_result.keys())}, "
                    error_msg += f"Full result: {solver_result}"
                elif hasattr(solver_result, "__dict__"):
                    error_msg += f"Attributes: {list(vars(solver_result).keys())}, "
                    error_msg += f"Full result: {vars(solver_result)}"
                else:
                    error_msg += f"Result: {solver_result}"
                raise ValueError(error_msg)
            
            # Call visualization tool - MUST generate visualization
            plot_tool = tool_map.get("plot_time_series_field_from_file")
            if not plot_tool:
                # Try alternative tool names
                plot_tool = tool_map.get("plot_time_series_field")
                if not plot_tool:
                    available_tools = list(tool_map.keys())
                    raise ValueError(
                        f"Visualization tool not found. Available tools: {available_tools}. "
                        f"Expected 'plot_time_series_field_from_file' or 'plot_time_series_field'."
                    )
            
            # At this point, plot_tool must be available (error raised if not)
            # Call visualization tool - MUST always generate visualization
            # Generate filename: extract unique ID from data_file (e.g., "heat_1d_abc123.pkl" -> "abc123")
            data_basename = os.path.basename(data_file).split('.')[0]  # Remove extension
            # Remove pde_type and dim prefix if present (e.g., "heat_1d_abc123" -> "abc123")
            prefix = f"{pde_params.pde_type}_{pde_params.dim}d_"
            if data_basename.startswith(prefix):
                unique_id = data_basename[len(prefix):]
            else:
                # If no prefix, use the whole basename or extract last part after underscore
                parts = data_basename.split('_')
                unique_id = parts[-1] if len(parts) > 1 else data_basename
            
            # For field_name and unit: let plot_time_series_field_from_file use metadata from the pickle file
            # The solver stores the correct field_name in the TimeSeriesField meta (e.g., "von_mises_stress" for elasticity)
            # By passing None, we let the plot function read from the file's metadata
            # This ensures elasticity plots show "von Mises stress" or "von Mises strain", not "temperature"
            plot_args = {
                "data_file": data_file,
                "field_name": None,  # None means read from file metadata
                "unit": None,  # None means read from file metadata
                "output_dir": "plots",
                "filename": f"{pde_params.pde_type}_{pde_params.dim}d_{unique_id}.html",
            }
            plot_result = await plot_tool.ainvoke(plot_args)

            # Parse JSON string responses (MCP tools may return JSON strings)
            if isinstance(plot_result, str):
                try:
                    plot_result = json.loads(plot_result.strip())
                except json.JSONDecodeError:
                    # If it's not valid JSON, try to extract JSON object from the string
                    original_str = plot_result
                    start_idx = original_str.find('{')
                    end_idx = original_str.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = original_str[start_idx:end_idx]
                        plot_result = json.loads(json_str)
                    else:
                        raise ValueError(f"Could not parse plot result as JSON: {original_str[:200]}")

            # Extract html_path from plot result
            html_path = self._extract_value(plot_result, "html_path")
            
            if not html_path:
                # Debug: Print the actual plot result structure
                print(f"DEBUG: plot_result type: {type(plot_result)}")
                if isinstance(plot_result, dict):
                    print(f"DEBUG: plot_result keys: {list(plot_result.keys())}")
                    print(f"DEBUG: plot_result content: {plot_result}")
                else:
                    print(f"DEBUG: plot_result: {str(plot_result)[:200]}")
                raise ValueError(f"Plot tool did not return html_path. Result type: {type(plot_result)}, Result: {plot_result}")
            
            return {
                "solver_result": solver_result,
                "plot_result": plot_result,
                "data_file": data_file,
                "html_path": html_path,
                "summary": self._generate_summary(pde_params, solver_result, plot_result),
            }
        except Exception as e:
            return {
                "error": str(e),
                "solver_args": solver_args,
            }
    
    def _build_1d_args(self, params: PDEParameters) -> Dict[str, Any]:
        """Build arguments for 1D heat solver."""
        # Extract length from domain_size (handle both "length" and "L" keys, case-insensitive)
        domain = params.domain_size or {}
        # Handle nested domain_size key (parser might return {'domain_size': value})
        if "length" in domain or "L" in domain or "l" in domain:
            # Domain has correct structure - extract directly
            length = (domain.get("length") or domain.get("Length") or domain.get("L") or 
                      domain.get("l") or 2.0)  # Default to 2.0 meters if not specified
        elif "domain_size" in domain and isinstance(domain["domain_size"], (int, float)):
            # Nested structure with single value - use it as length
            length = float(domain["domain_size"])
        else:
            # Default value if nothing found
            length = 2.0
        
        nx = params.nx or 50
        diffusivity = params.diffusivity or 1.0
        
        # Boundary conditions - handle both uppercase and lowercase keys (case-insensitive)
        # CRITICAL: Check if keys exist (not just truthy values) because 0.0 is falsy but valid
        bc_values = params.bc_values or {}
        
        # Check various key formats and use first one found, otherwise use default
        T_left = 20.0  # default
        if "T_left" in bc_values:
            T_left = bc_values["T_left"]
        elif "t_left" in bc_values:
            T_left = bc_values["t_left"]
        elif "T_Left" in bc_values:
            T_left = bc_values["T_Left"]
        elif "left" in bc_values:
            T_left = bc_values["left"]
        elif "T_left_boundary" in bc_values:
            T_left = bc_values["T_left_boundary"]
        elif "t_left_boundary" in bc_values:
            T_left = bc_values["t_left_boundary"]
        
        T_right = 0.0  # default
        if "T_right" in bc_values:
            T_right = bc_values["T_right"]
        elif "t_right" in bc_values:
            T_right = bc_values["t_right"]
        elif "T_Right" in bc_values:
            T_right = bc_values["T_Right"]
        elif "right" in bc_values:
            T_right = bc_values["right"]
        elif "T_right_boundary" in bc_values:
            T_right = bc_values["T_right_boundary"]
        elif "t_right_boundary" in bc_values:
            T_right = bc_values["t_right_boundary"]
        
        # Time discretization
        dt = params.dt
        num_steps = params.num_steps
        
        # Calculate defaults if not provided
        if dt is None or num_steps is None:
            # Estimate characteristic time
            t_char = length ** 2 / (2 * diffusivity) if diffusivity > 0 else 0.1
            if dt is None:
                # Use dt = 0.01s for reasonable time step
                dt = min(t_char / 200, 0.01)  # Use 200 steps to reach t_char, but max 0.01s
            if num_steps is None:
                if params.total_time:
                    num_steps = max(int(params.total_time / dt), 100) if dt > 0 else 200
                else:
                    # Use more steps for smoother animation
                    num_steps = max(int(t_char / dt), 200) if dt > 0 else 200
        
        # Source term and steady-state mode
        source_type = params.source_type if params.source_type else "none"
        source_value = params.source_value if params.source_value is not None else 0.0
        steady = params.steady if params.steady is not None else False
        
        # Initial condition parameters
        initial_type = params.initial_type if params.initial_type else "constant"
        initial_amplitude = params.initial_amplitude if params.initial_amplitude is not None else 1.0
        initial_wavenumber = params.initial_wavenumber if params.initial_wavenumber is not None else 1.0
        
        # For constant initial condition, use initial_value as T_initial
        # For cosine/sine/zero, T_initial is still used as a reference but the actual initial condition is set by the type
        # Default T_initial based on initial_type
        if initial_type == "constant":
            # Use initial_value if provided, otherwise default to 0.0
            T_initial = params.initial_value if params.initial_value is not None else 0.0
        elif initial_type == "zero":
            # T_initial doesn't matter for zero type, but we still need it for the function signature
            T_initial = 0.0
        else:  # cosine or sine
            # For cosine/sine, T_initial is not used, but we need a default value
            # Use initial_amplitude as a reference, or default to 0.0
            T_initial = params.initial_value if params.initial_value is not None else 0.0
        
        return {
            "length": length,
            "nx": nx,
            "diffusivity": diffusivity,
            "T_left": T_left,
            "T_right": T_right,
            "T_initial": T_initial,
            "dt": dt,
            "num_steps": num_steps,
            "data_dir": "data",
            "steady": steady,
            "source_type": source_type,
            "source_value": source_value,
            "initial_type": initial_type,
            "initial_amplitude": initial_amplitude,
            "initial_wavenumber": initial_wavenumber,
        }
    
    def _build_2d_args(self, params: PDEParameters) -> Dict[str, Any]:
        """Build arguments for 2D heat solver."""
        domain = params.domain_size or {}
        # Handle nested domain_size key (parser might return {'domain_size': value})
        if "domain_size" in domain and isinstance(domain["domain_size"], (int, float)):
            # If domain_size is a single value, use it for both Lx and Ly
            size = float(domain["domain_size"])
            Lx = size
            Ly = size
        else:
            Lx = (domain.get("Lx") or domain.get("lx") or domain.get("width") or 
                  domain.get("Width") or domain.get("W") or 1.0)
            Ly = (domain.get("Ly") or domain.get("ly") or domain.get("height") or 
                  domain.get("Height") or domain.get("H") or 1.0)
        nx = params.nx or 30
        ny = params.ny or 30
        diffusivity = params.diffusivity or 1.0
        
        bc_values = params.bc_values
        T_boundary = bc_values.get("T_boundary") or bc_values.get("T_boundary_value") or 0.0
        
        # Time discretization - use smaller dt for smoother animation
        dt = params.dt
        num_steps = params.num_steps
        
        # Calculate defaults if not provided
        if dt is None or num_steps is None:
            # Estimate characteristic time for 2D (use average domain size)
            avg_size = (Lx + Ly) / 2
            t_char = avg_size ** 2 / (2 * diffusivity) if diffusivity > 0 else 0.1
            if dt is None:
                dt = min(t_char / 200, 0.01)  # Use dt = 0.01s
            if num_steps is None:
                if params.total_time:
                    num_steps = max(int(params.total_time / dt), 100) if dt > 0 else 200
                else:
                    num_steps = max(int(t_char / dt), 200) if dt > 0 else 200
        
        # Ensure defaults if still None
        dt = dt or 0.01
        num_steps = num_steps or 200
        
        # Source term and steady-state mode
        source_type = params.source_type if params.source_type else "none"
        source_value = params.source_value if params.source_value is not None else 0.0
        steady = params.steady if params.steady is not None else False
        
        # Initial condition parameters
        initial_type = params.initial_type if params.initial_type else "constant"
        initial_amplitude = params.initial_amplitude if params.initial_amplitude is not None else 1.0
        initial_wavenumber = params.initial_wavenumber if params.initial_wavenumber is not None else 1.0
        
        # For constant initial condition, use initial_value as T_initial
        if initial_type == "constant":
            T_initial = params.initial_value if params.initial_value is not None else 20.0
        elif initial_type == "zero":
            T_initial = 0.0
        else:  # cosine or sine
            T_initial = params.initial_value if params.initial_value is not None else 0.0
        
        return {
            "Lx": Lx,
            "Ly": Ly,
            "nx": nx,
            "ny": ny,
            "diffusivity": diffusivity,
            "T_boundary": T_boundary,
            "T_initial": T_initial,
            "dt": dt,
            "num_steps": num_steps,
            "data_dir": "data",
            "steady": steady,
            "source_type": source_type,
            "source_value": source_value,
            "initial_type": initial_type,
            "initial_amplitude": initial_amplitude,
            "initial_wavenumber": initial_wavenumber,
        }
    
    def _build_3d_args(self, params: PDEParameters) -> Dict[str, Any]:
        """Build arguments for 3D heat solver."""
        domain = params.domain_size or {}
        # Handle nested domain_size key (parser might return {'domain_size': value})
        if "domain_size" in domain and isinstance(domain["domain_size"], (int, float)):
            # If domain_size is a single value, use it for Lx, Ly, Lz
            size = float(domain["domain_size"])
            Lx = size
            Ly = size
            Lz = size
        else:
            Lx = (domain.get("Lx") or domain.get("lx") or domain.get("width") or 
                  domain.get("Width") or domain.get("W") or 1.0)
            Ly = (domain.get("Ly") or domain.get("ly") or domain.get("height") or 
                  domain.get("Height") or domain.get("H") or 1.0)
            Lz = (domain.get("Lz") or domain.get("lz") or domain.get("depth") or 
                  domain.get("Depth") or domain.get("D") or 1.0)
        nx = params.nx or 10
        ny = params.ny or 10
        nz = params.nz or 10
        diffusivity = params.diffusivity or 1.0
        
        bc_values = params.bc_values
        T_boundary = bc_values.get("T_boundary") or bc_values.get("T_boundary_value") or 0.0
        
        # Time discretization - use smaller dt for smoother animation
        dt = params.dt
        num_steps = params.num_steps
        
        # Calculate defaults if not provided
        if dt is None or num_steps is None:
            # Estimate characteristic time for 3D (use average domain size)
            avg_size = (Lx + Ly + Lz) / 3
            t_char = avg_size ** 2 / (2 * diffusivity) if diffusivity > 0 else 0.1
            if dt is None:
                dt = min(t_char / 200, 0.01)  # Use dt = 0.01s
            if num_steps is None:
                if params.total_time:
                    num_steps = max(int(params.total_time / dt), 100) if dt > 0 else 200
                else:
                    num_steps = max(int(t_char / dt), 200) if dt > 0 else 200
        
        # Ensure defaults if still None
        dt = dt or 0.01
        num_steps = num_steps or 200
        
        # Source term and steady-state mode
        source_type = params.source_type if params.source_type else "none"
        source_value = params.source_value if params.source_value is not None else 0.0
        steady = params.steady if params.steady is not None else False
        
        # Initial condition parameters
        initial_type = params.initial_type if params.initial_type else "constant"
        initial_amplitude = params.initial_amplitude if params.initial_amplitude is not None else 1.0
        initial_wavenumber = params.initial_wavenumber if params.initial_wavenumber is not None else 1.0
        
        # For constant initial condition, use initial_value as T_initial
        if initial_type == "constant":
            T_initial = params.initial_value if params.initial_value is not None else 20.0
        elif initial_type == "zero":
            T_initial = 0.0
        else:  # cosine or sine
            T_initial = params.initial_value if params.initial_value is not None else 0.0
        
        return {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "diffusivity": diffusivity,
            "T_boundary": T_boundary,
            "T_initial": T_initial,
            "dt": dt,
            "num_steps": num_steps,
            "data_dir": "data",
            "steady": steady,
            "source_type": source_type,
            "source_value": source_value,
            "initial_type": initial_type,
            "initial_amplitude": initial_amplitude,
            "initial_wavenumber": initial_wavenumber,
        }
    
    def _build_elasticity_1d_args(self, params: PDEParameters) -> Dict[str, Any]:
        """Build arguments for 1D elasticity solver."""
        domain = params.domain_size or {}
        # Handle nested domain_size key (parser might return {'domain_size': value})
        if "length" in domain or "L" in domain or "l" in domain:
            # Domain has correct structure - extract directly
            L = (domain.get("length") or domain.get("L") or domain.get("l") or 
                 domain.get("Length") or 1.0)
        elif "domain_size" in domain and isinstance(domain["domain_size"], (int, float)):
            # Nested structure with single value - use it as length
            L = float(domain["domain_size"])
        else:
            # Default value if nothing found
            L = 1.0
        nx = params.nx or 50
        
        # Material parameters
        E = params.young_modulus if params.young_modulus is not None else 210e9  # Default: steel
        area = params.material_params.get("area") or params.material_params.get("cross_sectional_area") or 1.0
        
        # Body force
        body_force = params.material_params.get("body_force") or params.material_params.get("body_force_x") or 0.0
        
        # Output quantity
        quantity = params.material_params.get("quantity") or "stress"
        if quantity not in ["stress", "strain"]:
            quantity = "stress"
        
        return {
            "L": L,
            "nx": nx,
            "E": E,
            "area": area,
            "body_force": body_force,
            "quantity": quantity,
            "data_dir": "data",
        }
    
    def _build_elasticity_2d_args(self, params: PDEParameters) -> Dict[str, Any]:
        """Build arguments for 2D elasticity solver."""
        domain = params.domain_size or {}
        # Handle nested domain_size key (parser might return {'domain_size': value})
        if "domain_size" in domain and isinstance(domain["domain_size"], (int, float)):
            # If domain_size is a single value, use it for both Lx and Ly
            size = float(domain["domain_size"])
            Lx = size
            Ly = size
        else:
            Lx = (domain.get("Lx") or domain.get("lx") or domain.get("width") or 
                  domain.get("Width") or domain.get("W") or 1.0)
            Ly = (domain.get("Ly") or domain.get("ly") or domain.get("height") or 
                  domain.get("Height") or domain.get("H") or 1.0)
        nx = params.nx or 30
        ny = params.ny or 30
        
        # Material parameters
        E = params.young_modulus if params.young_modulus is not None else 210e9  # Default: steel
        nu = params.poisson_ratio if params.poisson_ratio is not None else 0.3  # Default: steel
        
        # Body forces
        # Check if gravity is mentioned in material_params or if it needs to be applied
        gravity_mentioned = (params.material_params.get("gravity") or 
                           params.material_params.get("apply_gravity") or False)
        
        # If gravity is mentioned but body_fy not explicitly set, apply gravity
        if gravity_mentioned and params.material_params.get("body_fy") is None and params.material_params.get("body_force_y") is None:
            # Default steel density ≈ 7800 kg/m³
            density = params.density if params.density is not None else 7800.0
            # Gravity acts downward (negative y-direction)
            body_fy = -9.81 * density  # N/m³ (force per unit volume)
            body_fx = params.material_params.get("body_fx") or params.material_params.get("body_force_x") or 0.0
        else:
            # Use explicit body forces or defaults
            body_fx = params.material_params.get("body_fx") or params.material_params.get("body_force_x") or 0.0
            body_fy = params.material_params.get("body_fy") or params.material_params.get("body_force_y") or 0.0
        
        # Output quantity
        quantity = params.material_params.get("quantity") or "stress"
        if quantity not in ["stress", "strain"]:
            quantity = "stress"
        
        # Plane stress or strain
        plane_stress = params.material_params.get("plane_stress")
        if plane_stress is None:
            plane_stress = True  # Default to plane stress
        
        return {
            "Lx": Lx,
            "Ly": Ly,
            "nx": nx,
            "ny": ny,
            "E": E,
            "nu": nu,
            "body_fx": body_fx,
            "body_fy": body_fy,
            "quantity": quantity,
            "plane_stress": plane_stress,
            "data_dir": "data",
        }
    
    def _build_elasticity_3d_args(self, params: PDEParameters) -> Dict[str, Any]:
        """Build arguments for 3D elasticity solver."""
        domain = params.domain_size or {}
        # Handle nested domain_size key (parser might return {'domain_size': value})
        # First, check if domain itself has the correct keys (Lx, Ly, Lz)
        if "Lx" in domain or "lx" in domain or "width" in domain:
            # Domain has correct structure - extract directly
            Lx = (domain.get("Lx") or domain.get("lx") or domain.get("width") or 
                  domain.get("Width") or domain.get("W") or 1.0)
            Ly = (domain.get("Ly") or domain.get("ly") or domain.get("height") or 
                  domain.get("Height") or domain.get("H") or 1.0)
            Lz = (domain.get("Lz") or domain.get("lz") or domain.get("depth") or 
                  domain.get("Depth") or domain.get("D") or 1.0)
        elif "domain_size" in domain and isinstance(domain["domain_size"], (int, float)):
            # Nested structure with single value - use for all three (fallback)
            size = float(domain["domain_size"])
            Lx = size
            Ly = size
            Lz = size
        else:
            # Default values if nothing found
            Lx = 1.0
            Ly = 1.0
            Lz = 1.0
        nx = params.nx or 10
        ny = params.ny or 10
        nz = params.nz or 10
        
        # Material parameters
        E = params.young_modulus if params.young_modulus is not None else 210e9  # Default: steel
        nu = params.poisson_ratio if params.poisson_ratio is not None else 0.3  # Default: steel
        
        # Body forces
        # If user mentions gravity, apply gravitational body force
        apply_gravity = params.material_params.get("gravity") or False
        if apply_gravity or params.material_params.get("apply_gravity"):
            # Default steel density ≈ 7800 kg/m³
            density = params.density if params.density is not None else 7800.0
            body_fx = params.material_params.get("body_fx") or params.material_params.get("body_force_x") or 0.0
            body_fy = params.material_params.get("body_fy") or params.material_params.get("body_force_y") or 0.0
            body_fz = params.material_params.get("body_fz") or params.material_params.get("body_force_z") or (-9.81 * density)
        else:
            body_fx = params.material_params.get("body_fx") or params.material_params.get("body_force_x") or 0.0
            body_fy = params.material_params.get("body_fy") or params.material_params.get("body_force_y") or 0.0
            body_fz = params.material_params.get("body_fz") or params.material_params.get("body_force_z") or 0.0
        
        # Output quantity
        quantity = params.material_params.get("quantity") or "stress"
        if quantity not in ["stress", "strain"]:
            quantity = "stress"
        
        return {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "E": E,
            "nu": nu,
            "body_fx": body_fx,
            "body_fy": body_fy,
            "body_fz": body_fz,
            "quantity": quantity,
            "data_dir": "data",
        }
    
    def _generate_summary(self, params: PDEParameters, solver_result: Any, plot_result: Any) -> str:
        """
        Generate a summary of the simulation.
        IMPORTANT: This uses the current params object, which should be the merged parameters
        after a follow-up question. All fields in the summary reflect the current merged state.
        """
        # Handle solver_result (could be dict or object)
        if isinstance(solver_result, dict):
            data_file = solver_result.get('data_file', 'N/A')
        else:
            data_file = getattr(solver_result, 'data_file', 'N/A')
        
        # Handle plot_result (could be dict or object)
        if plot_result:
            if isinstance(plot_result, dict):
                html_path = plot_result.get('html_path', 'N/A')
            else:
                html_path = getattr(plot_result, 'html_path', 'N/A')
        else:
            html_path = 'N/A'
        
        # Format source term information (heat only)
        source_info = "none"
        if params.pde_type == "heat":
            if params.source_type and params.source_type != "none":
                source_info = f"{params.source_type} (value: {params.source_value})"
            elif params.source_value and params.source_value != 0.0:
                source_info = f"constant (value: {params.source_value})"
        
        # Format steady-state mode (heat only)
        mode_info = ""
        if params.pde_type == "heat":
            mode_info = "steady-state" if params.steady else "transient"
        elif params.pde_type == "elasticity":
            mode_info = "static"
        
        # Format material parameters based on PDE type
        if params.pde_type == "heat":
            material_info = f"Diffusivity: {params.diffusivity or 'default'}"
        elif params.pde_type == "elasticity":
            E_str = f"{params.young_modulus / 1e9:.1f} GPa" if params.young_modulus else "default"
            nu_str = f"{params.poisson_ratio}" if params.poisson_ratio is not None else "default"
            material_info = f"Young's Modulus: {E_str}, Poisson's Ratio: {nu_str}"
        else:
            material_info = "Material parameters: default"
        
        # Build summary lines - ensure all fields reflect current merged parameters
        summary_lines = [
            "Simulation Summary:",
            f"- PDE Type: {params.pde_type}",
            f"- Dimension: {params.dim}D",
            f"- Mode: {mode_info}" if mode_info else "",
            f"- Domain: {params.domain_size}",
            f"- Spatial Resolution: nx={params.nx or 'auto'}, ny={params.ny or 'auto'}, nz={params.nz or 'auto'}",
            f"- Material Parameters: {material_info}",
            f"- Boundary Conditions: {params.bc_values if params.bc_values else 'default'}",
        ]
        
        # Add problem-specific fields
        if params.pde_type == "heat":
            # Format initial condition based on type
            initial_condition_str = "default"
            if params.initial_type:
                if params.initial_type == "zero":
                    initial_condition_str = "zero"
                elif params.initial_type == "cosine":
                    amplitude = params.initial_amplitude if params.initial_amplitude is not None else 1.0
                    wavenumber = params.initial_wavenumber if params.initial_wavenumber is not None else 1.0
                    initial_condition_str = f"cosine (amplitude: {amplitude}, wavenumber: {wavenumber})"
                elif params.initial_type == "sine":
                    amplitude = params.initial_amplitude if params.initial_amplitude is not None else 1.0
                    wavenumber = params.initial_wavenumber if params.initial_wavenumber is not None else 1.0
                    initial_condition_str = f"sine (amplitude: {amplitude}, wavenumber: {wavenumber})"
                elif params.initial_type == "constant":
                    initial_condition_str = f"constant ({params.initial_value or 'default'})"
                else:
                    initial_condition_str = f"{params.initial_type} ({params.initial_value or 'default'})"
            elif params.initial_value is not None:
                initial_condition_str = f"constant ({params.initial_value})"
            
            summary_lines.extend([
                f"- Initial Condition: {initial_condition_str}",
                f"- Source Term: {source_info}",
                f"- Time Step: {params.dt or 'auto'}, Number of Steps: {params.num_steps or 'auto'}"
            ])
        elif params.pde_type == "elasticity":
            # Add body forces if present
            body_forces = []
            if params.material_params.get("body_force") or params.material_params.get("body_force_x"):
                body_forces.append(f"x: {params.material_params.get('body_force') or params.material_params.get('body_force_x', 0.0)}")
            if params.material_params.get("body_force_y"):
                body_forces.append(f"y: {params.material_params.get('body_force_y', 0.0)}")
            if params.material_params.get("body_force_z"):
                body_forces.append(f"z: {params.material_params.get('body_force_z', 0.0)}")
            if body_forces:
                summary_lines.append(f"- Body Forces: {', '.join(body_forces)}")
            quantity = params.material_params.get("quantity") or "stress"
            summary_lines.append(f"- Output Quantity: {quantity}")
        
        summary_lines.extend([
            "",
            "Results:",
            f"- Data file: {data_file}",
            f"- Visualization: {html_path}"
        ])
        
        summary = "\n".join([line for line in summary_lines if line])  # Filter empty lines
        return summary.strip()

    def _extract_value(self, obj: Any, target_key: str) -> Optional[Any]:
        """
        Recursively extract a value associated with target_key from nested structures.
        Handles dicts, lists, dataclasses, LangChain tool responses, and MCP JSON-RPC responses.
        """
        if obj is None:
            return None

        # Direct dictionary lookup
        if isinstance(obj, dict):
            # Check direct key match (case-insensitive for common keys)
            for key in obj.keys():
                if key.lower() == target_key.lower():
                    return obj[key]
            
            # Try exact match
            if target_key in obj:
                return obj[target_key]
            
            # Check common wrapper keys that MCP/LangChain might use
            for wrapper_key in ["content", "result", "data", "output", "response"]:
                if wrapper_key in obj and isinstance(obj[wrapper_key], (dict, list)):
                    result = self._extract_value(obj[wrapper_key], target_key)
                    if result is not None:
                        return result
            
            # Recursively search in all values
            for value in obj.values():
                result = self._extract_value(value, target_key)
                if result is not None:
                    return result
            return None

        # Lists / tuples
        if isinstance(obj, (list, tuple)):
            for item in obj:
                result = self._extract_value(item, target_key)
                if result is not None:
                    return result
            return None

        # Objects with attribute (try both exact and case-insensitive)
        if hasattr(obj, target_key):
            attr = getattr(obj, target_key)
            if attr is not None:
                return attr
        
        # Try case-insensitive attribute lookup
        target_lower = target_key.lower()
        for attr_name in dir(obj):
            if not attr_name.startswith('_') and attr_name.lower() == target_lower:
                attr = getattr(obj, attr_name)
                if attr is not None:
                    return attr

        # Inspect __dict__ for dataclasses / objects
        if hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                if key.lower() == target_key.lower():
                    return value
                result = self._extract_value(value, target_key)
                if result is not None:
                    return result

        return None


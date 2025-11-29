# fenics_mcp_server.py
"""
FEniCS + MCP Server (Heat Equation + Generic Plotter)
-----------------------------------------------------

提供五个 MCP 工具：

1. solve_heat_1D
   解 1D 热传导:
      u_t - k * u_xx = f(x, t), x ∈ (0, L)  (瞬态)
      或
      -k * u_xx = f(x)  (稳态，当 steady=True)

   - Dirichlet 边界：x=0 处 T_left, x=L 处 T_right
   - 可选热源：source_type ("none" 或 "constant"), source_value
   - 可选稳态模式：steady (bool)
   - 返回 SolveResult（包含数据文件路径和元数据），数据保存在 pickle 文件中

2. solve_heat_2D
   解 2D 热传导:
      u_t - k * Δu = f(x, y, t)  (瞬态)
      或
      -k * Δu = f(x, y)  (稳态，当 steady=True)

   - 边界为常数 Dirichlet: T_boundary
   - 可选热源：source_type ("none" 或 "constant"), source_value
   - 可选稳态模式：steady (bool)
   - 返回 SolveResult（包含数据文件路径和元数据）

3. solve_heat_3D
   解 3D 热传导:
      u_t - k * Δu = f(x, y, z, t)  (瞬态)
      或
      -k * Δu = f(x, y, z)  (稳态，当 steady=True)

   - 边界为常数 Dirichlet: T_boundary
   - 可选热源：source_type ("none" 或 "constant"), source_value
   - 可选稳态模式：steady (bool)
   - 返回 SolveResult（包含数据文件路径和元数据）

4. plot_time_series_field_from_file (推荐)
   从文件读取数据并生成 3D 动图：
   - 输入：solve_heat_* 返回的数据文件路径
   - 输出：一个带 Play/slider、可旋转缩放 + hover 显示数值的 HTML 文件路径
   - 瞬态模式：动画显示时间演化
   - 稳态模式：单帧显示稳态解

5. plot_time_series_field
   通用 3D 动图工具（直接传入数据数组）：
   - 输入：coords, values, times, dim, field_name, unit
   - 输出：一个带 Play/slider、可旋转缩放 + hover 显示数值的 HTML 文件路径
   - 注意：此工具会在 JSON 响应中传递大量数据，可能超出上下文限制
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import io
import contextlib
import logging
import os
import sys
import pickle
import uuid
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

# Suppress FEniCS/FFC logging before importing
os.environ["FFC_LOG_LEVEL"] = "ERROR"
os.environ["DOLFIN_LOG_LEVEL"] = "ERROR"
os.environ["UFL_LOG_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["PYTHONWARNINGS"] += ",ignore::DeprecationWarning"

# Suppress Python logging from FEniCS packages
logging.getLogger("FFC").setLevel(logging.CRITICAL)
logging.getLogger("dolfin").setLevel(logging.CRITICAL)
logging.getLogger("UFL").setLevel(logging.CRITICAL)
logging.getLogger("firedrake").setLevel(logging.CRITICAL)
logging.getLogger("ffc").setLevel(logging.CRITICAL)
logging.getLogger("ufl").setLevel(logging.CRITICAL)
# Suppress pkg_resources warnings
logging.getLogger("pkg_resources").setLevel(logging.CRITICAL)

# Capture stdout/stderr during FEniCS imports to prevent FFC messages
_original_stdout = sys.stdout
_original_stderr = sys.stderr
_silent_buffer = io.StringIO()
sys.stdout = _silent_buffer
sys.stderr = _silent_buffer

# Try to import mshr for cylindrical mesh generation (before FEniCS imports)
try:
    import mshr
    MSHR_AVAILABLE = True
except ImportError:
    MSHR_AVAILABLE = False

try:
    from dolfin import set_log_active, set_log_level, LogLevel, parameters, MeshFunction
    from fenics import (
        IntervalMesh,
        RectangleMesh,
        BoxMesh,
        Point,
        FunctionSpace,
        DirichletBC,
        Constant,
        Expression,
        project,
        TrialFunction,
        TestFunction,
        Function,
        near,
        inner,
        grad,
        dx,
        solve,
        lhs,
        rhs,
        tr,
        sqrt,
        Identity,
        dot,
        sym,
        VectorFunctionSpace,
        SubDomain,
    )
finally:
    # Restore stdout/stderr after imports
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    _silent_buffer.close()

# Set FEniCS parameters AFTER importing
set_log_active(False)
set_log_level(LogLevel.ERROR)
parameters["std_out_all_processes"] = False

# Set form compiler parameters (may vary by FEniCS version)
try:
    parameters["form_compiler"]["log_level"] = 50  # ERROR level
except (KeyError, TypeError):
    pass
try:
    parameters["form_compiler"]["cpp_optimize"] = True
except (KeyError, TypeError):
    pass
try:
    parameters["form_compiler"]["optimize"] = True
except (KeyError, TypeError):
    pass
try:
    parameters["form_compiler"]["representation"] = "uflacs"
except (KeyError, TypeError):
    pass  # Some FEniCS versions may not have this option

from mcp.server.fastmcp import FastMCP


# ─────────────────────────────────
# 1. 通用时间序列场数据结构
# ─────────────────────────────────

@dataclass
class TimeSeriesField:
    """
    通用的标量场时间序列表示，统一 1D / 2D / 3D 输出格式。

    coords: [N][3] 所有自由度/点嵌入 3D 坐标
    values: [Nt][N] 每个时间步的场值
    times: [Nt]     时间数组
    dim:   1 / 2 / 3 实际 PDE 维数（用于 meta & 标题）
    meta:  额外信息（变量名、单位、PDE 类型等）
    """
    coords: List[List[float]]
    values: List[List[float]]
    times: List[float]
    dim: int
    meta: Dict[str, Any]


@dataclass
class SolveResult:
    """求解结果：包含文件路径和元数据"""
    data_file: str  # 保存 TimeSeriesField 的 pickle 文件路径
    dim: int
    meta: Dict[str, Any]


@dataclass
class PlotResult:
    """画图结果：生成的交互式 HTML 文件路径"""
    html_path: str


# ─────────────────────────────────
# 2. 1D Heat 求解（直接输出 TimeSeriesField）
# ─────────────────────────────────

def _solve_heat_1d_raw(
    length: float,
    nx: int,
    diffusivity: float,
    T_left: float,
    T_right: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
    initial_wavenumber: float = 1.0,
) -> TimeSeriesField:
    """内部函数：用 FEniCS 解 1D 热方程，并直接构造 TimeSeriesField。"""

    # 把 FEniCS 的 stdout/stderr 全部重定向防止污染 JSONRPC
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        L = length
        kappa = diffusivity

        mesh = IntervalMesh(nx, 0.0, L)
        V = FunctionSpace(mesh, "P", 1)

        # 边界
        def left(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        def right(x, on_boundary):
            return on_boundary and near(x[0], L)

        bc_left = DirichletBC(V, Constant(T_left), left)
        bc_right = DirichletBC(V, Constant(T_right), right)
        bcs = [bc_left, bc_right]

        # 热源项
        if source_type == "constant":
            f = Constant(source_value)
        else:  # source_type == "none"
            f = Constant(0.0)

        u = TrialFunction(V)
        v = TestFunction(V)

        coords_1d = V.tabulate_dof_coordinates().reshape(-1)
        order = np.argsort(coords_1d)
        x_sorted = coords_1d[order]

        snapshots = []
        times = []

        if steady:
            # 稳态：-k Δu = f  →  ∫ k ∇u·∇v dx = ∫ f v dx
            a = kappa * inner(grad(u), grad(v)) * dx
            L_form = f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bcs)
            
            values = u_sol.vector().get_local()
            snapshots.append(values[order].tolist())
            times.append(0.0)
        else:
            # 瞬态：u_t - k Δu = f  →  backward Euler
            # ∫ u^{n+1} v dx + dt * k ∫ ∇u^{n+1}·∇v dx = ∫ u^n v dx + dt ∫ f v dx
            u_n = Function(V)
            
            # Set initial condition based on type
            if initial_type == "zero":
                u_n.assign(Constant(0.0))
            elif initial_type == "cosine":
                # u(x,0) = A * cos(k * x)
                # Project Expression onto function space to ensure proper interpolation
                A = initial_amplitude
                k = initial_wavenumber
                init_expr = Expression(f"{A} * cos({k} * x[0])", degree=2)
                u_n = project(init_expr, V)
            elif initial_type == "sine":
                # u(x,0) = A * sin(k * x)
                A = initial_amplitude
                k = initial_wavenumber
                init_expr = Expression(f"{A} * sin({k} * x[0])", degree=2)
                u_n = project(init_expr, V)
            else:  # initial_type == "constant" or default
                u_n.assign(Constant(T_initial))
            
            # Apply boundary conditions to initial condition
            # This ensures boundary values are correct, but interior should show cosine/sine
            for bc in bcs:
                bc.apply(u_n.vector())
            
            # Save initial condition (t=0) - this should show cosine/sine at interior points
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values[order].tolist())
            times.append(0.0)
            
            a = (u * v + dt * kappa * inner(grad(u), grad(v))) * dx
            L_form = u_n * v * dx + dt * f * v * dx
            
            u_sol = Function(V)

            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bcs)

                values = u_sol.vector().get_local()
                snapshots.append(values[order].tolist())
                times.append(t)

                u_n.assign(u_sol)
                L_form = u_n * v * dx + dt * f * v * dx

    # 嵌入到 3D：y=z=0
    coords = [[float(x), 0.0, 0.0] for x in x_sorted]
    field = TimeSeriesField(
        coords=coords,
        values=snapshots,
        times=times,
        dim=1,
        meta={
            "name": "temperature",
            "unit": "°C",
            "pde": "heat",
            "coordinate_system": "cartesian",
            "length": length,
            "source_type": source_type,
            "source_value": source_value,
            "steady": steady,
        },
    )
    return field


# ─────────────────────────────────
# 3. 2D Heat 求解（矩形区域，常数边界温度）
# ─────────────────────────────────

def _solve_heat_2d_raw(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    diffusivity: float,
    T_boundary: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
    initial_wavenumber: float = 1.0,
) -> TimeSeriesField:
    """用 FEniCS 解 2D 热方程，矩形 [0,Lx]×[0,Ly]，常数 Dirichlet 边界。"""

    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        kappa = diffusivity

        mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)
        V = FunctionSpace(mesh, "P", 1)

        # 整个外边界都为常数 Dirichlet
        def boundary(x, on_boundary):
            return on_boundary

        bc = DirichletBC(V, Constant(T_boundary), boundary)

        # 热源项
        if source_type == "constant":
            f = Constant(source_value)
        else:  # source_type == "none"
            f = Constant(0.0)

        u = TrialFunction(V)
        v = TestFunction(V)

        coords_2d = V.tabulate_dof_coordinates().reshape(-1, 2)
        snapshots = []
        times = []

        if steady:
            # 稳态：-k Δu = f  →  ∫ k ∇u·∇v dx = ∫ f v dx
            a = kappa * inner(grad(u), grad(v)) * dx
            L_form = f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bc)
            
            values = u_sol.vector().get_local()
            snapshots.append(values.tolist())
            times.append(0.0)
        else:
            # 瞬态：u_t - k Δu = f  →  backward Euler
            # ∫ u^{n+1} v dx + dt * k ∫ ∇u^{n+1}·∇v dx = ∫ u^n v dx + dt ∫ f v dx
            u_n = Function(V)
            
            # Set initial condition based on type
            if initial_type == "zero":
                u_n.assign(Constant(0.0))
            elif initial_type == "cosine":
                # u(x,y,0) = A * cos(k * x) * cos(k * y)
                A = initial_amplitude
                k = initial_wavenumber
                init_expr = Expression(f"{A} * cos({k} * x[0]) * cos({k} * x[1])", degree=2)
                u_n = project(init_expr, V)
            elif initial_type == "sine":
                # u(x,y,0) = A * sin(k * x) * sin(k * y)
                A = initial_amplitude
                k = initial_wavenumber
                init_expr = Expression(f"{A} * sin({k} * x[0]) * sin({k} * x[1])", degree=2)
                u_n = project(init_expr, V)
            else:  # initial_type == "constant" or default
                u_n.assign(Constant(T_initial))
            
            # Apply boundary conditions to initial condition
            bc.apply(u_n.vector())
            
            # Save initial condition (t=0) - this should show cosine/sine at interior points
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values.tolist())
            times.append(0.0)
            
            a = u * v * dx + dt * kappa * inner(grad(u), grad(v)) * dx
            L_form = u_n * v * dx + dt * f * v * dx
            
            u_sol = Function(V)

            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bc)

                values = u_sol.vector().get_local()
                snapshots.append(values.tolist())
                times.append(t)

                u_n.assign(u_sol)
                L_form = u_n * v * dx + dt * f * v * dx

    # 嵌入 3D：z=0
    coords = [[float(x), float(y), 0.0] for (x, y) in coords_2d]
    field = TimeSeriesField(
        coords=coords,
        values=snapshots,
        times=times,
        dim=2,
        meta={
            "name": "temperature",
            "unit": "°C",
            "pde": "heat",
            "coordinate_system": "cartesian",
            "Lx": Lx,
            "Ly": Ly,
            "source_type": source_type,
            "source_value": source_value,
            "steady": steady,
        },
    )
    return field


# ─────────────────────────────────
# 4. 3D Heat 求解（盒体区域，常数边界温度）
# ─────────────────────────────────

def _solve_heat_3d_raw(
    Lx: float,
    Ly: float,
    Lz: float,
    nx: int,
    ny: int,
    nz: int,
    diffusivity: float,
    T_boundary: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
    initial_wavenumber: float = 1.0,
    geometry_type: str = "box",  # "box" or "cylinder"
    cylinder_radius: Optional[float] = None,  # For cylindrical geometry
    T_left: Optional[float] = None,  # Directional BC: left face temperature
    T_right: Optional[float] = None,  # Directional BC: right face temperature
    T_side: Optional[float] = None,  # Side/wall boundary temperature (for cylinder)
    core_radius: Optional[float] = None,  # Radius of high-conductivity core
    core_diffusivity: Optional[float] = None,  # Diffusivity of core material
) -> TimeSeriesField:
    """
    解 3D 热方程，支持：
    - 立方体 [0,Lx]×[0,Ly]×[0,Lz] 或 圆柱体 (半径 cylinder_radius, 长度 Lx)
    - 统一边界温度 (T_boundary) 或 方向边界 (T_left, T_right, T_side)
    - 均匀材料或复合材料 (core_radius + core_diffusivity)
    """

    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        # Determine geometry and mesh
        if geometry_type == "cylinder" and cylinder_radius is not None:
            # Create cylindrical mesh
            if MSHR_AVAILABLE:
                # Use mshr for true cylindrical geometry
                cylinder_length = Lx  # Use Lx as the length of the cylinder
                center = Point(0.0, 0.0, 0.0)
                axis = Point(cylinder_length, 0.0, 0.0)
                cylinder = mshr.Cylinder(center, axis, cylinder_radius, cylinder_radius)
                # Generate mesh with appropriate resolution
                mesh_resolution = max(nx, int(cylinder_radius * 20))
                mesh = mshr.generate_mesh(cylinder, mesh_resolution)
            else:
                # Fallback: Use box mesh and mark cylindrical boundary
                # Approximate cylinder as a box with high aspect ratio
                # IMPORTANT: Even though we use BoxMesh, we still mark it as cylinder in metadata
                mesh = BoxMesh(Point(0.0, -cylinder_radius, -cylinder_radius), 
                              Point(Lx, cylinder_radius, cylinder_radius), 
                              nx, int(ny * cylinder_radius * 2), int(nz * cylinder_radius * 2))
                # Note: geometry_type="cylinder" will still be set in metadata below
        else:
            # Standard box mesh
            mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(Lx, Ly, Lz), nx, ny, nz)
        
        V = FunctionSpace(mesh, "P", 1)

        # Handle heterogeneous material properties
        has_composite = (core_radius is not None and core_diffusivity is not None)
        if has_composite:
            # Create subdomain marking for core region
            class CoreSubDomain(SubDomain):
                def inside(self, x, on_boundary):
                    r = sqrt(x[1]**2 + x[2]**2)  # Distance from axis
                    return r < core_radius
            
            # Mark subdomains: 0 = outer material, 1 = core material
            subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
            subdomains.set_all(0)  # Default to outer material
            core = CoreSubDomain()
            core.mark(subdomains, 1)
            
            # Create spatially-varying diffusivity function
            # Use DG0 (piecewise constant) space matching subdomain marking
            V_kappa = FunctionSpace(mesh, "DG", 0)
            kappa_func = Function(V_kappa)
            
            # Assign diffusivity values based on subdomain marking
            # For DG0 space, there's one dof per cell, aligned with subdomains array
            kappa_values = kappa_func.vector().get_local()
            subdomain_array = subdomains.array()
            
            # Assign diffusivity based on subdomain marking
            for i in range(len(kappa_values)):
                if subdomain_array[i] == 1:  # Core
                    kappa_values[i] = core_diffusivity
                else:  # Outer material
                    kappa_values[i] = diffusivity
            
            kappa_func.vector().set_local(kappa_values)
            kappa_func.vector().apply("insert")
            
            kappa = kappa_func  # Use function for spatially-varying kappa
        else:
            kappa = Constant(diffusivity)  # Uniform material

        # Handle boundary conditions: directional or uniform
        bcs = []
        use_directional_bc = (T_left is not None or T_right is not None or T_side is not None)
        
        if use_directional_bc:
            # Directional boundary conditions
            if geometry_type == "cylinder":
                # Cylindrical geometry: left end, right end, and curved surface
                def left_boundary_cylinder(x, on_boundary):
                    return on_boundary and near(x[0], 0.0)
                
                def right_boundary_cylinder(x, on_boundary):
                    return on_boundary and near(x[0], Lx)
                
                def side_boundary_cylinder(x, on_boundary):
                    if not on_boundary:
                        return False
                    r = sqrt(x[1]**2 + x[2]**2)
                    return not (near(x[0], 0.0) or near(x[0], Lx)) and near(r, cylinder_radius)
                
                if T_left is not None:
                    bcs.append(DirichletBC(V, Constant(T_left), left_boundary_cylinder))
                if T_right is not None:
                    bcs.append(DirichletBC(V, Constant(T_right), right_boundary_cylinder))
                if T_side is not None:
                    bcs.append(DirichletBC(V, Constant(T_side), side_boundary_cylinder))
                elif T_left is None and T_right is None:
                    # If only side BC specified, use it
                    pass
            else:
                # Box geometry: left face (x=0), right face (x=Lx), other faces
                def left_face(x, on_boundary):
                    return on_boundary and near(x[0], 0.0)
                
                def right_face(x, on_boundary):
                    return on_boundary and near(x[0], Lx)
                
                def other_faces(x, on_boundary):
                    if not on_boundary:
                        return False
                    return not (near(x[0], 0.0) or near(x[0], Lx))
                
                if T_left is not None:
                    bcs.append(DirichletBC(V, Constant(T_left), left_face))
                if T_right is not None:
                    bcs.append(DirichletBC(V, Constant(T_right), right_face))
                if T_side is not None:
                    bcs.append(DirichletBC(V, Constant(T_side), other_faces))
        else:
            # Uniform boundary condition on all boundaries
            def boundary(x, on_boundary):
                return on_boundary
            bcs = [DirichletBC(V, Constant(T_boundary), boundary)]

        # 热源项
        if source_type == "constant":
            f = Constant(source_value)
        else:  # source_type == "none"
            f = Constant(0.0)

        u = TrialFunction(V)
        v = TestFunction(V)
        
        # For cylindrical geometry, use radial weighting in the PDE form
        # In cylindrical coordinates: volume element is r dr dθ dz
        # Weak form should be weighted by r = sqrt(y² + z²)
        if geometry_type == "cylinder" and cylinder_radius is not None:
            # Define radial coordinate expression for weighting
            # r = sqrt(y² + z²) where y=x[1], z=x[2]
            r_expr = Expression("sqrt(x[1]*x[1] + x[2]*x[2])", degree=2)
        else:
            r_expr = Constant(1.0)  # No weighting for Cartesian

        coords_3d = V.tabulate_dof_coordinates().reshape(-1, 3)
        snapshots = []
        times = []

        if steady:
            # 稳态：-k Δu = f  
            # For cylindrical: ∫ k * r * ∇u·∇v dr dθ dz = ∫ r * f * v dr dθ dz
            # For Cartesian: ∫ k * ∇u·∇v dx = ∫ f * v dx
            a = kappa * r_expr * inner(grad(u), grad(v)) * dx
            L_form = r_expr * f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bcs)
            
            values = u_sol.vector().get_local()
            snapshots.append(values.tolist())
            times.append(0.0)
        else:
            # 瞬态：u_t - k Δu = f  →  backward Euler
            # ∫ u^{n+1} v dx + dt * k ∫ ∇u^{n+1}·∇v dx = ∫ u^n v dx + dt ∫ f v dx
            u_n = Function(V)
            
            # Set initial condition based on type
            if initial_type == "zero":
                u_n.assign(Constant(0.0))
            elif initial_type == "cosine":
                # u(x,y,z,0) = A * cos(k * x) * cos(k * y) * cos(k * z)
                A = initial_amplitude
                k = initial_wavenumber
                init_expr = Expression(f"{A} * cos({k} * x[0]) * cos({k} * x[1]) * cos({k} * x[2])", degree=2)
                u_n = project(init_expr, V)
            elif initial_type == "sine":
                # u(x,y,z,0) = A * sin(k * x) * sin(k * y) * sin(k * z)
                A = initial_amplitude
                k = initial_wavenumber
                init_expr = Expression(f"{A} * sin({k} * x[0]) * sin({k} * x[1]) * sin({k} * x[2])", degree=2)
                u_n = project(init_expr, V)
            else:  # initial_type == "constant" or default
                u_n.assign(Constant(T_initial))
            
            # Apply boundary conditions to initial condition
            for bc in bcs:
                bc.apply(u_n.vector())
            
            # Save initial condition (t=0) - this should show cosine/sine at interior points
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values.tolist())
            times.append(0.0)
            
            # 瞬态：u_t - k Δu = f  →  backward Euler
            # For cylindrical: ∫ r * u^{n+1} * v dr dθ dz + dt * k * ∫ r * ∇u^{n+1}·∇v dr dθ dz
            #                   = ∫ r * u^n * v dr dθ dz + dt * ∫ r * f * v dr dθ dz
            # For Cartesian: ∫ u^{n+1} v dx + dt * k ∫ ∇u^{n+1}·∇v dx = ∫ u^n v dx + dt ∫ f v dx
            a = r_expr * u * v * dx + dt * kappa * r_expr * inner(grad(u), grad(v)) * dx
            L_form = r_expr * u_n * v * dx + dt * r_expr * f * v * dx
            
            u_sol = Function(V)

            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bcs)

                values = u_sol.vector().get_local()
                snapshots.append(values.tolist())
                times.append(t)

                u_n.assign(u_sol)
                L_form = r_expr * u_n * v * dx + dt * r_expr * f * v * dx

    coords = [[float(x), float(y), float(z)] for (x, y, z) in coords_3d]
    
    # Build metadata with geometry and material info
    meta = {
        "name": "temperature",
        "unit": "°C",
        "pde": "heat",
        "coordinate_system": "cartesian" if geometry_type == "box" else "cylindrical",
        "Lx": Lx,
        "Ly": Ly if geometry_type == "box" else (cylinder_radius * 2 if cylinder_radius else Ly),
        "Lz": Lz if geometry_type == "box" else (cylinder_radius * 2 if cylinder_radius else Lz),
        "geometry_type": geometry_type,
        "source_type": source_type,
        "source_value": source_value,
        "steady": steady,
    }
    
    if geometry_type == "cylinder" and cylinder_radius is not None:
        meta["cylinder_radius"] = cylinder_radius
    
    if use_directional_bc:
        if T_left is not None:
            meta["T_left"] = T_left
        if T_right is not None:
            meta["T_right"] = T_right
        if T_side is not None:
            meta["T_side"] = T_side
    else:
        meta["T_boundary"] = T_boundary
    
    if has_composite:
        meta["core_radius"] = core_radius
        meta["core_diffusivity"] = core_diffusivity
        meta["base_diffusivity"] = diffusivity
    else:
        meta["diffusivity"] = diffusivity
    
    field = TimeSeriesField(
        coords=coords,
        values=snapshots,
        times=times,
        dim=3,
        meta=meta,
    )
    return field


# ─────────────────────────────────
# 4.6 Coordinate System Specific Raw Solvers
# ─────────────────────────────────

def _solve_heat_1d_cylindrical_raw(
    r_inner: float,
    r_outer: float,
    nr: int,
    diffusivity: float,
    T_inner: float,
    T_outer: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> TimeSeriesField:
    """
    Internal function: Solve 1D radial heat equation in cylindrical coordinates.
    
    The heat equation in cylindrical coordinates (1D radial):
        u_t = k * (1/r) * d/dr (r * du/dr)
    
    Weak form (weighted by r):
        ∫ r * u_t * v dr = k * ∫ r * grad(u) · grad(v) dr
    
    For steady state:
        k * ∫ r * grad(u) · grad(v) dr = ∫ r * f * v dr
    """
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        kappa = diffusivity
        
        # Create mesh in radial coordinate [r_inner, r_outer]
        mesh = IntervalMesh(nr, r_inner, r_outer)
        V = FunctionSpace(mesh, "P", 1)
        
        # Define radial coordinate as Expression
        r_expr = Expression("x[0]", degree=1)
        r = project(r_expr, V)  # Project r onto function space for use in forms
        
        # Boundary conditions
        bcs = []
        if r_inner > 1e-10:  # Only apply inner BC if r_inner is not zero
            def inner_boundary(x, on_boundary):
                return on_boundary and near(x[0], r_inner)
            bcs.append(DirichletBC(V, Constant(T_inner), inner_boundary))
        
        def outer_boundary(x, on_boundary):
            return on_boundary and near(x[0], r_outer)
        bcs.append(DirichletBC(V, Constant(T_outer), outer_boundary))
        
        # Source term
        if source_type == "constant":
            f = Constant(source_value)
        else:
            f = Constant(0.0)
        
        u = TrialFunction(V)
        v = TestFunction(V)
        
        # Radial weighting: multiply by r (cylindrical coordinate factor)
        # For cylindrical 1D: the volume element is r * dr
        r_weighted = Function(V)
        r_weighted.vector()[:] = r.vector()[:]
        
        coords_1d = V.tabulate_dof_coordinates().reshape(-1)
        order = np.argsort(coords_1d)
        r_sorted = coords_1d[order]
        
        snapshots = []
        times = []
        
        if steady:
            # Steady state: k * ∫ r * grad(u) · grad(v) dr = ∫ r * f * v dr
            # We need to use a weighted measure. In FEniCS, we can create a custom measure.
            # However, a simpler approach: use r as a function and multiply in the form.
            # For cylindrical coordinates, we use: k * r * grad(u) · grad(v)
            # But we need to integrate with measure r*dx. 
            # We'll use an approximation: multiply the integrand by r
            # Since r is a function, we'll create a measure that includes r weighting
            # For now, use a simpler approach: multiply diffusivity by r at each point
            
            # Create r-weighted form: ∫ k * r * grad(u) · grad(v) dx
            # where r is the radial coordinate at each point
            r_vals = r.vector().get_local()
            
            # Alternative: Use UFL to create r-weighted form directly
            # The weak form should be: ∫ k * r * grad(u) · grad(v) dx
            # where r is Expression("x[0]", degree=1)
            a = kappa * r_expr * inner(grad(u), grad(v)) * dx
            L_form = r_expr * f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bcs)
            
            values = u_sol.vector().get_local()
            snapshots.append(values[order].tolist())
            times.append(0.0)
        else:
            # Transient: ∫ r * u^{n+1} * v dr + dt * k * ∫ r * grad(u^{n+1}) · grad(v) dr
            #              = ∫ r * u^n * v dr + dt * ∫ r * f * v dr
            u_n = Function(V)
            
            if initial_type == "constant":
                u_n.assign(Constant(T_initial))
            else:
                u_n.assign(Constant(T_initial))
            
            for bc in bcs:
                bc.apply(u_n.vector())
            
            # Save initial condition
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values[order].tolist())
            times.append(0.0)
            
            a = r_expr * u * v * dx + dt * kappa * r_expr * inner(grad(u), grad(v)) * dx
            L_form = r_expr * u_n * v * dx + dt * r_expr * f * v * dx
            
            u_sol = Function(V)
            
            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bcs)
                
                values = u_sol.vector().get_local()
                snapshots.append(values[order].tolist())
                times.append(t)
                
                u_n.assign(u_sol)
                L_form = r_expr * u_n * v * dx + dt * r_expr * f * v * dx
        
        # Embed in 3D: represent as radial coordinate along x-axis
        coords = [[float(r_val), 0.0, 0.0] for r_val in r_sorted]
        
        field = TimeSeriesField(
            coords=coords,
            values=snapshots,
            times=times,
            dim=1,
            meta={
                "name": "temperature",
                "unit": "°C",
                "pde": "heat",
                "coordinate_system": "cylindrical",
                "geometry_type": "cylinder" if r_inner < 1e-10 else "annulus",
                "r_inner": r_inner,
                "r_outer": r_outer,
                "source_type": source_type,
                "source_value": source_value,
                "steady": steady,
            },
        )
        return field


def _solve_heat_1d_spherical_raw(
    r_inner: float,
    r_outer: float,
    nr: int,
    diffusivity: float,
    T_inner: float,
    T_outer: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> TimeSeriesField:
    """
    Internal function: Solve 1D radial heat equation in spherical coordinates.
    
    The heat equation in spherical coordinates (1D radial):
        u_t = k * (1/r²) * d/dr (r² * du/dr)
    
    Weak form (weighted by r²):
        ∫ r² * u_t * v dr = k * ∫ r² * grad(u) · grad(v) dr
    
    For steady state:
        k * ∫ r² * grad(u) · grad(v) dr = ∫ r² * f * v dr
    """
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        kappa = diffusivity
        
        # Create mesh in radial coordinate [r_inner, r_outer]
        mesh = IntervalMesh(nr, r_inner, r_outer)
        V = FunctionSpace(mesh, "P", 1)
        
        # Define radial coordinate squared as Expression
        r_expr = Expression("x[0]", degree=1)
        r_squared_expr = Expression("x[0] * x[0]", degree=2)
        
        # Boundary conditions
        bcs = []
        if r_inner > 1e-10:  # Only apply inner BC if r_inner is not zero
            def inner_boundary(x, on_boundary):
                return on_boundary and near(x[0], r_inner)
            bcs.append(DirichletBC(V, Constant(T_inner), inner_boundary))
        
        def outer_boundary(x, on_boundary):
            return on_boundary and near(x[0], r_outer)
        bcs.append(DirichletBC(V, Constant(T_outer), outer_boundary))
        
        # Source term
        if source_type == "constant":
            f = Constant(source_value)
        else:
            f = Constant(0.0)
        
        u = TrialFunction(V)
        v = TestFunction(V)
        
        coords_1d = V.tabulate_dof_coordinates().reshape(-1)
        order = np.argsort(coords_1d)
        r_sorted = coords_1d[order]
        
        snapshots = []
        times = []
        
        if steady:
            # Steady state: k * ∫ r² * grad(u) · grad(v) dr = ∫ r² * f * v dr
            a = kappa * r_squared_expr * inner(grad(u), grad(v)) * dx
            L_form = r_squared_expr * f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bcs)
            
            values = u_sol.vector().get_local()
            snapshots.append(values[order].tolist())
            times.append(0.0)
        else:
            # Transient: ∫ r² * u^{n+1} * v dr + dt * k * ∫ r² * grad(u^{n+1}) · grad(v) dr
            #              = ∫ r² * u^n * v dr + dt * ∫ r² * f * v dr
            u_n = Function(V)
            
            if initial_type == "constant":
                u_n.assign(Constant(T_initial))
            else:
                u_n.assign(Constant(T_initial))
            
            for bc in bcs:
                bc.apply(u_n.vector())
            
            # Save initial condition
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values[order].tolist())
            times.append(0.0)
            
            a = r_squared_expr * u * v * dx + dt * kappa * r_squared_expr * inner(grad(u), grad(v)) * dx
            L_form = r_squared_expr * u_n * v * dx + dt * r_squared_expr * f * v * dx
            
            u_sol = Function(V)
            
            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bcs)
                
                values = u_sol.vector().get_local()
                snapshots.append(values[order].tolist())
                times.append(t)
                
                u_n.assign(u_sol)
                L_form = r_squared_expr * u_n * v * dx + dt * r_squared_expr * f * v * dx
        
        # Embed in 3D: represent as radial coordinate along x-axis
        coords = [[float(r_val), 0.0, 0.0] for r_val in r_sorted]
        
        field = TimeSeriesField(
            coords=coords,
            values=snapshots,
            times=times,
            dim=1,
            meta={
                "name": "temperature",
                "unit": "°C",
                "pde": "heat",
                "coordinate_system": "spherical",
                "geometry_type": "sphere" if r_inner < 1e-10 else "spherical_shell",
                "r_inner": r_inner,
                "r_outer": r_outer,
                "source_type": source_type,
                "source_value": source_value,
                "steady": steady,
            },
        )
        return field


def _solve_heat_2d_cylindrical_raw(
    r_inner: float,
    r_outer: float,
    z_length: float,
    nr: int,
    nz: int,
    diffusivity: float,
    T_boundary: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> TimeSeriesField:
    """
    Internal function: Solve 2D axisymmetric heat equation in cylindrical coordinates (r-z plane).
    
    The heat equation in axisymmetric cylindrical coordinates (r, z):
        u_t = k * [(1/r) * d/dr (r * du/dr) + d²u/dz²]
    
    Weak form (weighted by r, volume element dV = r dr dz):
        ∫ r * u_t * v dr dz = k * ∫ r * grad(u) · grad(v) dr dz
    """
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        kappa = diffusivity
        
        # Create mesh in (r, z) plane: [r_inner, r_outer] × [0, z_length]
        # x[0] = r (radial), x[1] = z (axial)
        mesh = RectangleMesh(Point(r_inner, 0.0), Point(r_outer, z_length), nr, nz)
        V = FunctionSpace(mesh, "P", 1)
        
        # Define radial coordinate as Expression for weighting
        r_expr = Expression("x[0]", degree=1)  # x[0] is the radial coordinate
        
        # Boundary conditions: all boundaries at T_boundary
        def boundary(x, on_boundary):
            return on_boundary
        bcs = [DirichletBC(V, Constant(T_boundary), boundary)]
        
        # Source term
        if source_type == "constant":
            f = Constant(source_value)
        else:
            f = Constant(0.0)
        
        u = TrialFunction(V)
        v = TestFunction(V)
        
        coords_2d = V.tabulate_dof_coordinates().reshape(-1, 2)
        snapshots = []
        times = []
        
        if steady:
            # Steady state: k * ∫ r * grad(u) · grad(v) dr dz = ∫ r * f * v dr dz
            a = kappa * r_expr * inner(grad(u), grad(v)) * dx
            L_form = r_expr * f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bcs)
            
            values = u_sol.vector().get_local()
            snapshots.append(values.tolist())
            times.append(0.0)
        else:
            # Transient: ∫ r * u^{n+1} * v dr dz + dt * k * ∫ r * grad(u^{n+1}) · grad(v) dr dz
            #              = ∫ r * u^n * v dr dz + dt * ∫ r * f * v dr dz
            u_n = Function(V)
            
            if initial_type == "constant":
                u_n.assign(Constant(T_initial))
            else:
                u_n.assign(Constant(T_initial))
            
            for bc in bcs:
                bc.apply(u_n.vector())
            
            # Save initial condition
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values.tolist())
            times.append(0.0)
            
            a = r_expr * u * v * dx + dt * kappa * r_expr * inner(grad(u), grad(v)) * dx
            L_form = r_expr * u_n * v * dx + dt * r_expr * f * v * dx
            
            u_sol = Function(V)
            
            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bcs)
                
                values = u_sol.vector().get_local()
                snapshots.append(values.tolist())
                times.append(t)
                
                u_n.assign(u_sol)
                L_form = r_expr * u_n * v * dx + dt * r_expr * f * v * dx
        
        # Embed in 3D: (r, z) -> (x=r*cos(0), y=r*sin(0), z=z) for visualization
        # For axisymmetric, we show as (r, 0, z) initially
        coords = [[float(r), 0.0, float(z)] for (r, z) in coords_2d]
        
        field = TimeSeriesField(
            coords=coords,
            values=snapshots,
            times=times,
            dim=2,
            meta={
                "name": "temperature",
                "unit": "°C",
                "pde": "heat",
                "coordinate_system": "cylindrical",
                "geometry_type": "cylinder" if r_inner < 1e-10 else "annular_cylinder",
                "r_inner": r_inner,
                "r_outer": r_outer,
                "z_length": z_length,
                "source_type": source_type,
                "source_value": source_value,
                "steady": steady,
            },
        )
        return field


def _solve_heat_2d_spherical_raw(
    r_inner: float,
    r_outer: float,
    nr: int,
    ntheta: int,
    diffusivity: float,
    T_boundary: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> TimeSeriesField:
    """
    Internal function: Solve 2D axisymmetric heat equation in spherical coordinates (r-θ plane).
    
    The heat equation in axisymmetric spherical coordinates (r, θ):
        u_t = k * [(1/r²) * d/dr (r² * du/dr) + (1/(r² sin θ)) * d/dθ (sin θ * du/dθ)]
    
    Weak form (weighted by r² sin θ, volume element dV = r² sin θ dr dθ):
        ∫ r² sin θ * u_t * v dr dθ = k * ∫ r² sin θ * [grad_r(u)·grad_r(v) + (1/r²) grad_θ(u)·grad_θ(v)] dr dθ
    """
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        kappa = diffusivity
        
        # Create mesh in (r, θ) plane: [r_inner, r_outer] × [0, π]
        # x[0] = r (radial), x[1] = θ (polar angle)
        mesh = RectangleMesh(Point(r_inner, 0.0), Point(r_outer, np.pi), nr, ntheta)
        V = FunctionSpace(mesh, "P", 1)
        
        # Define weighting expressions
        r_expr = Expression("x[0]", degree=1)  # Radial coordinate
        r_squared_expr = Expression("x[0] * x[0]", degree=2)
        sin_theta_expr = Expression("sin(x[1])", degree=2)
        r_squared_sin_theta_expr = Expression("x[0] * x[0] * sin(x[1])", degree=2)
        
        # Boundary conditions: all boundaries at T_boundary
        def boundary(x, on_boundary):
            return on_boundary
        bcs = [DirichletBC(V, Constant(T_boundary), boundary)]
        
        # Source term
        if source_type == "constant":
            f = Constant(source_value)
        else:
            f = Constant(0.0)
        
        u = TrialFunction(V)
        v = TestFunction(V)
        
        coords_2d = V.tabulate_dof_coordinates().reshape(-1, 2)
        snapshots = []
        times = []
        
        if steady:
            # Steady state: The Laplacian in spherical coordinates has r² and sin(θ) factors
            # Weak form: k * ∫ r² sin(θ) * grad(u) · grad(v) dr dθ = ∫ r² sin(θ) * f * v dr dθ
            # Note: grad(u) already accounts for both r and θ components
            a = kappa * r_squared_sin_theta_expr * inner(grad(u), grad(v)) * dx
            L_form = r_squared_sin_theta_expr * f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bcs)
            
            values = u_sol.vector().get_local()
            snapshots.append(values.tolist())
            times.append(0.0)
        else:
            # Transient
            u_n = Function(V)
            
            if initial_type == "constant":
                u_n.assign(Constant(T_initial))
            else:
                u_n.assign(Constant(T_initial))
            
            for bc in bcs:
                bc.apply(u_n.vector())
            
            # Save initial condition
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values.tolist())
            times.append(0.0)
            
            a = r_squared_sin_theta_expr * u * v * dx + dt * kappa * r_squared_sin_theta_expr * inner(grad(u), grad(v)) * dx
            L_form = r_squared_sin_theta_expr * u_n * v * dx + dt * r_squared_sin_theta_expr * f * v * dx
            
            u_sol = Function(V)
            
            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bcs)
                
                values = u_sol.vector().get_local()
                snapshots.append(values.tolist())
                times.append(t)
                
                u_n.assign(u_sol)
                L_form = r_squared_sin_theta_expr * u_n * v * dx + dt * r_squared_sin_theta_expr * f * v * dx
        
        # Embed in 3D: Convert (r, θ) to Cartesian (x, y, z)
        # For axisymmetric, we can visualize by setting φ=0: (r sin θ, 0, r cos θ)
        coords = []
        for (r, theta) in coords_2d:
            x_cart = r * np.sin(theta)
            y_cart = 0.0  # φ = 0 for axisymmetric
            z_cart = r * np.cos(theta)
            coords.append([float(x_cart), float(y_cart), float(z_cart)])
        
        field = TimeSeriesField(
            coords=coords,
            values=snapshots,
            times=times,
            dim=2,
            meta={
                "name": "temperature",
                "unit": "°C",
                "pde": "heat",
                "coordinate_system": "spherical",
                "geometry_type": "sphere" if r_inner < 1e-10 else "spherical_shell",
                "r_inner": r_inner,
                "r_outer": r_outer,
                "source_type": source_type,
                "source_value": source_value,
                "steady": steady,
            },
        )
        return field


def _solve_heat_3d_spherical_raw(
    r_inner: float,
    r_outer: float,
    nr: int,
    ntheta: int,
    nphi: int,
    diffusivity: float,
    T_boundary: float,
    T_initial: float,
    dt: float,
    num_steps: int,
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> TimeSeriesField:
    """
    Internal function: Solve 3D heat equation in spherical coordinates (r, θ, φ).
    
    The heat equation in full 3D spherical coordinates:
        u_t = k * [(1/r²) * d/dr (r² * du/dr) + (1/(r² sin θ)) * d/dθ (sin θ * du/dθ) 
                  + (1/(r² sin² θ)) * d²u/dφ²]
    
    Weak form (weighted by r² sin θ, volume element dV = r² sin θ dr dθ dφ):
        ∫ r² sin θ * u_t * v dr dθ dφ = k * ∫ r² sin θ * grad(u) · grad(v) dr dθ dφ
    """
    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        kappa = diffusivity
        
        # Create mesh in (r, θ, φ) space: [r_inner, r_outer] × [0, π] × [0, 2π]
        # x[0] = r (radial), x[1] = θ (polar), x[2] = φ (azimuthal)
        mesh = BoxMesh(
            Point(r_inner, 0.0, 0.0),
            Point(r_outer, np.pi, 2.0 * np.pi),
            nr, ntheta, nphi
        )
        V = FunctionSpace(mesh, "P", 1)
        
        # Define weighting expressions for spherical coordinates
        r_expr = Expression("x[0]", degree=1)  # Radial coordinate
        r_squared_expr = Expression("x[0] * x[0]", degree=2)
        sin_theta_expr = Expression("sin(x[1])", degree=2)  # sin(θ)
        r_squared_sin_theta_expr = Expression("x[0] * x[0] * sin(x[1])", degree=2)
        
        # Boundary conditions: all boundaries at T_boundary
        def boundary(x, on_boundary):
            return on_boundary
        bcs = [DirichletBC(V, Constant(T_boundary), boundary)]
        
        # Source term
        if source_type == "constant":
            f = Constant(source_value)
        else:
            f = Constant(0.0)
        
        u = TrialFunction(V)
        v = TestFunction(V)
        
        coords_3d = V.tabulate_dof_coordinates().reshape(-1, 3)
        snapshots = []
        times = []
        
        if steady:
            # Steady state: k * ∫ r² sin(θ) * grad(u) · grad(v) dr dθ dφ = ∫ r² sin(θ) * f * v dr dθ dφ
            a = kappa * r_squared_sin_theta_expr * inner(grad(u), grad(v)) * dx
            L_form = r_squared_sin_theta_expr * f * v * dx
            
            u_sol = Function(V)
            solve(a == L_form, u_sol, bcs)
            
            values = u_sol.vector().get_local()
            snapshots.append(values.tolist())
            times.append(0.0)
        else:
            # Transient
            u_n = Function(V)
            
            if initial_type == "constant":
                u_n.assign(Constant(T_initial))
            else:
                u_n.assign(Constant(T_initial))
            
            for bc in bcs:
                bc.apply(u_n.vector())
            
            # Save initial condition
            initial_values = u_n.vector().get_local()
            snapshots.append(initial_values.tolist())
            times.append(0.0)
            
            a = r_squared_sin_theta_expr * u * v * dx + dt * kappa * r_squared_sin_theta_expr * inner(grad(u), grad(v)) * dx
            L_form = r_squared_sin_theta_expr * u_n * v * dx + dt * r_squared_sin_theta_expr * f * v * dx
            
            u_sol = Function(V)
            
            for n in range(num_steps):
                t = (n + 1) * dt
                solve(a == L_form, u_sol, bcs)
                
                values = u_sol.vector().get_local()
                snapshots.append(values.tolist())
                times.append(t)
                
                u_n.assign(u_sol)
                L_form = r_squared_sin_theta_expr * u_n * v * dx + dt * r_squared_sin_theta_expr * f * v * dx
        
        # Convert spherical coordinates (r, θ, φ) to Cartesian (x, y, z) for visualization
        # x = r sin(θ) cos(φ)
        # y = r sin(θ) sin(φ)
        # z = r cos(θ)
        coords = []
        for (r, theta, phi) in coords_3d:
            x_cart = r * np.sin(theta) * np.cos(phi)
            y_cart = r * np.sin(theta) * np.sin(phi)
            z_cart = r * np.cos(theta)
            coords.append([float(x_cart), float(y_cart), float(z_cart)])
        
        field = TimeSeriesField(
            coords=coords,
            values=snapshots,
            times=times,
            dim=3,
            meta={
                "name": "temperature",
                "unit": "°C",
                "pde": "heat",
                "coordinate_system": "spherical",
                "geometry_type": "sphere" if r_inner < 1e-10 else "spherical_shell",
                "r_inner": r_inner,
                "r_outer": r_outer,
                "source_type": source_type,
                "source_value": source_value,
                "steady": steady,
            },
        )
        return field

# ─────────────────────────────────
# 4.5 1D linear elasticity (axial bar) solver: stress / strain
# ─────────────────────────────────

def _solve_elasticity_1d_static(
    L: float,
    nx: int,
    E: float,
    area: float,
    body_force: float,
    quantity: str = "stress",  # "stress" or "strain"
) -> TimeSeriesField:
    """
    Solve a 1D axial bar linear elasticity *static* problem and output
    either the stress or the strain as a scalar field.

    Model:
        - Geometry: interval [0, L]
        - Material: linear elastic, Young's modulus E, cross-sectional area 'area'
        - PDE:  -(EA u_x)_x = body_force
        - Boundary conditions:
            * x = 0: fixed displacement, u(0) = 0
            * x = L: natural (free) boundary

    Variables:
        - u(x): axial displacement
        - strain:  ε = du/dx
        - stress:  σ = E * ε

    Parameters:
        L          : bar length
        nx         : number of mesh cells
        E          : Young's modulus
        area       : cross-sectional area
        body_force : body force per unit length (aligned with the bar)
        quantity   : "stress" or "strain" to choose the output field

    Returns:
        TimeSeriesField with:
          - dim = 1
          - coords in 3D embedding [x, 0, 0]
          - values: single time snapshot of stress/strain field
          - times: [0.0]
          - meta: includes model parameters and field name/unit
    """

    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        # Mesh and function space
        mesh = IntervalMesh(nx, 0.0, L)
        V = FunctionSpace(mesh, "P", 1)  # scalar displacement space

        # Trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        EA = E * area

        # Weak form:
        # ∫ EA u_x v_x dx = ∫ body_force v dx
        a = EA * inner(grad(u), grad(v)) * dx
        L_form = Constant(body_force) * v * dx

        # Boundary condition: fixed at x = 0, u(0) = 0
        def left_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        bc = DirichletBC(V, Constant(0.0), left_boundary)

        # Solve for displacement u
        u_sol = Function(V)
        solve(a == L_form, u_sol, bc)

        # Strain ε = du/dx
        eps_expr = grad(u_sol)[0]  # 1D: only x-component
        eps = project(eps_expr, V)

        # Stress σ = E * ε
        sigma_expr = E * eps_expr
        sigma = project(sigma_expr, V)

        # Select output field
        if quantity == "strain":
            field_name = "axial_strain"
            unit = "-"  # dimensionless
            field_func = eps
        else:
            field_name = "axial_stress"
            unit = "Pa"
            field_func = sigma

        # Extract coordinates and values, sorted by x for nicer 1D visualization
        coords_1d = V.tabulate_dof_coordinates().reshape(-1)
        order = np.argsort(coords_1d)
        x_sorted = coords_1d[order]

        values = field_func.vector().get_local()
        values_sorted = values[order]

        # Embed in 3D: (x, 0, 0)
        coords = [[float(x), 0.0, 0.0] for x in x_sorted]
        snapshots = [values_sorted.tolist()]  # single snapshot (static problem)
        times = [0.0]

    field = TimeSeriesField(
        coords=coords,
        values=snapshots,
        times=times,
        dim=1,
        meta={
            "name": field_name,
            "unit": unit,
            "pde": "elasticity_1d",
            "L": L,
            "E": E,
            "area": area,
            "body_force": body_force,
            "quantity": quantity,
        },
    )
    return field

# ─────────────────────────────────
# 4.6 2D linear elasticity (plane stress/strain) static solver
# ─────────────────────────────────

def _solve_elasticity_2d_static(
    Lx: float,
    Ly: float,
    nx: int,
    ny: int,
    E: float,
    nu: float,
    body_fx: float = 0.0,
    body_fy: float = 0.0,
    quantity: str = "stress",    # "stress" or "strain"
    plane_stress: bool = True,   # True -> plane stress, False -> plane strain
) -> TimeSeriesField:
    """
    Solve a 2D *static* linear elasticity problem on a rectangular domain and
    output a scalar field derived from the displacement: von Mises equivalent
    stress or von Mises equivalent strain.

    Model:
        - Geometry: rectangle [0, Lx] × [0, Ly]
        - Unknown: displacement u(x, y) ∈ R^2
        - Strain:  ε(u) = sym(grad(u))
        - Stress:  σ(u) = λ tr(ε) I + 2 μ ε
        - Equilibrium: -div(σ) = b

        Here we use either:
          * plane stress
          * or plane strain   (controlled by 'plane_stress' flag)

    Boundary conditions (simple demo setup):
        - Left edge x = 0 is clamped: u = (0, 0)
        - Other boundaries are traction-free (natural)

    Parameters:
        Lx, Ly       : domain size in x and y
        nx, ny       : number of cells in x and y
        E            : Young's modulus
        nu           : Poisson's ratio
        body_fx, body_fy : body-force components (per unit volume/area)
        quantity     : "stress" -> von Mises equivalent stress [Pa]
                       "strain" -> von Mises equivalent strain [-]
        plane_stress : True for plane stress, False for plane strain

    Returns:
        TimeSeriesField with:
            dim   = 2
            coords: 2D coordinates embedded in 3D (x, y, 0)
            values: scalar field (von Mises stress or strain)
            times : [0.0]  (static problem, single snapshot)
            meta  : model parameters and field description
    """

    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        # Mesh and function spaces
        mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)
        V = VectorFunctionSpace(mesh, "P", 1)   # displacement space (2D vector)
        Vs = FunctionSpace(mesh, "P", 1)        # scalar space for projected fields

        # Helper: strain and stress
        def epsilon(u):
            return sym(grad(u))

        def sigma(u):
            eps = epsilon(u)
            if plane_stress:
                # Plane stress Lamé parameters
                mu = E / (2.0 * (1.0 + nu))
                lam = E * nu / (1.0 - nu**2)
            else:
                # Plane strain Lamé parameters
                mu = E / (2.0 * (1.0 + nu))
                lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            d = u.geometric_dimension()
            return lam * tr(eps) * Identity(d) + 2.0 * mu * eps

        # Trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        # Body force
        b = Constant((body_fx, body_fy))

        # Weak form: ∫ σ(u) : ε(v) dx = ∫ b · v dx
        a = inner(sigma(u), epsilon(v)) * dx
        L_form = dot(b, v) * dx

        # Boundary condition: clamp left edge x = 0
        def left_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        bc = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)

        # Solve for displacement
        u_sol = Function(V)
        solve(a == L_form, u_sol, bc)

        # Compute strain tensor and stress tensor
        eps_tensor = epsilon(u_sol)
        sig_tensor = sigma(u_sol)

        # Compute von Mises equivalent stress/strain in 2D
        d = u_sol.geometric_dimension()
        I = Identity(d)

        # Deviatoric strain/stress
        eps_dev = eps_tensor - (1.0 / 3.0) * tr(eps_tensor) * I
        sig_dev = sig_tensor - (1.0 / 3.0) * tr(sig_tensor) * I

        if quantity == "strain":
            # von Mises equivalent strain (J2 measure)
            eq_expr = sqrt(2.0 / 3.0 * inner(eps_dev, eps_dev))
            field_name = "von_mises_strain"
            unit = "-"
        else:
            # von Mises equivalent stress (J2 measure)
            eq_expr = sqrt(3.0 / 2.0 * inner(sig_dev, sig_dev))
            field_name = "von_mises_stress"
            unit = "Pa"

        # Project scalar field to P1 space
        field_scalar = project(eq_expr, Vs)

        # Coordinates and values
        coords_2d = Vs.tabulate_dof_coordinates().reshape(-1, 2)
        values = field_scalar.vector().get_local()

        coords = [[float(x), float(y), 0.0] for (x, y) in coords_2d]
        snapshots = [values.tolist()]  # single static snapshot
        times = [0.0]

    field = TimeSeriesField(
        coords=coords,
        values=snapshots,
        times=times,
        dim=2,
        meta={
            "name": field_name,
            "unit": unit,
            "pde": "elasticity_2d",
            "Lx": Lx,
            "Ly": Ly,
            "E": E,
            "nu": nu,
            "body_fx": body_fx,
            "body_fy": body_fy,
            "quantity": quantity,
            "plane_stress": plane_stress,
        },
    )
    return field

# ─────────────────────────────────
# 4.7 3D linear elasticity static solver (von Mises stress/strain)
# ─────────────────────────────────

def _solve_elasticity_3d_static(
    Lx: float,
    Ly: float,
    Lz: float,
    nx: int,
    ny: int,
    nz: int,
    E: float,
    nu: float,
    body_fx: float = 0.0,
    body_fy: float = 0.0,
    body_fz: float = 0.0,
    quantity: str = "stress",  # "stress" or "strain"
) -> TimeSeriesField:
    """
    Solve a 3D *static* linear elasticity problem on a rectangular box and
    output a scalar field derived from the displacement: von Mises equivalent
    stress or von Mises equivalent strain.

    Model:
        - Geometry: box [0, Lx] × [0, Ly] × [0, Lz]
        - Unknown: displacement u(x, y, z) ∈ R^3
        - Strain:  ε(u) = sym(grad(u))        (symmetric gradient)
        - Stress:  σ(u) = λ tr(ε) I + 2 μ ε   (isotropic linear elasticity)
        - Equilibrium: -div(σ) = b

    Boundary conditions (demo setup):
        - Left face x = 0 is clamped: u = (0, 0, 0)
        - Other faces are traction-free (natural BC)

    Parameters:
        Lx, Ly, Lz : box dimensions
        nx, ny, nz : mesh resolution in x, y, z
        E          : Young's modulus
        nu         : Poisson's ratio
        body_fx    : body force in x direction
        body_fy    : body force in y direction
        body_fz    : body force in z direction
        quantity   : "stress" -> von Mises equivalent stress [Pa]
                     "strain" -> von Mises equivalent strain [-]

    Returns:
        TimeSeriesField with:
            dim   = 3
            coords: 3D coordinates (x, y, z)
            values: scalar field (von Mises stress or strain)
            times : [0.0]  (static problem, single snapshot)
            meta  : model parameters and field description
    """

    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        # Mesh and function spaces
        mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(Lx, Ly, Lz), nx, ny, nz)
        V = VectorFunctionSpace(mesh, "P", 1)   # displacement space (3D vector)
        Vs = FunctionSpace(mesh, "P", 1)        # scalar space for projected fields

        # Strain and stress definitions
        def epsilon(u):
            return sym(grad(u))

        def sigma(u):
            # 3D isotropic linear elasticity Lamé parameters
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            d = u.geometric_dimension()
            eps = epsilon(u)
            return lam * tr(eps) * Identity(d) + 2.0 * mu * eps

        # Trial and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        # Body force vector
        b = Constant((body_fx, body_fy, body_fz))

        # Weak form: ∫ σ(u) : ε(v) dx = ∫ b · v dx
        a = inner(sigma(u), epsilon(v)) * dx
        L_form = dot(b, v) * dx

        # Boundary condition: clamp the face x = 0
        def left_boundary(x, on_boundary):
            return on_boundary and near(x[0], 0.0)

        bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), left_boundary)

        # Solve for displacement
        u_sol = Function(V)
        solve(a == L_form, u_sol, bc)

        # Strain and stress tensors
        eps_tensor = epsilon(u_sol)
        sig_tensor = sigma(u_sol)

        d = u_sol.geometric_dimension()
        I = Identity(d)

        # Deviatoric parts
        eps_dev = eps_tensor - (1.0 / 3.0) * tr(eps_tensor) * I
        sig_dev = sig_tensor - (1.0 / 3.0) * tr(sig_tensor) * I

        # von Mises equivalent strain/stress based on J2 invariant
        if quantity == "strain":
            eq_expr = sqrt(2.0 / 3.0 * inner(eps_dev, eps_dev))
            field_name = "von_mises_strain"
            unit = "-"
        else:
            eq_expr = sqrt(3.0 / 2.0 * inner(sig_dev, sig_dev))
            field_name = "von_mises_stress"
            unit = "Pa"

        # Project scalar field to P1 space
        field_scalar = project(eq_expr, Vs)

        # Coordinates and values
        coords_3d = Vs.tabulate_dof_coordinates().reshape(-1, 3)
        values = field_scalar.vector().get_local()

        coords = [[float(x), float(y), float(z)] for (x, y, z) in coords_3d]
        snapshots = [values.tolist()]  # single static snapshot
        times = [0.0]

    field = TimeSeriesField(
        coords=coords,
        values=snapshots,
        times=times,
        dim=3,
        meta={
            "name": field_name,
            "unit": unit,
            "pde": "elasticity_3d",
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "E": E,
            "nu": nu,
            "body_fx": body_fx,
            "body_fy": body_fy,
            "body_fz": body_fz,
            "quantity": quantity,
        },
    )
    return field


# ─────────────────────────────────
# 5. MCP Server & Tools
# ─────────────────────────────────

mcp = FastMCP("FEniCS-Heat")


@mcp.tool()
def solve_heat_1D(
    length: float = 2.0,
    nx: int = 50,
    diffusivity: float = 1.0,
    T_left: float = 20.0,
    T_right: float = 0.0,
    T_initial: float = 0.0,
    dt: float = 0.01,
    num_steps: int = 50,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
    initial_wavenumber: float = 1.0,
) -> SolveResult:
    """
    解 1D 热方程:
        u_t - k * u_xx = f(x, t), x in (0, length)  (瞬态)
        或
        -k * u_xx = f(x)  (稳态，当 steady=True)

    参数:
      - length: 杆长
      - nx: 网格点数
      - diffusivity: 热扩散系数 k
      - T_left, T_right: 两端 Dirichlet 边界温度
      - T_initial: 初始温度（整个内部，仅用于瞬态）
      - dt: 时间步长（仅用于瞬态）
      - num_steps: 时间步数量（仅用于瞬态）
      - data_dir: 数据保存目录
      - steady: 如果为 True，求解稳态问题；否则求解瞬态问题
      - source_type: 热源类型，可选 "none" 或 "constant"
      - source_value: 热源值（当 source_type="constant" 时使用）
                   注意：为了看到明显效果，建议使用较大的值（如 10-100），
                   特别是在瞬态问题中，因为每步贡献为 dt * source_value

    输出:
      - SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_1d_raw(
        length=length,
        nx=nx,
        diffusivity=diffusivity,
        T_left=T_left,
        T_right=T_right,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
        initial_wavenumber=initial_wavenumber,
    )
    
    # Save to file to avoid large JSON responses
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_1d_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def solve_heat_2D(
    Lx: float = 1.0,
    Ly: float = 1.0,
    nx: int = 30,
    ny: int = 30,
    diffusivity: float = 1.0,
    T_boundary: float = 0.0,
    T_initial: float = 20.0,
    dt: float = 0.01,
    num_steps: int = 50,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
    initial_wavenumber: float = 1.0,
) -> SolveResult:
    """
    解 2D 矩形区域 [0,Lx]×[0,Ly] 上的热方程:
        u_t - k * Δu = f(x, y, t)  (瞬态)
        或
        -k * Δu = f(x, y)  (稳态，当 steady=True)

    - 整个外边界为常数 Dirichlet: T_boundary。
    - steady: 如果为 True，求解稳态问题；否则求解瞬态问题
    - source_type: 热源类型，可选 "none" 或 "constant"
    - source_value: 热源值（当 source_type="constant" 时使用）
                   注意：为了看到明显效果，建议使用较大的值（如 10-100），
                   特别是在瞬态问题中，因为每步贡献为 dt * source_value
    - 返回 SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_2d_raw(
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        diffusivity=diffusivity,
        T_boundary=T_boundary,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
        initial_wavenumber=initial_wavenumber,
    )
    
    # Save to file to avoid large JSON responses
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_2d_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def solve_heat_3D_spherical(
    r_inner: float = 0.1,
    r_outer: float = 1.0,
    nr: int = 20,
    ntheta: int = 20,
    nphi: int = 20,
    diffusivity: float = 1.0,
    T_boundary: float = 20.0,
    T_initial: float = 20.0,
    dt: float = 0.01,
    num_steps: int = 50,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> SolveResult:
    """
    解 3D 球坐标热方程 (Full 3D heat transfer in spherical coordinates, r-θ-φ):
        u_t - k * [(1/r²) * d/dr (r² * du/dr) + (1/(r² sin θ)) * d/dθ (sin θ * du/dθ) 
             + (1/(r² sin² θ)) * d²u/dφ²] = f(r, θ, φ, t),
        r ∈ (r_inner, r_outer), θ ∈ (0, π), φ ∈ (0, 2π)  (瞬态)
        或
        -k * [Laplacian in spherical coordinates] = f(r, θ, φ)  (稳态)
    
    用于求解完整三维球体的热传导问题（非轴对称）。
    
    参数:
      - r_inner: 内半径 (m)，如果为0则求解实心球
      - r_outer: 外半径 (m)
      - nr, ntheta, nphi: 径向、极角和方位角网格点数
      - diffusivity: 热扩散系数 k
      - T_boundary: 边界 Dirichlet 温度
      - T_initial: 初始温度 (瞬态)
      - dt: 时间步长
      - num_steps: 时间步数量
      - steady: 如果为 True，求解稳态问题
      - source_type: 热源类型 ("none" 或 "constant")
      - source_value: 热源值
    
    输出:
      - SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_3d_spherical_raw(
        r_inner=r_inner,
        r_outer=r_outer,
        nr=nr,
        ntheta=ntheta,
        nphi=nphi,
        diffusivity=diffusivity,
        T_boundary=T_boundary,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
    )
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_3d_spherical_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def solve_heat_3D(
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    nx: int = 10,
    ny: int = 10,
    nz: int = 10,
    diffusivity: float = 1.0,
    T_boundary: float = 0.0,
    T_initial: float = 20.0,
    dt: float = 0.01,
    num_steps: int = 20,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
    initial_wavenumber: float = 1.0,
    geometry_type: str = "box",  # "box" or "cylinder"
    cylinder_radius: Optional[float] = None,  # For cylindrical geometry
    T_left: Optional[float] = None,  # Directional BC: left face temperature
    T_right: Optional[float] = None,  # Directional BC: right face temperature
    T_side: Optional[float] = None,  # Side/wall boundary temperature (for cylinder)
    core_radius: Optional[float] = None,  # Radius of high-conductivity core
    core_diffusivity: Optional[float] = None,  # Diffusivity of core material (higher = more conductive)
) -> SolveResult:
    """
    解 3D 热方程，支持多种几何形状和边界条件:
        u_t - k * Δu = f(x, y, z, t)  (瞬态)
        或
        -k * Δu = f(x, y, z)  (稳态，当 steady=True)

    **几何形状:**
    - geometry_type="box": 立方体 [0,Lx]×[0,Ly]×[0,Lz]
    - geometry_type="cylinder": 圆柱体 (半径 cylinder_radius, 长度 Lx, 沿 x 轴)

    **边界条件:**
    - 统一边界: T_boundary (应用于所有边界)
    - 方向边界: T_left, T_right, T_side (分别应用于左端、右端、侧面)

    **材料属性:**
    - 均匀材料: diffusivity (单一热扩散系数)
    - 复合材料: base diffusivity + core_radius + core_diffusivity (高导热核心区域)

    - steady: 如果为 True，求解稳态问题；否则求解瞬态问题
    - source_type: 热源类型，可选 "none" 或 "constant"
    - source_value: 热源值（当 source_type="constant" 时使用）
    - 返回 SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_3d_raw(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nx=nx,
        ny=ny,
        nz=nz,
        diffusivity=diffusivity,
        T_boundary=T_boundary,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
        initial_wavenumber=initial_wavenumber,
        geometry_type=geometry_type,
        cylinder_radius=cylinder_radius,
        T_left=T_left,
        T_right=T_right,
        T_side=T_side,
        core_radius=core_radius,
        core_diffusivity=core_diffusivity,
    )
    
    # Save to file to avoid large JSON responses
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_3d_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


# ─────────────────────────────────
# 5. Coordinate System Specific Solvers
# ─────────────────────────────────

@mcp.tool()
def solve_heat_1D_cylindrical(
    r_inner: float = 0.1,
    r_outer: float = 1.0,
    nr: int = 50,
    diffusivity: float = 1.0,
    T_inner: float = 100.0,
    T_outer: float = 20.0,
    T_initial: float = 20.0,
    dt: float = 0.01,
    num_steps: int = 50,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> SolveResult:
    """
    解 1D 柱坐标径向热方程 (Radial heat transfer in cylindrical coordinates):
        u_t - k * (1/r) * d/dr (r * du/dr) = f(r, t), r ∈ (r_inner, r_outer)  (瞬态)
        或
        -k * (1/r) * d/dr (r * du/dr) = f(r)  (稳态，当 steady=True)
    
    用于求解圆柱体或环形区域的径向热传导问题。
    
    参数:
      - r_inner: 内半径 (m)，如果为0则求解实心圆柱
      - r_outer: 外半径 (m)
      - nr: 径向网格点数
      - diffusivity: 热扩散系数 k
      - T_inner: 内边界 Dirichlet 温度 (当 r_inner > 0)
      - T_outer: 外边界 Dirichlet 温度
      - T_initial: 初始温度 (瞬态)
      - dt: 时间步长
      - num_steps: 时间步数量
      - steady: 如果为 True，求解稳态问题
      - source_type: 热源类型 ("none" 或 "constant")
      - source_value: 热源值
    
    输出:
      - SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_1d_cylindrical_raw(
        r_inner=r_inner,
        r_outer=r_outer,
        nr=nr,
        diffusivity=diffusivity,
        T_inner=T_inner,
        T_outer=T_outer,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
    )
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_1d_cylindrical_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def solve_heat_1D_spherical(
    r_inner: float = 0.1,
    r_outer: float = 1.0,
    nr: int = 50,
    diffusivity: float = 1.0,
    T_inner: float = 100.0,
    T_outer: float = 20.0,
    T_initial: float = 20.0,
    dt: float = 0.01,
    num_steps: int = 50,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> SolveResult:
    """
    解 1D 球坐标径向热方程 (Radial heat transfer in spherical coordinates):
        u_t - k * (1/r²) * d/dr (r² * du/dr) = f(r, t), r ∈ (r_inner, r_outer)  (瞬态)
        或
        -k * (1/r²) * d/dr (r² * du/dr) = f(r)  (稳态，当 steady=True)
    
    用于求解球体或同心球壳的径向热传导问题。
    
    参数:
      - r_inner: 内半径 (m)，如果为0则求解实心球
      - r_outer: 外半径 (m)
      - nr: 径向网格点数
      - diffusivity: 热扩散系数 k
      - T_inner: 内边界 Dirichlet 温度 (当 r_inner > 0)
      - T_outer: 外边界 Dirichlet 温度
      - T_initial: 初始温度 (瞬态)
      - dt: 时间步长
      - num_steps: 时间步数量
      - steady: 如果为 True，求解稳态问题
      - source_type: 热源类型 ("none" 或 "constant")
      - source_value: 热源值
    
    输出:
      - SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_1d_spherical_raw(
        r_inner=r_inner,
        r_outer=r_outer,
        nr=nr,
        diffusivity=diffusivity,
        T_inner=T_inner,
        T_outer=T_outer,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
    )
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_1d_spherical_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def solve_heat_2D_cylindrical(
    r_inner: float = 0.1,
    r_outer: float = 1.0,
    z_length: float = 2.0,
    nr: int = 30,
    nz: int = 30,
    diffusivity: float = 1.0,
    T_boundary: float = 20.0,
    T_initial: float = 20.0,
    dt: float = 0.01,
    num_steps: int = 50,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> SolveResult:
    """
    解 2D 柱坐标轴对称热方程 (Axisymmetric heat transfer in cylindrical coordinates, r-z plane):
        u_t - k * [(1/r) * d/dr (r * du/dr) + d²u/dz²] = f(r, z, t), 
        r ∈ (r_inner, r_outer), z ∈ (0, z_length)  (瞬态)
        或
        -k * [(1/r) * d/dr (r * du/dr) + d²u/dz²] = f(r, z)  (稳态)
    
    用于求解轴对称圆柱体的热传导问题（横截面为环形，沿 z 轴方向延伸）。
    
    参数:
      - r_inner: 内半径 (m)，如果为0则求解实心圆柱
      - r_outer: 外半径 (m)
      - z_length: 轴向长度 (m)
      - nr, nz: 径向和轴向网格点数
      - diffusivity: 热扩散系数 k
      - T_boundary: 边界 Dirichlet 温度
      - T_initial: 初始温度 (瞬态)
      - dt: 时间步长
      - num_steps: 时间步数量
      - steady: 如果为 True，求解稳态问题
      - source_type: 热源类型 ("none" 或 "constant")
      - source_value: 热源值
    
    输出:
      - SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_2d_cylindrical_raw(
        r_inner=r_inner,
        r_outer=r_outer,
        z_length=z_length,
        nr=nr,
        nz=nz,
        diffusivity=diffusivity,
        T_boundary=T_boundary,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
    )
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_2d_cylindrical_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def solve_heat_2D_spherical(
    r_inner: float = 0.1,
    r_outer: float = 1.0,
    nr: int = 30,
    ntheta: int = 30,
    diffusivity: float = 1.0,
    T_boundary: float = 20.0,
    T_initial: float = 20.0,
    dt: float = 0.01,
    num_steps: int = 50,
    data_dir: str = "data",
    steady: bool = False,
    source_type: str = "none",
    source_value: float = 0.0,
    initial_type: str = "constant",
    initial_amplitude: float = 1.0,
) -> SolveResult:
    """
    解 2D 球坐标轴对称热方程 (Axisymmetric heat transfer in spherical coordinates, r-θ plane):
        u_t - k * [(1/r²) * d/dr (r² * du/dr) + (1/(r² sin θ)) * d/dθ (sin θ * du/dθ)] = f(r, θ, t),
        r ∈ (r_inner, r_outer), θ ∈ (0, π)  (瞬态)
        或
        -k * [(1/r²) * d/dr (r² * du/dr) + (1/(r² sin θ)) * d/dθ (sin θ * du/dθ)] = f(r, θ)  (稳态)
    
    用于求解轴对称球体的热传导问题。
    
    参数:
      - r_inner: 内半径 (m)，如果为0则求解实心球
      - r_outer: 外半径 (m)
      - nr, ntheta: 径向和角度网格点数
      - diffusivity: 热扩散系数 k
      - T_boundary: 边界 Dirichlet 温度
      - T_initial: 初始温度 (瞬态)
      - dt: 时间步长
      - num_steps: 时间步数量
      - steady: 如果为 True，求解稳态问题
      - source_type: 热源类型 ("none" 或 "constant")
      - source_value: 热源值
    
    输出:
      - SolveResult，包含数据文件路径，可用 plot_time_series_field_from_file 绘图
    """
    field = _solve_heat_2d_spherical_raw(
        r_inner=r_inner,
        r_outer=r_outer,
        nr=nr,
        ntheta=ntheta,
        diffusivity=diffusivity,
        T_boundary=T_boundary,
        T_initial=T_initial,
        dt=dt,
        num_steps=num_steps,
        steady=steady,
        source_type=source_type,
        source_value=source_value,
        initial_type=initial_type,
        initial_amplitude=initial_amplitude,
    )
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"heat_2d_spherical_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(field, f)
    
    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def solve_elasticity_1D_static(
    L: float = 1.0,
    nx: int = 50,
    E: float = 210e9,          # Young's modulus (Pa)
    area: float = 1.0,         # Cross-sectional area (m^2)
    body_force: float = 0.0,   # Body force per unit length (along the bar)
    quantity: str = "stress",  # "stress" or "strain"
    data_dir: str = "data",
) -> SolveResult:
    """
    Solve a 1D axial bar linear elasticity *static* problem and output either
    the axial stress or axial strain as a scalar field, compatible with the
    existing plotting tools.

    Model:
        - Geometry: interval [0, L]
        - Boundary conditions:
            * x = 0: fixed, u(0) = 0
            * x = L: free (natural boundary)
        - PDE:  -(EA u_x)_x = body_force
        - Material: Young's modulus E, cross-sectional area 'area'

    Parameters:
        L          : bar length
        nx         : number of mesh cells
        E          : Young's modulus (Pa)
        area       : cross-sectional area (m^2)
        body_force : body force per unit length (aligned with the bar),
                    e.g., gravity-like distributed load
        quantity   :
            * "stress" -> output axial stress σ(x) [Pa]
            * "strain" -> output axial strain ε(x) [dimensionless]
        data_dir   : directory to save the result pickle file

    Returns:
        SolveResult:
            - data_file: path to a pickle containing a TimeSeriesField
            - dim      : 1
            - meta     : metadata including field name, unit, and model params

        The resulting data can be visualized using:
            plot_time_series_field_from_file(result.data_file)
    """
    field = _solve_elasticity_1d_static(
        L=L,
        nx=nx,
        E=E,
        area=area,
        body_force=body_force,
        quantity=quantity,
    )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"elasticity_1d_{quantity}_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename

    with open(filepath, "wb") as f:
        pickle.dump(field, f)

    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )

@mcp.tool()
def solve_elasticity_2D_static(
    Lx: float = 1.0,
    Ly: float = 1.0,
    nx: int = 30,
    ny: int = 30,
    E: float = 210e9,          # Young's modulus (Pa)
    nu: float = 0.3,           # Poisson's ratio
    body_fx: float = 0.0,      # x-component of body force
    body_fy: float = 0.0,      # y-component of body force
    quantity: str = "stress",  # "stress" or "strain"
    plane_stress: bool = True, # True -> plane stress, False -> plane strain
    data_dir: str = "data",
) -> SolveResult:
    """
    Solve a 2D *static* linear elasticity problem on a rectangular domain and
    output a scalar field (von Mises equivalent stress or strain) that can be
    visualized with the existing plotting tools.

    Model:
        - Geometry: rectangle [0, Lx] × [0, Ly]
        - Unknown:  displacement u(x, y) ∈ R^2
        - Strain:   ε(u) = sym(grad(u))
        - Stress:   σ(u) = λ tr(ε) I + 2 μ ε
        - Equilibrium: -div(σ) = b

        Boundary conditions (demo setup):
            * Left edge x = 0 is clamped: u = (0, 0)
            * Other edges are traction-free (natural BC)

        Lamé parameters (λ, μ) are computed using:
            * plane stress  (plane_stress=True), or
            * plane strain  (plane_stress=False)

    Parameters:
        Lx, Ly     : domain size
        nx, ny     : mesh resolution
        E          : Young's modulus (Pa)
        nu         : Poisson's ratio
        body_fx    : body force in x direction (N/m³) - if 0, no load is applied
        body_fy    : body force in y direction (N/m³) - if 0, no load is applied
                    Note: To see visible deformation, apply a nonzero body force or traction.
                    For example, gravity: body_fy = -9.81 * density (where density ≈ 7800 kg/m³ for steel)
        quantity   :
            * "stress" -> von Mises equivalent stress [Pa]
            * "strain" -> von Mises equivalent strain [-]
        plane_stress : True for plane stress, False for plane strain
        data_dir   : directory where the result pickle will be written

    Note:
        With zero body forces and only the left edge clamped, the solution will be zero
        displacement everywhere (physically correct for a free body with no loads).
        To observe deformation, apply nonzero body forces or modify boundary conditions.

    Returns:
        SolveResult:
            - data_file: path to a pickle containing a TimeSeriesField
            - dim      : 2
            - meta     : metadata (field name, unit, and model parameters)

        The result can be visualized via:
            plot_time_series_field_from_file(result.data_file)
    """
    field = _solve_elasticity_2d_static(
        Lx=Lx,
        Ly=Ly,
        nx=nx,
        ny=ny,
        E=E,
        nu=nu,
        body_fx=body_fx,
        body_fy=body_fy,
        quantity=quantity,
        plane_stress=plane_stress,
    )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"elasticity_2d_{quantity}_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename

    with open(filepath, "wb") as f:
        pickle.dump(field, f)

    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )

@mcp.tool()
def solve_elasticity_3D_static(
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    nx: int = 10,
    ny: int = 10,
    nz: int = 10,
    E: float = 210e9,          # Young's modulus (Pa)
    nu: float = 0.3,           # Poisson's ratio
    body_fx: float = 0.0,      # body force in x direction
    body_fy: float = 0.0,      # body force in y direction
    body_fz: float = 0.0,      # body force in z direction
    quantity: str = "stress",  # "stress" or "strain"
    data_dir: str = "data",
) -> SolveResult:
    """
    Solve a 3D *static* linear elasticity problem on a rectangular box and
    output a scalar field (von Mises equivalent stress or strain) compatible
    with the existing plotting tools.

    Model:
        - Geometry: box [0, Lx] × [0, Ly] × [0, Lz]
        - Unknown:  displacement u(x, y, z) ∈ R^3
        - Strain:   ε(u) = sym(grad(u))
        - Stress:   σ(u) = λ tr(ε) I + 2 μ ε
        - Equilibrium: -div(σ) = b

        Boundary conditions (demo setup):
            * Face x = 0 is clamped: u = (0, 0, 0)
            * Other faces are traction-free (natural BC)

    Parameters:
        Lx, Ly, Lz : box size
        nx, ny, nz : mesh resolution
        E          : Young's modulus (Pa)
        nu         : Poisson's ratio
        body_fx    : body force in x direction
        body_fy    : body force in y direction
        body_fz    : body force in z direction
        quantity   :
            * "stress" -> von Mises equivalent stress [Pa]
            * "strain" -> von Mises equivalent strain [-]
        data_dir   : directory to write the result pickle

    Returns:
        SolveResult:
            - data_file: path to a pickle containing a TimeSeriesField
            - dim      : 3
            - meta     : metadata (field name, unit, and model parameters)

        You can visualize the result via:
            plot_time_series_field_from_file(result.data_file)
    """
    field = _solve_elasticity_3d_static(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nx=nx,
        ny=ny,
        nz=nz,
        E=E,
        nu=nu,
        body_fx=body_fx,
        body_fy=body_fy,
        body_fz=body_fz,
        quantity=quantity,
    )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filename = f"elasticity_3d_{quantity}_{uuid.uuid4().hex[:8]}.pkl"
    filepath = data_path / filename

    with open(filepath, "wb") as f:
        pickle.dump(field, f)

    return SolveResult(
        data_file=str(filepath),
        dim=field.dim,
        meta=field.meta,
    )


@mcp.tool()
def plot_time_series_field_from_file(
    data_file: str,
    field_name: Optional[str] = None,
    unit: Optional[str] = None,
    output_dir: str = "plots",
    filename: Optional[str] = None,
) -> PlotResult:
    """
    从文件读取 TimeSeriesField 并生成 3D 动图（推荐使用）:
    
    输入:
      - data_file: solve_heat_* 返回的 pickle 文件路径
      - field_name: 物理量名称（默认从 meta 读取）
      - unit: 单位字符串（默认从 meta 读取）
      - output_dir, filename: 输出 HTML 路径设置
    
    输出:
      - PlotResult(html_path=...)
    """
    # Load field from file
    with open(data_file, "rb") as f:
        field = pickle.load(f)
    
    # Use meta values if not provided
    if field_name is None:
        field_name = field.meta.get("name", "u")
    if unit is None:
        unit = field.meta.get("unit", "")
    
    # Generate filename if not provided
    if filename is None:
        filename = f"{field.meta.get('pde', 'field')}_{field.dim}d_{uuid.uuid4().hex[:8]}.html"
    
    # Extract geometry information from metadata first (needed for domain bounds)
    geometry_type = field.meta.get("geometry_type", None)
    geometry_params = {}
    
    # CRITICAL: Check metadata for cylinder_radius FIRST - this is the most reliable indicator
    if field.dim == 3 and field.meta.get("cylinder_radius") is not None:
        # Metadata has cylinder_radius - definitely a cylinder
        geometry_type = "cylinder"
        geometry_params["cylinder_radius"] = float(field.meta.get("cylinder_radius"))
    elif geometry_type == "cylinder" and field.meta.get("cylinder_radius") is not None:
        # geometry_type says cylinder and metadata confirms
        geometry_params["cylinder_radius"] = float(field.meta.get("cylinder_radius"))
    
    # Default to "box" only if no geometry detected yet
    if geometry_type is None:
        geometry_type = "box"  # Will be overridden if detected as cylinder
    
    # If still not detected, try to infer from coordinates
    if field.dim == 3 and (geometry_type is None or geometry_type == "box"):
        # Use ALL coordinates for better detection
        coords_all = np.array(field.coords)
        x_coords = coords_all[:, 0]
        y_coords = coords_all[:, 1]
        z_coords = coords_all[:, 2]
        
        # Check if y and z are centered around 0 (cylindrical pattern)
        y_center = (y_coords.max() + y_coords.min()) / 2
        z_center = (z_coords.max() + z_coords.min()) / 2
        y_span = y_coords.max() - y_coords.min()
        z_span = z_coords.max() - z_coords.min()
        
        # More lenient detection: y and z centered around 0, similar spans
        y_centered = abs(y_center) < 0.2 * max(y_span, 1e-10)
        z_centered = abs(z_center) < 0.2 * max(z_span, 1e-10)
        spans_similar = abs(y_span - z_span) / max(y_span, z_span, 1e-10) < 0.4
        
        if y_centered and z_centered and spans_similar:
            # Calculate approximate radius from all points
            r_max = np.sqrt(y_coords**2 + z_coords**2).max()
            if r_max > 0:
                geometry_type = "cylinder"
                geometry_params["cylinder_radius"] = float(r_max)
                # Update metadata for future use
                field.meta["geometry_type"] = "cylinder"
                field.meta["cylinder_radius"] = float(r_max)
        
        # Check if all coordinates are centered around 0 (spherical pattern)
        x_center = (x_coords.max() + x_coords.min()) / 2
        if abs(x_center) < 0.1 and abs(y_center) < 0.1 and abs(z_center) < 0.1:
            r_max = np.sqrt(x_coords**2 + y_coords**2 + z_coords**2).max()
            if r_max > 0 and abs(y_span - z_span) / max(y_span, z_span) < 0.2:
                geometry_type = "sphere"
                geometry_params["sphere_radius"] = float(r_max)
                field.meta["geometry_type"] = "sphere"
                field.meta["r_outer"] = float(r_max)
    
    # Extract domain size from metadata if available (for 2D: Lx, Ly; for 3D: Lx, Ly, Lz)
    # Handle cylindrical/spherical geometries specially - coordinates are centered at origin
    domain_bounds = None
    
    if field.dim == 2:
        Lx = field.meta.get("Lx")
        Ly = field.meta.get("Ly")
        if Lx is not None and Ly is not None:
            domain_bounds = {"x_min": 0.0, "x_max": float(Lx), "y_min": 0.0, "y_max": float(Ly)}
    elif field.dim == 3:
        Lx = field.meta.get("Lx")
        Ly = field.meta.get("Ly")
        Lz = field.meta.get("Lz")
        
        if geometry_type == "cylinder" and field.meta.get("cylinder_radius") is not None:
            # Cylindrical geometry: coordinates are centered at origin
            # x: 0 to Lx (length along axis)
            # y, z: -radius to +radius (centered around 0)
            cylinder_radius = float(field.meta.get("cylinder_radius"))
            if Lx is not None:
                domain_bounds = {
                    "x_min": 0.0, "x_max": float(Lx),
                    "y_min": -cylinder_radius, "y_max": cylinder_radius,
                    "z_min": -cylinder_radius, "z_max": cylinder_radius
                }
            geometry_params["cylinder_radius"] = cylinder_radius
        elif geometry_type in ["sphere", "spherical_shell"]:
            # Spherical geometry: coordinates centered at origin
            # x, y, z: -radius to +radius (centered around 0)
            r_outer = field.meta.get("r_outer")
            if r_outer is not None:
                sphere_radius = float(r_outer)
            elif field.meta.get("sphere_radius") is not None:
                sphere_radius = float(field.meta.get("sphere_radius"))
            else:
                sphere_radius = 1.0  # Default
            
            domain_bounds = {
                "x_min": -sphere_radius, "x_max": sphere_radius,
                "y_min": -sphere_radius, "y_max": sphere_radius,
                "z_min": -sphere_radius, "z_max": sphere_radius
            }
            geometry_params["sphere_radius"] = sphere_radius
        elif Lx is not None and Ly is not None and Lz is not None:
            # Box geometry: coordinates from 0 to Lx, Ly, Lz
            domain_bounds = {
                "x_min": 0.0, "x_max": float(Lx),
                "y_min": 0.0, "y_max": float(Ly),
                "z_min": 0.0, "z_max": float(Lz)
            }
    elif field.dim == 1:
        length = field.meta.get("length")
        if length is not None:
            domain_bounds = {"x_min": 0.0, "x_max": float(length)}
    
    # Extract geometry parameters
    if geometry_type == "cylinder":
        if "cylinder_radius" not in geometry_params and field.meta.get("cylinder_radius") is not None:
            geometry_params["cylinder_radius"] = float(field.meta.get("cylinder_radius"))
    elif geometry_type in ["sphere", "spherical_shell"]:
        if "sphere_radius" not in geometry_params:
            r_outer = field.meta.get("r_outer")
            if r_outer is not None:
                geometry_params["sphere_radius"] = float(r_outer)
            elif field.meta.get("sphere_radius") is not None:
                geometry_params["sphere_radius"] = float(field.meta.get("sphere_radius"))
    
    # Ensure geometry_params is passed (even if empty) so it can be populated from metadata
    if geometry_params is None:
        geometry_params = {}
    
    return plot_time_series_field(
        coords=field.coords,
        values=field.values,
        times=field.times,
        dim=field.dim,
        field_name=field_name,
        unit=unit,
        output_dir=output_dir,
        filename=filename,
        domain_bounds=domain_bounds,
        geometry_type=geometry_type,
        geometry_params=geometry_params,  # Pass dict directly, not None
    )

def _plot_cylindrical_3d(
    coords_arr: np.ndarray,
    values_arr: np.ndarray,
    times_arr: np.ndarray,
    field_name: str,
    unit: str,
    cylinder_radius: float,
    Lx: float,
    vmin: float,
    vmax: float,
    frames: List,
    output_dir: str,
    filename: str,
) -> PlotResult:
    """
    Plot 3D cylindrical geometry using isosurface (Volume) rendering.
    Creates a volume visualization with isosurfaces showing the field distribution.
    """
    # Extract coordinates
    x_mesh = coords_arr[:, 0]
    y_mesh = coords_arr[:, 1]
    z_mesh = coords_arr[:, 2]
    
    # Create a regular 3D grid for volume rendering
    # Use cylindrical bounds: x in [0, Lx], y and z in [-radius, radius]
    # Aggressively reduced resolution to avoid Streamlit message size limits
    n_x = 20  # Number of points along the axis (reduced from 40)
    n_y = 15  # Number of points in y direction (reduced from 30)
    n_z = 15  # Number of points in z direction (reduced from 30)
    
    x_grid = np.linspace(0, Lx, n_x)
    y_grid = np.linspace(-cylinder_radius, cylinder_radius, n_y)
    z_grid = np.linspace(-cylinder_radius, cylinder_radius, n_z)
    
    # Create 3D grid
    Xi, Yi, Zi = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    # Filter points inside the cylinder (r <= radius)
    R_grid = np.sqrt(Yi**2 + Zi**2)
    inside_cylinder = R_grid <= cylinder_radius
    
    # Interpolate field values from mesh points to grid points for first time step
    v0 = values_arr[0]
    points = coords_arr
    
    # Interpolate to grid
    grid_points = np.column_stack([Xi.flatten(), Yi.flatten(), Zi.flatten()])
    v0_volume = griddata(
        points=points,
        values=v0,
        xi=grid_points,
        method='linear',
        fill_value=np.nan
    )
    
    # Handle NaN values using nearest neighbor
    nan_mask = np.isnan(v0_volume)
    if np.any(nan_mask):
        v0_volume_nearest = griddata(
            points=points,
            values=v0,
            xi=grid_points[nan_mask],
            method='nearest'
        )
        v0_volume[nan_mask] = v0_volume_nearest
    
    # Set values outside cylinder to NaN (will be excluded from volume)
    v0_volume = v0_volume.reshape(Xi.shape)
    v0_volume[~inside_cylinder] = np.nan
    
    # Replace NaN values with a value below isomin so Volume can render properly
    # Volume needs a structured grid - we can't filter out points
    v0_volume_flat = v0_volume.flatten()
    nan_mask_flat = np.isnan(v0_volume_flat)
    if np.any(nan_mask_flat):
        # Set NaN values to a value below the minimum so they're not rendered
        v0_volume_flat[nan_mask_flat] = vmin - (vmax - vmin) * 0.1
        v0_volume = v0_volume_flat.reshape(Xi.shape)
    
    # Create base volume/isosurface trace
    # Use full grid structure - Volume renderer needs this
    base_trace = go.Volume(
        x=Xi.flatten(),
        y=Yi.flatten(),
        z=Zi.flatten(),
        value=v0_volume.flatten(),
        isomin=vmin,  # This will exclude the fill values we set
        isomax=vmax,
        opacity=0.4,  # Slightly more opaque for better visibility
        surface_count=7,  # Number of isosurfaces (reduced to save memory)
        colorscale="Viridis",
        colorbar=dict(title=f"{field_name} {unit}".strip()),
        hovertemplate=(
            "x = %{x:.3f} m<br>"
            "y = %{y:.3f} m<br>"
            "z = %{z:.3f} m<br>"
            f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
        ),
    )
    
    # Create frames for animation
    # Limit number of frames to avoid Streamlit message size limits
    # Aggressively limit frames to keep HTML file size under 50MB
    max_frames = 30  # Maximum number of frames (reduced from 50)
    Nt = len(times_arr)
    if Nt > max_frames:
        # Subsample frames evenly
        frame_indices = np.linspace(0, Nt - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(Nt)
    
    frame_list = []
    for idx in frame_indices:
        vi = values_arr[idx]
        
        # Interpolate values at grid points for this time step
        vi_volume = griddata(
            points=points,
            values=vi,
            xi=grid_points,
            method='linear',
            fill_value=np.nan
        )
        
        # Handle NaN values
        nan_mask = np.isnan(vi_volume)
        if np.any(nan_mask):
            vi_volume_nearest = griddata(
                points=points,
                values=vi,
                xi=grid_points[nan_mask],
                method='nearest'
            )
            vi_volume[nan_mask] = vi_volume_nearest
        
        # Reshape and mask outside cylinder
        vi_volume = vi_volume.reshape(Xi.shape)
        vi_volume[~inside_cylinder] = np.nan
        
        # Replace NaN values with a value below isomin so Volume can render properly
        vi_volume_flat = vi_volume.flatten()
        nan_mask_flat = np.isnan(vi_volume_flat)
        if np.any(nan_mask_flat):
            vi_volume_flat[nan_mask_flat] = vmin - (vmax - vmin) * 0.1
            vi_volume = vi_volume_flat.reshape(Xi.shape)
        
        # Create volume trace for this frame
        frame_trace = go.Volume(
            x=Xi.flatten(),
            y=Yi.flatten(),
            z=Zi.flatten(),
            value=vi_volume.flatten(),
            isomin=vmin,  # This will exclude the fill values we set
            isomax=vmax,
            opacity=0.4,
            surface_count=7,  # Reduced to save memory and avoid Streamlit message size limits
            colorscale="Viridis",
            colorbar=dict(title=f"{field_name} {unit}".strip()),
            hovertemplate=(
                "x = %{x:.3f} m<br>"
                "y = %{y:.3f} m<br>"
                "z = %{z:.3f} m<br>"
                f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
            ),
            showscale=False,  # Only show colorbar on base trace
        )
        
        frame_list.append(go.Frame(
            data=[frame_trace],
            name=f"t={times_arr[idx]:.3f}"
        ))
    
    # Set up scene with proper aspect ratio for cylinder
    camera = dict(eye=dict(x=2.0, y=1.5, z=1.5))
    
    fig = go.Figure(
        data=[base_trace],  # Solid surface representation
        frames=frame_list,
        layout=go.Layout(
            title=f"3D Cylindrical {field_name} (Radius={cylinder_radius:.3f}m, Length={Lx:.3f}m)",
            scene=dict(
                xaxis_title="x (m) - Axis",
                yaxis_title="y (m)",
                zaxis_title="z (m)",
                xaxis=dict(range=[0, Lx], showgrid=True),
                yaxis=dict(range=[-cylinder_radius*1.1, cylinder_radius*1.1], showgrid=True),
                zaxis=dict(range=[-cylinder_radius*1.1, cylinder_radius*1.1], showgrid=True),
                aspectmode="data",  # Preserve cylinder proportions
                camera=camera,
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=True,
                x=1.10,
                y=1.15,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }]),
                    dict(label="Pause", method="animate", args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }]),
                ],
            )],
            sliders=[dict(
                active=0,
                pad={"t": 50},
                currentvalue={"prefix": "Time: "},
                steps=[
                    dict(
                        args=[[f"t={times_arr[idx]:.3f}"], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                        label=f"{times_arr[idx]:.3f}",
                        method="animate",
                    )
                    for idx in frame_indices
                ],
            )],
        ),
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    html_path = output_path / filename
    
    # Use CDN instead of embedding Plotly JS to reduce file size significantly
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    return PlotResult(html_path=str(html_path))


def _plot_spherical_3d(
    coords_arr: np.ndarray,
    values_arr: np.ndarray,
    times_arr: np.ndarray,
    field_name: str,
    unit: str,
    sphere_radius: float,
    vmin: float,
    vmax: float,
    frames: List,
    output_dir: str,
    filename: str,
) -> PlotResult:
    """
    Plot 3D spherical geometry using isosurface (Volume) rendering.
    Creates a volume visualization with isosurfaces showing the field distribution.
    """
    # Extract coordinates
    x_mesh = coords_arr[:, 0]
    y_mesh = coords_arr[:, 1]
    z_mesh = coords_arr[:, 2]
    
    # Create a regular 3D grid for volume rendering
    # Use spherical bounds: x, y, z in [-radius, radius]
    # Aggressively reduced resolution to avoid Streamlit message size limits
    n_points = 15  # Number of points in each direction (reduced from 30)
    
    x_grid = np.linspace(-sphere_radius, sphere_radius, n_points)
    y_grid = np.linspace(-sphere_radius, sphere_radius, n_points)
    z_grid = np.linspace(-sphere_radius, sphere_radius, n_points)
    
    # Create 3D grid
    Xi, Yi, Zi = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    # Filter points inside the sphere (r <= radius)
    R_grid = np.sqrt(Xi**2 + Yi**2 + Zi**2)
    inside_sphere = R_grid <= sphere_radius
    
    # Interpolate field values from mesh points to grid points for first time step
    v0 = values_arr[0]
    points = coords_arr
    
    # Interpolate to grid
    grid_points = np.column_stack([Xi.flatten(), Yi.flatten(), Zi.flatten()])
    v0_volume = griddata(
        points=points,
        values=v0,
        xi=grid_points,
        method='linear',
        fill_value=np.nan
    )
    
    # Handle NaN values using nearest neighbor
    nan_mask = np.isnan(v0_volume)
    if np.any(nan_mask):
        v0_volume_nearest = griddata(
            points=points,
            values=v0,
            xi=grid_points[nan_mask],
            method='nearest'
        )
        v0_volume[nan_mask] = v0_volume_nearest
    
    # Set values outside sphere to NaN (will be excluded from volume)
    v0_volume = v0_volume.reshape(Xi.shape)
    v0_volume[~inside_sphere] = np.nan
    
    # Replace NaN values with a value below isomin so Volume can render properly
    # Volume needs a structured grid - we can't filter out points
    v0_volume_flat = v0_volume.flatten()
    nan_mask_flat = np.isnan(v0_volume_flat)
    if np.any(nan_mask_flat):
        # Set NaN values to a value below the minimum so they're not rendered
        v0_volume_flat[nan_mask_flat] = vmin - (vmax - vmin) * 0.1
        v0_volume = v0_volume_flat.reshape(Xi.shape)
    
    # Create base volume/isosurface trace
    # Use full grid structure - Volume renderer needs this
    base_trace = go.Volume(
        x=Xi.flatten(),
        y=Yi.flatten(),
        z=Zi.flatten(),
        value=v0_volume.flatten(),
        isomin=vmin,  # This will exclude the fill values we set
        isomax=vmax,
        opacity=0.4,  # Slightly more opaque for better visibility
        surface_count=7,  # Number of isosurfaces (reduced to save memory)
        colorscale="Viridis",
        colorbar=dict(title=f"{field_name} {unit}".strip()),
        hovertemplate=(
            "x = %{x:.3f} m<br>"
            "y = %{y:.3f} m<br>"
            "z = %{z:.3f} m<br>"
            f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
        ),
    )
    
    # Create frames for animation
    # Limit number of frames to avoid Streamlit message size limits
    max_frames = 30  # Maximum number of frames (aggressive limit)
    Nt = len(times_arr)
    if Nt > max_frames:
        # Subsample frames evenly
        frame_indices = np.linspace(0, Nt - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(Nt)
    
    frame_list = []
    for idx in frame_indices:
        vi = values_arr[idx]
        
        # Interpolate values at grid points for this time step
        vi_volume = griddata(
            points=points,
            values=vi,
            xi=grid_points,
            method='linear',
            fill_value=np.nan
        )
        
        # Handle NaN values
        nan_mask = np.isnan(vi_volume)
        if np.any(nan_mask):
            vi_volume_nearest = griddata(
                points=points,
                values=vi,
                xi=grid_points[nan_mask],
                method='nearest'
            )
            vi_volume[nan_mask] = vi_volume_nearest
        
        # Reshape and mask outside sphere
        vi_volume = vi_volume.reshape(Xi.shape)
        vi_volume[~inside_sphere] = np.nan
        
        # Replace NaN values with a value below isomin so Volume can render properly
        vi_volume_flat = vi_volume.flatten()
        nan_mask_flat = np.isnan(vi_volume_flat)
        if np.any(nan_mask_flat):
            vi_volume_flat[nan_mask_flat] = vmin - (vmax - vmin) * 0.1
            vi_volume = vi_volume_flat.reshape(Xi.shape)
        
        # Create volume trace for this frame
        frame_trace = go.Volume(
            x=Xi.flatten(),
            y=Yi.flatten(),
            z=Zi.flatten(),
            value=vi_volume.flatten(),
            isomin=vmin,  # This will exclude the fill values we set
            isomax=vmax,
            opacity=0.4,
            surface_count=7,  # Reduced to save memory and avoid Streamlit message size limits
            colorscale="Viridis",
            colorbar=dict(title=f"{field_name} {unit}".strip()),
            hovertemplate=(
                "x = %{x:.3f} m<br>"
                "y = %{y:.3f} m<br>"
                "z = %{z:.3f} m<br>"
                f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
            ),
            showscale=False,  # Only show colorbar on base trace
        )
        
        frame_list.append(go.Frame(
            data=[frame_trace],
            name=f"t={times_arr[idx]:.3f}"
        ))
    
    # Set up scene with proper aspect ratio for sphere
    camera = dict(eye=dict(x=2.0, y=2.0, z=2.0))
    
    fig = go.Figure(
        data=[base_trace],  # Isosurface volume representation
        frames=frame_list,
        layout=go.Layout(
            title=f"3D Spherical {field_name} (Radius={sphere_radius:.3f}m)",
            scene=dict(
                xaxis_title="x (m)",
                yaxis_title="y (m)",
                zaxis_title="z (m)",
                xaxis=dict(range=[-sphere_radius*1.1, sphere_radius*1.1], showgrid=True),
                yaxis=dict(range=[-sphere_radius*1.1, sphere_radius*1.1], showgrid=True),
                zaxis=dict(range=[-sphere_radius*1.1, sphere_radius*1.1], showgrid=True),
                aspectmode="data",  # Preserve sphere proportions
                camera=camera,
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=True,
                x=1.10,
                y=1.15,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }]),
                    dict(label="Pause", method="animate", args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    }]),
                ],
            )],
            sliders=[dict(
                active=0,
                pad={"t": 50},
                currentvalue={"prefix": "Time: "},
                steps=[
                    dict(
                        args=[[f"t={times_arr[idx]:.3f}"], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                        label=f"{times_arr[idx]:.3f}",
                        method="animate",
                    )
                    for idx in frame_indices
                ],
            )],
        ),
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    html_path = output_path / filename
    
    # Use CDN instead of embedding Plotly JS to reduce file size significantly
    fig.write_html(str(html_path), include_plotlyjs='cdn')
    return PlotResult(html_path=str(html_path))


@mcp.tool()
def plot_time_series_field(
    coords: List[List[float]],
    values: List[List[float]],
    times: List[float],
    dim: int = 1,
    field_name: str = "u",
    unit: str = "",
    output_dir: str = "plots",
    filename: str = "field_timeseries_3d.html",
    domain_bounds: Optional[Dict[str, float]] = None,
    geometry_type: Optional[str] = None,  # "box", "cylinder", "sphere"
    geometry_params: Optional[Dict[str, float]] = None,  # e.g., {"cylinder_radius": 0.5}
) -> PlotResult:
    """
    通用场可视化工具（根据维度自动选择图像类型）:

    - dim = 1: 2D 折线图
        x 轴: 物理坐标 x
        y 轴: 场值 (value)
        带时间 slider / Play

    - dim = 2: 3D 曲面
        x, y 轴: 空间坐标
        z / color: 场值 (value)
        带时间 slider / Play

    - dim = 3: 3D 体渲染 (volume / isosurface)
        x, y, z: 空间坐标
        color: 场值
        带时间 slider / Play

    输入:
      - coords: [N][3] 点坐标（统一 3D 嵌入）
      - values: [Nt][N] 时间序列标量场
      - times: [Nt] 时间数组
      - dim: 1/2/3 实际 PDE 空间维度
      - field_name: 物理量名称 (如 'temperature', 'von Mises stress')
      - unit: 单位字符串
      - output_dir, filename: 输出 HTML 路径设置

    输出:
      - PlotResult(html_path=...)
    """
    coords_arr = np.array(coords, dtype=float)  # (N, 3)
    values_arr = np.array(values, dtype=float)  # (Nt, N)
    times_arr = np.array(times, dtype=float)    # (Nt,)

    if values_arr.ndim != 2:
        raise ValueError(f"'values' must have shape (Nt, N), got {values_arr.shape}")
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(f"'coords' must have shape (N, 3), got {coords_arr.shape}")
    if times_arr.ndim != 1 or times_arr.shape[0] != values_arr.shape[0]:
        raise ValueError(f"'times' must have length Nt={values_arr.shape[0]}, got {times_arr.shape}")

    Nt, Npts = values_arr.shape
    if Nt == 0 or Npts == 0:
        raise ValueError(f"Empty data: Nt={Nt}, Npts={Npts}")

    x = coords_arr[:, 0]
    y = coords_arr[:, 1]
    z = coords_arr[:, 2]

    vmin = float(values_arr.min())
    vmax = float(values_arr.max())

    # ===== ROUTE TO SPECIALIZED PLOTTERS FIRST (BEFORE DEFAULT) =====
    # CRITICAL: If metadata says it's a cylinder, use cylindrical plotter IMMEDIATELY
    # This check happens BEFORE any default plotting logic
    is_cylinder = False
    cylinder_radius = None
    
    if dim == 3:
        y_coords = coords_arr[:, 1]
        z_coords = coords_arr[:, 2]
        x_coords = coords_arr[:, 0]
        
        # Calculate statistics
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        z_min, z_max = float(z_coords.min()), float(z_coords.max())
        y_center = (y_max + y_min) / 2
        z_center = (z_max + z_min) / 2
        y_span = y_max - y_min
        z_span = z_max - z_min
        
        # PRIORITY 1: Check geometry_type parameter first (most reliable - from metadata)
        # If metadata says "cylinder", ALWAYS use cylindrical plotter - NO QUESTIONS ASKED
        geometry_type_str = str(geometry_type).lower() if geometry_type else ""
        if geometry_type_str == "cylinder":
            is_cylinder = True
            # Try to get radius from geometry_params first
            if geometry_params is not None and isinstance(geometry_params, dict) and "cylinder_radius" in geometry_params:
                cylinder_radius = geometry_params["cylinder_radius"]
                print(f"[PLOT DEBUG] PRIORITY 1: geometry_type='cylinder' + geometry_params has radius={cylinder_radius}")
            else:
                # Calculate radius from coordinates as fallback
                r_max = np.sqrt(y_coords**2 + z_coords**2).max()
                if r_max > 0:
                    cylinder_radius = float(r_max)
                    print(f"[PLOT DEBUG] PRIORITY 1: geometry_type='cylinder', calculated radius from coords={cylinder_radius:.3f}")
                else:
                    # Last resort: use span-based estimate
                    cylinder_radius = max(y_span, z_span) / 2.0
                    print(f"[PLOT DEBUG] PRIORITY 1: geometry_type='cylinder', estimated radius from span={cylinder_radius:.3f}")
        
        # PRIORITY 2: Check geometry_params (from metadata) even if geometry_type not set
        elif geometry_params is not None and "cylinder_radius" in geometry_params:
            is_cylinder = True
            if not cylinder_radius:
                r_max = np.sqrt(y_coords**2 + z_coords**2).max()
                if r_max > 0:
                    cylinder_radius = float(r_max)
            print(f"[PLOT DEBUG] Detected cylinder from geometry_type: radius={cylinder_radius}")
        
        # PRIORITY 3: ULTRA-LENIENT coordinate-based detection
        # Check if y and z are roughly centered around 0 (cylindrical pattern)
        if not is_cylinder:
            y_centered = abs(y_center) < 0.5 * max(y_span, 1e-10)  # Very lenient
            z_centered = abs(z_center) < 0.5 * max(z_span, 1e-10)  # Very lenient
            
            # Check if y and z spans are reasonable (similar or within 2.5x)
            spans_reasonable = max(y_span, z_span) / max(min(y_span, z_span), 1e-10) < 2.5
            
            # If coordinates suggest cylinder, use cylindrical plotter
            if y_centered and z_centered and spans_reasonable and y_span > 0 and z_span > 0:
                r_max = np.sqrt(y_coords**2 + z_coords**2).max()
                if r_max > 0:
                    cylinder_radius = float(r_max)
                    is_cylinder = True
                    print(f"[PLOT DEBUG] Detected cylinder from coordinates: y_center={y_center:.3f}, z_center={z_center:.3f}, radius={cylinder_radius:.3f}")
    
    # Debug output
    print(f"[PLOT DEBUG] Final decision: is_cylinder={is_cylinder}, cylinder_radius={cylinder_radius}, geometry_type={geometry_type}")
    
    # If cylinder detected, route immediately to cylindrical plotter
    # CRITICAL: Always route if geometry_type says cylinder, even if radius calculation failed
    if is_cylinder:
        if cylinder_radius is None or cylinder_radius <= 0:
            # Last resort: estimate radius from coordinate span
            if dim == 3:
                y_span = float(y_coords.max() - y_coords.min())
                z_span = float(z_coords.max() - z_coords.min())
                cylinder_radius = max(y_span, z_span) / 2.0
                print(f"[PLOT DEBUG] WARNING: Using estimated radius from span: {cylinder_radius:.3f}")
            else:
                cylinder_radius = 0.5  # Default fallback
        
        # Extract domain information for cylindrical geometry
        # For cylinder: domain_bounds should have x_max (length) and radius from geometry_params
        if domain_bounds:
            Lx = domain_bounds.get("x_max") or domain_bounds.get("Lx") or float(x.max())
        else:
            # Try to get from geometry_params or calculate from coordinates
            if geometry_params and "h" in geometry_params:
                Lx = geometry_params["h"]
            else:
                Lx = float(x.max())
        
        # Ensure radius is set (from geometry_params or calculated)
        if not cylinder_radius or cylinder_radius <= 0:
            if geometry_params and "r2" in geometry_params:
                cylinder_radius = geometry_params["r2"]
            elif geometry_params and "cylinder_radius" in geometry_params:
                cylinder_radius = geometry_params["cylinder_radius"]
            else:
                # Calculate from coordinates as last resort
                r_max = np.sqrt(y_coords**2 + z_coords**2).max()
                cylinder_radius = float(r_max) if r_max > 0 else 0.5
        
        print(f"[PLOT DEBUG] ROUTING TO CYLINDRICAL PLOTTER: radius={cylinder_radius:.3f}, Lx={Lx:.3f}")
        return _plot_cylindrical_3d(
            coords_arr=coords_arr,
            values_arr=values_arr,
            times_arr=times_arr,
            field_name=field_name,
            unit=unit,
            cylinder_radius=cylinder_radius,
            Lx=Lx,
            vmin=vmin,
            vmax=vmax,
            frames=[],  # Will be created inside
            output_dir=output_dir,
            filename=filename,
        )
    
    # Check for spherical geometry
    is_sphere = False
    sphere_radius = None
    
    if geometry_type in ["sphere", "spherical_shell"]:
        if geometry_params and "sphere_radius" in geometry_params:
            sphere_radius = geometry_params["sphere_radius"]
            is_sphere = True
    
    if is_sphere:
        # Extract domain information for spherical geometry
        # For sphere: domain_bounds should have radius from geometry_params
        if not sphere_radius or sphere_radius <= 0:
            if geometry_params and "r2" in geometry_params:
                sphere_radius = geometry_params["r2"]
            elif geometry_params and "sphere_radius" in geometry_params:
                sphere_radius = geometry_params["sphere_radius"]
            else:
                # Calculate from coordinates as last resort
                r_max = np.sqrt(x**2 + y**2 + z**2).max()
                sphere_radius = float(r_max) if r_max > 0 else 1.0
        
        print(f"[PLOT DEBUG] ROUTING TO SPHERICAL PLOTTER: radius={sphere_radius:.3f}")
        return _plot_spherical_3d(
            coords_arr=coords_arr,
            values_arr=values_arr,
            times_arr=times_arr,
            field_name=field_name,
            unit=unit,
            sphere_radius=sphere_radius,
            vmin=vmin,
            vmax=vmax,
            frames=[],  # Will be created inside
            output_dir=output_dir,
            filename=filename,
        )
    
    # Default plotting continues below for box/cartesian geometries
    # 保存所有帧用于动画
    frames = []

    # ─────────────────────────────
    # 1D: x → 坐标, y → value 的 2D 折线图
    # ─────────────────────────────
    if dim == 1:
        # 为了画得更顺滑，对 x 排序
        order = np.argsort(x)
        x_sorted = x[order]

        v0 = values_arr[0][order]

        base_trace = go.Scatter(
            x=x_sorted,
            y=v0,
            mode="lines",
            line=dict(width=3),
            hovertemplate=(
                "x = %{x:.3e}<br>"
                + f"{field_name} = %{{y:.3f}} {unit}<extra></extra>"
            ),
            name=f"t={times_arr[0]:.3f}",
        )

        for i in range(Nt):
            vi = values_arr[i][order]
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=x_sorted,
                        y=vi,
                        mode="lines",
                        line=dict(width=3),
                        hovertemplate=(
                            "x = %{x:.3e}<br>"
                            + f"{field_name} = %{{y:.3f}} {unit}<extra></extra>"
                        ),
                        name=f"t={times_arr[i]:.3f}",
                    )
                ],
                name=f"t={times_arr[i]:.3f}",
            )
            frames.append(frame)

        # x, y 轴范围
        x_range = [float(x_sorted.min()), float(x_sorted.max())]
        y_range = [float(values_arr.min()), float(values_arr.max())]

        fig = go.Figure(
            data=[base_trace],
            frames=frames,
            layout=go.Layout(
                title=f"1D {field_name} vs x (time series)",
                xaxis=dict(
                    title="x (m)",
                    range=x_range,
                    showgrid=True,
                ),
                yaxis=dict(
                    title=f"{field_name} {unit}".strip(),
                    range=y_range,
                    showgrid=True,
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=True,
                        x=1.10,
                        y=1.15,
                        xanchor="right",
                        yanchor="top",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 50, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    {
                        "active": 0,
                        "pad": {"t": 50},
                        "currentvalue": {"prefix": "Time: "},
                        "steps": [
                            {
                                "args": [
                                    [f"t={times_arr[i]:.3f}"],
                                    {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": f"{times_arr[i]:.3f}",
                                "method": "animate",
                            }
                            for i in range(Nt)
                        ],
                    }
                ],
            ),
        )

    # ─────────────────────────────
    # 2D: x, y 为平面坐标，z/color 为 value 的 3D 曲面
    # ─────────────────────────────
    elif dim == 2:
        # 生成规则网格
        # Use domain bounds if available, otherwise use coordinate min/max
        if domain_bounds:
            x_min = domain_bounds.get("x_min", x.min())
            x_max = domain_bounds.get("x_max", x.max())
            y_min = domain_bounds.get("y_min", y.min())
            y_max = domain_bounds.get("y_max", y.max())
        else:
            x_min = float(x.min())
            x_max = float(x.max())
            y_min = float(y.min())
            y_max = float(y.max())
        
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        xi = np.linspace(x_min, x_max, len(unique_x))
        yi = np.linspace(y_min, y_max, len(unique_y))
        Xi, Yi = np.meshgrid(xi, yi)

        points = np.column_stack((x, y))
        v0 = values_arr[0]

        # 插值为规则网格
        v0_grid = griddata(points, v0, (Xi, Yi), method="linear", fill_value=np.nan)
        v0_grid_filled = griddata(points, v0, (Xi, Yi), method="nearest")
        v0_grid = np.where(np.isnan(v0_grid), v0_grid_filled, v0_grid)

        base_trace = go.Surface(
            x=Xi,
            y=Yi,
            z=v0_grid,           # 高度 = 值
            surfacecolor=v0_grid,
            colorscale="Viridis",
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=f"{field_name} {unit}".strip()),
            hovertemplate=(
                "x = %{x:.3e}, y = %{y:.3e}<br>"
                + f"{field_name} = %{{z:.3f}} {unit}<extra></extra>"
            ),
        )

        for i in range(Nt):
            vi = values_arr[i]
            vi_grid = griddata(points, vi, (Xi, Yi), method="linear", fill_value=np.nan)
            vi_grid_filled = griddata(points, vi, (Xi, Yi), method="nearest")
            vi_grid = np.where(np.isnan(vi_grid), vi_grid_filled, vi_grid)

            frame = go.Frame(
                data=[
                    go.Surface(
                        x=Xi,
                        y=Yi,
                        z=vi_grid,
                        surfacecolor=vi_grid,
                        colorscale="Viridis",
                        cmin=vmin,
                        cmax=vmax,
                        colorbar=dict(title=f"{field_name} {unit}".strip()),
                        hovertemplate=(
                            "x = %{x:.3e}, y = %{y:.3e}<br>"
                            + f"{field_name} = %{{z:.3f}} {unit}<extra></extra>"
                        ),
                    )
                ],
                name=f"t={times_arr[i]:.3f}",
            )
            frames.append(frame)

        # 轴范围 - use domain bounds if available to ensure exact boundaries
        if domain_bounds:
            x_range = [domain_bounds.get("x_min", float(x.min())), domain_bounds.get("x_max", float(x.max()))]
            y_range = [domain_bounds.get("y_min", float(y.min())), domain_bounds.get("y_max", float(y.max()))]
        else:
            x_range = [float(x.min()), float(x.max())]
            y_range = [float(y.min()), float(y.max())]
        z_range = [float(vmin), float(vmax)]

        # Calculate aspect ratio for 2D surface plots:
        # - Aspect ratio depends ONLY on x and y geometric extents (NOT on z/field values)
        # - z-axis uses a fixed visual factor (0.6) relative to max spatial dimension
        # - This ensures the scene fills the canvas and surfaces don't look flat or squashed
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        z_span = z_range[1] - z_range[0]
        
        # Base scale: use max of x and y span to normalize spatial dimensions
        spatial_base = max(x_span, y_span) if (x_span > 0 and y_span > 0) else 1.0
        
        # Fixed visual factor for z-axis (0.6 means z appears 60% of max spatial dimension)
        z_visual_factor = 0.6
        
        if x_span > 0 and y_span > 0 and spatial_base > 0:
            # x and y maintain their ratio relative to spatial_base (preserves domain shape)
            # z uses fixed visual factor (independent of actual z range)
            aspect_ratio = dict(
                x=x_span / spatial_base,  # Preserves x:y ratio for domain shape
                y=y_span / spatial_base,  # Preserves x:y ratio for domain shape
                z=z_visual_factor  # Fixed factor so surface doesn't look flat or squashed
            )
        else:
            aspect_ratio = dict(x=1, y=1, z=z_visual_factor)

        camera = dict(eye=dict(x=1.5, y=1.5, z=1.2))

        fig = go.Figure(
            data=[base_trace],
            frames=frames,
            layout=go.Layout(
                title=f"2D {field_name} surface (x, y → space; z/color → value)",
                scene=dict(
                    xaxis_title="x (m)",
                    yaxis_title="y (m)",
                    zaxis_title=f"{field_name} {unit}".strip(),
                    xaxis=dict(range=x_range, showgrid=True, autorange=False),
                    yaxis=dict(range=y_range, showgrid=True, autorange=False),
                    zaxis=dict(range=z_range, showgrid=True, autorange=False),
                    aspectmode="manual",  # Manual aspect ratio: x,y based on geometry, z fixed visual factor
                    aspectratio=aspect_ratio,  # Preserves domain shape, fixed z visual factor
                    camera=camera,
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=True,
                        x=1.10,
                        y=1.15,
                        xanchor="right",
                        yanchor="top",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 50, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    {
                        "active": 0,
                        "pad": {"t": 50},
                        "currentvalue": {"prefix": "Time: "},
                        "steps": [
                            {
                                "args": [
                                    [f"t={times_arr[i]:.3f}"],
                                    {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": f"{times_arr[i]:.3f}",
                                "method": "animate",
                            }
                            for i in range(Nt)
                        ],
                    }
                ],
            ),
        )

    # ─────────────────────────────
    # 3D: 体渲染 / isosurface
    # ─────────────────────────────
    else:
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        unique_z = np.unique(z)

        xi = np.linspace(x.min(), x.max(), len(unique_x))
        yi = np.linspace(y.min(), y.max(), len(unique_y))
        zi = np.linspace(z.min(), z.max(), len(unique_z))
        Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing="ij")

        points = np.column_stack((x, y, z))
        v0 = values_arr[0]

        v0_vol = griddata(points, v0, (Xi, Yi, Zi), method="linear", fill_value=np.nan)
        v0_vol_filled = griddata(points, v0, (Xi, Yi, Zi), method="nearest")
        v0_vol = np.where(np.isnan(v0_vol), v0_vol_filled, v0_vol)

        base_trace = go.Volume(
            x=Xi.flatten(),
            y=Yi.flatten(),
            z=Zi.flatten(),
            value=v0_vol.flatten(),
            isomin=vmin,
            isomax=vmax,
            opacity=0.3,
            surface_count=7,  # Reduced to save memory and avoid Streamlit message size limits
            colorscale="Viridis",
            colorbar=dict(title=f"{field_name} {unit}".strip()),
            hovertemplate=(
                "x = %{x:.3e}, y = %{y:.3e}, z = %{z:.3e}<br>"
                + f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
            ),
        )

        for i in range(Nt):
            vi = values_arr[i]
            vi_vol = griddata(points, vi, (Xi, Yi, Zi), method="linear", fill_value=np.nan)
            vi_vol_filled = griddata(points, vi, (Xi, Yi, Zi), method="nearest")
            vi_vol = np.where(np.isnan(vi_vol), vi_vol_filled, vi_vol)

            frame = go.Frame(
                data=[
                    go.Volume(
                        x=Xi.flatten(),
                        y=Yi.flatten(),
                        z=Zi.flatten(),
                        value=vi_vol.flatten(),
                        isomin=vmin,
                        isomax=vmax,
                        opacity=0.3,
                        surface_count=7,  # Reduced to save memory and avoid Streamlit message size limits
                        colorscale="Viridis",
                        colorbar=dict(title=f"{field_name} {unit}".strip()),
                        hovertemplate=(
                            "x = %{x:.3e}, y = %{y:.3e}, z = %{z:.3e}<br>"
                            + f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
                        ),
                    )
                ],
                name=f"t={times_arr[i]:.3f}",
            )
            frames.append(frame)

        camera = dict(eye=dict(x=1.5, y=1.5, z=1.5))
        
        # Set axis ranges if domain_bounds provided
        scene_dict = {
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
            "aspectmode": "data",  # Maintain actual domain proportions
            "camera": camera,
        }
        
        if domain_bounds:
            scene_dict["xaxis"] = dict(
                range=[domain_bounds.get("x_min", float(x.min())), domain_bounds.get("x_max", float(x.max()))],
                showgrid=True,
                autorange=False
            )
            scene_dict["yaxis"] = dict(
                range=[domain_bounds.get("y_min", float(y.min())), domain_bounds.get("y_max", float(y.max()))],
                showgrid=True,
                autorange=False
            )
            scene_dict["zaxis"] = dict(
                range=[domain_bounds.get("z_min", float(z.min())), domain_bounds.get("z_max", float(z.max()))],
                showgrid=True,
                autorange=False
            )

        fig = go.Figure(
            data=[base_trace],
            frames=frames,
            layout=go.Layout(
                title=f"3D {field_name} volume",
                scene=scene_dict,
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=True,
                        x=1.10,
                        y=1.15,
                        xanchor="right",
                        yanchor="top",
                        buttons=[
                            dict(
                                label="Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {"duration": 50, "redraw": True},
                                        "fromcurrent": True,
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                            dict(
                                label="Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                            ),
                        ],
                    )
                ],
                sliders=[
                    {
                        "active": 0,
                        "pad": {"t": 50},
                        "currentvalue": {"prefix": "Time: "},
                        "steps": [
                            {
                                "args": [
                                    [f"t={times_arr[i]:.3f}"],
                                    {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    },
                                ],
                                "label": f"{times_arr[i]:.3f}",
                                "method": "animate",
                            }
                            for i in range(Nt)
                        ],
                    }
                ],
            ),
        )

    # ─────────────────────────────
    # 写 HTML + 注入 CSS 放大 modebar 按钮
    # ─────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    html_str = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )

    custom_css = """
<style>
.modebar-btn svg {
    width: 28px;
    height: 28px;
}
.modebar {
    font-size: 16px;
}
</style>
"""
    if "</head>" in html_str:
        html_str = html_str.replace("</head>", custom_css + "\n</head>")
    else:
        html_str = custom_css + html_str

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return PlotResult(html_path=str(out_path))


@mcp.tool()
def plot_time_series_field_old(
    coords: List[List[float]],
    values: List[List[float]],
    times: List[float],
    dim: int = 1,
    field_name: str = "u",
    unit: str = "",
    output_dir: str = "plots",
    filename: str = "field_timeseries_3d.html",
) -> PlotResult:
    """
    通用 3D 动图工具：

    输入:
      - coords: [N][3] 点坐标（统一 3D 嵌入）
      - values: [Nt][N] 时间序列标量场
      - times: [Nt] 时间数组
      - dim: 1/2/3 实际 PDE 空间维度（只用于标题/说明）
      - field_name: 物理量名称 (如 'Temperature', 'Displacement')
      - unit: 单位字符串
      - output_dir, filename: 输出 HTML 路径设置

    输出:
      - PlotResult(html_path=...)
    """
    coords_arr = np.array(coords, dtype=float)  # (N, 3)
    values_arr = np.array(values, dtype=float)  # (Nt, N)
    times_arr = np.array(times, dtype=float)    # (Nt,)

    x = coords_arr[:, 0]
    y = coords_arr[:, 1]
    z = coords_arr[:, 2]

    Nt, Npts = values_arr.shape
    vmin = float(values_arr.min())
    vmax = float(values_arr.max())

    # Validate data exists
    if Nt == 0 or Npts == 0:
        raise ValueError(f"Empty data: Nt={Nt}, Npts={Npts}")
    
    # Determine if this is essentially 1D (points along x-axis only)
    is_1d = dim == 1 or (np.std(y) < 1e-12 and np.std(z) < 1e-12)
    
    # 初始帧
    v0 = values_arr[0]
    
    # Create traces and frames based on dimension
    if is_1d or dim == 1:
        # 1D: Use thick line with varying color (no scatter points)
        base_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(
                width=8,  # Thick line
                color=v0,
                colorscale="Viridis",
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(title=f"{field_name} {unit}".strip()),
            ),
            customdata=v0,
            hovertemplate=(
                "x = %{x:.3e}<br>"
                + f"{field_name} = %{{customdata:.3f}} {unit}<extra></extra>"
            ),
        )
        
        frames = []
        for i in range(Nt):
            vi = values_arr[i]
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        line=dict(
                            width=8,
                            color=vi,
                            colorscale="Viridis",
                            cmin=vmin,
                            cmax=vmax,
                            colorbar=dict(title=f"{field_name} {unit}".strip()),
                        ),
                        customdata=vi,
                        hovertemplate=(
                            "x = %{x:.3e}<br>"
                            + f"{field_name} = %{{customdata:.3f}} {unit}<extra></extra>"
                        ),
                    )
                ],
                name=f"t={times_arr[i]:.3f}",
            )
            frames.append(frame)
            
    elif dim == 2:
        # 2D: Create surface plot (interpolate scattered points to regular grid)
        # Get unique x and y coordinates and create grid
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        
        # Create regular grid
        xi = np.linspace(x.min(), x.max(), len(unique_x))
        yi = np.linspace(y.min(), y.max(), len(unique_y))
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate values to grid for initial frame
        points = np.column_stack((x, y))
        v0_grid = griddata(points, v0, (Xi, Yi), method='linear', fill_value=np.nan)
        # Fill NaN values with nearest neighbor
        v0_grid_filled = griddata(points, v0, (Xi, Yi), method='nearest')
        v0_grid = np.where(np.isnan(v0_grid), v0_grid_filled, v0_grid)
        
        base_trace = go.Surface(
            x=Xi,
            y=Yi,
            z=v0_grid,  # Height represents field value
            surfacecolor=v0_grid,
            colorscale="Viridis",
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title=f"{field_name} {unit}".strip()),
            hovertemplate=(
                "x = %{x:.3e}, y = %{y:.3e}<br>"
                + f"{field_name} = %{{z:.3f}} {unit}<extra></extra>"
            ),
        )
        
        frames = []
        for i in range(Nt):
            vi = values_arr[i]
            vi_grid = griddata(points, vi, (Xi, Yi), method='linear', fill_value=np.nan)
            vi_grid_filled = griddata(points, vi, (Xi, Yi), method='nearest')
            vi_grid = np.where(np.isnan(vi_grid), vi_grid_filled, vi_grid)
            
            frame = go.Frame(
                data=[
                    go.Surface(
                        x=Xi,
                        y=Yi,
                        z=vi_grid,  # Height represents field value
                        surfacecolor=vi_grid,
                        colorscale="Viridis",
                        cmin=vmin,
                        cmax=vmax,
                        colorbar=dict(title=f"{field_name} {unit}".strip()),
                        hovertemplate=(
                            "x = %{x:.3e}, y = %{y:.3e}<br>"
                            + f"{field_name} = %{{z:.3f}} {unit}<extra></extra>"
                        ),
                    )
                ],
                name=f"t={times_arr[i]:.3f}",
            )
            frames.append(frame)
            
    else:  # dim == 3
        # 3D: Create volume/isosurface plot (interpolate to regular 3D grid)
        # Create regular 3D grid with reduced resolution to avoid Streamlit message size limits
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        unique_z = np.unique(z)
        
        # Limit grid resolution to maximum 20 points per dimension to reduce data size (aggressive reduction)
        max_points = 20
        nx_grid = min(len(unique_x), max_points)
        ny_grid = min(len(unique_y), max_points)
        nz_grid = min(len(unique_z), max_points)
        
        xi = np.linspace(x.min(), x.max(), nx_grid)
        yi = np.linspace(y.min(), y.max(), ny_grid)
        zi = np.linspace(z.min(), z.max(), nz_grid)
        Xi, Yi, Zi = np.meshgrid(xi, yi, zi, indexing='ij')
        
        # Interpolate values to 3D grid for initial frame
        points = np.column_stack((x, y, z))
        v0_volume = griddata(points, v0, (Xi, Yi, Zi), method='linear', fill_value=np.nan)
        v0_volume_filled = griddata(points, v0, (Xi, Yi, Zi), method='nearest')
        v0_volume = np.where(np.isnan(v0_volume), v0_volume_filled, v0_volume)
        
        base_trace = go.Volume(
            x=Xi.flatten(),
            y=Yi.flatten(),
            z=Zi.flatten(),
            value=v0_volume.flatten(),
            isomin=vmin,
            isomax=vmax,
            opacity=0.3,  # Make it semi-transparent
            surface_count=7,  # Reduced to save memory and avoid Streamlit message size limits  # Number of isosurfaces
            colorscale="Viridis",
            colorbar=dict(title=f"{field_name} {unit}".strip()),
            hovertemplate=(
                "x = %{x:.3e}, y = %{y:.3e}, z = %{z:.3e}<br>"
                + f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
            ),
        )
        
        # Limit number of frames to avoid Streamlit message size limits
        max_frames = 30  # Maximum number of frames (aggressive limit)
        if Nt > max_frames:
            # Subsample frames evenly
            frame_indices = np.linspace(0, Nt - 1, max_frames, dtype=int)
        else:
            frame_indices = np.arange(Nt)
        
        frames = []
        for idx in frame_indices:
            vi = values_arr[idx]
            vi_volume = griddata(points, vi, (Xi, Yi, Zi), method='linear', fill_value=np.nan)
            vi_volume_filled = griddata(points, vi, (Xi, Yi, Zi), method='nearest')
            vi_volume = np.where(np.isnan(vi_volume), vi_volume_filled, vi_volume)
            
            frame = go.Frame(
                data=[
                    go.Volume(
                        x=Xi.flatten(),
                        y=Yi.flatten(),
                        z=Zi.flatten(),
                        value=vi_volume.flatten(),
                        isomin=vmin,
                        isomax=vmax,
                        opacity=0.3,
                        surface_count=7,  # Reduced to save memory and avoid Streamlit message size limits
                        colorscale="Viridis",
                        colorbar=dict(title=f"{field_name} {unit}".strip()),
                        hovertemplate=(
                            "x = %{x:.3e}, y = %{y:.3e}, z = %{z:.3e}<br>"
                            + f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
                        ),
                    )
                ],
                name=f"t={times_arr[idx]:.3f}",
            )
            frames.append(frame)

    # Calculate axis ranges for better visualization
    x_range = [float(x.min()), float(x.max())] if len(x) > 0 else [0, 1]
    y_range = [float(y.min()), float(y.max())] if len(y) > 0 else [-0.1, 0.1]
    z_range = [float(z.min()), float(z.max())] if len(z) > 0 else [-0.1, 0.1]
    
    # Calculate range sizes
    x_span = abs(x_range[1] - x_range[0])
    y_span = abs(y_range[1] - y_range[0])
    z_span = abs(z_range[1] - z_range[0])
    
    # Ensure ranges are not zero (add small padding relative to scale)
    if x_span < 1e-15:
        x_padding = max(1e-15, abs(x_range[0]) * 0.1 if x_range[0] != 0 else 1e-15)
        x_range = [x_range[0] - x_padding, x_range[1] + x_padding]
        x_span = 2 * x_padding
    if y_span < 1e-15:
        y_padding = max(0.01, abs(x_range[0]) * 0.1 if x_range[0] != 0 else 0.01)
        y_range = [y_range[0] - y_padding, y_range[1] + y_padding]
        y_span = 2 * y_padding
    if z_span < 1e-15:
        z_padding = max(0.01, abs(x_range[0]) * 0.1 if x_range[0] != 0 else 0.01)
        z_range = [z_range[0] - z_padding, z_range[1] + z_padding]
        z_span = 2 * z_padding
    
    # Calculate aspect ratio (normalize by x_span to avoid division issues)
    base_span = max(x_span, 1e-15)
    aspect_ratio = dict(
        x=1.0,
        y=max(0.1, y_span / base_span),
        z=max(0.1, z_span / base_span)
    )
    
    # Configure axes based on dimension - hide irrelevant axes
    if dim == 1:
        # 1D: Only show x-axis, hide y and z axes
        xaxis_config = dict(range=x_range, type="linear", title="x (m)", visible=True, showgrid=True)
        yaxis_config = dict(range=y_range, type="linear", title="y (m)", visible=False, showgrid=False, showbackground=False)
        zaxis_config = dict(range=z_range, type="linear", title="z (m)", visible=False, showgrid=False, showbackground=False)
        axis_title_x = "x (m)"
        axis_title_y = ""
        axis_title_z = ""
    elif dim == 2:
        # 2D: Show x and y axes, hide z axis
        xaxis_config = dict(range=x_range, type="linear", title="x (m)", visible=True, showgrid=True)
        yaxis_config = dict(range=y_range, type="linear", title="y (m)", visible=True, showgrid=True)
        zaxis_config = dict(range=z_range, type="linear", title="z (m)", visible=False, showgrid=False, showbackground=False)
        axis_title_x = "x (m)"
        axis_title_y = "y (m)"
        axis_title_z = ""
    else:
        # 3D: Show all axes
        xaxis_config = dict(range=x_range, type="linear", title="x (m)", visible=True, showgrid=True)
        yaxis_config = dict(range=y_range, type="linear", title="y (m)", visible=True, showgrid=True)
        zaxis_config = dict(range=z_range, type="linear", title="z (m)", visible=True, showgrid=True)
        axis_title_x = "x (m)"
        axis_title_y = "y (m)"
        axis_title_z = "z (m)"
    
    fig = go.Figure(
        data=[base_trace],
        layout=go.Layout(
            title=f"{dim}D {field_name} Time Series (embedded in 3D)",
            scene=dict(
                xaxis_title=axis_title_x,
                yaxis_title=axis_title_y,
                zaxis_title=axis_title_z,
                xaxis=xaxis_config,
                yaxis=yaxis_config,
                zaxis=zaxis_config,
                aspectmode="manual",
                aspectratio=aspect_ratio,
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    x=1.15,
                    y=1.15,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},  # Faster frame rate for smoother animation
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                {
                    "active": 0,
                    "pad": {"t": 50},
                    "currentvalue": {"prefix": "Time: "},
                    "steps": [
                        {
                            "args": [
                                [f"t={times_arr[i]:.3f}"],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": f"{times_arr[i]:.3f}",
                            "method": "animate",
                        }
                        for i in range(Nt)
                    ],
                }
            ],
        ),
        frames=frames,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # 生成 HTML，并放大 modebar 图标，开启 scrollZoom
    html_str = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )

    custom_css = """
<style>
.modebar-btn svg {
    width: 28px;
    height: 28px;
}
.modebar {
    font-size: 16px;
}
</style>
"""
    if "</head>" in html_str:
        html_str = html_str.replace("</head>", custom_css + "\n</head>")
    else:
        html_str = custom_css + html_str

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    return PlotResult(html_path=str(out_path))


if __name__ == "__main__":
    mcp.run(transport="stdio")

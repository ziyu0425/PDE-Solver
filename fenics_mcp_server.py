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

try:
    from dolfin import set_log_active, set_log_level, LogLevel, parameters
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
) -> TimeSeriesField:
    """用 FEniCS 解 3D 热方程，立方体 [0,Lx]×[0,Ly]×[0,Lz]，常数 Dirichlet 边界。"""

    buf_stdout = io.StringIO()
    buf_stderr = io.StringIO()
    with contextlib.redirect_stdout(buf_stdout), contextlib.redirect_stderr(buf_stderr):
        kappa = diffusivity

        mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(Lx, Ly, Lz), nx, ny, nz)
        V = FunctionSpace(mesh, "P", 1)

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

        coords_3d = V.tabulate_dof_coordinates().reshape(-1, 3)
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

    coords = [[float(x), float(y), float(z)] for (x, y, z) in coords_3d]
    field = TimeSeriesField(
        coords=coords,
        values=snapshots,
        times=times,
        dim=3,
        meta={
            "name": "temperature",
            "unit": "°C",
            "pde": "heat",
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
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
) -> SolveResult:
    """
    解 3D 立方体 [0,Lx]×[0,Ly]×[0,Lz] 上的热方程:
        u_t - k * Δu = f(x, y, z, t)  (瞬态)
        或
        -k * Δu = f(x, y, z)  (稳态，当 steady=True)

    - 整个外边界为常数 Dirichlet: T_boundary。
    - steady: 如果为 True，求解稳态问题；否则求解瞬态问题
    - source_type: 热源类型，可选 "none" 或 "constant"
    - source_value: 热源值（当 source_type="constant" 时使用）
                   注意：为了看到明显效果，建议使用较大的值（如 10-100），
                   特别是在瞬态问题中，因为每步贡献为 dt * source_value
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
    
    # Extract domain size from metadata if available (for 2D: Lx, Ly; for 3D: Lx, Ly, Lz)
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
        if Lx is not None and Ly is not None and Lz is not None:
            domain_bounds = {
                "x_min": 0.0, "x_max": float(Lx),
                "y_min": 0.0, "y_max": float(Ly),
                "z_min": 0.0, "z_max": float(Lz)
            }
    elif field.dim == 1:
        length = field.meta.get("length")
        if length is not None:
            domain_bounds = {"x_min": 0.0, "x_max": float(length)}
    
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
    )

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
            surface_count=10,
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
                        surface_count=10,
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

        fig = go.Figure(
            data=[base_trace],
            frames=frames,
            layout=go.Layout(
                title=f"3D {field_name} volume",
                scene=dict(
                    xaxis_title="x (m)",
                    yaxis_title="y (m)",
                    zaxis_title="z (m)",
                    aspectmode="data",  # Maintain actual domain proportions (boxes stay box-shaped)
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
        # Create regular 3D grid
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        unique_z = np.unique(z)
        
        xi = np.linspace(x.min(), x.max(), len(unique_x))
        yi = np.linspace(y.min(), y.max(), len(unique_y))
        zi = np.linspace(z.min(), z.max(), len(unique_z))
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
            surface_count=10,  # Number of isosurfaces
            colorscale="Viridis",
            colorbar=dict(title=f"{field_name} {unit}".strip()),
            hovertemplate=(
                "x = %{x:.3e}, y = %{y:.3e}, z = %{z:.3e}<br>"
                + f"{field_name} = %{{value:.3f}} {unit}<extra></extra>"
            ),
        )
        
        frames = []
        for i in range(Nt):
            vi = values_arr[i]
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
                        surface_count=10,
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

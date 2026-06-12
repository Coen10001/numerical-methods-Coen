


# Imports
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------
# DATA CONTAINERS
# ----------------------------------------------------------------------
# Define lightweight data classes for the computational grid and
# boundary conditions, including cell-center coordinates and basic BCs.

@dataclass
class Grid:
    nx: int
    ny: int
    lx: float
    ly: float

    def __post_init__(self):
        # Set uniform cell sizes in x and y based on domain length and number of cells
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        # Cell-center coordinates in x and y
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        # 2D meshgrid of cell centers (Fortran-style indexing: X[i,j], Y[i,j])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

@dataclass
class BoundarySpec:
    u_in: float # u_in: prescribed inflow velocity at the left boundary
    p_out: float # p_out: prescribed static pressure at the outlet (right boundary)

# ----------------------------------------------------------------------
# OBSTACLE GEOMETRY
# ----------------------------------------------------------------------
# Define lightweight data classes for the computational grid and
# boundary conditions, including cell-center coordinates and basic BCs.

class SquareObstacleMask:
    def __init__(self, grid: Grid, side: float, xc: float, yc: float):
        # Build a boolean mask for a square solid obstacle centered at (xc, yc)
        h = 0.5 * side
        # Cells whose centers fall inside the square are marked as solid
        self.solid = (
            (np.abs(grid.X - xc) <= h) &
            (np.abs(grid.Y - yc) <= h)
        )
        # Complement of the solid mask gives the fluid region
        self.fluid = ~self.solid

        # Face masks used to zero fluxes on faces cut by the obstacle
        self.face_x = np.zeros((grid.nx + 1, grid.ny), dtype=bool)
        self.face_y = np.zeros((grid.nx, grid.ny + 1), dtype=bool)

        # Populate face_x and face_y where a solid–fluid interface crosses a face
        self._build_face_cuts(grid)

    def _build_face_cuts(self, grid: Grid):
        # Mark vertical faces where adjacent cells differ (solid vs fluid)
        for i in range(1, grid.nx):
            self.face_x[i, :] = self.solid[i - 1, :] ^ self.solid[i, :]

        # Mark horizontal faces where adjacent cells differ (solid vs fluid)
        for j in range(1, grid.ny):
            self.face_y[:, j] = self.solid[:, j - 1] ^ self.solid[:, j]

# ----------------------------------------------------------------------
# PRESSURE PROJECTION
# ----------------------------------------------------------------------
# Assemble and factorize the Poisson operator for pressure and apply a
# projection step that enforces incompressibility via a pressure solve.

class PressureProjector:
    def __init__(self, grid: Grid, bc: BoundarySpec):
        # Store grid and boundary data
        self.grid = grid
        self.bc = bc
        # Assemble discrete Poisson operator for pressure correction
        self.matrix = self._assemble_matrix()
        # Pre-factorize the sparse matrix for fast repeated solves
        self.solve_linear = spla.factorized(self.matrix.tocsc())

    def _idx(self, i: int, j: int) -> int:
        # Map (i,j) cell index to 1D linear index in column-major ordering
        return i + j * self.grid.nx

    def _assemble_matrix(self):
        # Build Laplacian matrix with Neumann/Dirichlet-like treatments at boundaries
        nx, ny = self.grid.nx, self.grid.ny
        dx2, dy2 = self.grid.dx**2, self.grid.dy**2
        rows, cols, data = [], [], []

        def put(r, c, v):
            # Helper to append one nonzero entry
            rows.append(r)
            cols.append(c)
            data.append(v)

        for j in range(ny):
            for i in range(nx):
                I = self._idx(i, j)

                # Outlet: enforce p = p_out via identity row
                if i == nx - 1:
                    put(I, I, 1.0)
                    continue

                diag = 0.0

                # x-direction neighbors (inlet and interior)
                if i == 0:
                    put(I, self._idx(i + 1, j), 1.0 / dx2)
                    diag -= 1.0 / dx2
                else:
                    put(I, self._idx(i - 1, j), 1.0 / dx2)
                    diag -= 1.0 / dx2

                put(I, self._idx(i + 1, j), 1.0 / dx2)
                diag -= 1.0 / dx2

                # y-direction neighbors with simple one-sided handling at top/bottom
                if j == 0:
                    put(I, self._idx(i, j + 1), 1.0 / dy2)
                    diag -= 1.0 / dy2
                elif j == ny - 1:
                    put(I, self._idx(i, j - 1), 1.0 / dy2)
                    diag -= 1.0 / dy2
                else:
                    put(I, self._idx(i, j - 1), 1.0 / dy2)
                    put(I, self._idx(i, j + 1), 1.0 / dy2)
                    diag -= 2.0 / dy2

                # Diagonal entry for this row
                put(I, I, diag)

        # Return assembled CSR matrix of size (nx*ny) x (nx*ny)
        N = nx * ny
        return sps.csr_matrix((data, (rows, cols)), shape=(N, N))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        # Solve Poisson equation for pressure given divergence-based rhs
        nx, ny = self.grid.nx, self.grid.ny
        # Flatten rhs using Fortran ordering to match index mapping
        b = rhs.reshape(nx * ny, order="F").copy()

        # Impose outlet pressure directly in the right-hand side
        for j in range(ny):
            b[self._idx(nx - 1, j)] = self.bc.p_out
        p = self.solve_linear(b).reshape((nx, ny), order="F")

        # Solve linear system and reshape back to 2D field
        p[0, :] = p[1, :]
        p[-1, :] = self.bc.p_out
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        return p

# ----------------------------------------------------------------------
# FLOW SOLVER
# ----------------------------------------------------------------------
# Implement an incompressible Navier–Stokes solver with explicit
# convection, viscous stress evaluation, pressure projection, and
# time stepping, including force and drag coefficient evaluation.

class OpenFlowSolver:
    def __init__(self, cfg: dict):
        # --- Solver configuration and object construction ---
        # Copy configuration, build grid, boundary spec, and obstacle geometry.
        self.cfg = dict(cfg)
        # Construct grid and boundary condition objects
        self.grid = Grid(cfg["Nx"], cfg["Ny"], cfg["Lx"], cfg["Ly"])
        self.bc = BoundarySpec(cfg["U_in"], cfg["pressure_outlet"])
        
        # --- Fluid properties and reference scales ---
        # Set viscosity, density, and reference scales for drag coefficient.
        # (optional alternative: compute nu from Re and U_in)
        #self.nu = cfg["U_in"] * cfg["square_side"] / cfg["Re"] # use for fixed reynolds
        self.nu = cfg["nu"] # use for fixed viscosity and fluid properties
        self.rho = cfg["rho"]
        self.mu = self.rho * self.nu

        # Reference scales for drag coefficient computation
        # 0.5 * rho * U^2 * D  (per m out of plane)
        self.D_ref = cfg["square_side"]
        self.U_ref = cfg["U_in"]
        self.F_ref = 0.5 * self.rho * self.U_ref**2 * self.D_ref

        # --- Time step selection ---
        # Choose time step from CFL and diffusive stability constraints (unless given)
        self.dt = self._choose_dt(cfg.get("dt"), cfg["CFL"])

        # --- Obstacle geometry and pressure projector ---
        # Build obstacle mask and pressure projector
        self.obstacle = SquareObstacleMask(self.grid, cfg["square_side"], cfg["square_xc"], cfg["square_yc"])
        self.projector = PressureProjector(self.grid, self.bc)

    def _choose_dt(self, dt_manual, cfl):
        # If user provides dt explicitly, just use it
        if dt_manual is not None:
            return float(dt_manual)
        # Convective stability limit based on inlet speed and grid spacing
        u = max(abs(self.bc.u_in), 1.0e-12)
        conv = cfl / (u / self.grid.dx + u / self.grid.dy)
        # Diffusive stability limit for explicit viscous update
        diff = 0.5 / (max(self.nu, 1.0e-12) * (1.0 / self.grid.dx**2 + 1.0 / self.grid.dy**2))

        # Use the more restrictive of convective and diffusive limits by taking the min value
        return min(conv, diff)

    # --- State allocation and boundary conditions ---
    # Initialize fields and enforce BCs and obstacle no-slip.
    def allocate_state(self):
        # Initialize velocity field: uniform inflow in x, zero in y
        u = np.full((self.grid.nx, self.grid.ny), self.bc.u_in, dtype=float)
        # Initialize pressure and velocity field to zero
        v = np.zeros_like(u)
        p = np.zeros_like(u)

        # Enforce boundary conditions and obstacle no-slip
        self.apply_cell_constraints(u, v)

        # Compute corresponding normal fluxes at cell faces
        fx, fy = self.cell_to_normal_fluxes(u, v)

        return {"u": u, "v": v, "p": p, "fx": fx, "fy": fy}

    def apply_cell_constraints(self, u: np.ndarray, v: np.ndarray):
        # Inlet: prescribed horizontal velocity and zero vertical velocity
        u[0, :] = self.bc.u_in
        v[0, :] = 0.0

        # Outlet: simple zero-gradient (copy from previous interior cell)
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]

        # Top and bottom: symmetry / slip for u, no-penetration for v
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        v[:, 0] = 0.0
        v[:, -1] = 0.0

        # Inside obstacle: enforce no-slip (zero velocity)
        u[self.obstacle.solid] = 0.0
        v[self.obstacle.solid] = 0.0

    # --- Flux and transport velocity construction ---
    # Map cell-centered velocities to face-normal and transport fluxes.
    def cell_to_normal_fluxes(self, u: np.ndarray, v: np.ndarray):
        # Convert cell-centered velocities to normal face fluxes (staggered grid)
        nx, ny = self.grid.nx, self.grid.ny
        fx = np.zeros((nx + 1, ny), dtype=float)
        fy = np.zeros((nx, ny + 1), dtype=float)

        # Interior faces: arithmetic average of neighboring cell values
        fx[1:nx, :] = 0.5 * (u[:-1, :] + u[1:, :])
        fy[:, 1:ny] = 0.5 * (v[:, :-1] + v[:, 1:])

        # Inlet and outlet faces in x
        fx[0, :] = self.bc.u_in
        fx[-1, :] = u[-1, :]

        # Top and bottom faces in y: no normal flow
        fy[:, 0] = 0.0
        fy[:, -1] = 0.0

        # Zero flux across faces intersected by the obstacle
        fx[self.obstacle.face_x] = 0.0
        fy[self.obstacle.face_y] = 0.0
        return fx, fy

    def cell_to_transport_faces(self, u: np.ndarray, v: np.ndarray):
        # Build velocities at faces used for convective transport of momentum
        nx, ny = self.grid.nx, self.grid.ny
        ux = np.zeros((nx + 1, ny), dtype=float)
        vx = np.zeros((nx + 1, ny), dtype=float)
        uy = np.zeros((nx, ny + 1), dtype=float)
        vy = np.zeros((nx, ny + 1), dtype=float)

        # Average velocities to faces in x- and y-directions
        ux[1:nx, :] = 0.5 * (u[:-1, :] + u[1:, :])
        vx[1:nx, :] = 0.5 * (v[:-1, :] + v[1:, :])
        uy[:, 1:ny] = 0.5 * (u[:, :-1] + u[:, 1:])
        vy[:, 1:ny] = 0.5 * (v[:, :-1] + v[:, 1:])

        # Boundary face velocities (inlet, outlet, top, bottom)
        ux[0, :] = self.bc.u_in
        vx[0, :] = 0.0
        ux[-1, :] = u[-1, :]
        vx[-1, :] = v[-1, :]
        uy[:, 0] = u[:, 0]
        uy[:, -1] = u[:, -1]
        vy[:, 0] = 0.0
        vy[:, -1] = 0.0

        # Zero transport on faces intersected by the obstacle
        ux[self.obstacle.face_x] = 0.0
        vx[self.obstacle.face_x] = 0.0
        uy[self.obstacle.face_y] = 0.0
        vy[self.obstacle.face_y] = 0.0
        return ux, vx, uy, vy

    # --- Divergence and pressure gradients ---
    # Support routines for projection and diagnostics.
    def divergence(self, fx: np.ndarray, fy: np.ndarray):
        # Compute discrete divergence of face-normal fluxes on each cell
        return (fx[1:, :] - fx[:-1, :]) / self.grid.dx + (fy[:, 1:] - fy[:, :-1]) / self.grid.dy

    def face_pressure_gradients(self, p: np.ndarray):
        # Compute pressure gradients on faces from cell-centered pressure
        nx, ny = self.grid.nx, self.grid.ny
        dpdx = np.zeros((nx + 1, ny), dtype=float)
        dpdy = np.zeros((nx, ny + 1), dtype=float)

        # Interior faces: centered difference
        dpdx[1:nx, :] = (p[1:, :] - p[:-1, :]) / self.grid.dx
        dpdy[:, 1:ny] = (p[:, 1:] - p[:, :-1]) / self.grid.dy

        # Homogeneous gradient at domain boundaries (no normal pressure gradient)
        dpdx[0, :] = 0.0
        dpdx[-1, :] = 0.0
        dpdy[:, 0] = 0.0
        dpdy[:, -1] = 0.0
        return dpdx, dpdy

    # --- Viscous stresses and obstacle forces ---
    # Compute stress tensor and integrate hydrodynamic force on the obstacle.
    def viscous_operator(self, u: np.ndarray, v: np.ndarray, drop_obstacle: bool=True):
        # Assemble viscous stress components at faces and their divergence
        dx, dy, mu = self.grid.dx, self.grid.dy, self.mu
        nx, ny = self.grid.nx, self.grid.ny

        # Velocity gradients at cell centers
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dy)
        dv_dx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dx)
        dv_dx[0, :] = (v[1, :] - v[0, :]) / dx
        dv_dx[-1, :] = 0.0

        # Normal and shear stresses on x- and y-faces
        txx = np.zeros((nx + 1, ny), dtype=float)
        txy_x = np.zeros((nx + 1, ny), dtype=float)
        txy_y = np.zeros((nx, ny + 1), dtype=float)
        tyy = np.zeros((nx, ny + 1), dtype=float)

        # Normal stress in x-direction on vertical faces
        txx[1:nx, :] = 2.0 * mu * (u[1:, :] - u[:-1, :]) / dx
        txx[0, :] = 0.0
        txx[-1, :] = 0.0

        # Shear stress on vertical faces (uses averaged gradients)
        txy_x[1:nx, :] = mu * (0.5 * (du_dy[:-1, :] + du_dy[1:, :]) + (v[1:, :] - v[:-1, :]) / dx)
        txy_x[0, :] = mu * (du_dy[0, :] + dv_dx[0, :])
        txy_x[-1, :] = 0.0

        # Gradients on horizontal faces for shear and normal stresses
        du_dy_face = np.zeros((nx, ny + 1), dtype=float)
        dv_dy_face = np.zeros((nx, ny + 1), dtype=float)
        dv_dx_face = np.zeros((nx, ny + 1), dtype=float)
        du_dy_face[:, 1:ny] = (u[:, 1:] - u[:, :-1]) / dy
        dv_dy_face[:, 1:ny] = (v[:, 1:] - v[:, :-1]) / dy
        dv_dx_face[:, 1:ny] = 0.5 * (dv_dx[:, :-1] + dv_dx[:, 1:])

        # Shear and normal stress on horizontal faces
        txy_y = mu * (du_dy_face + dv_dx_face)
        tyy = 2.0 * mu * dv_dy_face
        txy_y[:, 0] = 0.0
        txy_y[:, -1] = 0.0

        # Optionally zero stresses on faces cut by obstacle to avoid double-counting
        if drop_obstacle:
            txx[self.obstacle.face_x] = 0.0
            txy_x[self.obstacle.face_x] = 0.0
            txy_y[self.obstacle.face_y] = 0.0
            tyy[self.obstacle.face_y] = 0.0

        # Optionally zero stresses on faces cut by obstacle to avoid double-counting
        Lu = (txx[1:, :] - txx[:-1, :]) / dx + (txy_y[:, 1:] - txy_y[:, :-1]) / dy
        Lv = (txy_x[1:, :] - txy_x[:-1, :]) / dx + (tyy[:, 1:] - tyy[:, :-1]) / dy
        return Lu, Lv, txx, txy_x, txy_y, tyy
    
    def obstacle_force_from_stress(self, txx, txy_x, txy_y, tyy, p):
        """
        Compute integrated force on the square obstacle from viscous stresses.

        Returns
        -------
        Fx : float  # force in +x (drag)
        Fy : float  # force in +y (lift)
        """
        dx, dy = self.grid.dx, self.grid.dy
        solid = self.obstacle.solid

        # Vertical faces at the left and right edges of the solid
        # left boundary: faces where cell (i, j) is solid and (i-1, j) is fluid
        # right boundary: faces where cell (i-1, j) is solid and (i, j) is fluid
        left = np.zeros_like(txx, dtype=bool)
        right = np.zeros_like(txx, dtype=bool)

        for i in range(1, self.grid.nx):
            for j in range(self.grid.ny):
                # Left faces: solid cell with fluid neighbor to the left
                if solid[i, j] and not solid[i - 1, j]:
                    left[i, j] = True  # outward normal is (-1, 0)
                # Right faces: solid neighbor to left and fluid cell here
                if solid[i - 1, j] and not solid[i, j]:
                    right[i, j] = True  # outward normal is (+1, 0)

        # Horizontal faces at bottom and top edges of the solid
        bottom = np.zeros_like(tyy, dtype=bool)
        top = np.zeros_like(tyy, dtype=bool)

        for i in range(self.grid.nx):
            for j in range(1, self.grid.ny):
                # Bottom faces: solid cell with fluid neighbor below
                if solid[i, j] and not solid[i, j - 1]:
                    bottom[i, j] = True  # outward normal is (0, -1)
                # Top faces: solid cell below and fluid cell above
                if solid[i, j - 1] and not solid[i, j]:
                    top[i, j] = True     # outward normal is (0, +1)

        Fx = 0.0
        Fy = 0.0

        # Add viscous contributions on all obstacle faces using traction = sigma·n
        # Left faces: n = (-1, 0)
        # traction t = sigma·n => tx = -tau_xx, ty = -tau_xy
        Fx += np.sum(-txx[left] * dy)
        Fy += np.sum(-txy_x[left] * dy)

        # Right faces: n = (+1, 0)
        # traction: tx = +tau_xx, ty = +tau_xy
        Fx += np.sum(+txx[right] * dy)
        Fy += np.sum(+txy_x[right] * dy)

        # Bottom faces: n = (0, -1)
        # traction: tx = -tau_xy, ty = -tau_yy
        Fx += np.sum(-txy_y[bottom] * dx)
        Fy += np.sum(-tyy[bottom] * dx)

        # Top faces: n = (0, +1)
        # traction: tx = +tau_xy, ty = +tau_yy
        Fx += np.sum(+txy_y[top] * dx)
        Fy += np.sum(+tyy[top] * dx)

        # ---- pressure contribution on obstacle faces ----
        # We take pressure from the fluid cell adjacent to each obstacle-face

        # Left faces: solid[i,j] true and neighbor (i-1,j) is fluid
        # Normal n = (-1, 0) => traction t_p = -p * n => tx = -p*(-1) = +p, ty = 0
        p_left = np.zeros_like(txx)
        for i in range(1, self.grid.nx):
            for j in range(self.grid.ny):
                if left[i, j]:
                    p_left[i, j] = p[i - 1, j]

        Fx += np.sum(+p_left[left] * dy)

        # Right faces: solid[i-1,j] true and neighbor (i,j) is fluid
        # n = (+1, 0) => t_p = -p * n => tx = -p, ty = 0
        p_right = np.zeros_like(txx)
        for i in range(1, self.grid.nx):
            for j in range(self.grid.ny):
                if right[i, j]:
                    p_right[i, j] = p[i, j]

        Fx += np.sum(-p_right[right] * dy)

        # Bottom faces: solid[i,j] true and neighbor (i,j-1) is fluid
        # n = (0, -1) => t_p = -p * n => ty = -p*(-1) = +p
        p_bottom = np.zeros_like(tyy)
        for i in range(self.grid.nx):
            for j in range(1, self.grid.ny):
                if bottom[i, j]:
                    p_bottom[i, j] = p[i, j - 1]

        Fy += np.sum(+p_bottom[bottom] * dx)

        # Top faces: solid[i,j-1] true and neighbor (i,j) is fluid
        # n = (0, +1) => t_p = -p * n => ty = -p
        p_top = np.zeros_like(tyy)
        for i in range(self.grid.nx):
            for j in range(1, self.grid.ny):
                if top[i, j]:
                    p_top[i, j] = p[i, j]

        Fy += np.sum(-p_top[top] * dx)

        return float(Fx), float(Fy)

    # --- Predictor / corrector (projection) step ---
    # Advance velocities explicitly and project onto a divergence-free field.
    def predictor(self, state):
        # Copy current velocities and re-enforce boundary conditions
        u = state["u"].copy()
        v = state["v"].copy()
        self.apply_cell_constraints(u, v)

        # Current face-normal fluxes and transport velocities
        fx = state["fx"]
        fy = state["fy"]
        ux, vx, uy, vy = self.cell_to_transport_faces(u, v)

        # Explicit convective term using flux-form divergence
        conv_u = (fx[1:, :] * ux[1:, :] - fx[:-1, :] * ux[:-1, :]) / self.grid.dx
        conv_u += (fy[:, 1:] * uy[:, 1:] - fy[:, :-1] * uy[:, :-1]) / self.grid.dy

        conv_v = (fx[1:, :] * vx[1:, :] - fx[:-1, :] * vx[:-1, :]) / self.grid.dx
        conv_v += (fy[:, 1:] * vy[:, 1:] - fy[:, :-1] * vy[:, :-1]) / self.grid.dy

        # Viscous term from stress divergence (without obstacle masking)
        Lu, Lv, _, _, _, _ = self.viscous_operator(u, v, drop_obstacle=True)

        # Explicit Euler predictor step for velocity
        u_star = u + self.dt * (-conv_u + Lu / self.rho)
        v_star = v + self.dt * (-conv_v + Lv / self.rho)

        # Re-apply boundary conditions and obstacle no-slip
        self.apply_cell_constraints(u_star, v_star)
        return u_star, v_star

    def project(self, u_star: np.ndarray, v_star: np.ndarray):
        # Build provisional fluxes from predicted velocities
        fx_star, fy_star = self.cell_to_normal_fluxes(u_star, v_star)

        # Right-hand side for pressure Poisson equation: scaled divergence of fluxes
        rhs = (self.rho / self.dt) * self.divergence(fx_star, fy_star)

        # Solve for pressure correction
        p = self.projector.solve(rhs)

        # Pressure gradients at faces
        dpdx, dpdy = self.face_pressure_gradients(p)

        # Correct fluxes to enforce mass conservation
        fx = fx_star - (self.dt / self.rho) * dpdx
        fy = fy_star - (self.dt / self.rho) * dpdy

        # Reinstate boundary conditions and obstacle cut-faces
        fx[0, :] = self.bc.u_in
        fx[-1, :] = fx[-2, :]
        fy[:, 0] = 0.0
        fy[:, -1] = 0.0
        fx[self.obstacle.face_x] = 0.0
        fy[self.obstacle.face_y] = 0.0

        # Correct cell-centered velocities using averaged pressure gradients
        u = u_star - (self.dt / self.rho) * 0.5 * (dpdx[:-1, :] + dpdx[1:, :])
        v = v_star - (self.dt / self.rho) * 0.5 * (dpdy[:, :-1] + dpdy[:, 1:])

        # Apply boundary conditions again after correction
        self.apply_cell_constraints(u, v)

        # Return updated state plus rhs and final divergence for diagnostics
        return {"u": u, "v": v, "p": p, "fx": fx, "fy": fy}, rhs, self.divergence(fx, fy)

    def advance(self, state):
        # One full time step: predictor followed by pressure projection
        u_star, v_star = self.predictor(state)
        return self.project(u_star, v_star)

    # --- Run loop and diagnostics ---
    # Advance in time, record convergence metrics, forces, snapshots, and steady means.
    def run(self, final_time: float, print_every: int = 200, tolerance: float | None = None,
        snapshot_times=None):
        # Integrate in time up to final_time with given print and convergence options
        nsteps = int(np.ceil(final_time / self.dt))
        # Allocate initial state and perform an initial projection to make it divergence-free
        state = self.allocate_state()
        state, rhs, div = self.project(state["u"], state["v"])

        # Raises error if NaN output is detected, prevents wasted run
        if not np.isfinite(state["u"]).all() or not np.isfinite(state["v"]).all() or not np.isfinite(state["p"]).all():
            raise FloatingPointError(f"NaN/Inf detected at step {step}, t={step*self.dt:.4f}")

        # Storage for monitoring convergence and force histories
        history = {"time": [], "max_div": [], "max_delta": [], "u_out": []}
        history["Fx"] = []
        history["Fy"] = []
        history["Cd"] = []

        # Snapshot steps from needed times (in seconds)
        snapshot_steps = {}
        if snapshot_times is not None:
            for t in snapshot_times:
                k = int(round(t / self.dt))
                if 1 <= k <= nsteps:
                    snapshot_steps[k] = float(t)

        snapshots = {}

        for step in range(1, nsteps + 1):
            # Save previous velocities for computing max change
            u_prev = state["u"].copy()
            v_prev = state["v"].copy()

            # Advance one time step
            state, rhs, div = self.advance(state)

            # Instantaneous total force on the obstacle (viscous + pressure)
            Lu_tmp, Lv_tmp, txx, txy_x, txy_y, tyy = self.viscous_operator(
                state["u"], state["v"], drop_obstacle=False
            )
            Fx, Fy = self.obstacle_force_from_stress(txx, txy_x, txy_y, tyy, state["p"])

            # Drag coefficient based on reference scale
            Cd = Fx / self.F_ref if self.F_ref != 0.0 else 0.0

            # Diagnostics: max change in velocity and divergence
            max_delta = float(max(np.max(np.abs(state["u"] - u_prev)), np.max(np.abs(state["v"] - v_prev))))
            max_div = float(np.max(np.abs(div)))
            u_out = float(np.mean(state["u"][-1, :])) # mean outlet velocity

            # Store snapshot if this step is close to a requested time
            if step in snapshot_steps:
                snapshots[step] = {
                    "time": snapshot_steps[step],
                    "u": state["u"].copy(),
                    "v": state["v"].copy(),
                }

            # Append history
            history["time"].append(step * self.dt)
            history["max_div"].append(max_div)
            history["max_delta"].append(max_delta)
            history["u_out"].append(u_out)
            history["Fx"].append(Fx)
            history["Fy"].append(Fy)
            history["Cd"].append(Cd)

            # Periodic progress output
            if step == 1 or step % print_every == 0 or step == nsteps:
                print(
                    f"iter={step:6d} | t={step*self.dt:10.4f} | "
                    f"u_out={u_out:9.5f} | max|div|={max_div:9.2e} | dmax={max_delta:9.2e}"
                )

            # Optional early stop if solution has converged to steady state
            if tolerance is not None and step > 20 and max_delta < tolerance:
                break

        # --- Steady-state statistics (last 20% of the run) ---
        n = len(history["time"])
        if n > 0:
            start_idx = int(0.8 * n)  # start at 80% of the recorded steps
        else:
            start_idx = 0

        Fx_mean_steady = float(np.mean(history["Fx"][start_idx:]))
        Fy_mean_steady = float(np.mean(history["Fy"][start_idx:]))
        Cd_mean_steady = float(np.mean(history["Cd"][start_idx:]))

        # Convert lists to arrays for existing keys only
        for key in list(history.keys()):
            history[key] = np.asarray(history[key])

        # Attach steady-state means as scalars
        history["Fx_mean_steady"] = Fx_mean_steady
        history["Fy_mean_steady"] = Fy_mean_steady
        history["Cd_mean_steady"] = Cd_mean_steady

        return state, history, snapshots

# ----------------------------------------------------------------------
# POST-PROCESSING
# ----------------------------------------------------------------------
# Provide utility routines for derived flow quantities such as vorticity
# and for organizing simulation output into histories and snapshots.

def compute_vorticity(u: np.ndarray, v: np.ndarray, grid: Grid):
    # Compute scalar vorticity omega = dv/dx - du/dy on cell centers
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)

    # Centered differences in the interior, one-sided at boundaries
    dvdx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * grid.dx)
    dvdx[0, :] = (v[1, :] - v[0, :]) / grid.dx
    dvdx[-1, :] = (v[-1, :] - v[-2, :]) / grid.dx
    dudy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * grid.dy)
    dudy[:, 0] = 0.0
    dudy[:, -1] = 0.0
    return dvdx - dudy

# ----------------------------------------------------------------------
# MAIN PROGRAM + PLOTTING
# ----------------------------------------------------------------------
# Define simulation parameters, run the solver for multiple inlet
# velocities, print a validation table, and generate plots of drag
# history, drag coefficients, flow fields, and convergence measures.

if __name__ == "__main__":
    # Base physical and numerical parameters shared across runs
    base_params = {
        "Lx": 40, #Volume lengt
        "Ly": 30, #Volume height
        "Nx": 160, #Horizontal number of elements
        "Ny": 120, #Vertical number of elements
        "rho": 1025, #density of salt water
        #"Re": 100.0, # comment out for fixed viscosity and fluid property case
        "nu": 0.05, # fixed kinematic viscosity
        "pressure_outlet": 0.0,
        "square_side": 9.4,  # obstacle side length [m]
        "square_xc": 15.0,   # place obstacle 15m from inlet
        "square_yc": 15.0,   # mid-height
        "dt": None,
        "CFL": 0.10,
    }

    # Range of inlet velocities to sweep for validation
    inlet_velocities = [0.5, 1.0, 1.5, 2.0]
    final_time = 100.0
    snapshot_times = [10.0, 50.0, 100.0]

    all_results = []

    for U_in in inlet_velocities:
        # Clone base parameters and update inlet velocit
        params = dict(base_params)
        params["U_in"] = U_in

        # Set up solver and compute corresponding Reynolds number
        solver = OpenFlowSolver(params)
        Re_case = U_in * params["square_side"] / params["nu"]

        # Print run header with grid, fluid and time-step information
        print("=" * 72)
        print(f"RUN FOR U_in = {U_in:.3f} m/s")
        print("=" * 72)
        print(f"Grid : {solver.grid.nx} x {solver.grid.ny}")
        print(f"dx, dy : {solver.grid.dx:.6e}, {solver.grid.dy:.6e}")
        print(f"rho : {solver.rho:.6e}")
        print(f"nu : {solver.nu:.6e}")
        print(f"mu  : {solver.mu:.6e}")
        print(f"Re  : {Re_case:.6e}")
        print(f"dt : {solver.dt:.6e}")

        # Run simulation, collect state, history and wake snapshots
        state, history, snapshots = solver.run(
                final_time=final_time,
                print_every=200,
                tolerance=1.0e-8,
                snapshot_times=snapshot_times,
            )

        # Store summary data for this inlet velocity
        all_results.append({
            "U_in": U_in,
            "Re": Re_case,
            "solver": solver,
            "state": state,
            "history": history,
            "snapshots": snapshots,
            "Fx_mean": history["Fx_mean_steady"],
            "Cd_mean": history["Cd_mean_steady"],
        })

        # Print steady-state mean forces and drag coefficient
        print(f"Steady Fx (last 20%) = {history['Fx_mean_steady']:.6e} N/m")
        print(f"Steady Fy (last 20%) = {history['Fy_mean_steady']:.6e} N/m")
        print(f"Steady Cd (last 20%) = {history['Cd_mean_steady']:.3f}")

    # --- validation tabel ---
    # (inlet velocity, mean drag force, mean drag coeff.)
    print("\nValidation table")
    print(f"{'U_in [m/s]':>12} {'Re [-]':>14} {'Mean Drag Fx [N/m]':>20} {'Mean Cd [-]':>15}")
    print("-" * 68)

    U_vals = []
    Re_vals = []
    Fx_means = []
    Cd_means = []

    for res in all_results:
        U_vals.append(res["U_in"])
        Re_vals.append(res["Re"])
        Fx_means.append(res["Fx_mean"])
        Cd_means.append(res["Cd_mean"])
        print(f"{res['U_in']:12.3f} {res['Re']:14.6e} {res['Fx_mean']:20.6e} {res['Cd_mean']:15.6f}")

    U_vals = np.array(U_vals)
    Re_vals = np.array(Re_vals)
    Fx_means = np.array(Fx_means)
    Cd_means = np.array(Cd_means)

    # --- validation plots ---
    # Drag force history for different inlet velocities
    plt.figure(figsize=(8, 5))
    for res in all_results:
        plt.plot(
            res["history"]["time"],
            res["history"]["Fx"],
            label=fr"$U_{{in}}={res['U_in']:.2f}$ m/s"
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Drag force Fx [N/m]")
    plt.title("Drag force history for different inlet velocities")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Mean drag coefficient vs inlet velocity
    plt.figure(figsize=(7, 4.5))
    plt.plot(U_vals, Cd_means, "o-", lw=2)
    plt.xlabel("Inlet velocity U_in [m/s]")
    plt.ylabel("Mean drag coefficient Cd [-]")
    plt.title("Mean drag coefficient versus inlet velocity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Wake snapshots ---
    # Pick one representative inlet velocity case
    res = all_results[-1]
    solver = res["solver"]
    snapshots = res["snapshots"]

    X, Y = solver.grid.X, solver.grid.Y
    solid = solver.obstacle.solid.astype(float)

    seed_y = np.linspace(0.05 * solver.grid.ly, 0.95 * solver.grid.ly, 20)
    seed_x = np.full_like(seed_y, 0.1 * solver.grid.lx)

    steps_sorted = sorted(snapshots.keys())

    fig, axes = plt.subplots(1, len(steps_sorted), figsize=(6 * len(steps_sorted), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, step in zip(axes, steps_sorted):
        snap = snapshots[step]
        u_s = snap["u"]
        v_s = snap["v"]
        speed_s = np.hypot(u_s, v_s)

        cf = ax.contourf(X, Y, speed_s, levels=40, cmap="viridis")
        fig.colorbar(cf, ax=ax, label="Speed [m/s]")

        u_plot = u_s.copy()
        v_plot = v_s.copy()
        u_plot[solid > 0.5] = 0.0
        v_plot[solid > 0.5] = 0.0

        ax.streamplot(
            solver.grid.x,
            solver.grid.y,
            u_plot.T,
            v_plot.T,
            start_points=np.vstack([seed_x, seed_y]).T,
            density=1.2,
            linewidth=0.8,
            color="k",
        )

        ax.contour(X, Y, solid, levels=[0.5], colors="w", linewidths=1.5)
        ax.set_title(f"t = {snap['time']:.1f} s, U_in = {res['U_in']:.2f} m/s")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

    plt.show()

    vort = compute_vorticity(state["u"], state["v"], solver.grid)

    # --- streamline development plot ---
    X, Y = solver.grid.X, solver.grid.Y
    solid = solver.obstacle.solid.astype(float)

    # Choose a fixed seeding region (for example, vertical line near inlet)
    seed_y = np.linspace(0.05 * solver.grid.ly, 0.95 * solver.grid.ly, 20)
    seed_x = np.full_like(seed_y, 0.1 * solver.grid.lx)

    # Order snapshots by time
    steps_sorted = sorted(snapshots.keys())
    n_snap = len(steps_sorted)
    ncols = 3
    nrows = int(np.ceil(n_snap / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, step in zip(axes, steps_sorted):
        snap = snapshots[step]
        u_s = snap["u"]
        v_s = snap["v"]

        speed_s = np.hypot(u_s, v_s)
        cf = ax.contourf(X, Y, speed_s, levels=40)
        fig.colorbar(cf, ax=ax)

        # Mask the obstacle for streamlines
        u_plot = u_s.copy()
        v_plot = v_s.copy()
        u_plot[solid > 0.5] = 0.0
        v_plot[solid > 0.5] = 0.0

        ax.streamplot(
            solver.grid.x,
            solver.grid.y,
            u_plot.T,
            v_plot.T,
            start_points=np.vstack([seed_x, seed_y]).T,
            density=1.2,
            linewidth=0.7,
            color="k",
        )

        ax.contour(X, Y, solid, levels=[0.5], colors="w", linewidths=1.2)
        ax.set_title(f"t = {snap['time']:.1f} s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    # Turn off unused axes if any
    for ax in axes[n_snap:]:
        ax.axis("off")

    speed = np.hypot(state["u"], state["v"])
    X, Y = solver.grid.X, solver.grid.Y
    solid = solver.obstacle.solid.astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fields = [state["u"], state["p"], vort, np.log10(np.abs(speed) + 1.0e-14)]
    titles = ["u velocity", "pressure", "vorticity", "log10 speed"]

    for ax, field, title in zip(axes.ravel(), fields, titles):
        c = ax.contourf(X, Y, field, levels=40)
        fig.colorbar(c, ax=ax)
        ax.contour(X, Y, solid, levels=[0.5], colors="k", linewidths=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.figure(figsize=(8, 4))
    plt.semilogy(history["time"], history["max_div"] + 1.0e-30, label="max divergence")
    plt.semilogy(history["time"], history["max_delta"] + 1.0e-30, label="max change")
    plt.xlabel("time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

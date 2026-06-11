from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------
# DATA CONTAINERS
# ----------------------------------------------------------------------

@dataclass
class Grid:
    nx: int
    ny: int
    lx: float
    ly: float

    def __post_init__(self):
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

@dataclass
class BoundarySpec:
    u_in: float
    p_out: float

# ----------------------------------------------------------------------
# OBSTACLE GEOMETRY
# ----------------------------------------------------------------------

class SquareObstacleMask:
    def __init__(self, grid: Grid, side: float, xc: float, yc: float):
        h = 0.5 * side
        self.solid = (
            (np.abs(grid.X - xc) <= h) &
            (np.abs(grid.Y - yc) <= h)
        )
        self.fluid = ~self.solid
        self.face_x = np.zeros((grid.nx + 1, grid.ny), dtype=bool)
        self.face_y = np.zeros((grid.nx, grid.ny + 1), dtype=bool)
        self._build_face_cuts(grid)

    def _build_face_cuts(self, grid: Grid):
        for i in range(1, grid.nx):
            self.face_x[i, :] = self.solid[i - 1, :] ^ self.solid[i, :]
        for j in range(1, grid.ny):
            self.face_y[:, j] = self.solid[:, j - 1] ^ self.solid[:, j]

# ----------------------------------------------------------------------
# PRESSURE PROJECTION
# ----------------------------------------------------------------------

class PressureProjector:
    def __init__(self, grid: Grid, bc: BoundarySpec):
        self.grid = grid
        self.bc = bc
        self.matrix = self._assemble_matrix()
        self.solve_linear = spla.factorized(self.matrix.tocsc())

    def _idx(self, i: int, j: int) -> int:
        return i + j * self.grid.nx

    def _assemble_matrix(self):
        nx, ny = self.grid.nx, self.grid.ny
        dx2, dy2 = self.grid.dx**2, self.grid.dy**2
        rows, cols, data = [], [], []

        def put(r, c, v):
            rows.append(r)
            cols.append(c)
            data.append(v)

        for j in range(ny):
            for i in range(nx):
                I = self._idx(i, j)

                if i == nx - 1:
                    put(I, I, 1.0)
                    continue

                diag = 0.0

                if i == 0:
                    put(I, self._idx(i + 1, j), 1.0 / dx2)
                    diag -= 1.0 / dx2
                else:
                    put(I, self._idx(i - 1, j), 1.0 / dx2)
                    diag -= 1.0 / dx2

                put(I, self._idx(i + 1, j), 1.0 / dx2)
                diag -= 1.0 / dx2

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

                put(I, I, diag)

        N = nx * ny
        return sps.csr_matrix((data, (rows, cols)), shape=(N, N))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        nx, ny = self.grid.nx, self.grid.ny
        b = rhs.reshape(nx * ny, order="F").copy()
        for j in range(ny):
            b[self._idx(nx - 1, j)] = self.bc.p_out
        p = self.solve_linear(b).reshape((nx, ny), order="F")
        p[0, :] = p[1, :]
        p[-1, :] = self.bc.p_out
        p[:, 0] = p[:, 1]
        p[:, -1] = p[:, -2]
        return p

# ----------------------------------------------------------------------
# FLOW SOLVER
# ----------------------------------------------------------------------

class OpenFlowSolver:
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.grid = Grid(cfg["Nx"], cfg["Ny"], cfg["Lx"], cfg["Ly"])
        self.bc = BoundarySpec(cfg["U_in"], cfg["pressure_outlet"])
        self.nu = cfg["U_in"] * cfg["square_side"] / cfg["Re"]
        self.rho = cfg["rho"]
        self.mu = self.rho * self.nu

        # reference drag scale: 0.5 * rho * U^2 * D  (per m out of plane)
        self.D_ref = cfg["square_side"]
        self.U_ref = cfg["U_in"]
        self.F_ref = 0.5 * self.rho * self.U_ref**2 * self.D_ref

        self.dt = self._choose_dt(cfg.get("dt"), cfg["CFL"])
        self.obstacle = SquareObstacleMask(self.grid, cfg["square_side"], cfg["square_xc"], cfg["square_yc"])
        self.projector = PressureProjector(self.grid, self.bc)

    def _choose_dt(self, dt_manual, cfl):
        if dt_manual is not None:
            return float(dt_manual)
        u = max(abs(self.bc.u_in), 1.0e-12)
        conv = cfl / (u / self.grid.dx + u / self.grid.dy)
        diff = 0.5 / (max(self.nu, 1.0e-12) * (1.0 / self.grid.dx**2 + 1.0 / self.grid.dy**2))
        return min(conv, diff)

    def allocate_state(self):
        u = np.full((self.grid.nx, self.grid.ny), self.bc.u_in, dtype=float)
        v = np.zeros_like(u)
        p = np.zeros_like(u)
        self.apply_cell_constraints(u, v)
        fx, fy = self.cell_to_normal_fluxes(u, v)
        return {"u": u, "v": v, "p": p, "fx": fx, "fy": fy}

    def apply_cell_constraints(self, u: np.ndarray, v: np.ndarray):
        u[0, :] = self.bc.u_in
        v[0, :] = 0.0
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        u[self.obstacle.solid] = 0.0
        v[self.obstacle.solid] = 0.0

    def cell_to_normal_fluxes(self, u: np.ndarray, v: np.ndarray):
        nx, ny = self.grid.nx, self.grid.ny
        fx = np.zeros((nx + 1, ny), dtype=float)
        fy = np.zeros((nx, ny + 1), dtype=float)
        fx[1:nx, :] = 0.5 * (u[:-1, :] + u[1:, :])
        fy[:, 1:ny] = 0.5 * (v[:, :-1] + v[:, 1:])
        fx[0, :] = self.bc.u_in
        fx[-1, :] = u[-1, :]
        fy[:, 0] = 0.0
        fy[:, -1] = 0.0
        fx[self.obstacle.face_x] = 0.0
        fy[self.obstacle.face_y] = 0.0
        return fx, fy

    def cell_to_transport_faces(self, u: np.ndarray, v: np.ndarray):
        nx, ny = self.grid.nx, self.grid.ny
        ux = np.zeros((nx + 1, ny), dtype=float)
        vx = np.zeros((nx + 1, ny), dtype=float)
        uy = np.zeros((nx, ny + 1), dtype=float)
        vy = np.zeros((nx, ny + 1), dtype=float)

        ux[1:nx, :] = 0.5 * (u[:-1, :] + u[1:, :])
        vx[1:nx, :] = 0.5 * (v[:-1, :] + v[1:, :])
        uy[:, 1:ny] = 0.5 * (u[:, :-1] + u[:, 1:])
        vy[:, 1:ny] = 0.5 * (v[:, :-1] + v[:, 1:])

        ux[0, :] = self.bc.u_in
        vx[0, :] = 0.0
        ux[-1, :] = u[-1, :]
        vx[-1, :] = v[-1, :]
        uy[:, 0] = u[:, 0]
        uy[:, -1] = u[:, -1]
        vy[:, 0] = 0.0
        vy[:, -1] = 0.0

        ux[self.obstacle.face_x] = 0.0
        vx[self.obstacle.face_x] = 0.0
        uy[self.obstacle.face_y] = 0.0
        vy[self.obstacle.face_y] = 0.0
        return ux, vx, uy, vy

    def divergence(self, fx: np.ndarray, fy: np.ndarray):
        return (fx[1:, :] - fx[:-1, :]) / self.grid.dx + (fy[:, 1:] - fy[:, :-1]) / self.grid.dy

    def face_pressure_gradients(self, p: np.ndarray):
        nx, ny = self.grid.nx, self.grid.ny
        dpdx = np.zeros((nx + 1, ny), dtype=float)
        dpdy = np.zeros((nx, ny + 1), dtype=float)
        dpdx[1:nx, :] = (p[1:, :] - p[:-1, :]) / self.grid.dx
        dpdy[:, 1:ny] = (p[:, 1:] - p[:, :-1]) / self.grid.dy
        dpdx[0, :] = 0.0
        dpdx[-1, :] = 0.0
        dpdy[:, 0] = 0.0
        dpdy[:, -1] = 0.0
        return dpdx, dpdy

    def viscous_operator(self, u: np.ndarray, v: np.ndarray, drop_obstacle: bool=True):
        dx, dy, mu = self.grid.dx, self.grid.dy, self.mu
        nx, ny = self.grid.nx, self.grid.ny

        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        du_dy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dy)
        dv_dx[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * dx)
        dv_dx[0, :] = (v[1, :] - v[0, :]) / dx
        dv_dx[-1, :] = 0.0

        txx = np.zeros((nx + 1, ny), dtype=float)
        txy_x = np.zeros((nx + 1, ny), dtype=float)
        txy_y = np.zeros((nx, ny + 1), dtype=float)
        tyy = np.zeros((nx, ny + 1), dtype=float)

        txx[1:nx, :] = 2.0 * mu * (u[1:, :] - u[:-1, :]) / dx
        txx[0, :] = 0.0
        txx[-1, :] = 0.0

        txy_x[1:nx, :] = mu * (0.5 * (du_dy[:-1, :] + du_dy[1:, :]) + (v[1:, :] - v[:-1, :]) / dx)
        txy_x[0, :] = mu * (du_dy[0, :] + dv_dx[0, :])
        txy_x[-1, :] = 0.0

        du_dy_face = np.zeros((nx, ny + 1), dtype=float)
        dv_dy_face = np.zeros((nx, ny + 1), dtype=float)
        dv_dx_face = np.zeros((nx, ny + 1), dtype=float)
        du_dy_face[:, 1:ny] = (u[:, 1:] - u[:, :-1]) / dy
        dv_dy_face[:, 1:ny] = (v[:, 1:] - v[:, :-1]) / dy
        dv_dx_face[:, 1:ny] = 0.5 * (dv_dx[:, :-1] + dv_dx[:, 1:])

        txy_y = mu * (du_dy_face + dv_dx_face)
        tyy = 2.0 * mu * dv_dy_face
        txy_y[:, 0] = 0.0
        txy_y[:, -1] = 0.0

        if drop_obstacle:
            txx[self.obstacle.face_x] = 0.0
            txy_x[self.obstacle.face_x] = 0.0
            txy_y[self.obstacle.face_y] = 0.0
            tyy[self.obstacle.face_y] = 0.0

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
                if solid[i, j] and not solid[i - 1, j]:
                    left[i, j] = True  # outward normal is (-1, 0)
                if solid[i - 1, j] and not solid[i, j]:
                    right[i, j] = True  # outward normal is (+1, 0)

        # Horizontal faces at bottom and top edges of the solid
        bottom = np.zeros_like(tyy, dtype=bool)
        top = np.zeros_like(tyy, dtype=bool)

        for i in range(self.grid.nx):
            for j in range(1, self.grid.ny):
                if solid[i, j] and not solid[i, j - 1]:
                    bottom[i, j] = True  # outward normal is (0, -1)
                if solid[i, j - 1] and not solid[i, j]:
                    top[i, j] = True     # outward normal is (0, +1)

        Fx = 0.0
        Fy = 0.0

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

    def predictor(self, state):
        u = state["u"].copy()
        v = state["v"].copy()
        self.apply_cell_constraints(u, v)

        fx = state["fx"]
        fy = state["fy"]
        ux, vx, uy, vy = self.cell_to_transport_faces(u, v)

        conv_u = (fx[1:, :] * ux[1:, :] - fx[:-1, :] * ux[:-1, :]) / self.grid.dx
        conv_u += (fy[:, 1:] * uy[:, 1:] - fy[:, :-1] * uy[:, :-1]) / self.grid.dy

        conv_v = (fx[1:, :] * vx[1:, :] - fx[:-1, :] * vx[:-1, :]) / self.grid.dx
        conv_v += (fy[:, 1:] * vy[:, 1:] - fy[:, :-1] * vy[:, :-1]) / self.grid.dy

        Lu, Lv, _, _, _, _ = self.viscous_operator(u, v, drop_obstacle=True)

        u_star = u + self.dt * (-conv_u + Lu / self.rho)
        v_star = v + self.dt * (-conv_v + Lv / self.rho)
        self.apply_cell_constraints(u_star, v_star)
        return u_star, v_star

    def project(self, u_star: np.ndarray, v_star: np.ndarray):
        fx_star, fy_star = self.cell_to_normal_fluxes(u_star, v_star)
        rhs = (self.rho / self.dt) * self.divergence(fx_star, fy_star)
        p = self.projector.solve(rhs)
        dpdx, dpdy = self.face_pressure_gradients(p)

        fx = fx_star - (self.dt / self.rho) * dpdx
        fy = fy_star - (self.dt / self.rho) * dpdy
        fx[0, :] = self.bc.u_in
        fx[-1, :] = fx[-2, :]
        fy[:, 0] = 0.0
        fy[:, -1] = 0.0
        fx[self.obstacle.face_x] = 0.0
        fy[self.obstacle.face_y] = 0.0

        u = u_star - (self.dt / self.rho) * 0.5 * (dpdx[:-1, :] + dpdx[1:, :])
        v = v_star - (self.dt / self.rho) * 0.5 * (dpdy[:, :-1] + dpdy[:, 1:])
        self.apply_cell_constraints(u, v)

        return {"u": u, "v": v, "p": p, "fx": fx, "fy": fy}, rhs, self.divergence(fx, fy)

    def advance(self, state):
        u_star, v_star = self.predictor(state)
        return self.project(u_star, v_star)

    def run(self, final_time: float, print_every: int = 200, tolerance: float | None = None,
        snapshot_times=None):
        nsteps = int(np.ceil(final_time / self.dt))
        state = self.allocate_state()
        state, rhs, div = self.project(state["u"], state["v"])

        history = {"time": [], "max_div": [], "max_delta": [], "u_out": []}
        history["Fx"] = []
        history["Fy"] = []
        history["Cd"] = []

        # Snapshot steps from requested times (in seconds)
        snapshot_steps = {}
        if snapshot_times is not None:
            for t in snapshot_times:
                k = int(round(t / self.dt))
                if 1 <= k <= nsteps:
                    snapshot_steps[k] = float(t)

        snapshots = {}

        for step in range(1, nsteps + 1):
            u_prev = state["u"].copy()
            v_prev = state["v"].copy()
            state, rhs, div = self.advance(state)

            # Instantaneous total force on the obstacle (viscous + pressure)
            Lu_tmp, Lv_tmp, txx, txy_x, txy_y, tyy = self.viscous_operator(
                state["u"], state["v"], drop_obstacle=False
            )
            Fx, Fy = self.obstacle_force_from_stress(txx, txy_x, txy_y, tyy, state["p"])

            # Drag coefficient based on reference scale
            Cd = Fx / self.F_ref if self.F_ref != 0.0 else 0.0

            max_delta = float(max(np.max(np.abs(state["u"] - u_prev)), np.max(np.abs(state["v"] - v_prev))))
            max_div = float(np.max(np.abs(div)))
            u_out = float(np.mean(state["u"][-1, :]))

            # Store snapshot if this step is close to a requested time
            if step in snapshot_steps:
                snapshots[step] = {
                    "time": snapshot_steps[step],
                    "u": state["u"].copy(),
                    "v": state["v"].copy(),
                }

            history["time"].append(step * self.dt)
            history["max_div"].append(max_div)
            history["max_delta"].append(max_delta)
            history["u_out"].append(u_out)
            history["Fx"].append(Fx)
            history["Fy"].append(Fy)
            history["Cd"].append(Cd)

            if step == 1 or step % print_every == 0 or step == nsteps:
                print(
                    f"iter={step:6d} | t={step*self.dt:10.4f} | "
                    f"u_out={u_out:9.5f} | max|div|={max_div:9.2e} | dmax={max_delta:9.2e}"
                )

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

def compute_vorticity(u: np.ndarray, v: np.ndarray, grid: Grid):
    dvdx = np.zeros_like(v)
    dudy = np.zeros_like(u)
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

if __name__ == "__main__":
    params = {
        "Lx": 40,
        "Ly": 30,
        "Nx": 160,
        "Ny": 120,
        "rho": 1025,
        "U_in": 1.0,
        "Re": 100.0,
        "pressure_outlet": 0.0,
        "square_side": 9.4, ## obstacle side length [m]
        "square_xc": 15, # place obstacle 15m from inlet
        "square_yc": 15, # and mid-height = 15m
        "dt": None,
        "CFL": 0.20,
    }

    solver = OpenFlowSolver(params)

    print("=" * 72)
    print("ALTERNATIVE OPEN-BOUNDARY SQUARE-OBSTACLE SOLVER")
    print("=" * 72)
    print(f"Grid : {solver.grid.nx} x {solver.grid.ny}")
    print(f"dx, dy : {solver.grid.dx:.6e}, {solver.grid.dy:.6e}")
    print(f"nu : {solver.nu:.6e}")
    print(f"dt : {solver.dt:.6e}")
    print("Inlet  : prescribed velocity, dp/dx = 0")
    print("Outlet : prescribed pressure, du/dx = dv/dx = 0")
    print("Walls  : stress free with v = 0")
    print("Square : no-slip through obstacle-face flux cancellation")
    print("=" * 72)

    snapshot_times = [0.5, 1.0, 10.0, 30.0, 50.0, 100.0]
    state, history, snapshots = solver.run(
        final_time=100.0,  # must reach the last snapshot time
        print_every=200,
        tolerance=1.0e-8,
        snapshot_times=snapshot_times,
    )

    print(f"Steady Fx (last 20%) = {history['Fx_mean_steady']:.6e} N/m")
    print(f"Steady Fy (last 20%) = {history['Fy_mean_steady']:.6e} N/m")
    print(f"Steady Cd (last 20%) = {history['Cd_mean_steady']:.3f}")

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

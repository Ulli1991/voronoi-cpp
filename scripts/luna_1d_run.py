"""LUNA's 1D tests (Sod-Shock, Shu-Osher, Woodward-Colella) at LUNA's settings, timed.

Run from the LUNA repository with its environment (LUNA's src on the path):
    python luna_1d_run.py <sod|shu-osher|woodward-colella> <N> <hllc|hll> <out.npz> [LIMITER=1] [CFL=0.8] [steps=0]

Setups as LUNA's tests: Sod = Toro test 1 from the exact cell averages
(tests/test_hydro_accuracy_shocktube.py:96-99, exact_riemann.py:9, 105-113),
BOUNDARY 1, t = 0.25; Shu-Osher = initial_conditions (shu_osher.py:26-46,
exact cell averages), Lx = 10, BOUNDARY 1, t = 1.8 (test_hydro_hard_shu_osher.py:29, 62);
Woodward-Colella = initial_conditions (woodward_colella.py:14-27), BOUNDARY 2,
t = 0.038 (test_hydro_hard_woodward_colella.py:28, 63). gamma 1.4, ORDER 1,
FP64, CFL 0.8 (template_config.py:32), the last step shortened to t_end
(as tests/_helpers.py:186). steps > 0: stop after that many steps (throughput).

The loop is one jitted lax.while_loop around LUNA's compute_dt and step
(create_solver_distributed, setup/hydro.py:140-202, as tests/_helpers.py
evolve): compiled by a first call, then timed on a second call from the
same initial state. Writes rho, u, p (cell order), steps, wall_s.
"""
import sys
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

from luna.template_config import Config  # noqa: E402
from luna.ic_generation.ic import initial_conditions  # noqa: E402
from luna.hydro import exact_riemann as er  # noqa: E402
from luna.setup.hydro import create_solver_distributed  # noqa: E402
from luna.utils.grid_params import GridParams  # noqa: E402

G = 1.4
SETUPS = {
    "sod": dict(test_problem="Sod-Shock", Lx=1.0, BOUNDARY=1, t_end=0.25),
    "shu-osher": dict(test_problem="Shu-Osher", Lx=10.0, BOUNDARY=1, t_end=1.8),
    "woodward-colella": dict(test_problem="Woodward-Colella", Lx=1.0, BOUNDARY=2, t_end=0.038),
}


def main():
    problem, n, solver, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    extra = dict(a.split("=") for a in sys.argv[5:])
    limiter, cfl, max_steps = int(extra.get("LIMITER", 1)), float(extra.get("CFL", 0.8)), int(extra.get("steps", 0))
    setup = SETUPS[problem]
    cfg = Config(SPATIAL_DIMENSION=1, Nx=n, gamma=G, ORDER=1, LIMITER=limiter, riemann_solver=solver, CFL=cfl,
                 enable_x64=True, PLOT_AFTER_SIM=False, enable_checkpointing=False, **setup)
    cfg.validate()
    if problem == "sod":
        WL, WR, x0, _ = er.TORO[1]
        rho, u, p = er.cell_average(WL, WR, G, np.linspace(0.0, 1.0, n + 1), x0, 1e-12)
        U0 = np.stack([rho, rho * u, p / (G - 1) + 0.5 * rho * u * u])
    else:
        U0 = np.asarray(initial_conditions(cfg))
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("x",))
    compute_dt, step, sharding = create_solver_distributed(cfg, GridParams.from_config(cfg), mesh)
    t_end = setup["t_end"]
    cap = max_steps if max_steps > 0 else 10**9

    @jax.jit
    def advance(U):
        def body(s):
            U, t, k = s
            dt = jnp.minimum(compute_dt(U), t_end - t)
            return step(U, dt), t + dt, k + 1

        def go(s):
            return (s[1] < t_end * (1 - 1e-12)) & (s[2] < cap)

        return jax.lax.while_loop(go, body, (U, jnp.float64(0.0), jnp.int32(0)))

    U_start = jax.device_put(jnp.asarray(U0, dtype=jnp.float64), sharding)
    jax.block_until_ready(advance(U_start))  # compile (and a first run)
    t0 = time.perf_counter()
    U, t, k = jax.block_until_ready(advance(U_start))
    wall = time.perf_counter() - t0
    U = np.asarray(U)
    rho = U[0]
    u = U[1] / rho
    p = (G - 1) * (U[2] - 0.5 * rho * u * u)
    steps = int(k)
    np.savez(out, rho=rho, u=u, p=p, steps=steps, wall_s=wall, t=float(t), N=n, problem=problem, solver=solver,
             limiter=limiter, cfl=cfl, device=str(jax.devices()[0]))
    print(f"luna: problem={problem} N={n} solver={solver} LIMITER={limiter} CFL={cfl} steps={steps} t={float(t):.6g} "
          f"wall_s={wall:.4f} cell_updates_per_s={n * steps / wall:.4g} device={jax.devices()[0]}")


if __name__ == "__main__":
    main()

"""LUNA's Sedov 3D (scripts/validation/runs/sedov3d_run.py) with the whole box moving along x: a moving blast wave.

    cd <LUNA> && python <this file> <nx> <bulk km/s> <output dir> [Config field=value ...]

The initial state is LUNA's own Sedov-Blast-3D state (ic_generation/sedov3d.py) with the bulk velocity
added (momentum rho v, kinetic energy rho v^2 / 2), handed to luna.launch as a user state
(ic_generation/user_ic.py). The boundary is periodic (BOUNDARY=0), so the flow leaves and enters the box
like the Voronoi run's. Overrides as sedov3d_run.py takes them (field=value, e.g. ORDER=1
riemann_solver=hllc enable_x64=1).
"""
import sys
from dataclasses import fields, replace

import numpy as np

sys.path.insert(0, ".")
from scripts.validation.runs.sedov3d_run import config  # noqa: E402

from luna.ic_generation.sedov3d import sedov_blast_3d  # noqa: E402
from luna.setup.cli import launch  # noqa: E402


def main():
    nx, bulk, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    types = {f.name: f.type for f in fields(config(nx))}
    over = {}
    for arg in sys.argv[4:]:
        key, value = arg.split("=", 1)
        kind = types[key]
        over[key] = (value not in ("0", "False", "false")) if kind in (bool, "bool") else \
            int(value) if kind in (int, "int") else float(value) if kind in (float, "float") else value
    cfg = replace(config(nx), BOUNDARY=0, output_dir=out, **over)
    U = np.array(sedov_blast_3d(cfg), dtype=np.float64)
    rho = U[0]
    U[4] += 0.5 * rho * bulk * bulk + U[1] * bulk     # E of the gas at rest plus the bulk motion
    U[1] += rho * bulk
    launch(cfg, U0=U)


if __name__ == "__main__":
    main()

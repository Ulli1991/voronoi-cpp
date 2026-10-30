"""Sedov 3D: Voronoi snapshots (hydro_sedov) and LUNA snapshots against the Sedov-Taylor solution.

    python scripts/hydro_sedov_compare.py <out.png> <label>=<run dir> [<label>=<run dir> ...]

A run dir holds snapdir_NNN/ (one .hdf5 each: ours, or a one-GPU LUNA run).
Needs numpy, h5py, matplotlib and LUNA (luna.hydro.exact_sedov for the analytic
solution): run with LUNA's environment. Per run and snapshot it prints the
shock radius (peak of the radial density profile, as LUNA's
scripts/validation/plots/sedov3d_plot.py:shock_radius_from_snap, with its
binning: N/2 bins from 0 to the farthest cell), the analytic radius, the
largest cell density, the radius beyond which every cell is within 1e-6 of the
background, and at the last snapshot the volume-weighted L1 errors of density,
radial velocity and pressure against the analytic solution over r < 1.5 R_s
(not a LUNA metric: LUNA's l1_error is a per-cell mean on its grid), the same
for density over r < R_s / 2 and the smallest cell density. Plots the
final radial profiles and R_s(t).
"""
import glob
import os
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from luna.hydro import exact_sedov  # noqa: E402

# sedov3d_run.py:19-26, ic_generation/sedov3d.py:9-14
KPC, MSUN, YEAR = 3.08568e21, 1.98892e33, 3.15576e7
UNIT_L, UNIT_M, UNIT_V = KPC, 1e10 * MSUN, 1e5
UNIT_T = UNIT_L / UNIT_V
UNIT_RHO = UNIT_M / UNIT_L**3
UNIT_E = UNIT_M * UNIT_V**2
RHO0 = 1.6726219e-24 / UNIT_RHO
P0 = 1.380649e-16 * 1e4 / (UNIT_RHO * UNIT_V**2)
E0 = 1e51 / UNIT_E
GAMMA = 5.0 / 3.0


def load(path):
    """Cells of one snapshot file: positions relative to the box centre, volume, rho, v, P, time, box."""
    with h5py.File(path, "r") as f:
        c = f["cells"]
        a = dict(c.attrs)
        rho = c["density"][...]
        vel = np.stack([c["velocity_x"][...], c["velocity_y"][...], c["velocity_z"][...]])
        P = c["pressure"][...]
        L = float(a["Lx"])
        if "x" in c:  # Voronoi: generators and volumes
            pos = np.stack([c["x"][...], c["y"][...], c["z"][...]])
            vol = c["volume"][...]
        else:  # LUNA grid, one shard
            n = rho.shape[0]
            dx = L / n
            centres = (np.arange(n) + 0.5) * dx
            X, Y, Z = np.meshgrid(centres, centres, centres, indexing="ij")
            pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()])
            vol = np.full(n**3, dx**3)
            rho, P = rho.ravel(), P.ravel()
            vel = vel.reshape(3, -1)
    d = pos - 0.5 * L
    d -= L * np.round(d / L)
    r = np.sqrt((d**2).sum(axis=0))
    vr = (vel * d).sum(axis=0) / np.where(r > 0, r, 1)
    return dict(time=float(a["time"]), L=L, r=r, vol=vol, rho=rho, vr=vr, P=P, n=rho.size)


def profile(s, field, nbins=None):
    """Mean of a field in N/2 radial bins from 0 to the farthest cell (LUNA metrics.radial_profile)."""
    nbins = nbins or int(round(s["n"] ** (1 / 3))) // 2
    edges = np.linspace(0.0, s["r"].max(), nbins + 1)
    total, _ = np.histogram(s["r"], bins=edges, weights=field)
    count, _ = np.histogram(s["r"], bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), total / np.maximum(count, 1)


def analyse(label, root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "snapdir_*"))):
        s = load(sorted(glob.glob(os.path.join(d, "*.hdf5")))[0])
        if s["time"] <= 0:
            continue
        rb, rho_b = profile(s, s["rho"])
        R = rb[int(np.argmax(rho_b))]
        Ra = exact_sedov.solution(np.array([0.0]), s["time"], E0, RHO0, GAMMA)[3]
        quiet = (np.abs(s["rho"] / RHO0 - 1) > 1e-6) | (np.abs(s["P"] / P0 - 1) > 1e-6)
        rows.append((s, R, Ra, s["rho"].max(), s["r"][quiet].max()))
        print(f"{label}: t={s['time'] * UNIT_T / YEAR:8.1f} yr  R_s={R * 1e3:6.3f} pc  analytic={Ra * 1e3:6.3f} pc  "
              f"max_rho/rho0={s['rho'].max() / RHO0:6.3f}  disturbed_r={rows[-1][4] * 1e3:6.3f} pc  "
              f"half_box={s['L'] * 500:5.1f} pc")
    s, R, Ra = rows[-1][0], rows[-1][1], rows[-1][2]
    rho_a, vr_a, P_a, _ = exact_sedov.solution(s["r"], s["time"], E0, RHO0, GAMMA)
    inside = s["r"] < 1.5 * Ra
    w = s["vol"][inside] / s["vol"][inside].sum()
    l1 = {
        "rho": (w * np.abs(s["rho"] - rho_a)[inside]).sum() / RHO0,
        "v_r": (w * np.abs(s["vr"] - vr_a)[inside]).sum() / vr_a.max(),
        "P": (w * np.abs(s["P"] - P_a)[inside]).sum() / P_a.max(),
    }
    centre = s["r"] < 0.5 * Ra
    wc = s["vol"][centre] / s["vol"][centre].sum()
    l1_centre = (wc * np.abs(s["rho"] - rho_a)[centre]).sum() / RHO0
    print(f"{label}: final R_s/analytic={R / Ra:.4f}  peak profile rho/rho0={profile(s, s['rho'])[1].max() / RHO0:.3f}  "
          f"L1(rho)/rho0={l1['rho']:.4f}  L1(v_r)/v2={l1['v_r']:.4f}  L1(P)/p2={l1['P']:.4f}  "
          f"min rho/rho0={s['rho'].min() / RHO0:.3e}  L1(rho, r<R_s/2)/rho0={l1_centre:.4f}")
    return rows


def main():
    out = sys.argv[1]
    runs = [a.split("=", 1) for a in sys.argv[2:]]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    results = {label: analyse(label, root) for label, root in runs}
    first = next(iter(results.values()))[-1][0]
    r = np.linspace(1e-6, first["r"].max(), 2000)
    rho_a, vr_a, P_a, Ra = exact_sedov.solution(r, first["time"], E0, RHO0, GAMMA)
    for k, (name, exact) in enumerate((("rho", rho_a), ("vr", vr_a), ("P", P_a))):
        ax[k].plot(r * 1e3, exact, "k-", lw=1, label="Sedov-Taylor")
        for label, rows in results.items():
            s = rows[-1][0]
            rb, fb = profile(s, s[name])
            ax[k].plot(rb * 1e3, fb, lw=1.2, label=label)
        ax[k].set_xlabel("r [pc]")
        ax[k].set_ylabel({"rho": "density", "vr": "v_r [km/s]", "P": "pressure"}[name] + " (code units)")
        ax[k].set_xlim(0, 25)
    ax[0].legend(fontsize=8)
    for label, rows in results.items():
        ax[3].plot([row[0]["time"] * UNIT_T / YEAR for row in rows], [row[1] * 1e3 for row in rows], "o-", ms=3, label=label)
    t = np.linspace(first["time"] / 50, first["time"], 200)
    ax[3].plot(t * UNIT_T / YEAR, [exact_sedov.solution(np.array([0.0]), x, E0, RHO0, GAMMA)[3] * 1e3 for x in t], "k-",
               lw=1, label="analytic")
    ax[3].set_xlabel("t [yr]")
    ax[3].set_ylabel("R_s [pc]")
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()

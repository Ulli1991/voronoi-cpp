"""L1 errors of the 1D tests, ours (hydro_1d text files) and LUNA's (luna_1d_run.py npz), and plots.

    python compare_1d.py <runs dir> <reports dir> [margin=0.05]

Run with LUNA's environment (it imports luna for the exact Riemann solver and
the reference files, luna.reference_data.reference_path, which checks each
file's sha256 against luna-reference-solutions/MANIFEST.json).

Metrics, LUNA's:
- Sod: L1 = sum_i V_i |q_i - exact cell average over cell i| for rho, u, p
  (on N uniform cells LUNA's mean, tests/test_hydro_accuracy_shocktube.py,
  shocktube.py:138-139), the exact averages by LUNA's exact_riemann.cell_average.
- Shu-Osher, Woodward-Colella: our cells' mass, momentum and energy are
  deposited conservatively onto N uniform cells (the identity on a uniform
  line); rho against AthenaK PPMX 16384 (external; block mean onto N, LUNA's
  shu_osher.py:72-74: h sum |e|, woodward_colella.py:69-73: mean |e|), and
  rho, u, p against LUNA's own MC 65536 reference (regression only, its
  README: "not external"), block-averaged in conserved variables onto N.
PASS: each of ours (rho, u, p) <= LUNA's at the same N, solver and order
times (1 + margin).
"""
import glob
import json
import os
import sys

import numpy as np

from luna.hydro import exact_riemann as er
from luna.reference_data import reference_path

G = 1.4
DOMAIN = {"sod": (0.0, 1.0), "shu-osher": (-5.0, 5.0), "woodward-colella": (0.0, 1.0)}
ATHENAK = {"shu-osher": "athenak/shu_osher/so_ppmx_rk4_16384.npz",
           "woodward-colella": "athenak/woodward_colella/a9_ppmx_rk4_16384.npz"}
LUNA_REF = {"shu-osher": "luna/shu_osher/shu_osher_mc.npz", "woodward-colella": "luna/woodward_colella/woodward_colella_mc.npz"}
VARIANTS = ["static", "moving", "moving-arepo", "jittered"]


def conserved(rho, u, p):
    return np.stack([rho, rho * u, p / (G - 1) + 0.5 * rho * u * u])


def primitive(U):
    rho = U[0]
    u = U[1] / rho
    return rho, u, (G - 1) * (U[2] - 0.5 * rho * u * u)


def block(U, n):
    return U.reshape(3, n, -1).mean(axis=2)


def deposit(x, V, U, a, b, n):
    """Conservative deposit of extensive U * V on cells [x - V/2, x + V/2] onto n uniform cells of [a, b]."""
    edges_in = a + np.concatenate([[0.0], np.cumsum(V)])
    edges = np.linspace(a, b, n + 1)
    # Cumulative integrals at the uniform edges, by linear interpolation of the cumulative sums.
    out = np.empty((3, n))
    for v in range(3):
        cumulative = np.concatenate([[0.0], np.cumsum(U[v] * V)])
        at = np.interp(edges, edges_in, cumulative)
        out[v] = np.diff(at) / np.diff(edges)
    return out


def ours(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1], conserved(d[:, 2], d[:, 3], d[:, 4])


def errors(problem, n, x, V, U, refs):
    """(L1 rho, u, p) against the problem's reference; x, V None for LUNA's uniform cells."""
    a, b = DOMAIN[problem]
    if problem == "sod":
        WL, WR, x0, t = er.TORO[1]
        widths = V if V is not None else np.full(n, (b - a) / n)
        edges = a + np.concatenate([[0.0], np.cumsum(widths)])
        exact = er.cell_average(WL, WR, G, edges, x0, t)
        q = primitive(U)
        return [float(np.sum(widths * np.abs(q[k] - exact[k])) / (b - a)) for k in range(3)]
    if V is not None:
        U = deposit(x, V, U, a, b, n)
    rho, u, p = primitive(U)
    athenak, luna_ref = refs
    e_rho = rho - athenak.reshape(n, -1).mean(1)
    h = (b - a) / n
    scale = h if problem == "shu-osher" else 1.0 / n  # LUNA: h-weighted sum (SO), mean (WC)
    r_rho, r_u, r_p = primitive(block(conserved(*luna_ref), n))
    return [float(scale * np.abs(e_rho).sum()), float(scale * np.abs(u - r_u).sum()), float(scale * np.abs(p - r_p).sum()),
            float(scale * np.abs(rho - r_rho).sum())]


def main():
    runs, reports = sys.argv[1], sys.argv[2]
    margin = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    os.makedirs(reports, exist_ok=True)
    refs = {}
    for p in ATHENAK:
        ak = np.load(reference_path(ATHENAK[p]))["ref_rho"]
        lr = np.load(reference_path(LUNA_REF[p]))
        refs[p] = (ak, (lr["rho"], lr["u"], lr["p"]))
    rows = []
    for path in sorted(glob.glob(f"{runs}/luna_*_*_hll*.npz")):
        name = os.path.basename(path)[5:-4]
        if "steps" in name:
            continue
        problem, n, solver = name.rsplit("_", 2)
        n = int(n)
        d = np.load(path)
        luna = errors(problem, n, None, None, conserved(d["rho"], d["u"], d["p"]), refs.get(problem))
        luna_rate = n * int(d["steps"]) / float(d["wall_s"])
        for variant in VARIANTS:
            f = f"{runs}/ours_{problem}_{n}_{variant}_{solver}.txt"
            if not os.path.exists(f):
                continue
            x, V, U = ours(f)
            mine = errors(problem, n, x - DOMAIN[problem][0] - 0.5 * V + DOMAIN[problem][0], V, U, refs.get(problem))
            ok = all(m <= l * (1 + margin) for m, l in zip(mine[:3], luna[:3]))
            rows.append(dict(problem=problem, N=n, solver=solver, variant=variant, ours=mine, luna=luna,
                             ratio=[m / l if l > 0 else float("nan") for m, l in zip(mine, luna)], passed=bool(ok),
                             luna_steps=int(d["steps"]), luna_rate=luna_rate))
    with open(f"{reports}/dim1d_errors.json", "w") as out:
        json.dump(rows, out, indent=1)
    print(f"| problem | N | solver | ours | L1 rho | L1 u | L1 p | LUNA rho | LUNA u | LUNA p | ratio rho/u/p | PASS (margin {margin:.0%}) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        o, l = r["ours"], r["luna"]
        print(f"| {r['problem']} | {r['N']} | {r['solver']} | {r['variant']} | {o[0]:.4g} | {o[1]:.4g} | {o[2]:.4g} | "
              f"{l[0]:.4g} | {l[1]:.4g} | {l[2]:.4g} | {r['ratio'][0]:.3f}/{r['ratio'][1]:.3f}/{r['ratio'][2]:.3f} | "
              f"{'PASS' if r['passed'] else 'FAIL'} |")
    plots(runs, reports, rows)


def plots(runs, reports, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    colour = {"static": "#2a78d6", "moving": "#eb6834", "luna": "#1baf7a"}
    shown = {"sod": 256, "shu-osher": 512, "woodward-colella": 512}
    for problem, n in shown.items():
        fig, (ax, conv) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw=dict(width_ratios=[1.6, 1]))
        a, b = DOMAIN[problem]
        if problem == "sod":
            WL, WR, x0, t = er.TORO[1]
            edges = np.linspace(a, b, 4097)
            ref = er.cell_average(WL, WR, G, edges, x0, t)[0]
            xr = 0.5 * (edges[1:] + edges[:-1])
            label = "exact"
        else:
            ak = np.load(reference_path(ATHENAK[problem]))["ref_rho"]
            xr = a + (np.arange(ak.size) + 0.5) * (b - a) / ak.size
            ref, label = ak, "AthenaK PPMX 16384"
        ax.plot(xr, ref, color=ink, lw=1.0, label=label)
        luna = f"{runs}/luna_{problem}_{n}_hllc.npz"
        if os.path.exists(luna):
            d = np.load(luna)
            ax.plot(a + (np.arange(n) + 0.5) * (b - a) / n, d["rho"], "o", ms=3, mfc="none", color=colour["luna"],
                    label=f"LUNA MC, N={n}")
        for variant in ("static", "moving"):
            f = f"{runs}/ours_{problem}_{n}_{variant}_hllc.txt"
            if os.path.exists(f):
                dd = np.loadtxt(f)
                ax.plot(dd[:, 0], dd[:, 2], "-", lw=1.5 if variant == "static" else 2.0, color=colour[variant],
                        label=f"ours {variant}, N={n}", alpha=0.9)
        ax.set_xlabel("x", color=muted)
        ax.set_ylabel("density", color=muted)
        ax.set_title(f"{problem}: density at t_end, HLLC", color=ink, loc="left")
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
            conv.spines[side].set_visible(False)
        for variant, key in (("static", "static"), ("moving", "moving"), ("luna", None)):
            sel = sorted((r for r in rows if r["problem"] == problem and r["solver"] == "hllc" and r["variant"] == "static"),
                         key=lambda r: r["N"])
            if key:
                sel = sorted((r for r in rows if r["problem"] == problem and r["solver"] == "hllc" and r["variant"] == key),
                             key=lambda r: r["N"])
            if not sel:
                continue
            ns = [r["N"] for r in sel]
            err = [r["ours"][0] if key else r["luna"][0] for r in sel]
            conv.loglog(ns, err, "o-" if key else "s--", ms=8 if not key else 5, mfc="none" if not key else None,
                        lw=2, color=colour[variant], label=f"ours {variant}" if key else "LUNA MC")
        conv.set_xlabel("N", color=muted)
        conv.set_ylabel("L1 density error", color=muted)
        conv.set_title("convergence", color=ink, loc="left")
        conv.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{reports}/dim1d_{problem}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()

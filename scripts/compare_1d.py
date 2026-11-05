"""Field L1 of the 1D tests, ours (hydro_1d .npy) and LUNA's (luna_1d_run.py .npz), and GPU time to a target error.

    python compare_1d.py <runs dir> <out dir> [reference dir]
    python compare_1d.py ref <problem> <fine luna .npz> <reference dir>     # make ref_<problem>.npz, check it

Run with LUNA's environment. The norm is the one of research_notes/stage1_1d_2026-11-04.md: the integral of
|f_h - f_ref| dx / L of the code's reconstructed field, for rho, v and P:

- f_h is what the code reconstructs at t_end: ours, W_i + g_i (x - c_i) on [x_{i-1/2}, x_{i+1/2}] with the least-squares
  slope and AREPO's face limiter (finite_volume_1d.hpp, redone here on the output); LUNA's, W_i + s_i (x - x_i)
  with its MC slope (slopes.py:22-24; moncen). Both with the ghosts of their boundaries. "mean" is the piecewise
  constant field of the same run, the second line.
- f_ref: Sod, the exact solution (luna.hydro.exact_riemann.sample, the integral split at its waves); Shu-Osher and
  Woodward-Colella, the finest run of <reference dir> (ref_<problem>.npz, keys edges, rho, u, p: piecewise constant on
  its cells, the reference is ~8x finer than anything compared to it).
- The integral is by Gauss-Legendre on the union of the cell edges of both fields (and the waves), so nothing is
  sampled at generators or at a common grid.

T(eps) is the GPU-seconds to the target error, log-log interpolated between two of a code's own runs bracketing eps
(no extrapolation; "-" if the runs do not bracket it). W(eps) the same for cell updates.
"""
import glob
import json
import os
import sys

import numpy as np

G = 1.4
DOMAIN = {"sod": (0.0, 1.0), "shu-osher": (-5.0, 5.0), "woodward-colella": (0.0, 1.0)}
BOUNDARY = {"sod": "outflow", "shu-osher": "outflow", "woodward-colella": "reflective"}
GL_X, GL_W = np.polynomial.legendre.leggauss(4)
SUB = 4
VARS = ("rho", "v", "P")
# Shu-Osher, moving: the region all generators still cover at t_end; to its left is the inflowing gas of the grown end cell.
REGION = {"shu-osher": (0.0, 5.0)}
# Two targets per test and variable: one every series reaches if it can, one near the end of the ladders.
TARGETS = {"sod": {"rho": (3e-5, 1.2e-5), "v": (3e-5, 1e-5), "P": (1.3e-5, 5e-6)},
           "shu-osher": {"rho": (1e-3, 1e-4), "v": (3e-4, 3e-5), "P": (1e-3, 1e-4)},
           "woodward-colella": {"rho": (5e-3, 1e-3), "v": (2e-3, 1e-3), "P": (5e-2, 2.5e-2)}}


def ghosts(W, boundary):
    """The two ghost rows' values next to W (3, n): left, right, as LUNA's halo (halo_local.py)."""
    left, right = W[:, 0].copy(), W[:, -1].copy()
    if boundary == "reflective":
        left[1], right[1] = -left[1], -right[1]
    return left, right


def mc_slopes(W, h, boundary):
    """LUNA's limited slopes per unit length: moncen (slopes.py:22-24) on the uniform line with its ghosts."""
    l, r = ghosts(W, boundary)
    pad = np.concatenate([l[:, None], W, r[:, None]], axis=1)
    dl, dr = pad[:, 1:-1] - pad[:, :-2], pad[:, 2:] - pad[:, 1:-1]
    a, b, c = 2 * dl, 2 * dr, 0.5 * (dl + dr)
    same = (a * b > 0) & (a * c > 0)
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c))), 0.0) / h


def face_limited_slopes(gen, cen, edges, W, boundary):
    """Our gradients: the fit through the two neighbours, AREPO's limit_gradient at the face centroids, left then right."""
    l, r = ghosts(W, boundary)
    a, b = edges[0], edges[-1]
    c_pad = np.concatenate([[2 * a - cen[0]], cen, [2 * b - cen[-1]]])
    pad = np.concatenate([l[:, None], W, r[:, None]], axis=1)
    dL, dR = c_pad[:-2] - cen, c_pad[2:] - cen
    g = (pad[:, 2:] - pad[:, :-2]) / (dR - dL)
    lo, hi = np.minimum(pad[:, :-2], pad[:, 2:]), np.maximum(pad[:, :-2], pad[:, 2:])
    for r_face in (edges[:-1] - cen, edges[1:] - cen):
        dp = g * r_face
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(hi > W, (hi - W) / dp, 0.0)
            down = np.where(lo < W, (lo - W) / dp, 0.0)
        g = np.where((dp > 0) & (W + dp > hi), g * up, np.where((dp < 0) & (W + dp < lo), g * down, g))
    return g


class Field:
    """Piecewise linear (reconstructed) or constant (mean) on cells [edges[i], edges[i + 1]]."""

    def __init__(self, edges, W, slopes=None, centres=None):
        self.edges, self.W, self.slopes, self.centres = edges, W, slopes, centres

    def __call__(self, x):
        i = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, self.W.shape[1] - 1)
        if self.slopes is None:
            return self.W[:, i]
        return self.W[:, i] + self.slopes[:, i] * (x - self.centres[i])


def ours(path, problem):
    """(reconstructed, mean) fields of a hydro_1d output."""
    d = np.load(path)
    gen, cen, V = d[:, 0], d[:, 1], d[:, 2]
    W = d[:, 3:6].T
    a = DOMAIN[problem][0]
    edges = np.concatenate([[a], 0.5 * (gen[1:] + gen[:-1]), [DOMAIN[problem][1]]])
    g = face_limited_slopes(gen, cen, edges, W, BOUNDARY[problem])
    return Field(edges, W, g, cen), Field(edges, W)


def luna(path, problem):
    d = np.load(path)
    W = np.stack([d["rho"], d["u"], d["p"]])
    n = W.shape[1]
    a, b = DOMAIN[problem]
    edges = np.linspace(a, b, n + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    return Field(edges, W, mc_slopes(W, (b - a) / n, BOUNDARY[problem]), cen), Field(edges, W)


class Exact:
    """Sod at t = 0.25, from LUNA's exact solver."""

    def __init__(self):
        from luna.hydro import exact_riemann as er
        self.er = er
        self.WL, self.WR, self.x0, self.t = er.TORO[1]
        w = er.waves(self.WL, self.WR, G)
        self.breaks = self.x0 + self.t * np.array([*w["left"], w["u"], *w["right"]])

    def __call__(self, x):
        out = self.er.sample(self.WL, self.WR, G, (x - self.x0) / self.t)
        return np.stack(out[:3])


class Reference:
    """A finest run, piecewise constant on its cells."""

    def __init__(self, path):
        d = np.load(path)
        self.field = Field(d["edges"], np.stack([d["rho"], d["u"], d["p"]]))
        self.breaks = d["edges"]

    def __call__(self, x):
        return self.field(x)


def l1(field, ref, problem, region=None):
    """Integral of |f_h - f_ref| dx / L per variable, Gauss-Legendre on the union of the fields' breakpoints."""
    a, b = DOMAIN[problem]
    lo, hi = region if region else (a, b)
    points = np.unique(np.concatenate([[lo, hi], np.clip(field.edges, lo, hi), np.clip(ref.breaks, lo, hi)]))
    width = np.diff(points)
    keep = width > 0
    left, width = points[:-1][keep], width[keep]
    left = (left[:, None] + width[:, None] * np.arange(SUB) / SUB).ravel()  # |f_h - f_ref| has kinks inside a segment
    width = np.repeat(width / SUB, SUB)
    total = np.zeros(3)
    for start in range(0, left.size, 1 << 18):
        lo, w = left[start:start + (1 << 18)], width[start:start + (1 << 18)]
        x = lo[None, :] + 0.5 * w[None, :] * (GL_X[:, None] + 1)
        shape = x.shape
        f = field(x.ravel()) - ref(x.ravel())
        total += (np.abs(f).reshape(3, *shape) * GL_W[None, :, None] * 0.5 * w[None, None, :]).sum(axis=(1, 2))
    return total / (b - a)


def make_reference(problem, path, out):
    """ref_<problem>.npz from a fine LUNA run, with its distance to the stored AthenaK and LUNA references."""
    d = np.load(path)
    a, b = DOMAIN[problem]
    n = d["rho"].size
    os.makedirs(out, exist_ok=True)
    np.savez(f"{out}/ref_{problem}.npz", edges=np.linspace(a, b, n + 1), rho=d["rho"], u=d["u"], p=d["p"], N=n, t=d["t"])
    root = os.environ.get("LUNA_REFERENCE_SOLUTIONS", "/raven/u/uli/luna-reference-solutions")
    names = {"shu-osher": ("athenak/shu_osher/so_ppmx_rk4_16384.npz", "luna/shu_osher/shu_osher_mc.npz"),
             "woodward-colella": ("athenak/woodward_colella/a9_ppmx_rk4_16384.npz", "luna/woodward_colella/woodward_colella_mc.npz")}
    ak = np.load(f"{root}/{names[problem][0]}")["ref_rho"]
    stored = np.load(f"{root}/{names[problem][1]}")["rho"]
    for name, rho in (("AthenaK PPMX 16384", ak), ("LUNA MC 65536 on 8192", stored)):
        k = n // rho.size
        print(f"{problem}: N={n} t={float(d['t'])} rho block-averaged onto {rho.size}: L1 distance to {name} "
              f"{np.abs(d['rho'].reshape(rho.size, k).mean(1) - rho).mean():.3e}")


def interp(ns, errors, costs, eps):
    """log-log interpolation of cost at error eps between two runs bracketing it, in order of N."""
    for i in range(len(ns) - 1):
        e0, e1 = errors[i], errors[i + 1]
        if (e0 - eps) * (e1 - eps) <= 0 and e0 != e1:
            f = np.log(eps / e0) / np.log(e1 / e0)
            return float(np.exp(np.log(costs[i]) + f * (np.log(costs[i + 1]) - np.log(costs[i]))))
    return None


def main():
    if sys.argv[1] == "ref":
        make_reference(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    runs, reports = sys.argv[1], sys.argv[2]
    refs_dir = sys.argv[3] if len(sys.argv) > 3 else runs
    os.makedirs(reports, exist_ok=True)
    rows = []
    timing = {}
    for path in sorted(glob.glob(f"{runs}/*.json")):  # one JSON per run: series, problem, N, steps, wall_s, peak_bytes, ...
        meta = json.load(open(path))
        if os.path.basename(path).startswith("luna_rate_"):  # a throughput run, stopped after some steps
            continue
        timing[(meta["series"], meta["problem"], meta["N"])] = meta
    done = {}
    if os.path.exists(f"{reports}/dim1d_errors.json"):  # the integrals are slow: keep those of runs already compared
        done = {(r["series"], r["problem"], r["N"]): r for r in json.load(open(f"{reports}/dim1d_errors.json"))}
    refs = {"sod": Exact()}
    for p in ("shu-osher", "woodward-colella"):
        f = f"{refs_dir}/ref_{p}.npz"
        if os.path.exists(f):
            refs[p] = Reference(f)
    for (series, problem, n), meta in sorted(timing.items()):
        if problem not in refs:
            continue
        if (series, problem, n) in done:
            rows.append({**done[(series, problem, n)], **meta})  # the same bits, new timing
            continue
        path = os.path.join(runs, meta["file"])
        rec, mean = (luna if series.startswith("luna") else ours)(path, problem)
        row = dict(meta)
        row["L1"] = l1(rec, refs[problem], problem).tolist()
        row["L1_mean"] = l1(mean, refs[problem], problem).tolist()
        if problem in REGION:
            row["L1_region"] = l1(rec, refs[problem], problem, REGION[problem]).tolist()
        rows.append(row)
    with open(f"{reports}/dim1d_errors.json", "w") as out:
        json.dump(rows, out, indent=1)
    print("| test | series | N | steps | cell updates | T s | M updates/s | peak MB | L1 rho | L1 v | L1 P | mean rho | mean v | mean P |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['problem']} | {r['series']} | {r['N']} | {r['steps']} | {r['N'] * r['steps']:.3g} | {r['wall_s']:.3g} | "
              f"{r['N'] * r['steps'] / r['wall_s'] / 1e6:.0f} | {r.get('peak_MB', float('nan')):.3g} | "
              + " | ".join(f"{e:.3e}" for e in r["L1"] + r["L1_mean"]) + " |")
    print("\n## Against LUNA at the same N (ratio of field L1, reconstructed; <= 1: as good)\n")
    print("| test | series | N | rho | v | P | rho mean | v mean | P mean |")
    print("|---|---|---|---|---|---|---|---|---|")
    index = {(r["series"], r["problem"], r["N"]): r for r in rows}
    for r in rows:
        if r["series"].startswith("luna"):
            continue
        l = index.get(("luna", r["problem"], r["N"]))
        if l:
            ratios = [a / b for a, b in zip(r["L1"] + r["L1_mean"], l["L1"] + l["L1_mean"])]
            print(f"| {r['problem']} | {r['series']} | {r['N']} | " + " | ".join(f"{x:.3f}" for x in ratios) + " |")
    print("\n## Region of REGION only (Shu-Osher x > 0, the gas the moving cells cover), ratio to LUNA\n")
    for r in rows:
        l = index.get(("luna", r["problem"], r["N"]))
        if l and "L1_region" in r and not r["series"].startswith("luna"):
            ratios = [a / b for a, b in zip(r["L1_region"], l["L1_region"])]
            print(f"| {r['problem']} | {r['series']} | {r['N']} | " + " | ".join(f"{x:.3f}" for x in ratios) + " |")
    print("\n## Largest difference ours static - LUNA at the same N (uniform lattice: the same scheme)\n")
    for r in rows:
        if r["series"] != "static" or ("luna", r["problem"], r["N"]) not in index:
            continue
        a = np.load(os.path.join(runs, r["file"]))[:, 3:6].T
        d = np.load(os.path.join(runs, index[("luna", r["problem"], r["N"])]["file"]))
        print(f"| {r['problem']} | {r['N']} | {max(np.abs(a[k] - d[n]).max() for k, n in enumerate(('rho', 'u', 'p'))):.2e} |")
    print("\n## GPU-seconds and cell updates to the target error (interpolated between a code's own runs)\n")
    print("| test | variable | eps | series | T(eps) s | W(eps) cell updates |")
    print("|---|---|---|---|---|---|")
    for problem, targets in TARGETS.items():
        for k, eps in ((k, e) for k, name in enumerate(VARS) for e in targets[name]):
            name = VARS[k]
            for series in sorted({r["series"] for r in rows if r["problem"] == problem}):
                sel = sorted((r for r in rows if r["problem"] == problem and r["series"] == series), key=lambda r: r["N"])
                ns = [r["N"] for r in sel]
                errors = [r["L1"][k] for r in sel]
                t = interp(ns, errors, [r["wall_s"] for r in sel], eps)
                w = interp(ns, errors, [float(r["N"] * r["steps"]) for r in sel], eps)
                print(f"| {problem} | {name} | {eps:g} | {series} | {'-' if t is None else f'{t:.3g}'} | "
                      f"{'-' if w is None else f'{w:.3g}'} |")
    scores(rows)


def scores(rows):
    """The score of research_notes/accuracy_score_exploration_2026-11-04.md, S = 1 / (cost L1^x), x = (d + 1) / p, d = 1.

    Per test, series and variable: the order p fitted over the top of the ladder and the local one at its ends; then
    max / min of W L1^x and T L1^x over N, for p = 1 (x = 2), the card's choice, and p = the fitted order. W = cell updates
    (all N >= 2^10), T = GPU-seconds (N >= 2^16 only: below ~1e5 cells T is latency).
    """
    print("\n## Score flatness: max / min of cost L1^x over N; p fit over N >= 2^12\n")
    print("| test | variable | series | p fit | p first | p last | W, p=1 | W, p fit | T, p=1 | T, p fit | N range W |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    spread = lambda v: float(np.max(v) / np.min(v)) if len(v) > 1 else float("nan")
    for problem in DOMAIN:
        for series in sorted({r["series"] for r in rows if r["problem"] == problem}):
            sel = sorted((r for r in rows if r["problem"] == problem and r["series"] == series and r["N"] >= 1024), key=lambda r: r["N"])
            if len(sel) < 3:
                continue
            n = np.array([float(r["N"]) for r in sel])
            w = np.array([r["N"] * float(r["steps"]) for r in sel])
            t = np.array([r["wall_s"] for r in sel])
            for k, name in enumerate(VARS):
                e = np.array([r["L1"][k] for r in sel])
                top = n >= 4096
                fit = max(-np.polyfit(np.log(n[top]), np.log(e[top]), 1)[0], 0.01)
                local = -np.diff(np.log(e)) / np.diff(np.log(n))
                big = n >= 65536
                tspread = lambda x: spread(t[big] * e[big] ** x) if big.sum() >= 2 else float("nan")
                print(f"| {problem} | {name} | {series} | {fit:.2f} | {local[0]:.2f} | {local[-1]:.2f} | {spread(w * e ** 2):.3g} | "
                      f"{spread(w * e ** (2 / fit)):.3g} | {tspread(2.0):.3g} | {tspread(2 / fit):.3g} | {int(n[0])}-{int(n[-1])} |")


if __name__ == "__main__":
    main()

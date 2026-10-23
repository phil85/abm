"""
ABM cover: animated GIF + static PNG (1280x640, GitHub social-preview size).

Five clusters of points in the plane; ABM's maximum weight matching pairs
every cluster with its two farthest neighbours, so the matched edges bundle
into a pentagram. Every matched pair is oriented from a set A to a set B,
where A and B are each spread evenly over the whole point cloud (as in ABM's
bipartition), so strokes launch from everywhere at once and within every
bundle about half the strokes run each way. Each pair lights up when its edge
connects; then hold, fade, loop.

Usage:  python make_cover.py            -> frames/ + abm_cover.png
        then build the GIF with ffmpeg (see build_gif.sh)
Edit TEXT / PALETTE below to change wording or colours.
"""
import os
import numpy as np
import networkx as nx
from abm import run_abm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------- settings
W, H, DPI = 1280, 640, 100
FPS = 25
BG = "#0d1117"
PALETTE = ["#2dd4bf", "#38bdf8", "#a78bfa", "#f472b6", "#fbbf24"]
K, PER, SPREAD, SEED = 5, 40, 0.15, 3

TITLE = "ABM"
SUBTITLE = "Assignment-Based Matching"
TAGLINE = "Fast, near-optimal maximum weight matching\nfor large Euclidean datasets"
CHIPS = ["Millions of nodes", "Often < 1% from optimum", "High-dimensional data"]

# timeline (seconds)
SIMULTANEOUS = True      # True: all edges grow at once; False: launched in random order
if SIMULTANEOUS:
    T_INTRO, T_HOLD, T_FADE = 0.8, 3.0, 0.9
    EDGE_DUR = 1.4       # time for an edge to grow
    T_DRAW = EDGE_DUR
else:
    T_INTRO, T_DRAW, T_HOLD, T_FADE = 0.6, 3.6, 2.4, 0.9
    EDGE_DUR = 0.55
POP_DUR = 0.35           # endpoint "pop" when an edge connects

OUT_FRAMES = "frames"
OUT_PNG = "abm_cover.png"


# ---------------------------------------------------------------- data
def make_points():
    rng = np.random.default_rng(SEED)
    ang = np.pi / 2 + 2 * np.pi * np.arange(K) / K      # 0 = top, counter-clockwise
    C = np.c_[np.cos(ang), np.sin(ang)]
    pts = []
    for i in range(K):
        Z = rng.normal(size=(PER * 3, 2))
        Z = Z[np.linalg.norm(Z, axis=1) < 2.2][:PER]     # trim far outliers
        pts.append(C[i] + SPREAD * Z)
    P = np.vstack(pts)
    lab = np.repeat(np.arange(K), PER)
    return P, lab


def max_weight_matching(P):
    D = squareform(pdist(P))
    M = run_abm(P, params={'output_as_pairs': True})
    total = sum(D[i, j] for i, j in M)
    return [tuple(e) for e in M], total


# ---------------------------------------------------------------- A / B split
def split_AB(P, M, A=None, bw=None, sweeps=5, seed=SEED):
    """Orient every matched pair as (a, b) with a in set A and b in set B.

    If A is given (e.g. the set ABM used for its assignment step), pairs are
    simply oriented by membership. Otherwise each pair's direction is chosen so
    that A and B are both spread evenly over the point cloud: keep the local
    balance  sum_A k(x, a) - sum_B k(x, b)  close to zero everywhere (Gaussian
    kernel k, bandwidth ~ cluster spread). Greedy pass, then single-flip sweeps.
    """
    if A is not None:
        A = set(A)
        return [(i, j) if i in A else (j, i) for i, j in M]

    bw = SPREAD if bw is None else bw
    Kx = np.exp(-squareform(pdist(P, "sqeuclidean")) / (2 * bw**2))
    rng = np.random.default_rng(seed)
    E = [tuple(e) for e in M]
    bal = np.zeros(len(P))
    for k in rng.permutation(len(E)):
        i, j = E[k]
        g = bal @ (Kx[i] - Kx[j])
        if g > 0 or (g == 0 and rng.random() < 0.5):
            i, j = j, i
        E[k] = (i, j)
        bal += Kx[i] - Kx[j]
    for _ in range(sweeps):
        changed = False
        for k in rng.permutation(len(E)):
            i, j = E[k]
            d = Kx[i] - Kx[j]
            if bal @ d > d @ d:          # flipping (bal -> bal - 2d) lowers ||bal||^2
                E[k] = (j, i)
                bal -= 2 * d
                changed = True
        if not changed:
            break
    return E


# ---------------------------------------------------------------- schedule
def schedule(E, lab):
    """Start time for every stroke; each stroke grows from its A end to its B end.
    SIMULTANEOUS: all at once. Otherwise launched in a random order spread over
    the draw window, so no bundle or cluster goes first."""
    A = np.array([i for i, _ in E])
    share = np.bincount(lab[A], minlength=K) / np.bincount(lab, minlength=K)
    flows = {}
    for i, j in E:
        a, b = lab[i], lab[j]
        flows.setdefault((min(a, b), max(a, b)), [0, 0])[int(a > b)] += 1
    print("share of each cluster in A:", np.round(share, 2))
    print("strokes per bundle (low->high, high->low):",
          {f"{a}-{b}": v for (a, b), v in sorted(flows.items())})

    if SIMULTANEOUS:
        return [(e, T_INTRO) for e in E]
    rng = np.random.default_rng(SEED + 1)
    span = T_DRAW - EDGE_DUR
    order = rng.permutation(len(E))
    jitter = rng.random(len(E))
    return [(E[k], T_INTRO + (r + jitter[r]) / len(E) * span) for r, k in enumerate(order)]


# ---------------------------------------------------------------- drawing
def ease(t):
    t = np.clip(t, 0, 1)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def grad_segments(p, q, ca, cb, t, nseg=26):
    """Gradient polyline from p towards q, truncated at fraction t."""
    m = max(2, int(np.ceil(nseg * t)))
    s = np.linspace(0, t, m + 1)[:, None]
    pts = p + (q - p) * s
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    sm = (s[:-1] + s[1:]) / 2
    cols = np.asarray(ca) * (1 - sm) + np.asarray(cb) * sm
    return segs, cols


def setup_figure():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    # text block (left)
    x0 = 76 / W
    fig.text(x0 - 0.004, 0.60, TITLE, fontfamily="Inter Display", fontweight="bold",
             fontsize=104, color="#f0f6fc")
    fig.text(x0, 0.515, SUBTITLE, fontfamily="Inter", fontweight="medium",
             fontsize=27, color="#e6edf3")
    fig.text(x0, 0.375, TAGLINE, fontfamily="Inter", fontsize=16.5, color="#8b949e",
             linespacing=1.45)
    # chips
    renderer = fig.canvas.get_renderer()
    x = x0
    for c in CHIPS:
        t = fig.text(x, 0.255, c, fontfamily="Inter", fontweight="medium", fontsize=11.5,
                     color="#c9d1d9", va="center",
                     bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.9",
                               fc="#161b22", ec="#30363d", lw=1))
        bb = t.get_window_extent(renderer=renderer)
        x += (bb.width + 30) / W
    # plot area (right), square in pixels
    side = 640
    cx = 912
    ax = fig.add_axes([(cx - side / 2) / W, 0, side / W, 1])
    ax.set_facecolor(BG)
    lim = 1.62
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def main():
    P, lab = make_points()
    M, total = max_weight_matching(P)
    print(f"{len(P)} points, {len(M)} matched pairs, total weight {total:.2f}")
    E = split_AB(P, M)
    seq = schedule(E, lab)
    cols = [to_rgb(PALETTE[l]) for l in lab]

    fig, ax = setup_figure()
    lc = LineCollection([], linewidths=1.25, capstyle="round")
    ax.add_collection(lc)
    pts = ax.scatter(*P.T, s=16, c=cols, zorder=3, edgecolors=BG, linewidths=0.6)
    heads = ax.scatter([], [], s=10, c="white", zorder=4, linewidths=0)

    T_END = T_INTRO + T_DRAW + T_HOLD + T_FADE
    n_frames = int(round(T_END * FPS))
    os.makedirs(OUT_FRAMES, exist_ok=True)
    done_at = {}
    for (i, j), s in seq:
        done_at[i] = done_at[j] = s + EDGE_DUR

    def render(t, still=False):
        fade = 1.0
        if t > T_INTRO + T_DRAW + T_HOLD:
            fade = 1 - ease((t - T_INTRO - T_DRAW - T_HOLD) / T_FADE)
        segs, segc, hx = [], [], []
        for (i, j), s in seq:
            u = 1.0 if still else ease((t - s) / EDGE_DUR)
            if u <= 0:
                continue
            sg, c = grad_segments(P[i], P[j], cols[i], cols[j], u)
            segs.append(sg)
            segc.append(c)
            if 0 < u < 1:
                hx.append(P[i] + (P[j] - P[i]) * u)
        if segs:
            S = np.concatenate(segs)
            C = np.concatenate(segc)
            C = np.c_[C, np.full(len(C), 0.78 * fade)]
            lc.set_segments(S)
            lc.set_color(C)
        else:
            lc.set_segments([])
        heads.set_offsets(np.array(hx) if hx else np.empty((0, 2)))
        # points: dim until matched, pop when their edge connects
        size = np.full(len(P), 16.0)
        alpha = np.full(len(P), 0.45)
        for v, d in done_at.items():
            if still:
                alpha[v] = 1.0
                continue
            if t >= d:
                alpha[v] = 0.45 + 0.55 * fade
                k = (t - d) / POP_DUR
                if k < 1:
                    size[v] = 16 + 30 * np.sin(np.pi * k)
        rgba = np.c_[np.array(cols), alpha]
        pts.set_sizes(size)
        pts.set_facecolors(rgba)

    for f in range(n_frames):
        render(f / FPS)
        fig.savefig(f"{OUT_FRAMES}/f{f:04d}.png", facecolor=BG)
    print("frames:", n_frames)

    render(0, still=True)
    fig.savefig(OUT_PNG, facecolor=BG)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()

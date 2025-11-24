#!/usr/bin/env python3
"""
visualize_dp_results_plotly.py

Interactive visualization (Plotly) for:
 - measured_compute_grid.npz
 - measured_comm_profile.npz
 - dp_experiments_summary.csv

Generates interactive HTML files:
 - t_fwd_measured_surface.html
 - t_fwd_model_surface.html
 - comm_profile_plot.html
 - optimal_slicing_L_<L>.html (for each L in DP CSV)
 - slicing_comparison_L_<L>.html
 - Tfstar_vs_K.html
 - t_fwd_heatmap.html

Requires:
  pip install plotly pandas numpy
"""

import os
import numpy as np
import pandas as pd
import math
import json
import sys
import webbrowser
import plotly.graph_objects as go
import plotly.express as px

# Files (change if you used different names)
MEASURED_COMPUTE_NPZ = "measured_compute_grid.npz"
MEASURED_COMM_NPZ = "measured_comm_profile.npz"
DP_CSV = "dp_experiments_summary.csv"
PAPER_PATH = "/mnt/data/tokenlevel.pdf"  # uploaded paper path (developer note)

OUTPUT_DIR = "plots_plotly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_measured_compute(npzfile=MEASURED_COMPUTE_NPZ):
    if not os.path.exists(npzfile):
        print(f"[ERROR] {npzfile} not found.")
        return None
    data = np.load(npzfile, allow_pickle=True)
    keys = data["keys"]    # Nx2 array of (sum_prev, l)
    vals = data["vals"]    # N values
    measured = {}
    for (s, l), v in zip(keys, vals):
        measured[(int(s), int(l))] = float(v)
    return measured

def load_measured_comm(npzfile=MEASURED_COMM_NPZ):
    if not os.path.exists(npzfile):
        print(f"[WARNING] {npzfile} not found.")
        return {}
    data = np.load(npzfile, allow_pickle=True)
    sizes = data["sizes"]
    times = data["times"]
    return {int(s): float(t) for s,t in zip(sizes, times)}

def build_comm_fit_params(measured_comm):
    # Fit linear model time = a + slope * size
    if not measured_comm:
        return None
    sizes = np.array(sorted(measured_comm.keys()), dtype=float)
    times = np.array([measured_comm[s] for s in sizes], dtype=float)
    A = np.vstack([np.ones_like(sizes), sizes]).T
    sol, *_ = np.linalg.lstsq(A, times, rcond=None)
    a, slope = float(sol[0]), float(sol[1])
    return {"a": a, "slope": slope, "sizes": list(sizes), "times": list(times)}

def build_t_fwd_measured_and_model(measured_compute, comm_params, hidden_dim=16):
    # measured_compute: dict {(sum_prev, l): t}
    # comm_params: dict with a and slope
    # returns two callables: t_fwd_measured(l, sum_prev), t_fwd_model(l, sum_prev), and fitted c
    if measured_compute is None:
        return None, None, None

    # Fit analytic c from measured compute only (y ~ c * l*(s+l)*H)
    xs = []
    ys = []
    for (s, l), t in measured_compute.items():
        xs.append(l * (s + l) * hidden_dim)
        ys.append(t)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    c = 0.0
    if xs.size > 0:
        c = np.sum(xs * ys) / (np.sum(xs * xs) + 1e-30)

    def t_fwd_model(l_i, sum_prev):
        return float(c * l_i * (sum_prev + l_i) * hidden_dim)

    def t_fwd_measured(l_i, sum_prev):
        # nearest neighbor lookup on (sum_prev, l)
        key = (int(sum_prev), int(l_i))
        if key in measured_compute:
            comp = measured_compute[key]
        else:
            # find nearest measured pair
            bestk = None
            bestd = float("inf")
            for (s, l), v in measured_compute.items():
                d = (s - sum_prev)**2 + (l - l_i)**2
                if d < bestd:
                    bestd = d
                    bestk = (s, l)
            comp = measured_compute[bestk]
        comm_time = 0.0
        if comm_params is not None:
            msg_bytes = int(hidden_dim * l_i * 4)
            comm_time = comm_params["a"] + comm_params["slope"] * msg_bytes
        return float(comp + comm_time)

    return t_fwd_measured, t_fwd_model, c

def make_surface_plot(t_fwd_func, measured_compute, title, fname, max_sumprev=None, max_l=None, hidden_dim=16):
    # Build grid from measured keys if available; else use defaults
    if measured_compute:
        s_vals = sorted({s for (s,l) in measured_compute.keys()})
        l_vals = sorted({l for (s,l) in measured_compute.keys()})
    else:
        s_vals = list(range(0, 513, 32)) if max_sumprev is None else list(range(0, max_sumprev+1, max(1, max_sumprev//32)))
        l_vals = list(range(1, 129, max(1, (max_l or 128)//32)))

    # Create mesh
    S, L = np.meshgrid(s_vals, l_vals)
    Z = np.zeros_like(S, dtype=float)
    for i, li in enumerate(l_vals):
        for j, sj in enumerate(s_vals):
            Z[i,j] = t_fwd_func(li, sj)

    # Plotly surface
    fig = go.Figure(data=[go.Surface(x=S, y=L, z=Z, colorscale="Viridis")])
    fig.update_layout(title=title, scene=dict(xaxis_title='sum_prev', yaxis_title='l_i', zaxis_title='t_fwd (s)'))
    outpath = os.path.join(OUTPUT_DIR, fname)
    fig.write_html(outpath)
    print("Saved:", outpath)
    return outpath, fig

def make_comm_plot(comm_params, measured_comm, fname):
    if not measured_comm:
        print("No comm data to plot.")
        return None
    sizes = np.array(comm_params["sizes"])
    times = np.array(comm_params["times"])
    a = comm_params["a"]
    slope = comm_params["slope"]
    # predicted line
    xs = np.linspace(0, max(sizes)*1.1, 200)
    ys = a + slope * xs

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=times, mode='markers+lines', name='measured'))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='linear fit'))
    fig.update_layout(title="MPI comm measured vs linear fit", xaxis_title="message size (bytes)", yaxis_title="one-way time (s)")
    out = os.path.join(OUTPUT_DIR, fname)
    fig.write_html(out)
    print("Saved:", out)
    return out, fig

def plot_slicing_bar(slicing, title, fname):
    idx = list(range(1, len(slicing)+1))
    fig = px.bar(x=idx, y=slicing, labels={'x':'slice index', 'y':'slice length (tokens)'}, title=title)
    out = os.path.join(OUTPUT_DIR, fname)
    fig.write_html(out)
    print("Saved:", out)
    return out, fig

def plot_slicing_comparison(slicing_measured, slicing_model, title, fname):
    # Make cumulative x axis to visualize token distribution
    def expand_to_positions(slices):
        pos = []
        cur = 0
        for i, l in enumerate(slices):
            pos.append((cur, cur + l))
            cur += l
        return pos
    pos_m = expand_to_positions(slicing_measured)
    pos_s = expand_to_positions(slicing_model)

    # Build dataframe for Gantt-like chart
    rows = []
    for i,(a,b) in enumerate(pos_m):
        rows.append(dict(slice=i+1, start=a, finish=b, type="measured"))
    for i,(a,b) in enumerate(pos_s):
        rows.append(dict(slice=i+1, start=a, finish=b, type="model"))
    df = pd.DataFrame(rows)
    fig = px.timeline(df, x_start="start", x_end="finish", y="type", color="slice", title=title)
    fig.update_yaxes(autorange="reversed")
    out = os.path.join(OUTPUT_DIR, fname)
    fig.write_html(out)
    print("Saved:", out)
    return out, fig

def plot_dp_path_on_surface(t_fwd_func, slicing, title, fname, measured_compute, hidden_dim=16):
    """
    Plots the measured t_fwd surface and overlays the DP-chosen slicing path.
    - Surface: measured t_fwd(s_i, l_i)
    - Red polyline: DP slice choices (sum_prev_i, l_i, t_fwd)
    """

    # Compute sum_prev sequence for the slicing path
    sum_prev_list = []
    sp = 0
    for l in slicing:
        sum_prev_list.append(sp)
        sp += l

    # Build a combined grid for smoother surface
    s_vals = sorted({s for (s, l) in measured_compute.keys()})
    l_vals = sorted({l for (s, l) in measured_compute.keys()})

    # Build 2D grid for surface
    S, L = np.meshgrid(s_vals, l_vals)
    Z = np.zeros_like(S, dtype=float)

    for i, li in enumerate(l_vals):
        for j, sj in enumerate(s_vals):
            Z[i, j] = t_fwd_func(li, sj)

    # DP path z-values
    dp_z = [t_fwd_func(l_i, sp_i) for l_i, sp_i in zip(slicing, sum_prev_list)]

    # Build plot
    fig = go.Figure()

    # Surface
    fig.add_trace(go.Surface(
        x=S,
        y=L,
        z=Z,
        colorscale="Viridis",
        opacity=0.85,
        name="t_fwd surface"
    ))

    # DP path (in red)
    fig.add_trace(go.Scatter3d(
        x=sum_prev_list,
        y=slicing,
        z=dp_z,
        mode='markers+lines',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=4),
        name="DP optimal path"
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="sum_prev",
            yaxis_title="slice length (l_i)",
            zaxis_title="t_fwd (seconds)"
        )
    )

    outpath = os.path.join(OUTPUT_DIR, fname)
    fig.write_html(outpath)
    print("Saved DP path overlay:", outpath)
    return outpath, fig

def plot_Tstar_vs_K(t_fwd_func, L_values=[512,1024,2048], K_values=[2,4,8,16], fname="Tfstar_vs_K.html"):
    # For each L and K compute T* using DP (expensive for large L; use smaller grid if necessary)
    rows = []
    for L in L_values:
        for K in K_values:
            Tstar, slicing, tmax = find_optimal_slicing_scheme(L, K, t_fwd_func)
            rows.append(dict(L=L, K=K, Tstar=Tstar, n_slices=len(slicing)))
    df = pd.DataFrame(rows)
    fig = px.line(df, x="K", y="Tstar", color=df["L"].astype(str), markers=True,
                  title="T* vs K for different L (using provided t_fwd)")
    out = os.path.join(OUTPUT_DIR, fname)
    fig.write_html(out)
    print("Saved:", out)
    return out, fig

# Reuse DP functions from previous script (kept minimal here)
def find_optimal_slicing_given_tmax(L, t_fwd_func, tmax):
    S_star = [float('inf')] * (L + 1)
    S_star[0] = 0.0
    q = [0] * (L + 1)
    for i in range(1, L + 1):
        best_time = float('inf')
        best_k = 1
        for k in range(1, i + 1):
            sum_prev = i - k
            t_i = t_fwd_func(k, sum_prev)
            if t_i <= tmax:
                cand = S_star[i - k] + t_i
                if cand < best_time:
                    best_time = cand
                    best_k = k
        S_star[i] = best_time
        q[i] = best_k
    slicing = []
    i = L
    while i > 0:
        slicing.insert(0, q[i])
        i -= q[i]
    return S_star[L], slicing

def find_optimal_slicing_scheme(L, K, t_fwd_func):
    possible_tmax = set()
    for i in range(1, L + 1):
        for j in range(0, L + 1):
            if i + j <= L:
                possible_tmax.add(t_fwd_func(i, j))
    possible_tmax = sorted(list(possible_tmax))
    best_T_star = float('inf')
    best_slicing = None
    best_tmax = None
    for tmax in possible_tmax:
        S_star, slicing = find_optimal_slicing_given_tmax(L, t_fwd_func, tmax)
        if slicing and len(slicing) > 0:
            max_t_i = 0.0
            sum_prev = 0
            for l_i in slicing:
                t_i = t_fwd_func(l_i, sum_prev)
                max_t_i = max(max_t_i, t_i)
                sum_prev += l_i
            T_star = S_star + (K - 1) * max_t_i
            if T_star < best_T_star:
                best_T_star = T_star
                best_slicing = slicing
                best_tmax = max_t_i
    if best_slicing is None or len(best_slicing) == 0:
        uniform_size = L // K if K > 0 else L
        if uniform_size == 0:
            uniform_size = 1
        best_slicing = [uniform_size] * (L // uniform_size)
        remainder = L % uniform_size
        if remainder > 0:
            best_slicing.append(remainder)
    return best_T_star, best_slicing, best_tmax

# ------------------------
# Main flow
# ------------------------
def main():
    print("Loading measured compute grid:", MEASURED_COMPUTE_NPZ)
    measured_compute = load_measured_compute(MEASURED_COMPUTE_NPZ)
    print("Loading measured comm profile:", MEASURED_COMM_NPZ)
    measured_comm = load_measured_comm(MEASURED_COMM_NPZ)
    comm_params = build_comm_fit_params(measured_comm) if measured_comm else None

    if measured_compute is None:
        print("No measured compute data found. Please run the measurement script first.")
        return

    print("Building measured and model t_fwd functions...")
    t_fwd_measured, t_fwd_model, c = build_t_fwd_measured_and_model(measured_compute, comm_params, hidden_dim=16)
    print("Fitted c:", c)
    if comm_params:
        print("Comm linear params:", comm_params)

    # 1) surfaces
    print("Making measured t_fwd surface (interactive)...")
    out1, fig1 = make_surface_plot(t_fwd_measured, measured_compute, "t_fwd measured surface", "t_fwd_measured_surface.html")
    print("Making model t_fwd surface (interactive)...")
    out2, fig2 = make_surface_plot(t_fwd_model, measured_compute, "t_fwd model surface", "t_fwd_model_surface.html")

    # 1b) DP path overlay on surface (example for L=128)
    L_plot = 128
    print(f"Computing DP path overlay for L={L_plot}...")
    T_star, slicing_m, _ = find_optimal_slicing_scheme(L_plot, 4, t_fwd_measured)
    plot_dp_path_on_surface(
        t_fwd_func=t_fwd_measured,
        slicing=slicing_m,
        title=f"DP Path on Measured t_fwd Surface (L={L_plot})",
        fname=f"dp_path_L{L_plot}_measured.html",
        measured_compute=measured_compute
    )

    # 2) comm plot
    if comm_params:
        outc, figc = make_comm_plot(comm_params, measured_comm, "comm_profile_plot.html")

    # 3) load DP summary if exists
    if os.path.exists(DP_CSV):
        dp_df = pd.read_csv(DP_CSV)
    else:
        dp_df = None
        print("No dp_experiments_summary.csv found; we will still generate T* vs K using the model.")

    # 4) For each L present in dp_df or default Ls, plot slicing bars and comparisons
    Ls_to_plot = []
    if dp_df is not None:
        Ls_to_plot = sorted(dp_df["L"].unique())
    else:
        Ls_to_plot = [512, 1024, 2048]

    for L in Ls_to_plot:
        # compute measured slicing and model slicing using DP (expensive)
        print("Computing DP for L=", L, "using measured t_fwd...")
        Tm, slicing_m, _ = find_optimal_slicing_scheme(L, 4, t_fwd_measured)
        Ts, slicing_s, _ = find_optimal_slicing_scheme(L, 4, t_fwd_model)
        # bar charts
        fname = f"optimal_slicing_L_{L}.html"
        plot_slicing_bar(slicing_m, f"Optimal slicing (measured) L={L}", fname)
        fname2 = f"optimal_slicing_model_L_{L}.html"
        plot_slicing_bar(slicing_s, f"Optimal slicing (model) L={L}", fname2)
        # comparison
        fnamec = f"slicing_comp_L_{L}.html"
        plot_slicing_comparison(slicing_m, slicing_s, f"Slicing comparison L={L} (measured vs model)", fnamec)

    # 5) T* vs K plots (for both measured and model)
    print("Computing T* vs K (measured)...")
    plot_Tstar_vs_K(t_fwd_measured, L_values=[32,64,128], K_values=[2,4,8,16], fname="Tfstar_vs_K_measured.html")
    print("Computing T* vs K (model)...")
    plot_Tstar_vs_K(t_fwd_model, L_values=[32,64,128], K_values=[2,4,8,16], fname="Tfstar_vs_K_model.html")

    print("All plots saved in folder:", OUTPUT_DIR)
    print("Reference paper (local path):", PAPER_PATH)

    # Optionally open an index in browser
    index_path = os.path.join(OUTPUT_DIR, "index.html")
    html = "<html><body><h2>DP Visualization outputs</h2><ul>"
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        if fn.endswith(".html"):
            html += f'<li><a href="{fn}">{fn}</a></li>'
    html += "</ul></body></html>"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("Index page written to", index_path)
    # open index in default browser (comment/uncomment as needed)
    try:
        webbrowser.open("file://" + os.path.abspath(index_path))
    except Exception:
        pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
pipeline_measure_dp.py

One-file integrated script that:
 - measures compute time for slice-forward on CPU (NumPy transformer)
 - measures MPI point-to-point communication time (latency + bandwidth)
 - builds measured t_fwd(l_i, sum_prev) via interpolation on a measured grid
 - fits an analytical model t_fwd_model(l, sum_prev) = c * l * (sum_prev + l) * hidden_dim
 - runs DP to find optimal slicing using measured and model t_fwd
 - runs experiments for L in {512,1024,2048}
 - compares measured-based slicing with model-based slicing (simple metrics)

USAGE:
  # 1) To run non-MPI measurements + DP visualization locally:
    python pipeline_measure_dp.py --mode local --max_context_grid 512

  # 2) To measure MPI comm times, run with mpiexec using at least 2 ranks:
    mpiexec -n 2 python pipeline_measure_dp.py --mode mpi_comm_measure

  # 3) To run DP experiments using measured data (after measuring):
    python pipeline_measure_dp.py --mode experiments

  # 4) To run token-level pipeline using the measured slicing (requires mpiexec -n K):
    mpiexec -n 4 python pipeline_measure_dp.py --mode mpi_pipeline

Notes:
 - Measurements can be slow. For large experiments measure at moderate grid resolution and let the script interpolate.
 - The script stores measured results to local files (csv / npz) to avoid repeating expensive measurements.
 - Reference paper uploaded by user: /mnt/data/tokenlevel.pdf
"""

import time
import numpy as np
import argparse
import os
import sys
import math
from mpi4py import MPI
import csv

# ---------------------------
# User / hardware parameters
# ---------------------------
hidden_dim = 16
num_heads = 2
ffn_dim = 32
seq_length = 128  # default sequence length for MPI pipeline mode

# default sequence lengths to experiment on
EXPERIMENT_LS = [32,64,128]

# measurement grid defaults (adjust to tradeoff accuracy vs time)
DEFAULT_SUMPREV_GRID = [0, 16, 32, 64, 128, 256, 512]  # sum_prev values to sample
DEFAULT_SLICE_GRID = [1, 2, 4, 8, 16, 32, 64, 128]    # l_i values to sample

# measurement iterations
COMPUTE_WARMUP = 2
COMPUTE_ITERS = 6

# data files
MEASURED_COMPUTE_NPZ = "measured_compute_grid.npz"
MEASURED_COMM_NPZ = "measured_comm_profile.npz"
DP_OUTPUT_CSV = "dp_experiments_summary.csv"

# reference to uploaded paper (developer instruction)
PAPER_FILE_PATH = "/mnt/data/tokenlevel.pdf"

# ---------------------------
# Toy transformer (same as yours)
# ---------------------------
def split_heads(x, num_heads):
    return x.reshape(-1, num_heads, hidden_dim // num_heads)

def combine_heads(x):
    return x.reshape(x.shape[0], -1)

def transformer_layer_single_token(token_vec, layer_idx):
    """
    The toy transformer layer that accepts a single token vector (1D).
    This is the same forward implementation you had earlier.
    """
    np.random.seed(layer_idx)
    Wq, Wk, Wv, Wo = [np.random.randn(hidden_dim, hidden_dim) for _ in range(4)]
    bq, bk, bv, bo = [np.random.randn(hidden_dim) for _ in range(4)]

    Q = (token_vec @ Wq) + bq
    K = (token_vec @ Wk) + bk
    V = (token_vec @ Wv) + bv

    Qh = split_heads(Q, num_heads)
    Kh = split_heads(K, num_heads)
    Vh = split_heads(V, num_heads)

    attn_score = np.matmul(Qh, Kh.transpose(0,2,1)) / np.sqrt(hidden_dim // num_heads)
    attn_weights = np.exp(attn_score - np.max(attn_score))
    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

    attn_out = np.matmul(attn_weights, Vh)
    attn_out_cat = combine_heads(attn_out)
    out1 = np.tanh(attn_out_cat @ Wo + bo)

    W1 = np.random.randn(hidden_dim, ffn_dim)
    b1 = np.random.randn(ffn_dim)
    W2 = np.random.randn(ffn_dim, hidden_dim)
    b2 = np.random.randn(hidden_dim)
    ff_hidden = np.tanh(out1 @ W1 + b1)
    ff_out = ff_hidden @ W2 + b2

    normed = (ff_out - np.mean(ff_out)) / (np.std(ff_out) + 1e-5)
    return normed + token_vec

# ---------------------------
# More realistic compute timing: compute a slice (with preceding context)
# We will construct an array H of shape (sum_prev + l_i, hidden_dim),
# and compute attention outputs only for the last l_i tokens, using the
# naive O(N^2) token-by-token formula. This reflects the true forward cost
# of causal self-attention (as in the paper).
# ---------------------------
def compute_time_for_slice(sum_prev, l_i, warmup=COMPUTE_WARMUP, iters=COMPUTE_ITERS):
    """
    Returns average wallclock time to compute forward outputs for a slice of length l_i
    appended to a context of length sum_prev, using the toy transformer (single-token calls).
    This is expensive for large sum_prev; use a coarse grid.
    """
    total_len = sum_prev + l_i
    # Build random hidden states for the whole context+slice
    H = np.random.randn(total_len, hidden_dim).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        # simulate computing for last l_i tokens
        for t in range(sum_prev, total_len):
            # compute Q,K,V for token t, and keys/values for 0..t
            # We reuse transformer_layer_single_token semantics by passing token only;
            # this does not include actual context mixing inside that function;
            # so to simulate real cost, we implement attention-like operations here.
            # We'll implement a simple attention compute loop (vector ops) to reflect cost.
            q = H[t] @ np.random.randn(hidden_dim, hidden_dim)  # random projection
            # compute keys for 0..t
            _ = H[:t+1] @ np.random.randn(hidden_dim, hidden_dim)
    # Actual measurement
    times = []
    for _ in range(iters):
        t0 = time.time()
        for t in range(sum_prev, total_len):
            # naive attention compute for token t:
            # compute Q = H[t] @ Wq  , K = H[:t+1] @ Wk , V = H[:t+1] @ Wv
            # then scores = Q @ K.T  -> length t+1
            # attn = softmax(scores/sqrt(H)) ; out = attn @ V
            # We implement this with randomized matrices to keep cost realistic.
            Wq = np.random.randn(hidden_dim, hidden_dim)
            Wk = np.random.randn(hidden_dim, hidden_dim)
            Wv = np.random.randn(hidden_dim, hidden_dim)
            Q = H[t] @ Wq  # (hidden_dim,)
            Ks = H[:t+1] @ Wk  # (t+1, hidden_dim)
            Vs = H[:t+1] @ Wv  # (t+1, hidden_dim)
            # scores = Ks @ Q  (vector length t+1)
            scores = Ks @ Q
            # scale + softmax (we'll use stable exp)
            scaled = scores / math.sqrt(hidden_dim // num_heads)
            exps = np.exp(scaled - np.max(scaled))
            alphas = exps / (np.sum(exps) + 1e-12)
            out = alphas @ Vs  # (hidden_dim,)
            # pass through small FFN (simulate cost)
            _ = np.tanh(out @ np.random.randn(hidden_dim, ffn_dim) + np.random.randn(ffn_dim))
            _ = _ @ np.random.randn(ffn_dim, hidden_dim) + np.random.randn(hidden_dim)
        t1 = time.time()
        times.append(t1 - t0)
    return float(np.mean(times))

# ---------------------------
# MPI communication measurement
# ---------------------------
def measure_mpi_comm_profile(msg_sizes_bytes, comm=None, iters=5):
    """
    Measure one-way send time for a set of message sizes using blocking Send/Recv.
    Must be run with at least 2 ranks; returns a dict {size: time_seconds}.
    The returned time approximates latency + size/bandwidth.
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        if rank == 0:
            print("WARN: measure_mpi_comm_profile requires at least 2 MPI ranks")
        return {}

    results = {}
    for b in msg_sizes_bytes:
        total = 0.0
        # prepare buffer (float32)
        n_floats = max(1, b // 4)
        buf = np.random.randn(n_floats).astype(np.float32)
        for _ in range(iters):
            comm.Barrier()
            t0 = MPI.Wtime()
            if rank == 0:
                comm.Send([buf, MPI.FLOAT], dest=1, tag=999)
                comm.Recv([buf, MPI.FLOAT], source=1, tag=999)
            elif rank == 1:
                comm.Recv([buf, MPI.FLOAT], source=0, tag=999)
                comm.Send([buf, MPI.FLOAT], dest=0, tag=999)
            t1 = MPI.Wtime()
            total += (t1 - t0) / 2.0  # one-way
        avg = total / iters
        results[b] = avg
        # rank 0 prints small progress
        if rank == 0:
            print(f"Measured comm: {b} bytes -> {avg:.6e} s")
    return results

# ---------------------------
# Interpolation helpers
# ---------------------------
def interp_comm_time(measured_profile, msg_bytes):
    """
    Given measured points {size: time}, do a simple linear fit time = a + size/bandwidth.
    Return predicted time for msg_bytes.
    """
    sizes = np.array(sorted(list(measured_profile.keys())), dtype=np.float64)
    times = np.array([measured_profile[s] for s in sizes], dtype=np.float64)
    # Fit linear model: time = a + size / bw  -> treat as time vs size, slope = 1/bw
    # We'll fit time = a + slope * size
    A = np.vstack([np.ones_like(sizes), sizes]).T
    sol, _, _, _ = np.linalg.lstsq(A, times, rcond=None)
    a, slope = sol[0], sol[1]
    pred = a + slope * msg_bytes
    return float(pred), float(a), float(1.0 / slope if slope != 0 else float('inf'))

# ---------------------------
# DP algorithm (same as yours)
# ---------------------------
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
        sum_prev = 0
        max_t_i = 0.0
        total_time = 0.0
        for l_i in best_slicing:
            t_i = t_fwd_func(l_i, sum_prev)
            max_t_i = max(max_t_i, t_i)
            total_time += t_i
            sum_prev += l_i
        best_T_star = total_time + (K - 1) * max_t_i
        best_tmax = max_t_i
        print("Warning: No optimal slicing found, using uniform slicing as fallback")
    return best_T_star, best_slicing, best_tmax

# ---------------------------
# High-level workflows
# ---------------------------
def measure_compute_grid(sumprev_grid, slice_grid, out_npz=MEASURED_COMPUTE_NPZ):
    """
    Measure compute times for a grid of (sum_prev, l_i). Store results in NPZ.
    Returns dict measured[(sum_prev, l)] = time_seconds
    """
    print("Measuring compute grid: this can take long. Grid sizes:", len(sumprev_grid), len(slice_grid))
    measured = {}
    for s in sumprev_grid:
        for l in slice_grid:
            # Don't exceed a massive total length (s + l)
            print(f"Measuring compute for sum_prev={s}, l={l} ...", flush=True)
            t = compute_time_for_slice(s, l)
            measured[(s, l)] = t
            print(f"  -> {t:.6e} s")
    # Save to disk
    keys = np.array(list(measured.keys()), dtype=np.int64)
    vals = np.array([measured[k] for k in measured.keys()], dtype=np.float64)
    np.savez(out_npz, keys=keys, vals=vals)
    print("Saved compute measurements to", out_npz)
    return measured

def load_measured_compute_grid(npzfile=MEASURED_COMPUTE_NPZ):
    if not os.path.exists(npzfile):
        return None
    data = np.load(npzfile)
    keys = data["keys"]
    vals = data["vals"]
    measured = {}
    for (s, l), v in zip(keys, vals):
        measured[(int(s), int(l))] = float(v)
    return measured

def measure_and_fit_models(sumprev_grid=None, slice_grid=None, comm_msg_sizes=None):
    """
    Measure compute (grid) and communication profile, then:
      - build an interpolant for measured t_fwd (via nearest neighbor / simple bilinear)
      - fit the analytic model coefficient c via least squares:
          t_measured = c * l * (sum_prev + l) * hidden_dim
    Returns measured_compute_grid, measured_comm_profile, model_c, and a measured t_fwd function.
    """
    if sumprev_grid is None:
        sumprev_grid = DEFAULT_SUMPREV_GRID
    if slice_grid is None:
        slice_grid = DEFAULT_SLICE_GRID
    if comm_msg_sizes is None:
        comm_msg_sizes = [64, 256, 1024, 4096, 16384, 65536]

    # 1) compute grid (load if exists)
    measured_compute = load_measured_compute_grid()
    if measured_compute is None:
        measured_compute = measure_compute_grid(sumprev_grid, slice_grid)

    # 2) comm profile (only possible if MPI ranks >= 2)
    comm = MPI.COMM_WORLD
    measured_comm = {}
    rank = comm.Get_rank()
    if comm.Get_size() >= 2:
        # to avoid duplication, run measure on ranks 0/1 only
        measured_comm = measure_mpi_comm_profile(comm_msg_sizes, comm=comm)
        # Save comm to npz on rank 0
        if rank == 0:
            np.savez(MEASURED_COMM_NPZ, sizes=np.array(list(measured_comm.keys())), times=np.array(list(measured_comm.values())))
            print("Saved comm profile to", MEASURED_COMM_NPZ)
    else:
        if rank == 0:
            print("MPI size < 2: skipping MPI comm measurement. You can run with mpiexec -n 2 to measure comm.")
    comm.Barrier()

    # 3) Fit analytic model c using least squares on measured grid
    # Build arrays: y = measured_t ; x = l*(sum_prev + l) * hidden_dim
    xs = []
    ys = []
    for (s, l), t in measured_compute.items():
        xs.append(l * (s + l) * hidden_dim)
        ys.append(t)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    # Fit y = c * x
    c = 0.0
    if len(xs) > 0:
        # avoid zero division
        c = np.sum(xs * ys) / (np.sum(xs * xs) + 1e-30)
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Fitted analytic model coefficient c = {c:.6e} (so model t = c * l*(sum_prev+l)*hidden_dim)")

    # 4) Build measured t_fwd function via simple nearest-neighbor interpolation on measured grid
    # If measured_comm exists, build comm interpolation
    comm_interp = None
    comm_params = None
    if len(measured_comm) > 0:
        # Fit linear time = a + slope * size
        # get predicted function via interp_comm_time
        # store measured dict for interpolation
        comm_params = {}
        sizes = sorted(measured_comm.keys())
        times = [measured_comm[s] for s in sizes]
        # compute linear fit on sizes->times
        A = np.vstack([np.ones(len(sizes)), sizes]).T
        sol, _, _, _ = np.linalg.lstsq(A, np.array(times), rcond=None)
        a, slope = float(sol[0]), float(sol[1])
        comm_params["a"] = a
        comm_params["slope"] = slope
        comm_params["sizes"] = sizes
        comm_params["times"] = times
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Comm fit: time ~= {a:.3e} + {slope:.3e} * size (bytes)")

    # Create t_fwd_measured closure
    def t_fwd_measured(l_i, sum_prev):
        # nearest neighbor: if exact measured exists, return it; else find nearest measured s and l
        # fallback: use analytic model if nothing close
        key = (sum_prev, l_i)
        if key in measured_compute:
            comp = measured_compute[key]
        else:
            # find nearest measured by Euclidean distance on (s, l)
            bestk = None
            bestd = float("inf")
            for (s, l), v in measured_compute.items():
                d = (s - sum_prev)**2 + (l - l_i)**2
                if d < bestd:
                    bestd = d
                    bestk = (s, l)
            comp = measured_compute[bestk]
        # comm prediction: message size in bytes (float32)
        msg_bytes = int(hidden_dim * l_i * 4)
        comm_time = 0.0
        if comm_params is not None:
            comm_time = comm_params["a"] + comm_params["slope"] * msg_bytes
        # else fallback to zero comm (or a small default)
        return comp + comm_time

    # Create analytic model closure
    def t_fwd_model(l_i, sum_prev):
        return c * l_i * (sum_prev + l_i) * hidden_dim

    return measured_compute, measured_comm, t_fwd_measured, t_fwd_model, c

# ---------------------------
# Experiment runner: run DP for many L and compare
# ---------------------------
def run_experiments_and_compare(t_fwd_measured, t_fwd_model, Ks=[4], Ls=EXPERIMENT_LS, out_csv=DP_OUTPUT_CSV):
    rows = []
    for K in Ks:
        for L in Ls:
            print(f"\n=== Running DP for L={L}, K={K} (measured) ===")
            Tm, slicing_m, tmax_m = find_optimal_slicing_scheme(L, K, t_fwd_measured)
            print("Measured-based slicing:", slicing_m)
            print(f"Measured T*: {Tm:.6e}, tmax: {tmax_m:.6e}")

            print(f"\n=== Running DP for L={L}, K={K} (model) ===")
            Ts, slicing_s, tmax_s = find_optimal_slicing_scheme(L, K, t_fwd_model)
            print("Model-based slicing:", slicing_s)
            print(f"Model T*: {Ts:.6e}, tmax: {tmax_s:.6e}")

            # Comparison metrics
            # 1) number of slices
            n_m = len(slicing_m)
            n_s = len(slicing_s)
            # 2) L1 difference of slice-length vectors after padding/truncation
            def pad_list(a, n):
                if len(a) >= n:
                    return a[:n]
                return a + [0] * (n - len(a))
            maxlen = max(len(slicing_m), len(slicing_s))
            v1 = pad_list(slicing_m, maxlen)
            v2 = pad_list(slicing_s, maxlen)
            l1 = sum(abs(x - y) for x,y in zip(v1,v2))
            # 3) fraction of tokens placed in same slice indices (coarse)
            same_count = sum(1 for x,y in zip(v1,v2) if x == y)
            frac_same = same_count / maxlen if maxlen>0 else 1.0

            rows.append({
                "L": L, "K": K,
                "T_measured": Tm, "T_model": Ts,
                "tmax_measured": tmax_m, "tmax_model": tmax_s,
                "n_slices_measured": n_m, "n_slices_model": n_s,
                "slice_L1_diff": l1, "slice_frac_same": frac_same,
                "slicing_measured": str(slicing_m),
                "slicing_model": str(slicing_s)
            })
            # Save occasional CSV
            with open(out_csv, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(rows[-1])
    print("\nSaved experiment summaries to", out_csv)
    return rows

# ---------------------------
# MPI token-level pipeline runner (forward-only)
# ---------------------------
def run_token_level_pipeline_mpi(input_sequence, slicing_scheme=None):
    """
    Runs token-level pipeline on MPI ranks where rank i executes layer i.
    Assumes number of ranks == number of layers (K).
    This is a forward-only run and uses the toy transformer_layer_single_token.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    K = size
    L = input_sequence.shape[0]
    # obtain slicing via rank 0 if not provided
    if slicing_scheme is None and rank == 0:
        # simple default: equal slices
        slicing_scheme = [L // K] * K
        rem = L % K
        for i in range(rem):
            slicing_scheme[i] += 1
    slicing_scheme = comm.bcast(slicing_scheme if rank == 0 else None, root=0)
    if rank == 0:
        print("MPI pipeline running with ranks=", size, "slicing=", slicing_scheme)
    outputs = []
    slice_start = 0
    start_time = time.time()
    for slice_idx, slen in enumerate(slicing_scheme):
        slice_end = slice_start + slen
        for token_pos in range(slice_start, slice_end):
            if rank == 0:
                token_hidden = input_sequence[token_pos]
            else:
                token_hidden = comm.recv(source=rank-1, tag=1000 + slice_idx * 1000 + token_pos)
            # apply layer (toy)
            token_hidden = transformer_layer_single_token(token_hidden, layer_idx=rank)
            if rank < size - 1:
                comm.send(token_hidden, dest=rank+1, tag=1000 + slice_idx * 1000 + token_pos)
            else:
                outputs.append(token_hidden)
        slice_start = slice_end
    elapsed = time.time() - start_time
    if rank == size - 1:
        print(f"MPI pipeline forward finished. Elapsed: {elapsed:.6f}s. Outputs len={len(outputs)}")
    return outputs

# ---------------------------
# Command-line interface
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local","mpi_comm_measure","measure_and_fit","experiments","mpi_pipeline"], default="local",
                   help="Mode to run")
    p.add_argument("--sumprev_grid", nargs="+", type=int, default=DEFAULT_SUMPREV_GRID,
                   help="Grid of sum_prev values to measure")
    p.add_argument("--slice_grid", nargs="+", type=int, default=DEFAULT_SLICE_GRID,
                   help="Grid of slice lengths to measure")
    p.add_argument("--comm_msg_sizes", nargs="+", type=int, default=[64,256,1024,4096,16384,65536],
                   help="Message sizes (bytes) to measure for MPI")
    p.add_argument("--L_values", nargs="+", type=int, default=EXPERIMENT_LS, help="Sequence lengths for experiments")
    p.add_argument("--K_values", nargs="+", type=int, default=[4], help="Pipeline stage counts to test")
    return p.parse_args()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if args.mode == "local":
        # quick demo: load measured if available; else measure small grid and show DP for L=128
        measured_compute = load_measured_compute_grid()
        if measured_compute is None:
            if rank == 0:
                print("No measured compute grid found locally. Running a small local measurement (this can take time)...")
            measured_compute = measure_compute_grid(DEFAULT_SUMPREV_GRID[:3], DEFAULT_SLICE_GRID[:4])  # small grid
        # fit models
        measured_compute, measured_comm, t_fwd_measured, t_fwd_model, c = measure_and_fit_models(sumprev_grid=DEFAULT_SUMPREV_GRID[:3], slice_grid=DEFAULT_SLICE_GRID[:4])
        if rank == 0:
            # run DP for L=128 as quick check
            Tm, slicing_m, _ = find_optimal_slicing_scheme(128, 4, t_fwd_measured)
            Ts, slicing_s, _ = find_optimal_slicing_scheme(128, 4, t_fwd_model)
            print("Measured slicing (L=128):", slicing_m)
            print("Model slicing (L=128):   ", slicing_s)
            print("Reference paper file:", PAPER_FILE_PATH)
    elif args.mode == "mpi_comm_measure":
        # measure MPI comm times (requires mpiexec with at least 2 procs)
        measured_comm = measure_mpi_comm_profile(args.comm_msg_sizes, comm=comm)
        if rank == 0:
            np.savez(MEASURED_COMM_NPZ, sizes=np.array(list(measured_comm.keys())), times=np.array(list(measured_comm.values())))
            print("Saved measured comm profile to", MEASURED_COMM_NPZ)
    elif args.mode == "measure_and_fit":
        measured_compute, measured_comm, t_fwd_measured, t_fwd_model, c = measure_and_fit_models(sumprev_grid=args.sumprev_grid, slice_grid=args.slice_grid, comm_msg_sizes=args.comm_msg_sizes)
        if rank == 0:
            print("Measure+fit complete. c=", c)
            print("Saved measurements and fitted models. Paper ref:", PAPER_FILE_PATH)
    elif args.mode == "experiments":
        # load measured models if available
        measured_compute = load_measured_compute_grid()
        measured_comm = {}
        if os.path.exists(MEASURED_COMM_NPZ):
            dat = np.load(MEASURED_COMM_NPZ)
            measured_comm = {int(s): float(t) for s,t in zip(dat["sizes"], dat["times"])}
        if measured_compute is None:
            if rank == 0:
                print("No measured compute grid found. Please run measure_and_fit first. Exiting.")
            sys.exit(1)
        # create closures
        # reuse measure_and_fit to get fitted c and comm params
        measured_compute, measured_comm, t_fwd_measured, t_fwd_model, c = measure_and_fit_models(sumprev_grid=args.sumprev_grid, slice_grid=args.slice_grid, comm_msg_sizes=args.comm_msg_sizes)
        # run DP experiments
        if rank == 0:
            rows = run_experiments_and_compare(t_fwd_measured, t_fwd_model, Ks=args.K_values, Ls=args.L_values, out_csv=DP_OUTPUT_CSV)
            print("Experiment rows:", rows)
    elif args.mode == "mpi_pipeline":
        # Run MPI pipeline using a measured or model-derived slicing (rank 0 will compute DP if needed)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        K = size
        # try to load measured t_fwd
        measured_compute = load_measured_compute_grid()
        if measured_compute is None:
            if rank == 0:
                print("No measured compute grid found. Running small local measurement (will be used for slicing).")
            measured_compute = measure_compute_grid(DEFAULT_SUMPREV_GRID[:4], DEFAULT_SLICE_GRID[:5])
        # build t_fwd_measured closure (reuse measure_and_fit to also fit comm)
        measured_compute, measured_comm, t_fwd_measured, t_fwd_model, c = measure_and_fit_models(sumprev_grid=args.sumprev_grid, slice_grid=args.slice_grid, comm_msg_sizes=args.comm_msg_sizes)
        # rank 0 computes slicing for L=seq_length
        if rank == 0:
            T_star, slicing, tmax = find_optimal_slicing_scheme(seq_length, K, t_fwd_measured)
            print("Using measured slicing:", slicing)
        else:
            slicing = None
        slicing = comm.bcast(slicing, root=0)
        # build random input sequence on rank 0
        input_sequence = None
        if rank == 0:
            input_sequence = np.random.randn(seq_length, hidden_dim).astype(np.float32)
        else:
            input_sequence = np.empty((seq_length, hidden_dim), dtype=np.float32)
        # only rank 0 needs the input; alternative is to let rank0 feed tokens
        # We'll pass the sequence by broadcasting (cheapish)
        comm.Bcast(input_sequence, root=0)
        # Run token-level pipeline forward
        outputs = run_token_level_pipeline_mpi(input_sequence, slicing_scheme=slicing)
        if rank == 0:
            print("MPI pipeline invocation complete. Paper ref:", PAPER_FILE_PATH)
    else:
        if rank == 0:
            print("Unknown mode:", args.mode)

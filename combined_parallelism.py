#!/usr/bin/env python3
"""
combined_parallelism.py

Implements combining microbatch-based pipeline parallelism with token-level parallelism.
Compares:
1. Pure pipeline parallelism (no token-level slicing)
2. Combined pipeline + token-level parallelism (joint optimization)

Based on the approach:
- For each batch size b, run DP to get optimal T_b and s_b
- Solve 1D knapsack to find optimal batch slice sizes b_1, ..., b_D
- Compare with pure pipeline parallelism

Usage:
    python combined_parallelism.py --mode compare --B 64 --D 4 --K 4 --L 128
    python combined_parallelism.py --mode visualize --B 64 --D 4 --K 4 --L 128
"""

import numpy as np
import argparse
import os
import sys
import json
from typing import Dict, List, Tuple, Callable
import matplotlib.pyplot as plt
import pandas as pd

# Import DP functions from metricsdp
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metricsdp import (
        find_optimal_slicing_scheme,
        load_measured_compute_grid,
        MEASURED_COMPUTE_NPZ,
        MEASURED_COMM_NPZ,
        hidden_dim
    )
    
    # Helper functions for comm
    def load_measured_comm(npzfile=MEASURED_COMM_NPZ):
        if not os.path.exists(npzfile):
            return {}
        data = np.load(npzfile)
        sizes = data["sizes"]
        times = data["times"]
        return {int(s): float(t) for s, t in zip(sizes, times)}
    
    def build_comm_fit_params(measured_comm):
        if not measured_comm:
            return None
        sizes = np.array(sorted(measured_comm.keys()), dtype=float)
        times = np.array([measured_comm[s] for s in sizes], dtype=float)
        A = np.vstack([np.ones_like(sizes), sizes]).T
        sol, *_ = np.linalg.lstsq(A, times, rcond=None)
        a, slope = float(sol[0]), float(sol[1])
        return {"a": a, "slope": slope, "sizes": list(sizes), "times": list(times)}
        
except ImportError as e:
    print(f"Warning: Could not import from metricsdp: {e}")
    print("Using fallback implementations.")
    hidden_dim = 16
    MEASURED_COMPUTE_NPZ = "measured_compute_grid.npz"
    MEASURED_COMM_NPZ = "measured_comm_profile.npz"
    
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
    
    def find_optimal_slicing_scheme(L, K, t_fwd_func):
        # Fallback implementation
        uniform_size = L // K if K > 0 else L
        if uniform_size == 0:
            uniform_size = 1
        slicing = [uniform_size] * (L // uniform_size)
        remainder = L % uniform_size
        if remainder > 0:
            slicing.append(remainder)
        total_time = sum(t_fwd_func(l, sum(slicing[:i])) for i, l in enumerate(slicing))
        return total_time, slicing, max(t_fwd_func(l, sum(slicing[:i])) for i, l in enumerate(slicing))
    
    def load_measured_comm(npzfile=MEASURED_COMM_NPZ):
        return {}
    
    def build_comm_fit_params(measured_comm):
        return None

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_B = 64  # Total batch size
DEFAULT_D = 4   # Number of microbatches
DEFAULT_K = 4   # Number of pipeline stages
DEFAULT_L = 128 # Sequence length

# ---------------------------
# Helper: Build t_fwd function from measurements
# ---------------------------
def build_t_fwd_from_measurements(measured_compute, comm_params=None, hidden_dim=16):
    """Build t_fwd function from measured compute grid and comm params."""
    if measured_compute is None:
        # Fallback to simple model
        def t_fwd(l_i, sum_prev, batch_size=1):
            # Simple model: scales with batch_size
            base = l_i * (sum_prev + l_i) * hidden_dim * 1e-8
            comm = 0.001 + (hidden_dim * l_i * batch_size * 4) / 1e6
            return base * batch_size + comm
        return t_fwd
    
    def t_fwd(l_i, sum_prev, batch_size=1):
        # Nearest neighbor lookup
        key = (int(sum_prev), int(l_i))
        if key in measured_compute:
            comp = measured_compute[key]
        else:
            bestk = None
            bestd = float("inf")
            for (s, l), v in measured_compute.items():
                d = (s - sum_prev)**2 + (l - l_i)**2
                if d < bestd:
                    bestd = d
                    bestk = (s, l)
            comp = measured_compute[bestk]
        
        # Scale by batch size
        comp = comp * batch_size
        
        # Add communication (scales with batch_size)
        comm_time = 0.0
        if comm_params is not None:
            msg_bytes = int(hidden_dim * l_i * batch_size * 4)
            comm_time = comm_params["a"] + comm_params["slope"] * msg_bytes
        
        return comp + comm_time
    
    return t_fwd

# ---------------------------
# Step 1: DP for each batch size b
# ---------------------------
def compute_optimal_for_batch_size(b: int, L: int, K: int, t_fwd_func: Callable) -> Tuple[float, List[int]]:
    """
    For batch size b, compute optimal T_b and slicing scheme s_b.
    
    Args:
        b: Batch size
        L: Sequence length
        K: Number of pipeline stages
        t_fwd_func: Function t_fwd(l_i, sum_prev, batch_size)
    
    Returns:
        (T_b, s_b) where T_b is optimal time and s_b is slicing scheme
    """
    def t_fwd_wrapper(l_i, sum_prev):
        return t_fwd_func(l_i, sum_prev, batch_size=b)
    
    T_star, slicing, _ = find_optimal_slicing_scheme(L, K, t_fwd_wrapper)
    return T_star, slicing

# ---------------------------
# Step 2: 1D Knapsack solver for batch dimension
# ---------------------------
def solve_batch_knapsack(B: int, D: int, T_b_dict: Dict[int, float]) -> Tuple[List[int], float]:
    """
    Solve 1D knapsack: find b_1, ..., b_D such that:
    - b_1 + ... + b_D = B
    - minimize T_{b_1} + ... + T_{b_D}
    
    Args:
        B: Total batch size
        D: Number of microbatches
        T_b_dict: Dictionary mapping batch size b to optimal time T_b
    
    Returns:
        (batch_sizes, total_time) where batch_sizes is list [b_1, ..., b_D]
    """
    # DP: dp[i][j] = minimum total time using i microbatches to get total batch size j
    dp = [[float('inf')] * (B + 1) for _ in range(D + 1)]
    dp[0][0] = 0.0
    
    # Backtracking: parent[i][j] = (prev_i, prev_j, batch_size_used)
    parent = [[None] * (B + 1) for _ in range(D + 1)]
    
    for i in range(1, D + 1):
        for j in range(B + 1):
            # Try all possible batch sizes for this microbatch
            for b in range(1, min(j + 1, B + 1)):
                if b in T_b_dict:
                    prev_j = j - b
                    if prev_j >= 0 and dp[i-1][prev_j] != float('inf'):
                        candidate = dp[i-1][prev_j] + T_b_dict[b]
                        if candidate < dp[i][j]:
                            dp[i][j] = candidate
                            parent[i][j] = (i-1, prev_j, b)
    
    # Backtrack to get batch sizes
    if dp[D][B] == float('inf'):
        # Fallback: uniform splitting
        uniform_b = B // D
        remainder = B % D
        batch_sizes = [uniform_b] * D
        for i in range(remainder):
            batch_sizes[i] += 1
        total_time = sum(T_b_dict.get(b, float('inf')) for b in batch_sizes)
        return batch_sizes, total_time
    
    # Reconstruct solution
    batch_sizes = []
    i, j = D, B
    while i > 0 and parent[i][j] is not None:
        prev_i, prev_j, b = parent[i][j]
        batch_sizes.insert(0, b)
        i, j = prev_i, prev_j
    
    return batch_sizes, dp[D][B]

# ---------------------------
# Step 3: Pure pipeline parallelism (baseline)
# ---------------------------
def compute_pure_pipeline_time(B: int, D: int, K: int, L: int, t_fwd_func: Callable) -> float:
    """
    Compute time for pure pipeline parallelism (no token-level slicing).
    Each microbatch has batch_size = B/D and processes full sequence L.
    """
    batch_per_microbatch = B // D
    remainder = B % D
    
    # Time for one microbatch: process full sequence L with batch_size
    def t_fwd_wrapper(l_i, sum_prev):
        return t_fwd_func(l_i, sum_prev, batch_size=batch_per_microbatch)
    
    # For pure pipeline: one slice of length L
    t_single = t_fwd_wrapper(L, 0)
    
    # Pipeline time: (D + K - 1) * t_single (simplified)
    # More accurate: first microbatch takes K stages, remaining D-1 take 1 stage each
    pipeline_time = (K - 1) * t_single + D * t_single
    
    return pipeline_time

# ---------------------------
# Step 4: Combined approach
# ---------------------------
def compute_combined_parallelism(B: int, D: int, K: int, L: int, t_fwd_func: Callable) -> Tuple[float, List[int], Dict[int, List[int]]]:
    """
    Compute optimal combined pipeline + token-level parallelism.
    
    Returns:
        (total_time, batch_sizes, slicing_schemes)
        where batch_sizes = [b_1, ..., b_D]
        and slicing_schemes[b] = optimal token slicing for batch size b
    """
    # Step 1: For each possible batch size b, compute optimal T_b and s_b
    print(f"Computing optimal slicing for each batch size (1 to {B})...")
    T_b_dict = {}
    slicing_schemes = {}
    
    # Only compute for batch sizes that could be used (1 to B)
    for b in range(1, min(B + 1, 33)):  # Limit to avoid too many computations
        T_b, s_b = compute_optimal_for_batch_size(b, L, K, t_fwd_func)
        T_b_dict[b] = T_b
        slicing_schemes[b] = s_b
        if b % 8 == 0:
            print(f"  b={b}: T_b={T_b:.6f}s, slices={len(s_b)}")
    
    # Step 2: Solve knapsack for batch dimension
    print(f"Solving batch dimension knapsack (B={B}, D={D})...")
    batch_sizes, total_time = solve_batch_knapsack(B, D, T_b_dict)
    
    print(f"Optimal batch sizes: {batch_sizes}")
    print(f"Total time: {total_time:.6f}s")
    
    return total_time, batch_sizes, slicing_schemes

# ---------------------------
# Comparison function
# ---------------------------
def compare_approaches(B: int, D: int, K: int, L: int, t_fwd_func: Callable) -> Dict:
    """
    Compare pure pipeline vs combined approach.
    
    Returns dictionary with comparison results.
    """
    print("=" * 60)
    print("COMPARISON: Pure Pipeline vs Combined Pipeline+Token-Level")
    print("=" * 60)
    print(f"Configuration: B={B}, D={D}, K={K}, L={L}")
    print()
    
    # Pure pipeline
    print("Computing pure pipeline parallelism...")
    pure_time = compute_pure_pipeline_time(B, D, K, L, t_fwd_func)
    print(f"Pure pipeline time: {pure_time:.6f}s")
    print()
    
    # Combined
    print("Computing combined pipeline + token-level parallelism...")
    combined_time, batch_sizes, slicing_schemes = compute_combined_parallelism(B, D, K, L, t_fwd_func)
    print(f"Combined time: {combined_time:.6f}s")
    print()
    
    # Comparison
    speedup = pure_time / combined_time if combined_time > 0 else 0
    improvement = ((pure_time - combined_time) / pure_time * 100) if pure_time > 0 else 0
    
    print("=" * 60)
    print("RESULTS:")
    print(f"  Pure pipeline:        {pure_time:.6f}s")
    print(f"  Combined approach:    {combined_time:.6f}s")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  Improvement:          {improvement:.2f}%")
    print("=" * 60)
    
    return {
        "B": B,
        "D": D,
        "K": K,
        "L": L,
        "pure_pipeline_time": pure_time,
        "combined_time": combined_time,
        "speedup": speedup,
        "improvement_percent": improvement,
        "batch_sizes": batch_sizes,
        "slicing_schemes": {str(b): s for b, s in slicing_schemes.items()}
    }

# ---------------------------
# Visualization
# ---------------------------
def visualize_comparison(results: Dict, output_dir="plots_combined"):
    """Create visualization comparing pure vs combined approaches."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ["Pure Pipeline", "Combined\nPipeline+Token"]
    times = [results["pure_pipeline_time"], results["combined_time"]]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Pipeline Parallelism Comparison\n(B={results["B"]}, D={results["D"]}, K={results["K"]}, L={results["L"]})', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.4f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add speedup annotation
    speedup = results["speedup"]
    ax.text(0.5, 0.95, f'Speedup: {speedup:.2f}x\nImprovement: {results["improvement_percent"]:.1f}%',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    outpath = os.path.join(output_dir, "comparison_bar.png")
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.close()
    
    # Batch size distribution
    if "batch_sizes" in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        batch_sizes = results["batch_sizes"]
        microbatch_indices = list(range(1, len(batch_sizes) + 1))
        
        bars = ax.bar(microbatch_indices, batch_sizes, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Microbatch Index', fontsize=12)
        ax.set_ylabel('Batch Size', fontsize=12)
        ax.set_title('Optimal Batch Size Distribution\n(Combined Approach)', fontsize=14, fontweight='bold')
        ax.set_xticks(microbatch_indices)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars, batch_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{size}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        outpath = os.path.join(output_dir, "batch_size_distribution.png")
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {outpath}")
        plt.close()

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare pure vs combined pipeline parallelism")
    parser.add_argument("--mode", choices=["compare", "visualize", "both"], default="both",
                       help="Mode: compare, visualize, or both")
    parser.add_argument("--B", type=int, default=DEFAULT_B, help="Total batch size")
    parser.add_argument("--D", type=int, default=DEFAULT_D, help="Number of microbatches")
    parser.add_argument("--K", type=int, default=DEFAULT_K, help="Number of pipeline stages")
    parser.add_argument("--L", type=int, default=DEFAULT_L, help="Sequence length")
    parser.add_argument("--output", type=str, default="combined_results.json",
                       help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Load measurements
    print("Loading measured compute grid...")
    measured_compute = load_measured_compute_grid()
    measured_comm = load_measured_comm()
    comm_params = build_comm_fit_params(measured_comm) if measured_comm else None
    
    # Build t_fwd function
    t_fwd_func = build_t_fwd_from_measurements(measured_compute, comm_params, hidden_dim)
    
    # Run comparison
    if args.mode in ["compare", "both"]:
        results = compare_approaches(args.B, args.D, args.K, args.L, t_fwd_func)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Visualize
    if args.mode in ["visualize", "both"]:
        if args.mode == "visualize":
            # Load results if just visualizing
            if os.path.exists(args.output):
                with open(args.output, 'r') as f:
                    results = json.load(f)
            else:
                print(f"Error: {args.output} not found. Run with --mode compare first.")
                return
        visualize_comparison(results)

if __name__ == "__main__":
    main()


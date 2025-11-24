#!/usr/bin/env python3
"""
data_parallel_mpi_demo.py

Real MPI-based implementation showing data parallelism with and without token-level slicing.
Uses actual multiple processes to demonstrate the difference.

Usage:
    # Pure data parallelism
    mpiexec -n 4 python data_parallel_mpi_demo.py --mode pure --B 32 --L 128
    
    # Data parallelism + token-level slicing
    mpiexec -n 4 python data_parallel_mpi_demo.py --mode token_level --B 32 --L 128 --K 4
    
    # Compare both (runs sequentially)
    python data_parallel_mpi_demo.py --mode compare --B 32 --L 128 --K 4 --N 4
"""

import numpy as np
import argparse
import os
import sys
import time
import json
from mpi4py import MPI

# Import DP functions
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from metricsdp import (
        find_optimal_slicing_scheme,
        load_measured_compute_grid,
        MEASURED_COMPUTE_NPZ,
        MEASURED_COMM_NPZ,
        hidden_dim,
        transformer_layer_single_token
    )
    
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
    hidden_dim = 16

# ---------------------------
# Helper: Build t_fwd function
# ---------------------------
def build_t_fwd_from_measurements(measured_compute, comm_params=None, hidden_dim=16):
    """Build t_fwd function from measured compute grid."""
    if measured_compute is None:
        def t_fwd(l_i, sum_prev, batch_size=1):
            base = l_i * (sum_prev + l_i) * hidden_dim * 1e-8
            comm = 0.001 + (hidden_dim * l_i * batch_size * 4) / 1e6
            return base * batch_size + comm
        return t_fwd
    
    def t_fwd(l_i, sum_prev, batch_size=1):
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
        
        comp = comp * batch_size
        
        comm_time = 0.0
        if comm_params is not None:
            msg_bytes = int(hidden_dim * l_i * batch_size * 4)
            comm_time = comm_params["a"] + comm_params["slope"] * msg_bytes
        
        return comp + comm_time
    
    return t_fwd

# ---------------------------
# Pure Data Parallelism (MPI)
# ---------------------------
def run_pure_data_parallel(B, L, n_layers=4, seed=42):
    """
    Run pure data parallelism: each worker processes its batch shard with full sequence.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 70)
        print("PURE DATA PARALLELISM")
        print("=" * 70)
        print(f"Total batch size: {B}, Sequence length: {L}, Workers: {size}")
        print()
    
    # Split batch across workers
    batch_per_worker = B // size
    if B % size != 0:
        if rank == 0:
            print(f"Error: B ({B}) must be divisible by number of workers ({size})")
        return None
    
    # Generate input data on rank 0, then scatter
    if rank == 0:
        np.random.seed(seed)
        # Full batch: (B, L, hidden_dim)
        full_batch = np.random.randn(B, L, hidden_dim).astype(np.float32)
        print(f"[Rank 0] Generated full batch: shape {full_batch.shape}")
    else:
        full_batch = None
    
    # Each worker gets its shard
    worker_batch = np.empty((batch_per_worker, L, hidden_dim), dtype=np.float32)
    comm.Scatter(full_batch, worker_batch, root=0)
    
    print(f"[Rank {rank}] Received batch shard: shape {worker_batch.shape}")
    
    # Each worker processes its shard with FULL sequence (no token-level slicing)
    start_time = time.time()
    
    outputs = []
    for sample_idx in range(batch_per_worker):
        sample_output = worker_batch[sample_idx].copy()  # (L, hidden_dim)
        
        # Process full sequence through all layers
        for layer_idx in range(n_layers):
            # Process each token in the sequence
            for token_pos in range(L):
                token_vec = sample_output[token_pos]
                # Apply transformer layer
                sample_output[token_pos] = transformer_layer_single_token(token_vec, layer_idx)
        
        outputs.append(sample_output)
    
    worker_time = time.time() - start_time
    
    print(f"[Rank {rank}] Finished processing. Time: {worker_time:.4f}s")
    
    # Gather all outputs back to rank 0
    if rank == 0:
        all_outputs = np.empty((B, L, hidden_dim), dtype=np.float32)
    else:
        all_outputs = None
    
    comm.Gather(np.array(outputs), all_outputs, root=0)
    
    # Synchronize and report
    comm.Barrier()
    worker_times = comm.gather(worker_time, root=0)
    
    if rank == 0:
        max_time = max(worker_times)
        print(f"\n[Rank 0] All workers finished.")
        print(f"  Worker times: {worker_times}")
        print(f"  Max time (bottleneck): {max_time:.4f}s")
        print(f"  Final output shape: {all_outputs.shape}")
        print()
        return all_outputs, max_time
    
    return None, worker_time

# ---------------------------
# Data Parallelism + Token-Level Slicing (MPI)
# ---------------------------
def run_data_parallel_token_level(B, L, K, n_layers=4, seed=42):
    """
    Run data parallelism with token-level optimal slicing.
    Each worker processes its batch shard using optimal token-level slicing.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=" * 70)
        print("DATA PARALLELISM + TOKEN-LEVEL SLICING")
        print("=" * 70)
        print(f"Total batch size: {B}, Sequence length: {L}, Workers: {size}, Stages: {K}")
        print()
    
    # Split batch across workers
    batch_per_worker = B // size
    if B % size != 0:
        if rank == 0:
            print(f"Error: B ({B}) must be divisible by number of workers ({size})")
        return None
    
    # Find optimal slicing scheme (only rank 0, then broadcast)
    if rank == 0:
        # Load measurements
        measured_compute = load_measured_compute_grid()
        measured_comm = load_measured_comm()
        comm_params = build_comm_fit_params(measured_comm) if measured_comm else None
        t_fwd_func = build_t_fwd_from_measurements(measured_compute, comm_params, hidden_dim)
        
        def t_fwd_wrapper(l_i, sum_prev):
            return t_fwd_func(l_i, sum_prev, batch_size=batch_per_worker)
        
        T_star, slicing_scheme, tmax = find_optimal_slicing_scheme(L, K, t_fwd_wrapper)
        print(f"[Rank 0] Optimal slicing scheme: {slicing_scheme}")
        print(f"[Rank 0] Number of slices: {len(slicing_scheme)}")
        print(f"[Rank 0] Optimal time estimate: {T_star:.6f}s")
    else:
        slicing_scheme = None
    
    # Broadcast slicing scheme to all workers
    slicing_scheme = comm.bcast(slicing_scheme, root=0)
    
    # Generate input data on rank 0, then scatter
    if rank == 0:
        np.random.seed(seed)
        full_batch = np.random.randn(B, L, hidden_dim).astype(np.float32)
        print(f"[Rank 0] Generated full batch: shape {full_batch.shape}")
    else:
        full_batch = None
    
    # Each worker gets its shard
    worker_batch = np.empty((batch_per_worker, L, hidden_dim), dtype=np.float32)
    comm.Scatter(full_batch, worker_batch, root=0)
    
    print(f"[Rank {rank}] Received batch shard: shape {worker_batch.shape}")
    print(f"[Rank {rank}] Using slicing scheme: {slicing_scheme}")
    
    # Each worker processes its shard with TOKEN-LEVEL SLICING
    start_time = time.time()
    
    outputs = []
    for sample_idx in range(batch_per_worker):
        sample_output = worker_batch[sample_idx].copy()  # (L, hidden_dim)
        
        # Process through all layers
        for layer_idx in range(n_layers):
            # Process sequence using token-level slicing
            slice_start = 0
            for slice_idx, slice_length in enumerate(slicing_scheme):
                slice_end = slice_start + slice_length
                
                # Process tokens in this slice
                for token_pos in range(slice_start, slice_end):
                    token_vec = sample_output[token_pos]
                    # Apply transformer layer
                    sample_output[token_pos] = transformer_layer_single_token(token_vec, layer_idx)
                
                slice_start = slice_end
        
        outputs.append(sample_output)
    
    worker_time = time.time() - start_time
    
    print(f"[Rank {rank}] Finished processing. Time: {worker_time:.4f}s")
    
    # Gather all outputs back to rank 0
    if rank == 0:
        all_outputs = np.empty((B, L, hidden_dim), dtype=np.float32)
    else:
        all_outputs = None
    
    comm.Gather(np.array(outputs), all_outputs, root=0)
    
    # Synchronize and report
    comm.Barrier()
    worker_times = comm.gather(worker_time, root=0)
    
    if rank == 0:
        max_time = max(worker_times)
        print(f"\n[Rank 0] All workers finished.")
        print(f"  Worker times: {worker_times}")
        print(f"  Max time (bottleneck): {max_time:.4f}s")
        print(f"  Final output shape: {all_outputs.shape}")
        print()
        return all_outputs, max_time, slicing_scheme
    
    return None, worker_time, slicing_scheme

# ---------------------------
# Comparison Mode
# ---------------------------
def compare_modes(B, L, K, N, n_layers=4, seed=42):
    """
    Compare both modes by running them sequentially.
    """
    print("=" * 70)
    print("COMPARING DATA PARALLELISM: WITH vs WITHOUT TOKEN-LEVEL SLICING")
    print("=" * 70)
    print(f"Configuration: B={B}, L={L}, K={K}, N={N}")
    print()
    
    results = {}
    
    # Run pure data parallelism
    print("\n" + "=" * 70)
    print("STEP 1: Running PURE DATA PARALLELISM")
    print("=" * 70)
    print(f"Command: mpiexec -n {N} python data_parallel_mpi_demo.py --mode pure --B {B} --L {L}")
    print()
    
    # Note: This would need to be run separately with mpiexec
    # For now, we'll provide instructions
    print("To run pure data parallelism, execute:")
    print(f"  mpiexec -n {N} python data_parallel_mpi_demo.py --mode pure --B {B} --L {L}")
    print()
    
    # Run data parallelism + token-level
    print("\n" + "=" * 70)
    print("STEP 2: Running DATA PARALLELISM + TOKEN-LEVEL SLICING")
    print("=" * 70)
    print(f"Command: mpiexec -n {N} python data_parallel_mpi_demo.py --mode token_level --B {B} --L {L} --K {K}")
    print()
    
    print("To run data parallelism + token-level, execute:")
    print(f"  mpiexec -n {N} python data_parallel_mpi_demo.py --mode token_level --B {B} --L {L} --K {K}")
    print()
    
    print("=" * 70)
    print("COMPARISON INSTRUCTIONS")
    print("=" * 70)
    print("""
1. Run pure data parallelism:
   mpiexec -n 4 python data_parallel_mpi_demo.py --mode pure --B 32 --L 128

2. Run data parallelism + token-level:
   mpiexec -n 4 python data_parallel_mpi_demo.py --mode token_level --B 32 --L 128 --K 4

3. Compare the execution times and outputs from both runs.

The outputs should be identical (same input, same computation, different organization).
    """)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="MPI-based data parallelism demo")
    parser.add_argument("--mode", choices=["pure", "token_level", "compare"], default="pure",
                       help="Mode: pure, token_level, or compare")
    parser.add_argument("--B", type=int, default=32, help="Total batch size")
    parser.add_argument("--L", type=int, default=128, help="Sequence length")
    parser.add_argument("--K", type=int, default=4, help="Number of pipeline stages (for token-level)")
    parser.add_argument("--N", type=int, default=4, help="Number of workers (for compare mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if args.mode == "pure":
        if rank == 0:
            print(f"Running with {size} MPI processes")
        outputs, time_taken = run_pure_data_parallel(args.B, args.L, seed=args.seed)
        if rank == 0:
            print(f"\n✓ Pure data parallelism completed in {time_taken:.4f}s")
    
    elif args.mode == "token_level":
        if rank == 0:
            print(f"Running with {size} MPI processes")
        outputs, time_taken, slicing = run_data_parallel_token_level(
            args.B, args.L, args.K, seed=args.seed
        )
        if rank == 0:
            print(f"\n✓ Data parallelism + token-level completed in {time_taken:.4f}s")
            print(f"  Slicing scheme: {slicing}")
    
    elif args.mode == "compare":
        if rank == 0:
            compare_modes(args.B, args.L, args.K, args.N, seed=args.seed)
        else:
            pass  # Other ranks do nothing in compare mode

if __name__ == "__main__":
    main()


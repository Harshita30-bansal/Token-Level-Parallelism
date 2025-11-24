# Token-Level-Parallelism

This project implements and analyzes token-level parallelism for large transformer models — a technique inspired by TeraPipe that splits tokens, not microbatches, across pipeline stages.

Traditional pipeline parallelism divides a batch into microbatches, which still leaves pipeline bubbles and under-utilized devices. Our approach instead slices the sequence dimension itself. Each layer processes the input in carefully chosen token chunks so that all stages stay busy for more of the forward pass.

To choose these chunks, we implement a dynamic-programming (DP) token-slicing algorithm.
It uses two measured cost surfaces:

a compute grid that models forward-pass cost as sequence length grows

a communication profile that models MPI-style message latency and bandwidth

Because the DP algorithm explores all valid slice boundaries under these cost constraints, it finds a sequence of token slices that minimizes total end-to-end latency while respecting communication limits.
One of the key observations from our results is that the optimal slices naturally align with the MPI eager-message region, giving lower communication overhead.

The project then compares:

Data Parallelism

Pipeline Parallelism

Combined Parallelism

DP-optimized Token-Level Parallelism

Experiments show that token-level DP slicing significantly reduces latency compared to naïve pipeline execution.
We also provide plots showing the compute surface, communication regions, DP reconstruction path, and slice-size distribution — all included in the project presentation.

For full details, motivation, figures, and results, see the project report:
📄 NHPC_PRESENTATION.pptx (included in this repository).

This work demonstrates that token-level parallelism is an effective way to reduce pipeline bubbles and improve utilization when training or running large models on multi-device systems.

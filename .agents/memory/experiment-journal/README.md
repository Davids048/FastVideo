# Experiment Journal

Living log of all experiments. Each entry captures what was tried, the result,
and any insights. Newest entries go at the top.

## [2026-05-29] Experiment: ltx2-distilled-sequence-parallel-8x3-profile

- **Hypothesis**: Moving LTX-2 distilled profiling to low-res 8 steps plus high-res/refine 3 steps will be slower than the prior 5+2 setup, but 4-GPU SP plus stack overhead reductions may recover generation latency to the requested <=4.2s target.
- **Config**: model=FastVideo/LTX2-Distilled-Diffusers, lr=N/A, sp_size=1/2/4 baselines and 4 optimized, gpus=1/2/4, script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py
- **W&B run**: N/A
- **Duration**: Baselines and completed trials used 12 runs with 2 warmups except the no-VAE trial, which used 5 runs with 2 warmups. CUDA-node tests and the final 4-GPU profile used escalated Slurm overlap commands on job 4745 because sandboxed commands could not reach the node allocation.
- **Key metrics**: 1gpu baseline gen=6.534s e2e=6.634s sr=4.112s; 2gpu baseline gen=5.541s e2e=5.637s sr=2.641s; 4gpu baseline gen=4.335s e2e=4.433s sr=1.496s; best VAE-decode-included optimized 4gpu gen=4.207s e2e=4.207s; best latency-only decode-skip 4gpu gen=4.117s e2e=4.117s
- **Checkpoint**: N/A
- **Insight**: The <=4.2s target is achieved for the no-save/no-return latency-only contract by skipping discarded VAE pixel decode in `DecodingStage`; visible output paths still decode pixels. If a future target requires saved videos or returned frames, use the VAE-decode-included 4.207s near-miss as the honest starting point. `reduce-overhead` compile mode failed with a CUDA graph overwritten-tensor error, disabling VAE compile regressed generation to 5.521s, and a temporary TQDM-disable patch regressed to 4.411s and was reverted.
- **Status**: completed
- **Related lessons**: Detailed report and result snapshot: `.agents/memory/experiment-journal/2026-05-29_ltx2_sequence_parallel_8x3_profile.md`, `.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_8x3_profile_results.json`

## [2026-05-29] Experiment: ltx2-distilled-sequence-parallel-profile

- **Hypothesis**: Ulysses sequence parallelism on 2/4 GPUs with torch.compile should match or beat the supplied 1-GPU LTX-2 distilled profiling baseline; disabling all FP4 paths may isolate whether NVFP4 linear layers help or hurt this workload.
- **Config**: model=FastVideo/LTX2-Distilled-Diffusers, lr=N/A, sp_size=2 and 4, gpus=2 and 4, script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py
- **W&B run**: N/A
- **Duration**: Agent session about 39 minutes; full profiles used 12 runs each with 2 warmups on Slurm job 4745.
- **Key metrics**: fp4_g4 e2e=5.269s sr=1.185s, fp4_g2 e2e=5.304s sr=1.438s, no_fp4_g4 e2e=3.872s sr=1.007s, no_fp4_g2 e2e=4.536s sr=1.769s
- **Checkpoint**: N/A
- **Insight**: No-FP4 4-GPU SP was the only run that beat the provided 1-GPU 4.20s e2e baseline. FP4-linear SP improved SR latency versus the baseline but made total e2e slower. SP compile requires `fullgraph=False` because `LTXDistributedAttention.forward` is `torch.compiler.disable()`-wrapped.
- **Status**: completed
- **Related lessons**: Detailed report and committed JSON snapshots: `.agents/memory/experiment-journal/2026-05-29_ltx2_sequence_parallel_profile.md`

<!-- TEMPLATE — copy and fill for each new experiment:

## [YYYY-MM-DD] Experiment: <name>
- **Hypothesis**: <what you expected to learn>
- **Config**: model=..., lr=..., sp_size=..., gpus=..., script=...
- **W&B run**: <run_id or URL>
- **Duration**: <total wall time>
- **Key metrics**: loss=..., step_time=..., grad_norm=...
- **Checkpoint**: <path>
- **Insight**: <what was learned>
- **Status**: running | completed | failed | abandoned
- **Related lessons**: `.agents/lessons/<filename>.md`

-->

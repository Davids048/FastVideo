# Experiment Journal

Living log of all experiments. Each entry captures what was tried, the result,
and any insights. Newest entries go at the top.

## [2026-05-30] Experiment: ltx2-generation-speed-sweep

- **Hypothesis**: With output and compile settings fixed, the reduced matrix will identify whether GPU/SP count, FP4 linear quantization, denoising schedule, or compile fullgraph is the actual latency driver for LTX-2 generation.
- **Config**: model=FastVideo/LTX2-Distilled-Diffusers, lr=N/A, sp_size=1/2/4, gpus=1/2/4, script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py, schedules=5+2 and 8+3, fp4_linear=false/true, compile_fullgraph=false/true, fixed=nvfp4_fa4=false, stage_logging=true, save_video=false, return_frames=true, compile_dit=true, compile_vae=true, compile_text_encoder=true, backend=inductor, dynamic=false, tp_size=1, attention_backend=FLASH_ATTN, num_runs=12, warmup_runs=2
- **W&B run**: N/A
- **Duration**: 2026-05-30 19:26-20:48 UTC on Slurm job 4745, node `hpc-rack-2-8`, using serial `srun --overlap --jobid=4745` cells.
- **Key metrics**: 24/24 cells succeeded; best 5+2=`ltx2_speed_s5p2_g4_fp4off_fgoff` e2e=2.590s gen=2.490s sr=0.950s; best 8+3=`ltx2_speed_s8p3_g4_fp4off_fgoff` e2e=3.660s gen=3.566s sr=1.407s; no-FP4 4-GPU was fastest for both schedules; no MP4 files were written because `save_video=false`.
- **Checkpoint**: N/A
- **Insight**: No-FP4 scales best across 1/2/4 GPU SP for both schedules. FP4 helps single-GPU refine latency but hurts multi-GPU total latency because base denoising becomes slower. Fullgraph now runs successfully but is neutral to mildly negative for the fastest cells. SR latency is a subset of generation time, not an additional term to add after generation.
- **Status**: completed
- **Related lessons**: Detailed report and metrics artifact: `.agents/memory/experiment-journal/2026-05-30_ltx2_generation_speed_sweep.md`, `.agents/memory/experiment-journal/artifacts/2026-05-30_ltx2_generation_speed_sweep_results.json`

## [2026-05-30] Experiment: ltx2-generation-speed-design-space-cleanup

- **Hypothesis**: Reducing the LTX-2 speed matrix to denoising steps, GPUs/SP, FP4 linear, and fullgraph while fixing output and compile settings will make the next profiling pass comparable and easier to interpret.
- **Config**: model=FastVideo/LTX2-Distilled-Diffusers, lr=N/A, sp_size=1/2/4, gpus=1/2/4, script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py, tuning_dimensions=denoising_steps(5+2|8+3), gpus_sp(1|2|4), fp4_linear(false|true), compile_fullgraph(false|true); fixed=nvfp4_fa4=false, stage_logging=true, save_video=false, return_frames=true, compile_dit=true, compile_vae=true, compile_text_encoder=true, backend=inductor, dynamic=false, mode=default, tp_size=1
- **W&B run**: N/A
- **Duration**: Side-conversation cleanup pass on 2026-05-30; no model inference launched.
- **Key metrics**: 22 deprecated non-tracing run directories moved under `outputs_video/deprecated runs/`; tracing/profiler outputs intentionally left in place.
- **Checkpoint**: N/A
- **Insight**: The active generation-speed design space is now four dimensions: denoising steps, GPUs/SP simplified to `1`, `2`, `4`, FP4 linear, and compile fullgraph. Historical incompatible runs remain available under the deprecated folder but should not be used as current-matrix measurements.
- **Status**: completed
- **Related lessons**: Current landing note: `.agents/memory/experiment-journal/2026-05-30_ltx2_generation_speed_design_space_cleanup.md`

## [2026-05-29] Experiment: ltx2-distilled-block47-gpu-profile

- **Hypothesis**: A late transformer block will show whether the prior 8+3 stage split comes from SP communication overhead in low-res base denoising versus useful compute partitioning in high-res refine denoising.
- **Config**: model=FastVideo/LTX2-Distilled-Diffusers, lr=N/A, sp_size=1/2/4, gpus=1/2/4, script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py, schedule=8 base steps + 3 refine steps, block=transformer_blocks.47, fp4_linear=false, nvfp4_fa4=false, torch_compile=true, compile_text_encoder=true, compile_vae=true, save_video=false, return_frames=false, stage_logging=false
- **W&B run**: N/A
- **Duration**: Six Torch-profiler runs plus six Nsight-only runs on Slurm overlap steps using holder job 4745 and temporary allocation 4774; allocation 4774 was released after completion.
- **Key metrics**: block47 base rank-max mean=4.384ms on 1 GPU, 6.999ms on 2 GPUs, 7.747ms on 4 GPUs; Nsight base NCCL share=0.0%, 47.5%, 70.0%. block47 refine rank-max mean=27.992ms on 1 GPU, 16.961ms on 2 GPUs, 9.590ms on 4 GPUs; Nsight refine NCCL share=0.0%, 16.5%, 12.0%.
- **Checkpoint**: N/A
- **Insight**: The base-stage SP slowdown is communication dominated at block 47, while the refine-stage workload is large enough that SP compute reduction wins despite NCCL. No MP4 outputs were written because the run used `--no-save-video --no-return-frames`; all durable artifacts are configs, summaries, Torch traces, and Nsight reports under `outputs_video/ltx2_sp_profile/block47_gpu_profile/`.
- **Status**: completed
- **Related lessons**: Detailed report and compact metrics: `.agents/memory/experiment-journal/2026-05-29_ltx2_block47_gpu_profile.md`, `.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_block47_gpu_profile_summary.json`

## [2026-05-29] Experiment: ltx2-distilled-sequence-parallel-fullgraph-fix

- **Hypothesis**: LTX-2 sequence-parallel fullgraph compile fails only because the LTX-2-specific distributed attention forward is explicitly `torch.compiler.disable()` wrapped; removing that forced graph break should let `--compile-fullgraph` run.
- **Config**: model=FastVideo/LTX2-Distilled-Diffusers, lr=N/A, sp_size=4, gpus=4, script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py, schedule=8 base denoise steps + 3 refine steps, fp4_linear=false, nvfp4_fa4=false, torch_compile=true, compile_fullgraph=true
- **W&B run**: N/A
- **Duration**: One failing reproduction and one successful cold fullgraph smoke on Slurm job 4745, node `hpc-rack-2-8`, using `/home/hal-jundas/venvs/fv-shared`.
- **Key metrics**: before patch failed with `torch._dynamo.exc.Unsupported: Skip inlining torch.compiler.disable()d function LTXDistributedAttention.forward`; after patch exited 0 with cold generation=502.300934055s and e2e=502.301681672s. This is compile/correctness evidence, not warmed latency.
- **Checkpoint**: N/A
- **Insight**: Fullgraph was previously disabled for SP because of a forced Dynamo graph break, not because of a measured performance preference. Removing `@torch.compiler.disable` from `fastvideo/models/dits/ltx2.py::LTXDistributedAttention.forward` lets the tested 4-GPU LTX-2 SP path run with `--compile-fullgraph`.
- **Status**: completed
- **Related lessons**: Detailed report and committed config/summary snapshots: `.agents/memory/experiment-journal/2026-05-29_ltx2_fullgraph_sequence_parallel_fix.md`, `.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_fullgraph_sp_fix/after_remove_disable_ltxdistattn_g4_sp4/profile_config.json`, `.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_fullgraph_sp_fix/after_remove_disable_ltxdistattn_g4_sp4/profile_summary.json`

## [2026-05-29] Experiment: ltx2-distilled-sequence-parallel-8x3-profile

- **Hypothesis**: Moving LTX-2 distilled profiling to low-res 8 steps plus high-res/refine 3 steps will be slower than the prior 5+2 setup, but 4-GPU SP plus stack overhead reductions may recover generation latency to the requested <=4.2s target.
- **Config**: model=FastVideo/LTX2-Distilled-Diffusers, lr=N/A, sp_size=1/2/4 baselines and 4 optimized, gpus=1/2/4, script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py
- **W&B run**: N/A
- **Duration**: Baselines and completed trials used 12 runs with 2 warmups except the no-VAE trial, which used 5 runs with 2 warmups. CUDA-node tests and the final 4-GPU profile used escalated Slurm overlap commands on job 4745 because sandboxed commands could not reach the node allocation.
- **Key metrics**: 1gpu baseline gen=6.534s e2e=6.634s sr=4.112s; 2gpu baseline gen=5.541s e2e=5.637s sr=2.641s; 4gpu baseline gen=4.335s e2e=4.433s sr=1.496s; rejected latency-only decode-skip 4gpu gen=4.117s e2e=4.117s; corrected real-output return-frames 4gpu gen=4.337s e2e=4.437s
- **Checkpoint**: N/A
- **Insight**: Review rejected the target-achieved interpretation because skipping VAE pixel decode does not preserve the real video-frame-producing workload. The corrected run uses `--no-save-video --return-frames`, keeps decoded frames, and misses the <=4.2s generation target by 0.137s. Torch-profiler attribution shows the bottleneck order is base denoising, refine denoising, then VAE decode; `PostDecodeFrameProcessStage` explains e2e overhead but is after recorded generation time.
- **Status**: corrected; target not achieved for real decoded frames
- **Related lessons**: Detailed report, review decision, follow-up, and result snapshot: `.agents/memory/experiment-journal/2026-05-29_ltx2_sequence_parallel_8x3_profile.md`, `.agents/memory/experiment-journal/2026-05-29_ltx2_8x3_profile_review_decision.md`, `.agents/memory/experiment-journal/2026-05-29_ltx2_8x3_profile_review_followup.md`, `.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_8x3_profile_results.json`

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

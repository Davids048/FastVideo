# LTX-2 Distilled Block 47 GPU Profile

Date: 2026-05-29

Status: completed

## Goal

The prior LTX-2 distilled 8+3 stage breakdown showed an unusual split: the
base denoising stage was fastest on 1 GPU, while the refine denoising stage
scaled strongly on 2 and 4 GPUs. This run profiled the last transformer block,
`transformer_blocks.47`, separately in the base and refine stages across
1, 2, and 4 GPUs.

The mental model after this run is:

```text
base block at lower resolution
  1 GPU: compute only
  SP: smaller local sequence, but NCCL all-gather/sendrecv dominates

refine block at higher resolution
  1 GPU: much larger compute
  SP: communication exists, but compute reduction dominates
```

That explains why stage-level base denoising was slower under SP, while
stage-level refine denoising improved with SP.

## Code Changes

`fastvideo/models/dits/ltx2.py` now has env-gated instrumentation for a selected
LTX-2 block and stage. The gates are:

```text
FASTVIDEO_LTX2_BLOCK_PROFILE_INDEX=47
FASTVIDEO_LTX2_BLOCK_PROFILE_STAGE=base|refine
FASTVIDEO_LTX2_BLOCK_PROFILE_SKIP_OCCURRENCES=<n>
FASTVIDEO_LTX2_BLOCK_PROFILE_ACTIVE_OCCURRENCES=<n>
FASTVIDEO_LTX2_BLOCK_PROFILE_CAPTURE_RANGE=0|1
```

When enabled, the matching block call is wrapped in both
`torch.profiler.record_function("fastvideo.ltx2.block47.<stage>")` and an
explicit `torch.cuda.nvtx.range_push/pop` range with the same name. The stage
comes from `forward_context.forward_batch.extra["ltx2_fp4_stage_profile"]`,
which is set by the LTX-2 denoising code to `base` or `refine`.

The same file also sets
`LTX2Transformer3DModel._compile_conditions = LTX2VideoConfig()._compile_conditions`.
This keeps transformer compilation at the block level. That mattered because an
outer transformer fullgraph compile tried to trace `torch.cuda.nvtx.range_push`
and failed; block-level compile keeps the profiling wrapper outside each
compiled block. The SP=1 profile configs still record `compile_fullgraph=true`,
but with `_compile_conditions` that applies to each selected compiled submodule,
not to one giant outer transformer graph. SP=2 and SP=4 kept the baseline script
default `compile_fullgraph=false`.

`examples/inference/basic/basic_ltx2_distilled_fast_profile.py` now records the
block profiling env vars and torch profiler env vars in `profile_config.json`.

## Shared Config

All six main Torch-profiler runs used:

```text
script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py
model_id=FastVideo/LTX2-Distilled-Diffusers
validation_json=examples/training/finetune/ltx2/validation.json
prompt=first validation prompt
num_frames=121
num_inference_steps=8
refine_num_inference_steps=3
num_runs=4
warmup_runs=2
avg_window=2
fp4_linear=false
nvfp4_fa4=false
torch_compile=true
compile_text_encoder=true
compile_vae=true
stage_logging=false
save_video=false
return_frames=false
tp_size=1
```

The Torch profiler schedule was:

```text
FASTVIDEO_TORCH_PROFILER_WAIT_STEPS=2
FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS=0
FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS=2
FASTVIDEO_TORCH_PROFILE_REGIONS=profiler_region_inference_forward
FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES=0
FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
FASTVIDEO_TORCH_PROFILER_WITH_STACK=0
FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=0
```

The block occurrence windows were:

```text
base:   skip 16 block occurrences, record/filter next 16
refine: skip 6 block occurrences, record/filter next 6
```

These counts correspond to two warmup generations followed by two measured
generations: 8 base steps per generation and 3 refine steps per generation.

## Compute And Execution

Job `4745` was inspected before use. It was holding `hpc-rack-2-8` with 4 B200
GPUs via `/home/hal-jundas/codes/DiffusionNFT/scripts/slurm/train_sd3_multi_reward_hold.slurm`.
That script leaves the allocation alive with an infinite sleep when
`HOLD_AFTER_TRAIN=1`, so overlap steps do not release the node.

The profiles ran with `srun --overlap`. A second allocation, job `4774` on
`hpc-rack-2-2`, was requested to run cases in parallel. It was released after
the profiling finished. A final `squeue -j 4745,4774` showed only job `4745`
still running.

Python environment:

```text
/home/hal-jundas/venvs/fv-shared/bin/python
```

## Profiler Caveats

Torch profiler and Nsight Systems could not be active in the same process
because CUPTI reported multiple subscribers. The final Nsight pass therefore
ran separately with torch profiler disabled.

Named NVTX capture with `--capture-range=nvtx` was unreliable under the
multi-process executor and produced no usable report. The final Nsight pass used
full-process capture and then exported filtered stats with:

```text
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
  --filter-nvtx fastvideo.ltx2.block47.<stage>
```

The Torch trace table below uses rank-max mean block range duration from the
`record_function`/NVTX ranges. Treat that as a per-rank block range timing from
the trace, not a full CUDA critical-path proof. The Nsight table is the kernel
attribution sample inside the filtered block range. Nsight memops reports
returned no data for all six cases.

Stage logging was intentionally disabled, so `profile_summary.json` has
`stage_times={}`, `sr_forward_latency=null`, and `non_stage_overhead=null` for
these six runs.

## Output Layout

Main profile outputs are in:

```text
outputs_video/ltx2_sp_profile/block47_gpu_profile/<case>/
```

Each case has:

```text
profile_config.json
profile_summary.json
profile.log
torch_traces/*.pt.trace.json.gz
nsys/<case>.nsys-rep
nsys/<case>.sqlite
nsys/<case>_stats.txt
```

The Nsight-only script run summaries are in:

```text
outputs_video/ltx2_sp_profile/block47_gpu_profile_nsys_runs/<case>_nsys/
```

No output videos were stored. All runs used `--no-save-video --no-return-frames`.
The logs still print nominal target paths such as:

```text
outputs_video/ltx2_sp_profile/block47_gpu_profile/<case>/videos/output_ltx2_basic_t2v_run_<n>.mp4
outputs_video/ltx2_sp_profile/block47_gpu_profile_nsys_runs/<case>_nsys/videos/output_ltx2_basic_t2v_run_<n>.mp4
```

Those are not real files when `save_video=false`.

## Top-Level Run Metrics

These are sanity checks for the full profile-script run, not stage timings.
Each row averages runs 3 and 4 after two warmups.

| Case | GPUs / SP | Captured block stage | Avg generation (s) | Avg e2e (s) | Stage times |
| --- | ---: | --- | ---: | ---: | --- |
| `g1_sp1_base_block47` | 1 / 1 | base | 7.732579491 | 7.732776451 | unavailable |
| `g2_sp2_base_block47` | 2 / 2 | base | 7.781213711 | 7.781400191 | unavailable |
| `g4_sp4_base_block47` | 4 / 4 | base | 6.854927492 | 6.855182933 | unavailable |
| `g1_sp1_refine_block47` | 1 / 1 | refine | 7.833214986 | 7.833391787 | unavailable |
| `g2_sp2_refine_block47` | 2 / 2 | refine | 7.734417262 | 7.734741359 | unavailable |
| `g4_sp4_refine_block47` | 4 / 4 | refine | 6.649028208 | 6.649353168 | unavailable |

## Torch Trace Block 47 Results

The table reports the average of the max duration across ranks for each block
occurrence. Base cases have 16 measured occurrences per rank; refine cases have
6 measured occurrences per rank.

| Case | Trace files | Events per rank | Rank mean ms | Rank-max mean ms | Rank-max p50 ms | Rank-max min/max ms |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `g1_sp1_base_block47` | 1 | 16 | 4.384 | 4.384 | 4.378 | 4.234 / 4.589 |
| `g2_sp2_base_block47` | 2 | 16, 16 | 6.921, 6.830 | 6.999 | 6.916 | 6.843 / 7.622 |
| `g4_sp4_base_block47` | 4 | 16, 16, 16, 16 | 7.070, 7.319, 7.176, 7.624 | 7.747 | 7.584 | 7.259 / 8.672 |
| `g1_sp1_refine_block47` | 1 | 6 | 27.992 | 27.992 | 28.232 | 26.948 / 28.797 |
| `g2_sp2_refine_block47` | 2 | 6, 6 | 16.929, 16.906 | 16.961 | 16.888 | 16.671 / 17.571 |
| `g4_sp4_refine_block47` | 4 | 6, 6, 6, 6 | 7.607, 7.883, 9.590, 7.470 | 9.590 | 9.635 | 9.274 / 10.032 |

## Nsight Filtered Kernel Attribution

These rows summarize `nsys stats` output filtered to
`fastvideo.ltx2.block47.<stage>`. Percentages are by filtered CUDA GPU kernel
time in the exported stats file.

| Case | Filtered kernel total ms | NCCL % | FlashAttention % | GEMM/nvjet % | Triton % | Dominant kernels |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `g1_sp1_base_block47` | 4.162 | 0.0 | 29.4 | 55.8 | 14.8 | nvjet 34.8%, FlashAttention 23.3%, nvjet 16.7% |
| `g2_sp2_base_block47` | 5.519 | 47.5 | 12.2 | 28.3 | 12.0 | NCCL all-gather 28.6%, NCCL sendrecv 18.9%, nvjet 16.5% |
| `g4_sp4_base_block47` | 6.132 | 70.0 | 6.2 | 17.1 | 6.7 | NCCL sendrecv 36.7%, NCCL all-gather 33.4%, FlashAttention 5.1% |
| `g1_sp1_refine_block47` | 30.494 | 0.0 | 47.3 | 42.2 | 10.5 | FlashAttention 43.8%, nvjet 36.4%, nvjet 5.2% |
| `g2_sp2_refine_block47` | 17.382 | 16.5 | 36.1 | 32.6 | 14.8 | FlashAttention 33.5%, nvjet 23.5%, NCCL sendrecv 13.4% |
| `g4_sp4_refine_block47` | 9.054 | 12.0 | 35.5 | 37.3 | 15.1 | FlashAttention 32.8%, nvjet 14.0%, nvjet 7.7%, NCCL sendrecv 7.2% |

The base-stage SP result is the key evidence: NCCL rises from 0% on 1 GPU to
47.5% on 2 GPUs and 70.0% on 4 GPUs for block 47, while the trace range gets
slower. The refine-stage SP result is the opposite: the block range falls from
27.992 ms on 1 GPU to 9.590 ms on 4 GPUs, and NCCL remains a minority of the
filtered kernel time.

## Relationship To The Prior Stage Breakdown

The prior 8+3 stage breakdown is documented in:

```text
.agents/exploration/2026-05-29_ltx2_8x3_gpu_stage_breakdown.md
```

It measured:

| Stage | 1 GPU | 2 GPUs | 4 GPUs |
| --- | ---: | ---: | ---: |
| base denoising | 2.005970s | 2.469087s | 2.407451s |
| refine denoising | 4.076190s | 2.606739s | 1.461364s |

Block 47 matches that directionally. The base block is slower under SP because
communication overwhelms reduced compute at the lower base resolution. The
refine block is much larger on 1 GPU, and SP reduces enough compute for the
communication to pay off.

## Timing Model Note

`video_generation_time` in the profile script is not "low-resolution generation
only." It is the measured `generate_video` latency for the selected output mode,
which includes base denoise, upsample, refine denoise, audio/VAE decode work
that remains enabled, and synchronization/materialization inside the generator.

`sr_forward_latency` is a nested subset derived from stage timings when stage
logging is enabled. It should not be added to generation time as if the two are
independent sequential components. If an earlier summary appeared to have
`average e2e < average generation + average SR`, that is expected because SR is
already inside generation and because the stage timers and outer timers have
different scopes. In this block-profile run, SR latency is unavailable because
`--no-stage-logging` was used.

## Source Files And Config Paths

Plan and setup:

```text
.agents/exploration/2026-05-29_ltx2_block47_gpu_profiling_plan.md
.agents/exploration/2026-05-29_ltx2_8x3_gpu_stage_breakdown.md
```

Main profile configs and summaries:

```text
outputs_video/ltx2_sp_profile/block47_gpu_profile/g1_sp1_base_block47/profile_config.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g1_sp1_base_block47/profile_summary.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g2_sp2_base_block47/profile_config.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g2_sp2_base_block47/profile_summary.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g4_sp4_base_block47/profile_config.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g4_sp4_base_block47/profile_summary.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g1_sp1_refine_block47/profile_config.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g1_sp1_refine_block47/profile_summary.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g2_sp2_refine_block47/profile_config.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g2_sp2_refine_block47/profile_summary.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g4_sp4_refine_block47/profile_config.json
outputs_video/ltx2_sp_profile/block47_gpu_profile/g4_sp4_refine_block47/profile_summary.json
```

Nsight reports and stats:

```text
outputs_video/ltx2_sp_profile/block47_gpu_profile/<case>/nsys/<case>.nsys-rep
outputs_video/ltx2_sp_profile/block47_gpu_profile/<case>/nsys/<case>.sqlite
outputs_video/ltx2_sp_profile/block47_gpu_profile/<case>/nsys/<case>_stats.txt
```

Compact derived metrics for future agents:

```text
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_block47_gpu_profile_summary.json
```

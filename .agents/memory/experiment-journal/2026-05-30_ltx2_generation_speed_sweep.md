# LTX-2 Generation Speed Sweep

Date: 2026-05-30

Status: completed; 24/24 matrix cells succeeded.

## Goal

Run the reduced LTX-2 generation-speed matrix with comparable output and runtime
settings. The sweep replaces the old mixed-output, smoke, failed, and
hack-optimized measurements with a clean matrix over only denoising schedule,
GPU/SP count, FP4 linear quantization, and torch compile fullgraph.

The plan used for this run is committed at:

```text
ltx2_generation_speed_sweep_plan.md
```

The cleanup note that defined the reduced design space is:

```text
.agents/memory/experiment-journal/2026-05-30_ltx2_generation_speed_design_space_cleanup.md
```

## Compute And Execution

The run used Slurm job `4745` on `hpc-rack-2-8` with `srun --overlap --jobid=4745`.
The job record showed `BatchFlag=1`, `StepMgrEnabled=Yes`, `TimeLimit=UNLIMITED`,
`NodeList=hpc-rack-2-8`, `BatchHost=hpc-rack-2-8`, and four allocated GB200/B200
GPUs. The holder command was:

```text
/home/hal-jundas/codes/DiffusionNFT/scripts/slurm/train_sd3_multi_reward_hold.slurm
```

That means the node was held by the long-running batch allocation, and each
profile cell was only an overlapping Slurm step inside that allocation. Finishing
these `srun --overlap` steps did not release the node.

An idle 4-GPU GB200 node (`hpc-rack-2-6`) was visible, but it was not used. The
active plan required one active cell at a time for comparability, and splitting
the matrix after the serial runner had started would have introduced duplicate
cell and cross-node/cache variance risk.

The serial runner used per-run cache directories under `/tmp/hal-jundas/`:

```text
/tmp/hal-jundas/torchinductor-cache-<run_name>
/tmp/hal-jundas/triton-cache-<run_name>
/tmp/hal-jundas/cuda-cache-<run_name>
/tmp/hal-jundas/xdg-cache-<run_name>
```

It also unset the prior block-profiling and Torch-profiler environment toggles
before each cell so the block47 profiler instrumentation did not affect this
speed sweep.

## Fixed Config

Every successful `profile_config.json` was checked against the fixed config:

| Field                | Value                                                                      |
| -------------------- | -------------------------------------------------------------------------- |
| Model                | `FastVideo/LTX2-Distilled-Diffusers`                                       |
| `nvfp4_fa4`          | `false`                                                                    |
| Stage logging        | `true`                                                                     |
| Save video           | `false`                                                                    |
| Return frames        | `true`                                                                     |
| Torch compile        | `true`                                                                     |
| Compile text encoder | `true`                                                                     |
| Compile VAE          | `true`                                                                     |
| Compile backend      | `inductor`                                                                 |
| Compile dynamic      | `false`                                                                    |
| Compile mode         | default / `None`                                                           |
| Resolution           | `1920x1088`                                                                |
| Frames               | `121`                                                                      |
| Prompt               | first prompt in `examples/training/finetune/ltx2/validation.json`          |
| Seed                 | `10`                                                                       |
| Guidance             | `guidance_scale=1.0`, `refine_guidance_scale=1.0`, `refine_add_noise=true` |
| Attention backend    | `FLASH_ATTN`                                                               |
| TP size              | `1`                                                                        |
| Distributed executor | `mp`                                                                       |
| Protocol             | `num_runs=12`, `warmup_runs=2`, no `avg_window` override                   |

The row-specific fields also matched each run name: base/refine steps, `num_gpus`,
`sp_size`, `fp4_linear`, and `compile_fullgraph`.

## Output Locations

Raw outputs are under:

```text
outputs_video/ltx2_generation_speed_sweep/
```

Each run directory contains:

```text
profile_config.json
profile_summary.json
profile.log
profile_runs_partial.json
videos/
```

No `.mp4` files were written. The script still records a per-run `output_path`
inside each summary, but `save_video=false` means those MP4 paths are placeholders
and the decoded frames were returned in memory only because `return_frames=true`.
The `videos/` directories are therefore empty by design.

A compact metrics artifact was written to:

```text
.agents/memory/experiment-journal/artifacts/2026-05-30_ltx2_generation_speed_sweep_results.json
```

## Validation

Validation result:

```text
expected cells: 24
successful cells: 24
failed cells: 0
config mismatches: 0
saved MP4 files: 0
sweep_failure.txt files: 0
```

## Results: 5+2 Steps

| GPUs/SP | FP4 | fullgraph | e2e s | gen s |  SR s | base denoise s | refine denoise s | decode s | post s | overhead s |
| ------: | --- | --------- | ----: | ----: | ----: | -------------: | ---------------: | -------: | -----: | ---------: |
|       1 | off | off       | 4.490 | 4.396 | 2.734 |          1.245 |            2.698 |    0.330 |  0.094 |      0.026 |
|       1 | off | on        | 4.498 | 4.392 | 2.732 |          1.243 |            2.696 |    0.331 |  0.106 |      0.026 |
|       1 | on  | off       | 4.188 | 4.094 | 2.015 |          1.664 |            1.981 |    0.326 |  0.093 |      0.026 |
|       1 | on  | on        | 4.196 | 4.099 | 2.008 |          1.674 |            1.974 |    0.330 |  0.097 |      0.026 |
|       2 | off | off       | 3.263 | 3.166 | 1.655 |          1.072 |            1.621 |    0.341 |  0.096 |      0.034 |
|       2 | off | on        | 3.258 | 3.160 | 1.656 |          1.073 |            1.623 |    0.337 |  0.098 |      0.031 |
|       2 | on  | off       | 3.823 | 3.728 | 1.292 |          2.006 |            1.258 |    0.336 |  0.094 |      0.030 |
|       2 | on  | on        | 3.948 | 3.845 | 1.299 |          2.111 |            1.264 |    0.333 |  0.103 |      0.038 |
|       4 | off | off       | 2.590 | 2.490 | 0.950 |          1.108 |            0.916 |    0.334 |  0.100 |      0.034 |
|       4 | off | on        | 2.608 | 2.513 | 0.948 |          1.133 |            0.913 |    0.338 |  0.094 |      0.032 |
|       4 | on  | off       | 3.584 | 3.483 | 1.094 |          1.957 |            1.060 |    0.336 |  0.100 |      0.033 |
|       4 | on  | on        | 3.704 | 3.597 | 1.121 |          2.040 |            1.087 |    0.336 |  0.107 |      0.036 |

## Results: 8+3 Steps

| GPUs/SP | FP4 | fullgraph | e2e s | gen s |  SR s | base denoise s | refine denoise s | decode s | post s | overhead s |
| ------: | --- | --------- | ----: | ----: | ----: | -------------: | ---------------: | -------: | -----: | ---------: |
|       1 | off | off       | 6.589 | 6.495 | 4.087 |          1.994 |            4.050 |    0.330 |  0.093 |      0.026 |
|       1 | off | on        | 6.588 | 6.490 | 4.090 |          1.984 |            4.054 |    0.330 |  0.098 |      0.027 |
|       1 | on  | off       | 6.007 | 5.904 | 2.996 |          2.493 |            2.962 |    0.328 |  0.103 |      0.026 |
|       1 | on  | on        | 5.938 | 5.841 | 2.989 |          2.439 |            2.955 |    0.327 |  0.096 |      0.026 |
|       2 | off | off       | 4.800 | 4.705 | 2.472 |          1.800 |            2.438 |    0.339 |  0.095 |      0.029 |
|       2 | off | on        | 4.777 | 4.685 | 2.472 |          1.779 |            2.438 |    0.340 |  0.092 |      0.030 |
|       2 | on  | off       | 5.631 | 5.529 | 1.924 |          3.165 |            1.890 |    0.338 |  0.101 |      0.036 |
|       2 | on  | on        | 5.705 | 5.612 | 1.977 |          3.204 |            1.943 |    0.338 |  0.092 |      0.028 |
|       4 | off | off       | 3.660 | 3.566 | 1.407 |          1.727 |            1.373 |    0.337 |  0.093 |      0.033 |
|       4 | off | on        | 3.794 | 3.692 | 1.408 |          1.846 |            1.373 |    0.339 |  0.101 |      0.037 |
|       4 | on  | off       | 5.402 | 5.296 | 1.636 |          3.225 |            1.601 |    0.335 |  0.105 |      0.036 |
|       4 | on  | on        | 5.333 | 5.232 | 1.618 |          3.183 |            1.583 |    0.336 |  0.101 |      0.033 |

## Topline Findings

For `5+2`, the fastest cell was `ltx2_speed_s5p2_g4_fp4off_fgoff` with average
e2e latency `2.590s`, generation time `2.490s`, and SR forward latency `0.950s`.
The fullgraph-on counterpart was very close at `2.608s` e2e, but did not improve
latency.

For `8+3`, the fastest cell was `ltx2_speed_s8p3_g4_fp4off_fgoff` with average
e2e latency `3.660s`, generation time `3.566s`, and SR forward latency `1.407s`.
The fullgraph-on counterpart was slower at `3.794s` e2e.

No-FP4 runs scaled cleanly with GPU/SP count in both schedules. For `5+2`, no-FP4
fullgraph-off e2e moved from `4.490s` on 1 GPU to `3.263s` on 2 GPUs to `2.590s`
on 4 GPUs. For `8+3`, the same path moved from `6.589s` to `4.800s` to `3.660s`.

FP4 linear helped the single-GPU cases by reducing refine/SR latency, but it hurt
2-GPU and 4-GPU total latency. The reason is visible in the stage table: FP4
reduces `ltx2_refine_denoising_stage`, but makes `denoising_stage` much slower
in the multi-GPU cases. For example, `8+3` on 4 GPUs moves from no-FP4 e2e
`3.660s` to FP4 e2e `5.402s`/`5.333s`.

Compile fullgraph was mostly neutral to mildly negative in this matrix. It is no
longer disabled for correctness in these successful cells; it simply was not the
fastest option here.

## Fullgraph Context

The earlier reason fullgraph had to be disabled for LTX-2 SP was a forced Dynamo
graph break in `LTXDistributedAttention.forward`: that method was
`torch.compiler.disable()` wrapped, and fullgraph compile failed with a Dynamo
"Skip inlining torch.compiler.disable()d function" error. The previous fullgraph
fix removed that forced disable path, so this sweep could run both fullgraph off
and fullgraph on for every matrix cell.

In this sweep, fullgraph-off remains the best setting for the fastest cells, but
that is a measured latency result, not the old correctness blocker.

## Timing Definitions

`avg_sr_forward_latency` is not an extra phase to add after generation. It is a
subset of the generation path, specifically:

```text
ltx2_upsample_stage + ltx2_refine_denoising_stage
```

`avg_video_generation_time` is the overall generation interval reported by
`VideoGenerator`, including base denoising, SR/refine work, audio decode, and VAE
pixel decode. `avg_e2e_latency` adds the post-decode frame processing and small
bookkeeping overhead around that generation call.

So `avg_e2e_latency` can be smaller than `avg_video_generation_time +
avg_sr_forward_latency` because that sum double-counts SR. Average generation is
not average low-resolution generation. The closest low-resolution generation
component in these summaries is `stage_times.denoising_stage` plus tiny prompt
and latent preparation overhead.

## Assumptions And Modifications

No code changes were made during this sweep. The run used the current branch as
it existed after the earlier LTX-2 SP/fullgraph/profiling work.

The output contract was fixed to `--no-save-video --return-frames`. This keeps
the real decode-to-frames workload while avoiding MP4 encoding and file write
latency. Consequently, there are no output videos to inspect from this sweep.

The previous tracing/profiler outputs under `outputs_video/ltx2_sp_profile/` were
not touched. This sweep wrote only to `outputs_video/ltx2_generation_speed_sweep/`.

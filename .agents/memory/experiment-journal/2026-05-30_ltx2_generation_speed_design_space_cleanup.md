# LTX-2 Generation Speed Design Space Cleanup

Date: 2026-05-30

Status: current landing point for the next generation-speed profiling pass

## Goal

After the earlier LTX-2 speed and profiling runs, the config space had too many
moving parts. This cleanup narrowed the next experiment matrix to a small set of
tuning dimensions, made the rest fixed, and moved old incompatible run outputs
out of the active output tree.

The current plan is:

```text
generation-speed matrix
  vary only denoising steps, GPU/SP count, fp4_linear, and fullgraph
  keep output contract and compile/component settings fixed
  leave tracing/profiler artifacts untouched for diagnosis
```

## Tuning Dimensions

| Dimension name          | Options         |
| ----------------------- | --------------- |
| Denoising steps         | `5+2`, `8+3`    |
| GPUs / SP               | `1`, `2`, `4`   |
| FP4 linear quantization | `false`, `true` |
| Compile fullgraph       | `false`, `true` |

`GPUs / SP` is simplified to `1`, `2`, and `4`: each option means the GPU count
and SP size match, with `tp_size=1`.

## Fixed Dimensions

| Dimension name          | Fixed val                            |
| ----------------------- | ------------------------------------ |
| Model                   | `FastVideo/LTX2-Distilled-Diffusers` |
| `nvfp4_fa4`             | `false`                              |
| Stage logging           | `true`                               |
| Save video              | `false`                              |
| Return frames           | `true`                               |
| Compile DiT             | `true`                               |
| Compile VAE             | `true`                               |
| Compile text encoder    | `true`                               |
| Compile backend         | `inductor`                           |
| Compile dynamic         | `false`                              |
| Compile mode            | default / `None`                     |
| Code-path optimizations | none                                 |
| Resolution              | `1920x1088`                          |
| Frames                  | `121`                                |
| Prompt                  | first validation prompt              |
| Seed                    | `10`                                 |
| Guidance                | fixed script/default values          |
| Attention backend       | `FLASH_ATTN`                         |
| Tensor parallelism      | `tp_size=1`                          |
| Distributed executor    | `mp`                                 |
| Run protocol            | script standard profile protocol     |

## Cleanup Performed

Deprecated non-tracing runs were moved, not deleted, under:

```text
outputs_video/deprecated runs/
```

The move preserved the relative source layout. The active tree was cleaned of
empty folders after the move.

Moved deprecated run groups:

| Source group                                          | Moved run directories |
| ----------------------------------------------------- | --------------------: |
| `outputs_video/ltx2_sp_profile/full/`                 |                     5 |
| `outputs_video/ltx2_sp_profile/fullgraph_fix/`        |                     2 |
| `outputs_video/ltx2_sp_profile/smoke/`                |                     1 |
| `outputs_video/ltx2_sp_profile/steps8x3_baseline/`    |                     3 |
| `outputs_video/ltx2_sp_profile/steps8x3_optimized/`   |                     6 |
| `outputs_video/ltx2_sp_profile/steps8x3_real_output/` |                     1 |
| `outputs_video/ltx2_sp_profile/steps8x3_trials/`      |                     2 |
| `outputs_video/profile_runs/ltx2_sp_profile/`         |                     2 |

Reasons for deprecation were smoke/validation/failure status or incompatibility
with the fixed config, especially `return_frames=false`, `save_video=true`,
`stage_logging=false`, disabled compile components, reduced frame count, or
code-path optimization hacks.

## Runs Left In Place

Tracing and profiler outputs were intentionally not touched. Remaining active
outputs are diagnostic/profiler artifacts:

```text
outputs_video/ltx2_sp_profile/block47_gpu_profile/
outputs_video/ltx2_sp_profile/block47_gpu_profile_nsys_runs/
outputs_video/ltx2_sp_profile/block47_gpu_profile_smoke/
outputs_video/ltx2_sp_profile/block47_gpu_profile_smoke_nsys_runs/
outputs_video/ltx2_sp_profile/steps8x3_real_output/torch_profiler_return_frames_no_fp4_g4_compile_sp4_8x3_no_save_one_run/
outputs_video/ltx2_sp_profile/steps8x3_real_output/torch_profiler_return_frames_no_fp4_g4_compile_sp4_8x3_no_save_one_run_step/
outputs_video/ltx2_sp_profile/steps8x3_real_output/torch_profiler_return_frames_no_fp4_g4_compile_sp4_8x3_no_save_steady_run3/
outputs_video/ltx2_sp_profile/steps8x3_real_output/optimized_profiler_gated_return_frames_no_fp4_g4_compile_sp4_8x3_no_save/
```

At this landing point, the old non-tracing generation-speed runs should not be
used as current-matrix measurements. They are archived as historical context
under `outputs_video/deprecated runs/`. The next clean speed run should populate
the reduced design space above with the fixed dimensions enforced.

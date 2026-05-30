# LTX-2 Generation Speed Sweep Plan

Date: 2026-05-30

Worktree:

```text
/home/hal-jundas/codes/FastVideo-ltx2-sp-profile
```

Compute target:

```text
Slurm job 4745, using srun --overlap
```

## Goal

Run a clean LTX-2 generation-speed sweep over the reduced design space. This
pass should replace the old mixed-output, smoke, failed, and hack-optimized
measurements with comparable runs that share the same output contract and fixed
runtime settings.

Existing tracing/profiler outputs are diagnostic artifacts and should not be
moved or overwritten by this sweep.

## Tuning Dimensions

| Dimension name          | Options         |
| ----------------------- | --------------- |
| Denoising steps         | `5+2`, `8+3`    |
| GPUs / SP               | `1`, `2`, `4`   |
| FP4 linear quantization | `false`, `true` |
| Compile fullgraph       | `false`, `true` |

`GPUs / SP` is represented as `1`, `2`, or `4`; the GPU count and SP size are
the same for each run, and `tp_size=1`.

Total planned runs:

```text
2 denoising schedules * 3 GPU/SP sizes * 2 FP4 settings * 2 fullgraph settings = 24 runs
```

## Fixed Dimensions

| Dimension name          | Fixed val                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| Model                   | `FastVideo/LTX2-Distilled-Diffusers`                                                        |
| `nvfp4_fa4`             | `false`                                                                                     |
| Stage logging           | `true`                                                                                      |
| Save video              | `false`                                                                                     |
| Return frames           | `true`                                                                                      |
| Compile DiT             | `true`                                                                                      |
| Compile VAE             | `true`                                                                                      |
| Compile text encoder    | `true`                                                                                      |
| Compile backend         | `inductor`                                                                                  |
| Compile dynamic         | `false`                                                                                     |
| Compile mode            | default / `None`                                                                            |
| Benchmark-only optimizations | none                                                                                   |
| Resolution              | `1920x1088`                                                                                 |
| Frames                  | `121`                                                                                       |
| Prompt                  | first validation prompt from `examples/training/finetune/ltx2/validation.json`              |
| Seed                    | `10`                                                                                        |
| Guidance                | script defaults: `guidance_scale=1.0`, `refine_guidance_scale=1.0`, `refine_add_noise=true` |
| Attention backend       | `FLASH_ATTN`                                                                                |
| Tensor parallelism      | `tp_size=1`                                                                                 |
| Distributed executor    | `mp`                                                                                        |
| Run protocol            | script default profile protocol: `num_runs=12`, `warmup_runs=2`, no `avg_window` override   |

## Output Root

All sweep outputs should be written under the current worktree:

```text
outputs_video/ltx2_generation_speed_sweep/
```

Each run directory should contain the script's normal outputs:

```text
profile_config.json
profile_summary.json
profile.log
profile_runs_partial.json
videos/                 # expected to be empty because save_video=false
```

Do not write into `outputs_video/ltx2_sp_profile/` for this sweep. That tree now
contains retained tracing/profiler diagnostics and the deprecated historical
runs were moved to `outputs_video/deprecated runs/`.

## Run To Directory Mapping

```text
Run name                                  Steps  GPU/SP  FP4 linear  Fullgraph  Output directory
----------------------------------------  -----  ------  ----------  ---------  --------------------------------------------------------------------------------------
ltx2_speed_s5p2_g1_fp4off_fgoff          5+2    1       false       false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g1_fp4off_fgoff
ltx2_speed_s5p2_g1_fp4off_fgon           5+2    1       false       true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g1_fp4off_fgon
ltx2_speed_s5p2_g1_fp4on_fgoff           5+2    1       true        false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g1_fp4on_fgoff
ltx2_speed_s5p2_g1_fp4on_fgon            5+2    1       true        true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g1_fp4on_fgon
ltx2_speed_s5p2_g2_fp4off_fgoff          5+2    2       false       false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g2_fp4off_fgoff
ltx2_speed_s5p2_g2_fp4off_fgon           5+2    2       false       true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g2_fp4off_fgon
ltx2_speed_s5p2_g2_fp4on_fgoff           5+2    2       true        false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g2_fp4on_fgoff
ltx2_speed_s5p2_g2_fp4on_fgon            5+2    2       true        true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g2_fp4on_fgon
ltx2_speed_s5p2_g4_fp4off_fgoff          5+2    4       false       false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g4_fp4off_fgoff
ltx2_speed_s5p2_g4_fp4off_fgon           5+2    4       false       true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g4_fp4off_fgon
ltx2_speed_s5p2_g4_fp4on_fgoff           5+2    4       true        false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g4_fp4on_fgoff
ltx2_speed_s5p2_g4_fp4on_fgon            5+2    4       true        true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s5p2_g4_fp4on_fgon
ltx2_speed_s8p3_g1_fp4off_fgoff          8+3    1       false       false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g1_fp4off_fgoff
ltx2_speed_s8p3_g1_fp4off_fgon           8+3    1       false       true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g1_fp4off_fgon
ltx2_speed_s8p3_g1_fp4on_fgoff           8+3    1       true        false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g1_fp4on_fgoff
ltx2_speed_s8p3_g1_fp4on_fgon            8+3    1       true        true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g1_fp4on_fgon
ltx2_speed_s8p3_g2_fp4off_fgoff          8+3    2       false       false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g2_fp4off_fgoff
ltx2_speed_s8p3_g2_fp4off_fgon           8+3    2       false       true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g2_fp4off_fgon
ltx2_speed_s8p3_g2_fp4on_fgoff           8+3    2       true        false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g2_fp4on_fgoff
ltx2_speed_s8p3_g2_fp4on_fgon            8+3    2       true        true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g2_fp4on_fgon
ltx2_speed_s8p3_g4_fp4off_fgoff          8+3    4       false       false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g4_fp4off_fgoff
ltx2_speed_s8p3_g4_fp4off_fgon           8+3    4       false       true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g4_fp4off_fgon
ltx2_speed_s8p3_g4_fp4on_fgoff           8+3    4       true        false      outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g4_fp4on_fgoff
ltx2_speed_s8p3_g4_fp4on_fgon            8+3    4       true        true       outputs_video/ltx2_generation_speed_sweep/ltx2_speed_s8p3_g4_fp4on_fgon
```

## Execution Plan

Run the sweep serially for comparability. Parallel packing on the 4-GPU node can
change cache, memory, and interconnect contention, so a clean speed matrix should
use one active run at a time.

For each row:

1. Confirm the output directory does not already exist.
2. Set per-run cache directories under `/tmp/hal-jundas/`.
3. Launch with `srun --overlap --jobid=4745 -N1 -n1 --gpus=<GPU/SP>`.
4. Use the script's fixed settings plus the row-specific flags below.
5. Treat non-zero exit or missing `profile_summary.json` as a failed matrix cell.
6. Do not change flags to make a failed cell pass; record the failure as that
   cell's result.

Base command shape:

```text
srun --overlap --jobid=4745 -N1 -n1 --gpus=<GPU/SP> bash -lc '
  set -euo pipefail
  cd /home/hal-jundas/codes/FastVideo-ltx2-sp-profile

  export TOKENIZERS_PARALLELISM=false
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCHINDUCTOR_CACHE_DIR=/tmp/hal-jundas/torchinductor-cache-<run_name>
  export TRITON_CACHE_DIR=/tmp/hal-jundas/triton-cache-<run_name>
  export CUDA_CACHE_PATH=/tmp/hal-jundas/cuda-cache-<run_name>
  export XDG_CACHE_HOME=/tmp/hal-jundas/xdg-cache-<run_name>

  /home/hal-jundas/venvs/fv-shared/bin/python \
    examples/inference/basic/basic_ltx2_distilled_fast_profile.py \
    --num-gpus <GPU/SP> \
    <steps flags> \
    <fp4 flags> \
    <fullgraph flags>
'
```

Row-specific flag mapping:

```text
Steps 5+2  -> --num-inference-steps 5 --refine-num-inference-steps 2
Steps 8+3  -> --num-inference-steps 8 --refine-num-inference-steps 3

FP4 false  -> --no-fp4-linear
FP4 true   -> --fp4-linear

FG false   -> --no-compile-fullgraph
FG true    -> --compile-fullgraph
```

## Validation After Each Run

Each completed run should have:

```text
outputs_video/ltx2_generation_speed_sweep/<run_name>/profile_config.json
outputs_video/ltx2_generation_speed_sweep/<run_name>/profile_summary.json
outputs_video/ltx2_generation_speed_sweep/<run_name>/profile.log
```

For every successful run, verify `profile_config.json` records:

```text
fixed.save_video=false
fixed.return_frames=true
fixed.stage_logging=true
fixed.nvfp4_fa4=false
fixed.torch_compile=true
fixed.compile_text_encoder=true
fixed.compile_vae=true
fixed.compile_backend=inductor
fixed.compile_dynamic=false
fixed.tp_size=1
fixed.num_frames=121
fixed.seed=10
tuned.num_gpus=<GPU/SP>
tuned.sp_size=<GPU/SP>
tuned.num_inference_steps=<base steps>
tuned.refine_num_inference_steps=<refine steps>
tuned.fp4_linear=<row FP4 setting>
tuned.compile_fullgraph=<row fullgraph setting>
```

For each schedule/GPU/FP4/fullgraph cell, report:

```text
avg video_generation_time
avg e2e_latency
avg sr_forward_latency
stage_times.denoising_stage
stage_times.ltx2_refine_denoising_stage
stage_times.decoding_stage
stage_times.PostDecodeFrameProcessStage
```

## Notes

The fixed config intentionally uses `return_frames=true` and `save_video=false`
so the workload produces real decoded frames without MP4 encoding overhead.

Stage logging is intentionally on. This makes the latency slightly less minimal
than a pure timing-only run, but it keeps SR and stage breakdown available for
every matrix cell.

Fullgraph is a tuning dimension even for SP runs. If fullgraph fails for any
SP cell, the failed cell should stay in the matrix as a failure rather than
being silently converted to `fullgraph=false`.

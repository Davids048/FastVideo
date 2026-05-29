# LTX-2 Block 47 GPU Profiling Plan

## Goal

Profile why the LTX-2 8+3 no-FP4 baseline scales differently across stages:
the base denoising stage is fastest on 1 GPU, while the refine denoising stage
scales strongly from 1 to 2 to 4 GPUs.

The profiling should produce two kinds of artifacts for each case:

```text
torch profiler Chrome trace
Nsight Systems report and exported timeline/stat summaries
```

Both profilers should focus on one selected late transformer block, not the whole
pipeline. The selected block is `transformer_blocks.47`, because LTX-2 has
`num_layers=48` and block 47 is the final block.

## Fixed Experiment Config

Use the same prompt and generation config as the prior 8+3 baseline profiles:

```text
script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py
model_id=FastVideo/LTX2-Distilled-Diffusers
validation_json=examples/training/finetune/ltx2/validation.json
prompt=first entry from validation_json
num_frames=121
num_inference_steps=8
refine_num_inference_steps=3
fp4_linear=false
nvfp4_fa4=false
torch_compile=true
compile_text_encoder=true
compile_vae=true
save_video=false
return_frames=false
stage_logging=false
tp_size=1
```

Keep the same fullgraph policy as the baseline comparison:

```text
sp_size=1 uses the script default fullgraph=true
sp_size=2/4 use the script default fullgraph=false
```

This keeps the profiler runs comparable to the baseline logs where the unusual
stage scaling was observed. Fullgraph can be profiled later as a separate
experiment.

## Profiling Matrix

Run six profiling jobs:

| Case | GPUs | SP | Captured Stage | Selected Block |
|---|---:|---:|---|---:|
| `g1_sp1_base_block47` | 1 | 1 | base denoising | 47 |
| `g2_sp2_base_block47` | 2 | 2 | base denoising | 47 |
| `g4_sp4_base_block47` | 4 | 4 | base denoising | 47 |
| `g1_sp1_refine_block47` | 1 | 1 | refine denoising | 47 |
| `g2_sp2_refine_block47` | 2 | 2 | refine denoising | 47 |
| `g4_sp4_refine_block47` | 4 | 4 | refine denoising | 47 |

Each case should run:

```text
num_runs=4
warmup_runs=2
avg_window=2
```

The first two generations are compile/warmup. The profiler captures the two
post-warmup generations. For the base cases, block 47 appears once per base
denoising step, so the selected block should appear 16 times total across two
captured generations. For the refine cases, it should appear 6 times total.

## Required Instrumentation Patch

Add a temporary, env-gated profiler range around only block 47 in:

```text
fastvideo/models/dits/ltx2.py::_process_transformer_blocks
```

The existing loop is:

```text
for idx, block in enumerate(self.transformer_blocks):
    ...
    video, audio = block(...)
```

The instrumentation should be gated by environment variables so normal runs are
unchanged:

```text
FASTVIDEO_LTX2_BLOCK_PROFILE_INDEX=47
FASTVIDEO_LTX2_BLOCK_PROFILE_STAGE=base|refine
```

The current stage can be read from the existing forward context or stage profile
state used by LTX-2:

```text
ltx2_fp4_stage_profile=base   for the first denoising stage
ltx2_fp4_stage_profile=refine for the refine denoising stage
```

Only when both the block index and stage match should the block be wrapped in:

```text
torch.profiler.record_function("fastvideo.ltx2.block47.<stage>")
```

The range should be present for torch profiler and Nsight Systems. For Nsight
Systems, also add NVTX emission if `torch.cuda.nvtx.range_push/pop` is not
already emitted by `record_function` in the installed PyTorch build. The goal is
that Nsight can filter visually to the same `fastvideo.ltx2.block47.<stage>`
range.

Do not keep this as permanent broad instrumentation unless it remains fully
env-gated and has near-zero overhead when disabled.

## Torch Profiler Settings

Use torch profiler only for CUDA/CPU timeline and Chrome trace output. Keep the
trace small:

```text
FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES=0
FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY=0
FASTVIDEO_TORCH_PROFILER_WITH_STACK=0
FASTVIDEO_TORCH_PROFILER_WITH_FLOPS=0
FASTVIDEO_TORCH_PROFILER_WAIT_STEPS=2
FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS=0
FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS=2
FASTVIDEO_TORCH_PROFILE_REGIONS=profiler_region_inference_forward
FASTVIDEO_TORCH_PROFILER_DIR=<case_dir>/torch_traces
```

The pipeline-level profiler region gates collection to inference forwards. The
block-level `record_function` range is the marker to inspect in Chrome.

Because the profiler step currently advances once per inference forward, this
schedule means skip the two warmup generations and record the next two
generations.

## Nsight Systems Requirement

Nsight Systems is mandatory for these runs. First verify on the compute node:

```text
command -v nsys
nsys --version
```

Run each profiling case under `nsys profile` with a constrained capture:

```text
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=nvtx \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output <case_dir>/nsys/<case_name> \
  <python command>
```

The code should emit an NVTX capture range only for the selected block/stage
capture window, so Nsight does not trace the full compile/warmup or full
pipeline. If `--capture-range=nvtx` cannot be used reliably with multiple worker
processes, fall back to tracing the full post-warmup process but keep the torch
profiler block range and export filtered stats around the
`fastvideo.ltx2.block47.<stage>` NVTX range.

Export useful Nsight sidecars after each run:

```text
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum <case>.nsys-rep
nsys export --type sqlite --output <case_dir>/nsys/<case_name>.sqlite <case>.nsys-rep
```

## Output Layout

Use one directory per case:

```text
outputs_video/ltx2_sp_profile/block47_gpu_profile/
  g1_sp1_base_block47/
    profile.log
    profile_config.json
    profile_summary.json
    torch_traces/
    nsys/
  g2_sp2_base_block47/
  g4_sp4_base_block47/
  g1_sp1_refine_block47/
  g2_sp2_refine_block47/
  g4_sp4_refine_block47/
```

Expected trace files:

```text
torch_traces/*.pt.trace.json.gz
nsys/<case_name>.nsys-rep
nsys/<case_name>.sqlite
nsys/*stats*.txt
```

## Execution Notes

Run on the allocated Slurm node with the existing node access pattern:

```text
srun --overlap --jobid=4745 -N1 -n1 --gpus=<1|2|4> bash -lc '<env> <nsys profile> /home/hal-jundas/venvs/fv-shared/bin/python examples/inference/basic/basic_ltx2_distilled_fast_profile.py ...'
```

Use per-experiment cache directories to avoid cross-run cache permission issues:

```text
TORCHINDUCTOR_CACHE_DIR=/tmp/hal-jundas/torchinductor-cache-ltx2-block47-profile
TRITON_CACHE_DIR=/tmp/hal-jundas/triton-cache-ltx2-block47-profile
CUDA_CACHE_PATH=/tmp/hal-jundas/cuda-cache-ltx2-block47-profile
XDG_CACHE_HOME=/tmp/hal-jundas/xdg-cache-ltx2-block47-profile
TOKENIZERS_PARALLELISM=false
NCCL_ASYNC_ERROR_HANDLING=1
```

## Success Criteria

Each of the six cases is successful only if it has:

```text
exit_code=0
profile_summary.json with measured_runs=2
torch Chrome trace containing fastvideo.ltx2.block47.<stage>
nsys report containing the same block/stage NVTX range
Nsight kernel summary and NVTX summary exported to text or sqlite
```

The final comparison should report, for each GPU count and stage:

```text
block47 wall time from torch profiler range
dominant CUDA kernels from Nsight Systems
NCCL/all-to-all/all-gather time if present
GPU memory copy time if present
stage-level average from profile_summary.json for context
```

## Open Risks

Multi-process profiling may produce one trace per worker or only rank-local
traces depending on profiler behavior. That is acceptable if the rank is clearly
labeled, but the final report must say which rank each trace came from.

Nsight `--capture-range=nvtx` may require explicit `cudaProfilerStart/Stop` or
NVTX capture annotations that are process-local. If capture-range does not work
with the multiprocessing executor, use full-process Nsight capture for the
post-warmup profile command and rely on NVTX block ranges for filtering.

The trace should not include MP4 encoding or returned frame materialization,
because the command uses `--no-save-video --no-return-frames` and the target is
DiT GPU behavior.

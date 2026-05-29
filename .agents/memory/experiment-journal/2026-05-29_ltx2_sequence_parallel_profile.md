# LTX-2 Distilled Sequence Parallel Profiling - 2026-05-29

## Goal

The run tested whether `examples/inference/basic/basic_ltx2_distilled_fast_profile.py` could use
Ulysses sequence parallelism on 2 and 4 GPUs, with torch.compile enabled, and whether turning off
all FP4 paths changes latency. The comparison baseline supplied by the user was a 1-GPU profile with
average end-to-end latency 4.20s, average SR forward latency 2.215s, denoising 0.922s, refine
denoising 2.175s, and video save 0.544s over runs 3-12.

## Worktree And Environment

The task worktree is `/home/hal-jundas/codes/FastVideo-ltx2-sp-profile` on branch
`ltx2-sp-profile`, based on `upstream/main` at `afdb6fbf` (`[feat] Add MatrixGame3.0 (#1201)`).
The main checkout at `/home/hal-jundas/codes/FastVideo` was dirty before this work started, so the
task used a separate worktree and did not modify the main checkout.

The shared venv is `/home/hal-jundas/venvs/fv-shared`. Its editable `fastvideo` install points at
the task worktree. Local `flash-attn-cute` was installed from
`/home/hal-jundas/codes/flash-attention/flash_attn/cute` with `uv pip install --reinstall --no-deps`
because the local package pins a different torch build than the venv. `nvidia-cutlass-dsl` was then
upgraded to 4.5.2 because the older installed version failed to import local cute on the GPU node.

The profiling ran through overlap Slurm steps on existing job `4745`, allocated on `hpc-rack-2-8`.
The user described the node as B200-class; the runtime reported `NVIDIA GB200`. `/mnt/local` was not
available on that compute node, so all transient compile/cache state was moved to `/tmp/hal-jundas`:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/hal-jundas/torchinductor-cache-ltx2-sp-profile
TRITON_CACHE_DIR=/tmp/hal-jundas/triton-cache-ltx2-sp-profile
CUDA_CACHE_PATH=/tmp/hal-jundas/cuda-cache-ltx2-sp-profile
XDG_CACHE_HOME=/tmp/hal-jundas/xdg-cache-ltx2-sp-profile
```

## Code Changes

The modified script is
`examples/inference/basic/basic_ltx2_distilled_fast_profile.py`.

It now accepts CLI controls for `--num-gpus`, `--sp-size`, `--tp-size`, and
`--distributed-executor-backend`. For these profiles, tensor parallelism stayed off with
`tp_size=1`, and sequence parallelism used `sp_size=num_gpus`.

It also records one directory per run under `--profile-dir/--run-name`, with `profile.log`,
`profile_config.json`, `profile_runs_partial.json`, `profile_summary.json`, and generated videos.
The config recorder stores the resolved model paths, prompt, dimensions, quantization choices,
torch.compile kwargs, sequence-parallel config, package versions, git state, Slurm env, and cache
env vars.

Torch compile is enabled by default for the DiT, text encoder, and VAE. For SP runs,
`compile_fullgraph` defaults to false. A 4-GPU SP attempt with `fullgraph=True` failed because
`LTXDistributedAttention.forward` is wrapped with `torch.compiler.disable()`, and Dynamo cannot
inline a disabled function into a full graph.

FP4 linear layers are controlled by `--fp4-linear/--no-fp4-linear`. The implementation sets
`pipeline_config.dit_config.quant_config = NVFP4Config()` when enabled, and `None` when disabled.
FP4 Q/K FlashAttention-4 is controlled separately by `--nvfp4-fa4/--no-nvfp4-fa4`; every recorded
experiment below used `--no-nvfp4-fa4`.

## Common Run Config

All four full profiles used the first entry from
`examples/training/finetune/ltx2/validation.json`, prompt:

```text
A large metal cylinder is seen pressing down on a pile of Oreo cookies, flattening them as if they were under a hydraulic press.
```

The model was `FastVideo/LTX2-Distilled-Diffusers`, resolved locally to:

```text
/home/hal-jundas/.local/share/huggingface/hub/models--FastVideo--LTX2-Distilled-Diffusers/snapshots/0762ece944ea65f45cd3318981423e1670ff7225
```

The refine upsampler path was:

```text
/home/hal-jundas/.local/share/huggingface/hub/models--FastVideo--LTX2-Distilled-Diffusers/snapshots/0762ece944ea65f45cd3318981423e1670ff7225/spatial_upsampler
```

Shared profile parameters:

```text
num_runs=12
warmup_runs=2
measured_runs=10 (runs 3-12)
height=1088
width=1920
num_frames=121
num_inference_steps=5
refine_num_inference_steps=2
fps=24
seed=10
guidance_scale=1.0
refine_guidance_scale=1.0
save_video=true
attention_backend=FLASH_ATTN
torch_compile=true
compile_text_encoder=true
compile_vae=true
compile_backend=inductor
compile_fullgraph=false
compile_dynamic=false
distributed_executor_backend=mp
```

The profile logs repeatedly show ffmpeg pipe save failing with `Unrecognized option 'preset'` and
falling back to PyAV single-pass save. The fallback succeeded, but video-save latency is included in
end-to-end latency.

## Results

The best measured result was no-FP4 on 4 GPUs: 3.872s average end-to-end latency, faster than the
provided 1-GPU 4.20s baseline. The FP4-linear SP runs did not improve end-to-end latency. They
improved SR forward latency relative to the provided 1-GPU baseline, but total generation and video
save latency erased that improvement.

| Run | GPUs | SP | FP4 linear | Avg generation | Avg end-to-end | Avg SR forward |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `fp4_linear_g4_compile_sp_fullgraph_false` | 4 | 4 | yes | 4.133s | 5.269s | 1.185s |
| `fp4_linear_g2_compile_sp_fullgraph_false` | 2 | 2 | yes | 4.300s | 5.304s | 1.438s |
| `no_fp4_g4_compile_sp_fullgraph_false` | 4 | 4 | no | 2.900s | 3.872s | 1.007s |
| `no_fp4_g2_compile_sp_fullgraph_false` | 2 | 2 | no | 3.556s | 4.536s | 1.769s |

Stage averages over runs 3-12:

| Run | Denoising | Refine denoising | Upsample | Decode | Video save | Stage sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fp4_linear_g4_compile_sp_fullgraph_false` | 2.513s | 1.150s | 0.035s | 0.337s | 1.035s | 5.236s |
| `fp4_linear_g2_compile_sp_fullgraph_false` | 2.434s | 1.404s | 0.035s | 0.334s | 0.911s | 5.275s |
| `no_fp4_g4_compile_sp_fullgraph_false` | 1.459s | 0.973s | 0.034s | 0.339s | 0.881s | 3.841s |
| `no_fp4_g2_compile_sp_fullgraph_false` | 1.356s | 1.734s | 0.034s | 0.339s | 0.889s | 4.508s |

## Committed Artifact Snapshots

The compact JSON config and summary files were copied into agent memory so the exact run inputs and
per-run timings are available without relying on ignored output directories:

```text
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/fp4_linear_g4_compile_sp_fullgraph_false/profile_config.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/fp4_linear_g4_compile_sp_fullgraph_false/profile_summary.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/fp4_linear_g2_compile_sp_fullgraph_false/profile_config.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/fp4_linear_g2_compile_sp_fullgraph_false/profile_summary.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/no_fp4_g4_compile_sp_fullgraph_false/profile_config.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/no_fp4_g4_compile_sp_fullgraph_false/profile_summary.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/no_fp4_g2_compile_sp_fullgraph_false/profile_config.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_sp_profile/no_fp4_g2_compile_sp_fullgraph_false/profile_summary.json
```

The full local logs and generated videos remain in the ignored output directory:

```text
outputs_video/ltx2_sp_profile/full/<run_name>/profile.log
outputs_video/ltx2_sp_profile/full/<run_name>/videos/
```

There is also a failed exploratory run at
`outputs_video/ltx2_sp_profile/full/fp4_linear_g4_compile/` for the `fullgraph=True` attempt. It was
not committed as an artifact because it was not one of the requested completed profiles.

## Exact Launch Shape

Each run used an overlap step on job `4745`. The run-specific values were only GPU count, SP size,
run name, and FP4 flag:

```bash
srun --overlap --jobid=4745 -N1 -n1 --gres=gpu:<2-or-4> bash -lc '
  set -euo pipefail
  export TOKENIZERS_PARALLELISM=false
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCHINDUCTOR_CACHE_DIR=/tmp/hal-jundas/torchinductor-cache-ltx2-sp-profile
  export TRITON_CACHE_DIR=/tmp/hal-jundas/triton-cache-ltx2-sp-profile
  export CUDA_CACHE_PATH=/tmp/hal-jundas/cuda-cache-ltx2-sp-profile
  export XDG_CACHE_HOME=/tmp/hal-jundas/xdg-cache-ltx2-sp-profile
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"
  cd /home/hal-jundas/codes/FastVideo-ltx2-sp-profile
  /home/hal-jundas/venvs/fv-shared/bin/python examples/inference/basic/basic_ltx2_distilled_fast_profile.py \
    --profile-dir outputs_video/ltx2_sp_profile/full \
    --run-name <run_name> \
    --num-gpus <2-or-4> \
    --sp-size <2-or-4> \
    --tp-size 1 \
    --num-runs 12 \
    --warmup-runs 2 \
    --num-frames 121 \
    --num-inference-steps 5 \
    --refine-num-inference-steps 2 \
    <--fp4-linear-or---no-fp4-linear> \
    --no-nvfp4-fa4 \
    --torch-compile \
    --compile-text-encoder \
    --compile-vae \
    --no-compile-fullgraph
'
```

## Problems And Assumptions

The first attempt to use `/mnt/local/hal-jundas` for caches failed because `/mnt/local` did not
exist on `hpc-rack-2-8`. `/tmp/hal-jundas` worked.

The first 4-GPU SP compile attempt used `fullgraph=True` and failed before measurements. The
successful SP profiles used `fullgraph=False`, and the script now chooses that default whenever
`sp_size > 1`.

The "no FP4" profiles mean both linear FP4 and FP4 Q/K attention are disabled:
`--no-fp4-linear --no-nvfp4-fa4`. They still use the FlashAttention-4 backend for BF16 attention.

`pre-commit` initially failed because the shared pre-commit cache was locked. Retrying with
`PRE_COMMIT_HOME=/tmp/hal-jundas/pre-commit-cache-ltx2-sp-profile-2` and
`VIRTUALENV_OVERRIDE_APP_DATA=/tmp/hal-jundas/virtualenv-app-data-ltx2-sp-profile` passed.

## Validation

Checks run after the script and memory updates:

```text
AST parse of examples/inference/basic/basic_ltx2_distilled_fast_profile.py: passed
manual >120-character line check with awk: passed
git diff --check: passed
pre-commit run --files examples/inference/basic/basic_ltx2_distilled_fast_profile.py: passed
```

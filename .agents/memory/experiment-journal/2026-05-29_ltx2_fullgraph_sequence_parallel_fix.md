# LTX-2 Sequence-Parallel Fullgraph Fix

## Goal

This run checked whether the LTX-2 distilled sequence-parallel generation path can use
`torch.compile(fullgraph=True)` after the earlier profiling work had to force
`compile_fullgraph=False`.

The previous state was:

```text
LTX-2 transformer block
  -> LTXDistributedSelfAttention.forward
    -> LTXDistributedAttention.forward
      -> @torch.compiler.disable boundary
```

That explicit disabled boundary made Dynamo fail when `fullgraph=True` was requested. The
working state after this run is:

```text
LTX-2 transformer block
  -> LTXDistributedSelfAttention.forward
    -> LTXDistributedAttention.forward
      -> sequence-parallel all-to-all + attention backend captured in the compiled graph
```

## Code Change

`fastvideo/models/dits/ltx2.py` no longer decorates `LTXDistributedAttention.forward` with
`@torch.compiler.disable`.

No tensor math or communication logic was changed. The change only removes the forced graph break
from the LTX-2-specific distributed attention subclass. The shared base
`DistributedAttention.forward` is still disabled, so this result should be treated as LTX-2-specific
until other model paths are tested.

## Failure Reproduction Before Patch

The fullgraph failure was reproduced on Slurm job `4745`, node `hpc-rack-2-8`, from worktree:

```text
/home/hal-jundas/codes/FastVideo-ltx2-sp-profile
```

The Python environment was:

```text
/home/hal-jundas/venvs/fv-shared
```

The failing command used 4 GPUs, SP=4, TP=1, LTX-2 distilled, 8 base denoise steps plus 3 refine
steps, no FP4 linear path, no NVFP4 FA4 path, and explicit `--compile-fullgraph`:

```text
srun --overlap --jobid=4745 -N1 -n1 --gpus=4 bash -lc 'set -euo pipefail; export TOKENIZERS_PARALLELISM=false; export NCCL_ASYNC_ERROR_HANDLING=1; export TORCHINDUCTOR_CACHE_DIR=/tmp/hal-jundas/torchinductor-cache-ltx2-fullgraph; export TRITON_CACHE_DIR=/tmp/hal-jundas/triton-cache-ltx2-fullgraph; export CUDA_CACHE_PATH=/tmp/hal-jundas/cuda-cache-ltx2-fullgraph; export XDG_CACHE_HOME=/tmp/hal-jundas/xdg-cache-ltx2-fullgraph; mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"; cd /home/hal-jundas/codes/FastVideo-ltx2-sp-profile; /home/hal-jundas/venvs/fv-shared/bin/python examples/inference/basic/basic_ltx2_distilled_fast_profile.py --profile-dir outputs_video/ltx2_sp_profile/fullgraph_fix --run-name reproduce_fullgraph_break_g4_sp4 --num-gpus 4 --sp-size 4 --tp-size 1 --num-runs 1 --warmup-runs 0 --avg-window 1 --num-frames 121 --num-inference-steps 8 --refine-num-inference-steps 3 --no-fp4-linear --no-nvfp4-fa4 --torch-compile --compile-text-encoder --compile-vae --compile-fullgraph --no-save-video --no-return-frames --no-stage-logging'
```

It failed with exit code `131` and:

```text
torch._dynamo.exc.Unsupported: Skip inlining torch.compiler.disable()d function LTXDistributedAttention.forward
```

The relevant stack was:

```text
fastvideo/models/dits/ltx2.py:2689  transformer forward
fastvideo/models/dits/ltx2.py:2385  _process_transformer_blocks
fastvideo/models/dits/ltx2.py:2315  block call
fastvideo/models/dits/ltx2.py:1844  vx = vx + self.attn1(...)
fastvideo/models/dits/ltx2.py:1610  out, _ = self.attn(...)
```

## Successful Fullgraph Smoke After Patch

The post-patch smoke used the same node, environment, cache directories, and generation config:

```text
srun --overlap --jobid=4745 -N1 -n1 --gpus=4 bash -lc 'set -euo pipefail; export TOKENIZERS_PARALLELISM=false; export NCCL_ASYNC_ERROR_HANDLING=1; export TORCHINDUCTOR_CACHE_DIR=/tmp/hal-jundas/torchinductor-cache-ltx2-fullgraph; export TRITON_CACHE_DIR=/tmp/hal-jundas/triton-cache-ltx2-fullgraph; export CUDA_CACHE_PATH=/tmp/hal-jundas/cuda-cache-ltx2-fullgraph; export XDG_CACHE_HOME=/tmp/hal-jundas/xdg-cache-ltx2-fullgraph; mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME"; cd /home/hal-jundas/codes/FastVideo-ltx2-sp-profile; /home/hal-jundas/venvs/fv-shared/bin/python examples/inference/basic/basic_ltx2_distilled_fast_profile.py --profile-dir outputs_video/ltx2_sp_profile/fullgraph_fix --run-name after_remove_disable_ltxdistattn_g4_sp4 --num-gpus 4 --sp-size 4 --tp-size 1 --num-runs 1 --warmup-runs 0 --avg-window 1 --num-frames 121 --num-inference-steps 8 --refine-num-inference-steps 3 --no-fp4-linear --no-nvfp4-fa4 --torch-compile --compile-text-encoder --compile-vae --compile-fullgraph --no-save-video --no-return-frames --no-stage-logging'
```

Result:

```text
exit_code=0
generation_time=502.3009340548888s
e2e_latency=502.30168167175725s
measured_runs=1
compile_fullgraph=true
torch_compile_backend=inductor
torch_compile_dynamic=false
num_gpus=4
sp_size=4
tp_size=1
num_frames=121
num_inference_steps=8
refine_num_inference_steps=3
save_video=false
return_frames=false
stage_logging=false
```

The first base denoise step spent about `225s` compiling and executing; later base steps completed
almost immediately. The first refine step spent about `218s`; later refine steps also completed
quickly. This run is a fullgraph compile/correctness smoke, not a steady-state latency result, and
must not be compared against the warmed `4.337300085s` real-output profiling result.

No stage timings were collected because `--no-stage-logging` was used. The profile summary therefore
has an empty `stage_times` object and no SR forward latency.

## Artifacts

Live run directory:

```text
outputs_video/ltx2_sp_profile/fullgraph_fix/after_remove_disable_ltxdistattn_g4_sp4/
```

Committed snapshots:

```text
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_fullgraph_sp_fix/after_remove_disable_ltxdistattn_g4_sp4/profile_config.json
.agents/memory/experiment-journal/artifacts/2026-05-29_ltx2_fullgraph_sp_fix/after_remove_disable_ltxdistattn_g4_sp4/profile_summary.json
```

The run directory also contains `profile.log` and `profile_runs_partial.json`, but only the config
and final summary were committed as stable memory artifacts.

## Output Videos

No MP4 files were written for this smoke. The command used both `--no-save-video` and
`--no-return-frames`, so the `videos/` directory exists but is empty. The `output_path` in
`profile_summary.json` is the path that would have been used if saving were enabled.

For profiling runs that do save MP4s, videos are stored under:

```text
outputs_video/ltx2_sp_profile/<profile-dir>/<run-name>/videos/output_ltx2_basic_t2v_run_<n>.mp4
```

## Interpretation

The old reason for disabling fullgraph is now removed for this LTX-2 path. The reason was not a
performance decision; it was a forced graph-break in `LTXDistributedAttention.forward` that fullgraph
mode could not legally cross.

The profile script still defaults `compile_fullgraph` to false when `sp_size > 1`. After this fix,
future agents can explicitly pass `--compile-fullgraph` for the tested LTX-2 4-GPU SP path. Making
fullgraph the default for SP should be a separate decision because cold compile time is very large
and this run did not measure warmed steady-state latency.

## Assumptions And Risks

This was verified only for `FastVideo/LTX2-Distilled-Diffusers` with FlashAttention, BF16 attention,
`--no-fp4-linear`, `--no-nvfp4-fa4`, 4 GPUs, SP=4, TP=1, 121 frames, and the 8+3 denoise schedule.

The smoke used `--no-save-video --no-return-frames` to avoid file I/O and frame materialization, so it
proves the fullgraph transformer path can compile and run. It does not prove real-output latency,
MP4 save behavior, or fullgraph behavior with every optional instrumentation path.

# LTX-2 Distilled Sequence Parallel 8x3 Profile

Date: 2026-05-29

Status: running. Baselines and several optimization trials are measured, but the latest code optimization has not been GPU-timed because Slurm began failing with `Unable to contact slurm controller (connect failure)`.

## Goal

The user asked to change the LTX-2 distilled profile setup to low-resolution/base 8 denoising steps and high-resolution/refine 3 denoising steps, establish 1/2/4 GPU baselines, optimize generation latency across the stack, and reach generation time <= 4.2 seconds on either 2 or 4 GPUs if feasible. Saving videos is not important for the target.

The main latency model for this profile is:

```text
end-to-end
  = prompt/setup overhead
  + generation
  + optional save/return work

generation
  = low-res denoise
  + SR upsample
  + high-res refine denoise
  + audio decode
  + VAE decode
  + output materialization/synchronization

SR
  = SR upsample
  + high-res refine denoise
```

This is why `average generation + average SR` is larger than e2e: SR is already inside generation.

## Code Changes In This Run

`examples/inference/basic/basic_ltx2_distilled_fast_profile.py` now defaults to `--num-inference-steps 8` and `--refine-num-inference-steps 3`. The script also accepts `parse_args(argv)` for unit tests, records the refine/save/return/stage-logging settings in `profile_config.json`, records `FASTVIDEO_LOGGING_LEVEL` and `TQDM_DISABLE`, and adds `--stage-logging/--no-stage-logging` so latency-only runs can disable per-stage timers.

`fastvideo/entrypoints/video_generator.py` now avoids materializing decoded CPU `samples` and RGB `frames` when both `save_video=False` and `return_frames=False`. In that path it still synchronizes CUDA before recording generation time so latency-only profiling does not undercount asynchronous GPU work.

`tests/local_tests/ltx2/test_ltx2_profile_script_config.py` covers the profile-script defaults and config recording. `fastvideo/tests/entrypoints/test_video_generator.py` adds focused coverage for the output materialization optimization: latency-only calls skip CPU/frame work, `return_frames=True` still returns samples/frames, and `save_video=True` still materializes frames for writing.

## Shared Config

All completed 8x3 profile runs used:

```text
model_id=FastVideo/LTX2-Distilled-Diffusers
model_root=/home/hal-jundas/.local/share/huggingface/hub/models--FastVideo--LTX2-Distilled-Diffusers/snapshots/0762ece944ea65f45cd3318981423e1670ff7225
validation_json=/home/hal-jundas/codes/FastVideo-ltx2-sp-profile/examples/training/finetune/ltx2/validation.json
prompt="A large metal cylinder is seen pressing down on a pile of Oreo cookies, flattening them as if they were under a hydraulic press."
height=1088
width=1920
num_frames=121
fps=24
num_inference_steps=8
refine_num_inference_steps=3
guidance_scale=1.0
refine_guidance_scale=1.0
refine_add_noise=True
attention_backend=FLASH_ATTN
fp4_linear=False
nvfp4_fa4=False
torch_compile=True
compile_text_encoder=True
compile_vae=True
compile_backend=inductor
compile_fullgraph=False
compile_dynamic=False
save_video=False
return_frames=False
```

`compile_fullgraph=False` is required for SP runs because the fullgraph attempt failed in the prior profiling pass with:

```text
torch._dynamo.exc.Unsupported: Skip inlining torch.compiler.disable()d function LTXDistributedAttention.forward
```

Fullgraph cannot tolerate that graph break in the sequence-parallel distributed attention path.

## Baseline Results

Each baseline used 12 runs and skipped the first 2 warmups.

| Run | GPUs / SP | Avg generation | Avg e2e | Avg SR | Low-res denoise | Refine denoise | Post-decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_no_fp4_g1_compile_sp1_8x3_no_save` | 1 / 1 | 6.534s | 6.634s | 4.112s | 2.006s | 4.076s | 0.100s |
| `baseline_no_fp4_g2_compile_sp2_8x3_no_save` | 2 / 2 | 5.541s | 5.637s | 2.641s | 2.469s | 2.607s | 0.095s |
| `baseline_no_fp4_g4_compile_sp4_8x3_no_save` | 4 / 4 | 4.335s | 4.433s | 1.496s | 2.407s | 1.461s | 0.097s |

The best baseline is 4 GPUs, but it misses the <=4.2s generation target by about 0.135s.

Full baseline outputs are in:

```text
outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g1_compile_sp1_8x3_no_save/
outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g2_compile_sp2_8x3_no_save/
outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g4_compile_sp4_8x3_no_save/
```

Each directory has `profile_config.json`, `profile_summary.json`, and `profile.log`.

## Optimization Trials

The best completed measured trial is:

| Run | Change | Avg generation | Avg e2e | Result |
| --- | --- | ---: | ---: | --- |
| `optimized_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save` | disabled per-stage timing with `--no-stage-logging` | 4.276s | 4.376s | best measured; still 0.076s above target |
| `optimized_quiet_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save` | added `FASTVIDEO_LOGGING_LEVEL=WARNING` and `TQDM_DISABLE=1` | 4.346s | 4.457s | regressed |
| `trial_no_vae_compile_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save` | disabled VAE compile | 5.521s | 5.616s | regressed heavily |
| `trial_reduce_overhead_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save` | `--compile-mode reduce-overhead` | N/A | N/A | failed before measurements |

The `reduce-overhead` trial failed during the first run with a TorchDynamo CUDA graph error:

```text
accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
```

The next code-level optimization was to skip decoded output materialization in latency-only calls. That patch is present in the worktree, but it still needs a 4-GPU timed run after Slurm is available again.

Optimized/trial outputs are in:

```text
outputs_video/ltx2_sp_profile/steps8x3_optimized/optimized_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save/
outputs_video/ltx2_sp_profile/steps8x3_optimized/optimized_quiet_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save/
outputs_video/ltx2_sp_profile/steps8x3_trials/trial_no_vae_compile_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save/
outputs_video/ltx2_sp_profile/steps8x3_trials/trial_reduce_overhead_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save/
```

## Output Videos

The 8x3 baseline and optimization runs intentionally used `--no-save-video`, so they do not write MP4 outputs even though the profile metadata still contains target output paths.

The earlier 5x2 profiling pass did save videos under:

```text
outputs_video/ltx2_sp_profile/full/<run_name>/videos/output_ltx2_basic_t2v_run_<n>.mp4
```

## Validation

Local validation that does not require CUDA:

```text
/home/hal-jundas/venvs/fv-shared/bin/python -m pytest tests/local_tests/ltx2/test_ltx2_profile_script_config.py -q
5 passed

/home/hal-jundas/venvs/fv-shared/bin/python -c "<install in-memory fastvideo_kernel stubs>; pytest.main(['fastvideo/tests/entrypoints/test_video_generator.py', '-q'])"
23 passed

/home/hal-jundas/venvs/fv-shared/bin/python -c "compile(...)"
AST compile passed for 4 files

git diff --check
passed
```

The `fastvideo/tests/entrypoints/test_video_generator.py` login-node run used in-memory stubs for
`fastvideo_kernel` and `fastvideo_kernel.triton_kernels.sla_triton` so collection did not ask Triton
for an active CUDA driver. That validates the Python control flow and new output-materialization
tests, but it is not a replacement for a real CUDA-node pytest run.

Attempted validation that is currently blocked:

```text
/home/hal-jundas/venvs/fv-shared/bin/python -m pytest fastvideo/tests/entrypoints/test_video_generator.py tests/local_tests/ltx2/test_ltx2_profile_script_config.py -q
```

On the login node this fails during collection because importing `fastvideo_kernel` asks Triton for an active CUDA driver and gets `0 active drivers`.

```text
srun --overlap --jobid=4745 ...
```

This fails because job `4745` is expired/invalid or Slurm cannot confirm the allocation.

```text
sbatch --parsable -N1 --gres=gpu:4 --time=04:00:00 --wrap='sleep infinity'
```

This fails with `Unable to contact slurm controller (connect failure)`.

## Next Steps

When Slurm is reachable, run the focused `fastvideo/tests/entrypoints/test_video_generator.py` tests on a CUDA node and rerun the best 4-GPU no-stage-logging profile with the materialization patch. The target run should use the same 8x3/no-FP4/no-save config as `optimized_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save`.

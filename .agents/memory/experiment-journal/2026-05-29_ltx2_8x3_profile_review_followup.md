# LTX-2 8x3 Profile Review Follow-Up

Date: 2026-05-29

Status: corrected follow-up completed. The earlier `4.117228775s` target-achieving result is not valid for the real frame-producing workload because it skipped the main VAE pixel decode. The corrected real-output contract uses `--no-save-video --return-frames`, which excludes MP4 file I/O but still produces decoded frames for the caller. Under that contract, the best measured 4-GPU result in this follow-up is `4.337300085s` average generation time and `4.437282278s` average end-to-end latency, so the `<=4.2s` target was not achieved.

## Mental Model

The profile script disables MP4 saving to avoid measuring encoder and file I/O cost. That does not make pixel generation optional. The real video-serving boundary is:

```text
prompt
  -> text/audio conditioning
  -> low-resolution denoise
  -> latent upsample / SR
  -> high-resolution refine denoise
  -> VAE pixel decode
  -> frame tensor usable by caller
  -> optional MP4 save, excluded here
```

The rejected shortcut skipped the VAE pixel decode and frame construction, then reported a faster latency. That shortcut is only a latent/timing-only benchmark and must not be used as the real video generation SLA.

## Corrected Code Changes

`fastvideo/pipelines/stages/decoding.py` was corrected so normal non-latent video generation always runs the main VAE pixel decode, even when `save_video=False` and `return_frames=False`. The only remaining main decode bypass is `output_type="latent"`, where the caller explicitly requested latent output.

`fastvideo/tests/stages/test_decoding.py` now asserts that pixel frames are decoded even with no save and no returned frames, and separately asserts that latent output still bypasses pixel decode.

`fastvideo/profiler.py` and `fastvideo/pipelines/composed_pipeline_base.py` now support an inference-forward torch-profiler region with per-stage `record_function` ranges. Those stage ranges are gated behind an active profiler so normal latency runs do not pay profiler range overhead. Profiler shutdown is idempotent and calls `profiler.step()` after a top-level profiled region so trace files are flushed.

The earlier `VideoGenerator` materialization optimization remains: when both `save_video=False` and `return_frames=False`, CPU sample materialization and RGB frame construction are skipped after the decoded tensor is synchronized. That path is useful for internal latency-only timing, but it is not the corrected real-output contract.

## Shared Corrected Config

The corrected 4-GPU real-output runs used:

```text
script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py
model_id=FastVideo/LTX2-Distilled-Diffusers
model_root=/home/hal-jundas/.local/share/huggingface/hub/models--FastVideo--LTX2-Distilled-Diffusers/snapshots/0762ece944ea65f45cd3318981423e1670ff7225
validation_json=examples/training/finetune/ltx2/validation.json
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
return_frames=True
stage_logging=False for lowest-overhead corrected latency runs
```

The detailed config files are in each run directory as `profile_config.json`; exact run summaries are in `profile_summary.json`.

## Corrected Results

Each corrected latency result used 12 runs and skipped the first 2 warmups.

| Run | Output contract | Avg generation | Avg e2e | Target result | Directory |
| --- | --- | ---: | ---: | --- | --- |
| `baseline_return_frames_no_fp4_g4_compile_sp4_8x3_no_save` | decoded frames returned, no MP4 save | 4.353493176s | 4.450335541s | misses by 0.153493176s | `outputs_video/ltx2_sp_profile/steps8x3_real_output/baseline_return_frames_no_fp4_g4_compile_sp4_8x3_no_save/` |
| `optimized_profiler_gated_return_frames_no_fp4_g4_compile_sp4_8x3_no_save` | decoded frames returned, no MP4 save | 4.337300085s | 4.437282278s | misses by 0.137300085s | `outputs_video/ltx2_sp_profile/steps8x3_real_output/optimized_profiler_gated_return_frames_no_fp4_g4_compile_sp4_8x3_no_save/` |

The profiler-gated run improved normal timing slightly by avoiding unconditional `torch.profiler.record_function` overhead in non-profiler runs. It did not change model semantics and did not reach the target.

The earlier invalid result remains useful only as a negative control:

```text
optimized_latency_only_decode_skip_no_stage_logging_no_fp4_g4_compile_sp4_8x3_no_save
avg_generation=4.117228775s
avg_e2e=4.117467192s
invalid_for_real_video_frames=True
reason=skipped main VAE pixel decode
```

## Profiling Evidence

A first torch-profiler trace captured the first compiled pass and was useful only to confirm trace plumbing:

```text
trace_dir=/tmp/hal-jundas/ltx2_torch_profiler/real_output_g4_return_frames_one_run_step
profile_run=outputs_video/ltx2_sp_profile/steps8x3_real_output/torch_profiler_return_frames_no_fp4_g4_compile_sp4_8x3_no_save_one_run_step/
generation_time=71.76s
e2e_latency=71.88s
reason_not_latency_evidence=first run included compile/profiler overhead
```

The warmed profiler trace skipped two forwards and captured the third, but torch-profiler still slowed the captured forward to `9.565395680s`. Treat it as attribution evidence, not latency evidence:

```text
trace_dir=/tmp/hal-jundas/ltx2_torch_profiler/real_output_g4_return_frames_steady_run3
profile_run=outputs_video/ltx2_sp_profile/steps8x3_real_output/torch_profiler_return_frames_no_fp4_g4_compile_sp4_8x3_no_save_steady_run3/
captured_generation_time=9.565395680s
captured_e2e_latency=9.661771181s
```

The warmed trace stage ranges across ranks showed the bottleneck order:

```text
base denoising stage:        3.746s - 3.845s under profiler
refine denoising stage:      1.409s - 1.496s under profiler
VAE decoding stage:          0.335s - 0.345s under profiler
prompt encoding stage:       ~0.052s under profiler
audio decoding stage:        0.048s - 0.135s under profiler
latent upsample stage:       0.009s - 0.105s under profiler
largest CUDA comm ranges:    nccl:all_to_all and nccl:all_gather_into_tensor_coalesced
```

The lower-overhead stage-logging baseline from the earlier valid decoded path remains the best normal-stage breakdown:

```text
run=baseline_no_fp4_g4_compile_sp4_8x3_no_save
generation_time=4.335169821s
e2e_latency=4.432635242s
stage_sum_avg=4.400138776s
denoising_stage=2.407450876s
ltx2_upsample_stage=0.034380009s
ltx2_refine_denoising_stage=1.461363664s
decoding_stage=0.335916484s
prompt_encoding_stage=0.041930793s
audio_decoding_stage=0.021789587s
PostDecodeFrameProcessStage=0.096997136s
non_stage_overhead=0.032496466s
```

The actual bottleneck is denoising, not MP4 writing and not RGB frame-list conversion. `PostDecodeFrameProcessStage` explains most of why end-to-end latency is about `0.09s-0.13s` larger than generation time in return-frames runs, but it occurs after the recorded `generation_time`, so optimizing it would not close the generation-time target gap.

## Fullgraph Decision

`compile_fullgraph=False` is still required for sequence-parallel runs. The fullgraph attempt failed because TorchDynamo could not inline the `torch.compiler.disable()`-wrapped `LTXDistributedAttention.forward` path:

```text
torch._dynamo.exc.Unsupported: Skip inlining torch.compiler.disable()d function LTXDistributedAttention.forward
```

This is a graph-break constraint in the distributed attention path, not a benchmark preference.

## Output Videos

The corrected real-output runs used `--no-save-video`, so no MP4 files are written despite the profile metadata listing intended output paths. The returned frames exist in process for each call, but the profile script does not persist them to disk. The run directories contain:

```text
profile.log
profile_config.json
profile_runs_partial.json
profile_summary.json
```

Earlier 5x2 profiling runs that actually saved MP4 files stored them under:

```text
outputs_video/ltx2_sp_profile/full/<run_name>/videos/output_ltx2_basic_t2v_run_<n>.mp4
```

## Validation

Focused CUDA-node tests after the decode correction and profiler instrumentation changes:

```text
srun --overlap --jobid=4745 -N1 -n1 --gpus=1 bash -lc 'cd /home/hal-jundas/codes/FastVideo-ltx2-sp-profile && export TOKENIZERS_PARALLELISM=false && export TORCHINDUCTOR_CACHE_DIR=/tmp/hal-jundas/torchinductor-cache-ltx2-sp-profile && export TRITON_CACHE_DIR=/tmp/hal-jundas/triton-cache-ltx2-sp-profile && export CUDA_CACHE_PATH=/tmp/hal-jundas/cuda-cache-ltx2-sp-profile && export XDG_CACHE_HOME=/tmp/hal-jundas/xdg-cache-ltx2-sp-profile && mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" && /home/hal-jundas/venvs/fv-shared/bin/python -m pytest fastvideo/tests/stages/test_decoding.py fastvideo/tests/entrypoints/test_video_generator.py tests/local_tests/ltx2/test_ltx2_profile_script_config.py -q'
33 passed, 14 warnings in 0.79s
```

All Slurm/GPU work in this follow-up had to run outside the sandbox through escalated `srun --overlap --jobid=4745 ...` commands on `hpc-rack-2-8`. The Python environment was `/home/hal-jundas/venvs/fv-shared`.

## Interpretation Notes

`average generation + average SR` is not expected to equal end-to-end latency. SR is a subset of generation:

```text
generation = low-res denoise + SR upsample/refine + decode + required synchronization/materialization
e2e        = generation + post-decode frame processing + optional save/mux work
```

So adding average generation and average SR double-counts the SR work. Average generation is not average low-resolution generation; low-resolution generation is only the base denoising portion.

# LTX-2 8+3 GPU Stage Breakdown

## Summary

These numbers compare the prior LTX-2 8+3 no-FP4 baseline runs across 1, 2,
and 4 GPUs. They are averaged over measured runs 3-12, after skipping the first
2 warmup runs.

The interesting result is that the base denoising stage is fastest on 1 GPU,
while the refine denoising stage scales strongly with more GPUs.

## Stage Times

All values are seconds.

| Stage | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `input_validation_stage` | 0.000043 | 0.000044 | 0.000040 |
| `prompt_encoding_stage` | 0.040525 | 0.042603 | 0.041931 |
| `latent_preparation_stage` | 0.000246 | 0.000225 | 0.000203 |
| `denoising_stage` | 2.005970 | 2.469087 | 2.407451 |
| `ltx2_upsample_stage` | 0.036221 | 0.034435 | 0.034380 |
| `ltx2_refine_denoising_stage` | 4.076190 | 2.606739 | 1.461364 |
| `audio_decoding_stage` | 0.021346 | 0.022919 | 0.021790 |
| `decoding_stage` | 0.327429 | 0.337648 | 0.335916 |
| `PostDecodeFrameProcessStage` | 0.099859 | 0.094591 | 0.096997 |
| `total(stage sum avg)` | 6.607939 | 5.608361 | 4.400139 |

## Top-Level Metrics

All values are seconds.

| Metric | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---:|---:|
| `video_generation_time` | 6.533543 | 5.541380 | 4.335170 |
| `e2e_latency` | 6.634035 | 5.636504 | 4.432635 |
| `sr_forward_latency` | 4.112411 | 2.641174 | 1.495744 |
| `non_stage_overhead` | 0.026096 | 0.028142 | 0.032496 |

## Alarm Note

The base denoising stage does not scale with sequence parallelism in these
baseline runs:

```text
1 GPU base denoising: 2.005970s
2 GPU base denoising: 2.469087s
4 GPU base denoising: 2.407451s
```

That means the 2-GPU and 4-GPU base denoising runs are slower than the 1-GPU
base denoising run, despite using more GPUs. The refine denoising stage shows
the opposite behavior:

```text
1 GPU refine denoising: 4.076190s
2 GPU refine denoising: 2.606739s
4 GPU refine denoising: 1.461364s
```

This is the main reason to profile a late LTX-2 transformer block separately for
base and refine stages across 1, 2, and 4 GPUs.

## Derived Hybrid Estimate

If the base denoising stage used the 1-GPU timing and the upsample/refine path
used the 4-GPU timing, the stage-based estimate is:

```text
1-GPU denoising_stage              2.005970s
4-GPU ltx2_upsample_stage          0.034380s
4-GPU ltx2_refine_denoising_stage  1.461364s
4-GPU remaining generation stages  0.399947s
estimated generation time          3.901661s
estimated e2e latency              4.031154s
```

This estimate ignores any cost to move or repartition latents between the 1-GPU
base path and the 4-GPU refine path.

## Specific Config

The compared runs used the same 8+3 no-FP4 baseline configuration:

```text
script=examples/inference/basic/basic_ltx2_distilled_fast_profile.py
model_id=FastVideo/LTX2-Distilled-Diffusers
validation_json=examples/training/finetune/ltx2/validation.json
prompt=first entry from validation_json
num_frames=121
num_inference_steps=8
refine_num_inference_steps=3
num_runs=12
warmup_runs=2
avg_window=10
torch_compile=true
compile_text_encoder=true
compile_vae=true
compile_fullgraph=script default
fp4_linear=false
nvfp4_fa4=false
save_video=false
return_frames=false
stage_logging=true
tp_size=1
```

GPU-specific run settings:

| Run | GPUs | SP | Run Directory |
|---|---:|---:|---|
| 1-GPU baseline | 1 | 1 | `outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g1_compile_sp1_8x3_no_save/` |
| 2-GPU baseline | 2 | 2 | `outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g2_compile_sp2_8x3_no_save/` |
| 4-GPU baseline | 4 | 4 | `outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g4_compile_sp4_8x3_no_save/` |

The profile script default made `compile_fullgraph=true` for SP=1 and
`compile_fullgraph=false` for SP=2/4.

## Source Files

```text
outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g1_compile_sp1_8x3_no_save/profile.log
outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g1_compile_sp1_8x3_no_save/profile_summary.json

outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g2_compile_sp2_8x3_no_save/profile.log
outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g2_compile_sp2_8x3_no_save/profile_summary.json

outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g4_compile_sp4_8x3_no_save/profile.log
outputs_video/ltx2_sp_profile/steps8x3_baseline/baseline_no_fp4_g4_compile_sp4_8x3_no_save/profile_summary.json
```

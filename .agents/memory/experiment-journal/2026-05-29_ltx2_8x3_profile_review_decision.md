# Review Decision: LTX-2 8x3 Profiling Optimizations

Date: 2026-05-29

Decision: reject the target-achieved interpretation of the final optimization result for the real video-serving use case. The experiment assumed that `save_video=False` and `return_frames=False` meant generated frames were not required. That assumption is false. The profile script disabled saving only to avoid measuring file I/O and encoder cost; the actual system still needs the full generation stack to produce real decoded video frames.

## Mental Model

The LTX-2 pipeline has to generate final refined latents and then decode those latents into actual pixel frames. Saving an MP4 is a later output step. Disabling MP4 saving does not make VAE decoding or frame construction optional if the real product requirement is "generate video frames."

The correct benchmark boundary is:

```text
prompt
  -> text/audio conditioning
  -> low-resolution denoise
  -> latent upsample / SR
  -> high-resolution refine denoise
  -> VAE pixel decode
  -> frame tensor usable by the caller
  -> optional MP4 save, excluded from this benchmark
```

The invalid shortcut measured:

```text
prompt
  -> text/audio conditioning
  -> low-resolution denoise
  -> latent upsample / SR
  -> high-resolution refine denoise
  -> skip VAE pixel decode
  -> skip frame construction
  -> report timing
```

That shortcut is not a valid optimization for a workload that needs video frames. If this kind of shortcut were acceptable, the benchmark could also skip denoising and claim a faster number, which would clearly no longer measure video generation.

## Issues Found

### 1. False Benchmark Assumption

The optimization treated `save_video=False` and `return_frames=False` as a semantic signal that decoded video frames are unnecessary. In this experiment, that was only an artifact of the example profile script. The user wanted to exclude video file saving, not skip the core generation components needed to produce real frames.

Impact: the final `4.117228775s` result is not a valid result for the actual use case. It measures a latent/timing-only shortcut, not complete video-frame generation.

### 2. VAE Decode Skip Removes A Required Component

The change in `fastvideo/pipelines/stages/decoding.py` skips the main VAE pixel decode when `save_video=False`, `return_frames=False`, and decoded trajectories are not requested. For a real video generation benchmark, VAE decoding is part of the required component stack. It should not be skipped just because the example script does not save an MP4.

Impact: the target-achieving result is invalid for any SLA that requires generated frames. It should not be presented as "generation time" for the real workload.

Required action: revert this optimization for the production/default video path, or move it behind an explicit, clearly named latent-only benchmark mode that cannot be confused with video generation. Any report must label it as "latent-only, no decoded frames."

### 3. CPU Materialization / RGB Frame Construction Result Is Not Representative If Frames Are Required

The `VideoGenerator` materialization optimization skips CPU samples and RGB frame construction when neither saving nor returned frames are requested. That behavior may be acceptable for a true internal timing-only API, but it is not representative of a workload where callers need returned frames.

Impact: the `4.206548934s` near-miss result still does not represent the real return-frames path if the actual use case requires `return_frames=True`.

Required action: run a dedicated benchmark with the real output contract, at minimum with decoded pixel frames produced and returned in the same form the production caller needs. If MP4 saving is out of scope, keep MP4 writing disabled, but do not skip frame decode/construction needed by the caller.

### 4. Profiling Work Should Identify Bottlenecks, Not Remove Required Stages

The requested optimization approach is to analyze profiling output, identify bottlenecks, and improve the bottleneck while preserving the semantic work of video generation. Skipping VAE decode or frame construction is not equivalent to optimizing those components.

Valid examples of future optimization directions include faster VAE decode, overlapping decode/materialization with later work where semantics allow it, reducing unnecessary synchronization, improving tensor layout conversions, tuning compile settings that preserve output, improving sequence-parallel bottlenecks, or optimizing CPU frame conversion without dropping it.

## Results Reinterpretation

The results should be interpreted as:

```text
4.276347051s  valid no-stage-logging 4-GPU result for the original decoded path before output shortcuts
4.206548935s  near-miss when CPU output materialization/frame conversion are skipped; not representative if return_frames=True
4.117228775s  invalid for real video-frame generation; skips VAE pixel decode
```

The `4.117s` number must not be used to claim that the real video generation target was achieved.

## Required Actions

1. Revert or isolate the VAE decode skip in `fastvideo/pipelines/stages/decoding.py` so normal video generation always produces decoded frames.
2. Update the experiment memory and any final report to say the target was not achieved for the real video-frame-producing workload.
3. Re-run profiling with the real output contract: decoded frames must be produced, MP4 saving may remain disabled if file I/O is intentionally out of scope.
4. Use stage-level profiling to identify the actual bottleneck, then optimize that bottleneck without removing required pipeline stages.
5. Treat the no-stage-logging result as the only broadly valid optimization from this run unless a future production path explicitly does not need frames.

## Review Outcome

The code and memory from commit `b6cce56e` need follow-up correction before they can be considered valid for the user's intended workload. The review decision is to reject the "target achieved" conclusion and require a corrected benchmark and optimization pass that preserves actual frame generation.

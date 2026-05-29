from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.decoding import DecodingStage


def _fastvideo_args(output_type="pil"):
    return SimpleNamespace(
        disable_autocast=False,
        model_loaded={"vae": True},
        model_paths={"vae": "unused"},
        output_type=output_type,
        pipeline_config=SimpleNamespace(vae_precision="bf16", vae_tiling=False),
        vae_cpu_offload=False,
    )


def _batch(save_video=False, return_frames=False, return_trajectory_decoded=False):
    return ForwardBatch(
        data_type="video",
        latents=torch.ones((1, 16, 3, 4, 4)),
        save_video=save_video,
        return_frames=return_frames,
        return_trajectory_decoded=return_trajectory_decoded,
    )


def test_decoding_stage_skips_pixel_decode_for_latency_only_calls():
    batch = _batch(save_video=False, return_frames=False)
    stage = DecodingStage(vae=object())

    def fail_decode(*args, **kwargs):
        raise AssertionError("latency-only calls should not decode discarded pixel frames")

    stage.decode = fail_decode

    result = stage.forward(batch, _fastvideo_args())

    assert result.output is batch.latents


@pytest.mark.parametrize(
    ("save_video", "return_frames"),
    [
        (True, False),
        (False, True),
    ],
)
def test_decoding_stage_keeps_pixel_decode_for_visible_video_outputs(save_video, return_frames):
    batch = _batch(save_video=save_video, return_frames=return_frames)
    stage = DecodingStage(vae=object())
    calls = []

    def fake_decode(latents, fastvideo_args):
        calls.append(latents)
        return torch.ones((1, 3, 3, 8, 8), dtype=torch.bfloat16)

    stage.decode = fake_decode

    result = stage.forward(batch, _fastvideo_args())

    assert calls == [batch.latents]
    assert result.output.shape == (1, 3, 3, 8, 8)
    assert result.output.dtype == torch.float32


def test_decoding_stage_keeps_pixel_decode_for_decoded_trajectories():
    batch = _batch(return_trajectory_decoded=True)
    batch.trajectory_latents = torch.ones((1, 2, 16, 3, 4, 4))
    batch.trajectory_timesteps = [torch.tensor(2), torch.tensor(1)]
    stage = DecodingStage(vae=object())
    calls = []

    def fake_decode(latents, fastvideo_args):
        calls.append(latents)
        return torch.ones((1, 3, 3, 8, 8), dtype=torch.bfloat16)

    stage.decode = fake_decode

    result = stage.forward(batch, _fastvideo_args())

    assert calls[0] is batch.latents
    assert len(calls) == 3
    assert result.output.shape == (1, 3, 3, 8, 8)
    assert len(result.trajectory_decoded) == 2

# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest


def _load_profile_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "examples" / "inference" / "basic" / "basic_ltx2_distilled_fast_profile.py"
    spec = importlib.util.spec_from_file_location("basic_ltx2_distilled_fast_profile", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ltx2_profile_defaults_match_sweep_defaults():
    profile = _load_profile_module()

    args = profile.parse_args([])
    profile.validate_args(args)

    assert args.num_gpus == 1
    assert args.num_inference_steps == 8
    assert args.refine_num_inference_steps == 3
    assert args.fp4_linear is True
    assert args.compile_fullgraph is True


@pytest.mark.parametrize(
    ("base_steps", "refine_steps"),
    [
        (5, 2),
        (8, 3),
    ],
)
def test_ltx2_profile_accepts_only_sweep_step_pairs(base_steps, refine_steps):
    profile = _load_profile_module()

    args = profile.parse_args([
        "--num-inference-steps",
        str(base_steps),
        "--refine-num-inference-steps",
        str(refine_steps),
    ])

    profile.validate_args(args)


def test_ltx2_profile_rejects_mixed_step_pairs():
    profile = _load_profile_module()

    args = profile.parse_args([
        "--num-inference-steps",
        "5",
        "--refine-num-inference-steps",
        "3",
    ])

    with pytest.raises(ValueError, match="Expected 5\\+2 or 8\\+3"):
        profile.validate_args(args)


def test_ltx2_profile_run_name_uses_only_sweep_dimensions():
    profile = _load_profile_module()

    args = profile.parse_args([
        "--num-gpus",
        "4",
        "--num-inference-steps",
        "5",
        "--refine-num-inference-steps",
        "2",
        "--no-fp4-linear",
        "--no-compile-fullgraph",
    ])
    profile.validate_args(args)

    assert profile.run_name(args) == "ltx2_speed_s5p2_g4_fp4off_fgoff"


def test_ltx2_profile_config_records_tuned_and_fixed_settings():
    profile = _load_profile_module()

    args = profile.parse_args([
        "--num-gpus",
        "2",
        "--num-inference-steps",
        "5",
        "--refine-num-inference-steps",
        "2",
        "--no-fp4-linear",
        "--compile-fullgraph",
    ])
    profile.validate_args(args)
    compile_kwargs = profile.build_torch_compile_kwargs(args)
    config = profile.build_profile_config(
        args=args,
        model_root="/tmp/model",
        refine_upsampler_path=Path("/tmp/model/spatial_upsampler"),
        prompt="profile prompt",
        benchmark_entry={
            "height": 1088,
            "width": 1920,
        },
        torch_compile_kwargs=compile_kwargs,
        pipeline_config={
            "name": "dummy",
        },
    )

    assert config["tuned"] == {
        "num_inference_steps": 5,
        "refine_num_inference_steps": 2,
        "num_gpus": 2,
        "sp_size": 2,
        "fp4_linear": False,
        "compile_fullgraph": True,
    }
    assert config["fixed"]["save_video"] is False
    assert config["fixed"]["return_frames"] is True
    assert config["fixed"]["stage_logging"] is True
    assert config["fixed"]["nvfp4_fa4"] is False
    assert config["fixed"]["tp_size"] == 1
    assert config["resolved"]["torch_compile_kwargs"] == {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
    }

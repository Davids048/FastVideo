# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path


def _load_profile_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "examples" / "inference" / "basic" / "basic_ltx2_distilled_fast_profile.py"
    spec = importlib.util.spec_from_file_location("basic_ltx2_distilled_fast_profile", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ltx2_profile_defaults_use_eight_base_steps_and_three_refine_steps():
    profile = _load_profile_module()

    args = profile.parse_args([])
    profile.validate_args(args)

    assert args.num_inference_steps == 8
    assert args.refine_num_inference_steps == 3
    assert args.stage_logging is True


def test_ltx2_profile_sp_defaults_to_num_gpus_and_graph_break_compile():
    profile = _load_profile_module()

    args = profile.parse_args(["--num-gpus", "4"])
    profile.validate_args(args)

    assert args.sp_size == 4
    assert args.compile_fullgraph is False


def test_ltx2_profile_explicit_fullgraph_override_is_preserved():
    profile = _load_profile_module()

    args = profile.parse_args(["--num-gpus", "4", "--compile-fullgraph"])
    profile.validate_args(args)

    assert args.sp_size == 4
    assert args.compile_fullgraph is True


def test_ltx2_profile_config_records_base_and_refine_step_counts():
    profile = _load_profile_module()

    args = profile.parse_args(["--num-gpus", "2", "--no-save-video", "--no-stage-logging"])
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

    assert config["resolved"]["num_inference_steps"] == 8
    assert config["resolved"]["refine_num_inference_steps"] == 3
    assert config["resolved"]["save_video"] is False
    assert config["resolved"]["return_frames"] is False
    assert config["resolved"]["stage_logging_enabled"] is False
    assert config["resolved"]["sequence_parallel"]["sp_size"] == 2
    assert config["resolved"]["torch_compile_kwargs"]["fullgraph"] is False


def test_ltx2_profile_configure_environment_can_disable_stage_logging(monkeypatch):
    profile = _load_profile_module()

    monkeypatch.delenv("FASTVIDEO_STAGE_LOGGING", raising=False)
    args = profile.parse_args(["--no-stage-logging"])
    profile.validate_args(args)

    profile.configure_environment(args)

    assert profile.os.environ["FASTVIDEO_STAGE_LOGGING"] == "0"

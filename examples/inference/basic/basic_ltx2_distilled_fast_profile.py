# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
import torch._inductor.config

VALIDATION_JSON = (
    Path(__file__).resolve().parents[2] / "training" / "finetune" / "ltx2" / "validation.json"
)
DEFAULT_MODEL_ID = "FastVideo/LTX2-Distilled-Diffusers"
DEFAULT_PROFILE_DIR = Path(os.getenv("LTX2_PROFILE_DIR", "outputs_video/ltx2_distilled_fast_profile"))
DEFAULT_NUM_INFERENCE_STEPS = 8
DEFAULT_REFINE_NUM_INFERENCE_STEPS = 3

ENV_KEYS_TO_RECORD = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_CACHE_PATH",
    "FASTVIDEO_ATTENTION_BACKEND",
    "FASTVIDEO_LTX2_BLOCK_PROFILE_ACTIVE_OCCURRENCES",
    "FASTVIDEO_LTX2_BLOCK_PROFILE_CAPTURE_RANGE",
    "FASTVIDEO_LTX2_BLOCK_PROFILE_INDEX",
    "FASTVIDEO_LTX2_BLOCK_PROFILE_SKIP_OCCURRENCES",
    "FASTVIDEO_LTX2_BLOCK_PROFILE_STAGE",
    "FASTVIDEO_LOGGING_LEVEL",
    "FASTVIDEO_NVFP4_FA4",
    "FASTVIDEO_SR_LATENCY_STAGE_SUBSTR",
    "FASTVIDEO_STAGE_LOGGING",
    "FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS",
    "FASTVIDEO_TORCH_PROFILER_DIR",
    "FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES",
    "FASTVIDEO_TORCH_PROFILER_WAIT_STEPS",
    "FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS",
    "FASTVIDEO_TORCH_PROFILER_WITH_FLOPS",
    "FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY",
    "FASTVIDEO_TORCH_PROFILER_WITH_STACK",
    "FASTVIDEO_TORCH_PROFILE_REGIONS",
    "HF_HOME",
    "HF_HUB_CACHE",
    "LTX2_MODEL_PATH",
    "LTX2_PROFILE_DIR",
    "LTX2_REFINE_UPSAMPLER_PATH",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_ASYNC_ERROR_HANDLING",
    "PYTORCH_CUDA_ALLOC_CONF",
    "RANK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_PROCID",
    "SLURM_STEP_ID",
    "SLURM_STEP_NODELIST",
    "TOKENIZERS_PARALLELISM",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRANSFORMERS_CACHE",
    "TRITON_CACHE_DIR",
    "TQDM_DISABLE",
    "WORLD_SIZE",
    "XDG_CACHE_HOME",
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile LTX-2 distilled inference with optional Ulysses sequence parallelism.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=os.getenv("LTX2_MODEL_PATH", DEFAULT_MODEL_ID))
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--run-name", default=None, help="Subdirectory name under --profile-dir.")
    parser.add_argument("--validation-json", type=Path, default=VALIDATION_JSON)
    parser.add_argument("--num-gpus", type=int, default=int(os.getenv("FASTVIDEO_PROFILE_NUM_GPUS", "1")))
    parser.add_argument("--sp-size", type=int, default=None, help="Sequence parallel size. Defaults to num_gpus.")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--distributed-executor-backend", choices=("mp", "ray"), default="mp")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--num-runs", type=int, default=12)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--avg-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--refine-num-inference-steps", type=int, default=DEFAULT_REFINE_NUM_INFERENCE_STEPS)
    parser.add_argument("--refine-guidance-scale", type=float, default=1.0)
    parser.add_argument("--refine-add-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ltx2-vae-tiling", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--fp4-linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LTX-2 NVFP4 quantization for selected linear layers.",
    )
    parser.add_argument(
        "--nvfp4-fa4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable FP4 Q/K FlashAttention 4. Leave disabled for no-FP4 experiments.",
    )
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-text-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-vae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to true for single-GPU and false when sequence parallelism is enabled.",
    )
    parser.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--stage-logging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-stage timing logs. Disable for lowest-overhead latency runs.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.sp_size is None:
        args.sp_size = args.num_gpus
    if args.num_gpus < 1:
        raise ValueError(f"--num-gpus must be >= 1, got {args.num_gpus}")
    if args.tp_size < 1:
        raise ValueError(f"--tp-size must be >= 1, got {args.tp_size}")
    if args.sp_size < 1:
        raise ValueError(f"--sp-size must be >= 1, got {args.sp_size}")
    if args.sp_size > args.num_gpus or args.num_gpus % args.sp_size != 0:
        raise ValueError(f"--num-gpus ({args.num_gpus}) must be divisible by --sp-size ({args.sp_size})")
    if args.num_runs <= 0:
        raise ValueError(f"--num-runs must be > 0, got {args.num_runs}")
    if args.warmup_runs < 0:
        raise ValueError(f"--warmup-runs must be >= 0, got {args.warmup_runs}")
    if args.warmup_runs >= args.num_runs:
        raise ValueError("--warmup-runs must be smaller than --num-runs")
    if args.avg_window is not None and args.avg_window <= 0:
        raise ValueError(f"--avg-window must be > 0, got {args.avg_window}")
    if args.compile_fullgraph is None:
        args.compile_fullgraph = args.sp_size <= 1


def make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    linear_mode = "linear-fp4" if args.fp4_linear else "linear-bf16"
    fa4_mode = "fa4-fp4qk" if args.nvfp4_fa4 else "fa4-bf16qk"
    compile_mode = "compile" if args.torch_compile else "eager"
    run_name = args.run_name or f"{timestamp}_g{args.num_gpus}_sp{args.sp_size}_{linear_mode}_{fa4_mode}_{compile_mode}"
    run_dir = args.profile_dir / run_name
    if not run_dir.exists():
        return run_dir
    suffix = 2
    while True:
        candidate = args.profile_dir / f"{run_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def load_validation_entries(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [entry for entry in data["data"] if isinstance(entry, dict)]

    raise ValueError(f"Unsupported validation format in {path}. Expected {{'data': [...]}}.")


def extract_stage_times(result: dict) -> OrderedDict[str, float]:
    logging_info = result.get("logging_info")
    if logging_info is None:
        return OrderedDict()

    stages = getattr(logging_info, "stages", None)
    if not stages:
        return OrderedDict()

    stage_times: OrderedDict[str, float] = OrderedDict()
    for stage_name, stage_metrics in stages.items():
        stage_times[stage_name] = float(stage_metrics.get("execution_time", 0.0))
    return stage_times


def print_stage_breakdown(
    result: dict,
    run_idx: int,
    num_runs: int,
) -> float | None:
    stage_times = extract_stage_times(result)
    if not stage_times:
        print(f"[{run_idx}/{num_runs}] Stage breakdown unavailable: no stage timings")
        return None

    print(f"[{run_idx}/{num_runs}] Stage breakdown:")
    total = 0.0
    for stage_name, exec_time in stage_times.items():
        total += exec_time
        print(f"  - {stage_name}: {exec_time:.3f}s")
    print(f"  - total(stage sum): {total:.3f}s")
    return total


def extract_sr_forward_latency(
    result: dict,
) -> tuple[float | None, list[tuple[str, float]], list[str]]:
    stage_times = extract_stage_times(result)
    if not stage_times:
        return None, [], []

    stage_names = list(stage_times.keys())
    sr_match_substr = os.getenv("FASTVIDEO_SR_LATENCY_STAGE_SUBSTR", "").strip().lower()

    sr_stage_entries: list[tuple[str, float]] = []
    for stage_name, exec_time in stage_times.items():
        stage_name_l = stage_name.lower()
        if sr_match_substr:
            is_sr_stage = sr_match_substr in stage_name_l
        else:
            is_sr_stage = (
                "srdenoisingstage" in stage_name_l
                or "sr_denoising" in stage_name_l
                or "upsample" in stage_name_l
                or ("refine" in stage_name_l and "denois" in stage_name_l)
            )
        if is_sr_stage:
            sr_stage_entries.append((stage_name, exec_time))

    if not sr_stage_entries:
        return None, [], stage_names
    return sum(x[1] for x in sr_stage_entries), sr_stage_entries, stage_names


def collect_stage_times(
    result: dict,
    stage_times: dict[str, list[float]],
    stage_order: OrderedDict[str, None],
) -> None:
    for stage_name, exec_time in extract_stage_times(result).items():
        stage_order.setdefault(stage_name, None)
        stage_times.setdefault(stage_name, []).append(exec_time)


def print_stage_averages(
    stage_times: dict[str, list[float]],
    stage_order: OrderedDict[str, None],
    measured_runs: int,
) -> OrderedDict[str, float]:
    averages: OrderedDict[str, float] = OrderedDict()
    if measured_runs <= 0:
        return averages
    if not stage_times:
        print("No stage timings collected for measured runs.")
        return averages

    print(f"Average stage times over {measured_runs} measured runs:")
    total_avg = 0.0
    for stage_name in stage_order.keys():
        times = stage_times.get(stage_name, [])
        if not times:
            continue
        avg = sum(times) / len(times)
        averages[stage_name] = avg
        total_avg += avg
        print(f"  - {stage_name}: {avg:.3f}s")
    averages["total(stage sum avg)"] = total_avg
    print(f"  - total(stage sum avg): {total_avg:.3f}s")
    return averages


def resolve_refine_upsampler_path(model_root: str) -> Path:
    root = Path(model_root)
    candidates = [
        root / "spatial_upscaler",
        root / "spatial_upsampler",
    ]

    env_path = os.getenv("LTX2_REFINE_UPSAMPLER_PATH")
    if env_path:
        candidates.insert(0, Path(os.path.expandvars(os.path.expanduser(env_path))))

    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find an LTX2 refine upsampler directory.\n"
        "Checked:\n"
        f"{checked}"
    )


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.attention_backend
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1" if args.stage_logging else "0"
    os.environ["FASTVIDEO_NVFP4_FA4"] = "1" if args.nvfp4_fa4 else "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.nvfp4_fa4:
        os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")


def configure_inductor() -> None:
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False


def build_torch_compile_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    compile_kwargs = {
        "backend": args.compile_backend,
        "fullgraph": args.compile_fullgraph,
        "dynamic": args.compile_dynamic,
    }
    if args.compile_mode:
        compile_kwargs["mode"] = args.compile_mode
    return compile_kwargs


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(v) for v in value]
    get_name = getattr(value, "get_name", None)
    if callable(get_name):
        try:
            name = get_name()
        except Exception:
            name = None
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "name": name,
            "repr": repr(value),
        }
    return repr(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, str]:
    return {key: os.environ[key] for key in ENV_KEYS_TO_RECORD if key in os.environ}


def runtime_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        "packages": {
            "fastvideo": package_version("fastvideo"),
            "flash-attn": package_version("flash-attn"),
            "flash-attn-cute": package_version("flash-attn-cute"),
            "flashinfer-python": package_version("flashinfer-python"),
            "nvidia-cutlass-dsl": package_version("nvidia-cutlass-dsl"),
            "torch-c-dlpack-ext": package_version("torch-c-dlpack-ext"),
            "quack-kernels": package_version("quack-kernels"),
        },
        "git": {
            "worktree": str(Path(__file__).resolve().parents[3]),
            "branch": run_git(["branch", "--show-current"]),
            "head": run_git(["rev-parse", "HEAD"]),
            "upstream_main": run_git(["rev-parse", "upstream/main"]),
            "status_short": run_git(["status", "--short", "--branch"]),
        },
        "env": environment_snapshot(),
        "argv": sys.argv,
    }


def build_profile_config(
    *,
    args: argparse.Namespace,
    model_root: str,
    refine_upsampler_path: Path,
    prompt: str,
    benchmark_entry: dict[str, Any],
    torch_compile_kwargs: dict[str, Any],
    pipeline_config: Any,
) -> dict[str, Any]:
    height = args.height if args.height is not None else benchmark_entry.get("height", 1088)
    width = args.width if args.width is not None else benchmark_entry.get("width", 1920)
    return {
        "args": vars(args),
        "resolved": {
            "model_root": model_root,
            "refine_upsampler_path": str(refine_upsampler_path),
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "refine_num_inference_steps": args.refine_num_inference_steps,
            "fps": args.fps,
            "save_video": args.save_video,
            "return_frames": args.return_frames,
            "stage_logging_enabled": args.stage_logging,
            "linear_fp4_enabled": args.fp4_linear,
            "nvfp4_fa4_enabled": args.nvfp4_fa4,
            "torch_compile_enabled": args.torch_compile,
            "torch_compile_kwargs": torch_compile_kwargs,
            "sequence_parallel": {
                "num_gpus": args.num_gpus,
                "tp_size": args.tp_size,
                "sp_size": args.sp_size,
                "ulysses_sequence_parallel": args.sp_size > 1,
            },
        },
        "pipeline_config": pipeline_config,
        "runtime": runtime_snapshot(),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_dir = make_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    video_dir = run_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "profile.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        try:
            configure_environment(args)
            configure_inductor()

            from fastvideo import VideoGenerator
            from fastvideo.configs.pipelines.base import PipelineConfig
            from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
            from fastvideo.utils import maybe_download_model

            if not args.validation_json.exists():
                raise FileNotFoundError(f"Validation file not found: {args.validation_json}")

            validation_entries = load_validation_entries(args.validation_json)
            if not validation_entries:
                raise ValueError(f"No validation entries found in {args.validation_json}")

            benchmark_entry = validation_entries[0]
            prompt = benchmark_entry.get("caption")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("First validation entry is missing a usable caption")

            avg_window = args.avg_window if args.avg_window is not None else args.num_runs - args.warmup_runs
            measured_start_idx = max(args.warmup_runs, args.num_runs - avg_window)

            model_id = os.path.expandvars(os.path.expanduser(args.model_id))
            model_root = maybe_download_model(model_id)
            refine_upsampler_path = resolve_refine_upsampler_path(model_root)
            print(f"Profile run directory: {run_dir}")
            print(f"Using model root: {model_root}")
            print(f"Using refine upsampler: {refine_upsampler_path}")
            print(
                "Parallel config: "
                f"num_gpus={args.num_gpus}, tp_size={args.tp_size}, sp_size={args.sp_size}"
            )
            print(
                "FP4 config: "
                f"linear_fp4={args.fp4_linear}, nvfp4_fa4={args.nvfp4_fa4}"
            )

            pipeline_config = PipelineConfig.from_pretrained(model_root)
            pipeline_config.dit_config.quant_config = NVFP4Config() if args.fp4_linear else None
            torch_compile_kwargs = build_torch_compile_kwargs(args)

            profile_config = build_profile_config(
                args=args,
                model_root=model_root,
                refine_upsampler_path=refine_upsampler_path,
                prompt=prompt,
                benchmark_entry=benchmark_entry,
                torch_compile_kwargs=torch_compile_kwargs,
                pipeline_config=pipeline_config,
            )
            write_json(run_dir / "profile_config.json", profile_config)

            generator = VideoGenerator.from_pretrained(
                model_root,
                num_gpus=args.num_gpus,
                tp_size=args.tp_size,
                sp_size=args.sp_size,
                distributed_executor_backend=args.distributed_executor_backend,
                nvfp4_fa4=args.nvfp4_fa4,
                ltx2_refine_enabled=True,
                ltx2_refine_upsampler_path=str(refine_upsampler_path),
                refine_lora_path="",
                ltx2_refine_lora_path="",
                ltx2_refine_num_inference_steps=args.refine_num_inference_steps,
                ltx2_refine_guidance_scale=args.refine_guidance_scale,
                ltx2_refine_add_noise=args.refine_add_noise,
                pipeline_config=pipeline_config,
                enable_torch_compile=args.torch_compile,
                enable_torch_compile_text_encoder=args.torch_compile and args.compile_text_encoder,
                enable_torch_compile_vae=args.torch_compile and args.compile_vae,
                torch_compile_kwargs=torch_compile_kwargs,
                torch_compile_kwargs_vae=torch_compile_kwargs,
                dit_cpu_offload=False,
                text_encoder_cpu_offload=False,
                vae_cpu_offload=False,
                ltx2_vae_tiling=args.ltx2_vae_tiling,
            )

            run_times: list[float] = []
            e2e_times: list[float] = []
            sr_forward_times: list[float] = []
            non_stage_overhead_times: list[float] = []
            stage_times: dict[str, list[float]] = {}
            stage_order: OrderedDict[str, None] = OrderedDict()
            run_records: list[dict[str, Any]] = []

            try:
                for i in range(args.num_runs):
                    output_path = video_dir / f"output_ltx2_basic_t2v_run_{i + 1}.mp4"
                    if output_path.exists():
                        output_path.unlink()
                        print(f"[{i + 1}/{args.num_runs}] Removed existing file: {output_path}")

                    print(f"[{i + 1}/{args.num_runs}] Generating: {output_path}")
                    if os.environ.get("FASTVIDEO_STAGE_LOGGING") == "0" and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    start = time.perf_counter()
                    result = generator.generate_video(
                        prompt=prompt,
                        output_path=str(output_path),
                        fps=args.fps,
                        seed=args.seed,
                        save_video=args.save_video,
                        return_frames=args.return_frames,
                        guidance_scale=args.guidance_scale,
                        height=args.height if args.height is not None else benchmark_entry.get("height", 1088),
                        width=args.width if args.width is not None else benchmark_entry.get("width", 1920),
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                    )
                    if os.environ.get("FASTVIDEO_STAGE_LOGGING") == "0" and torch.cuda.is_available():
                        torch.cuda.synchronize()

                    elapsed = result.get("generation_time") if isinstance(result, dict) else None
                    e2e_elapsed = result.get("e2e_latency") if isinstance(result, dict) else None
                    if elapsed is None:
                        elapsed = time.perf_counter() - start
                    if e2e_elapsed is None:
                        e2e_elapsed = time.perf_counter() - start

                    run_times.append(elapsed)
                    e2e_times.append(e2e_elapsed)
                    print(f"[{i + 1}/{args.num_runs}] Generation time: {elapsed:.2f}s")
                    print(f"[{i + 1}/{args.num_runs}] End-to-end latency: {e2e_elapsed:.2f}s")

                    run_record: dict[str, Any] = {
                        "run_index": i + 1,
                        "measured": i >= measured_start_idx,
                        "generation_time": elapsed,
                        "e2e_latency": e2e_elapsed,
                        "output_path": str(output_path),
                    }

                    if isinstance(result, dict):
                        per_stage_times = extract_stage_times(result)
                        stage_sum = print_stage_breakdown(result, i + 1, args.num_runs)
                        run_record["stage_times"] = per_stage_times
                        if stage_sum is not None:
                            non_stage_overhead = e2e_elapsed - stage_sum
                            run_record["stage_sum"] = stage_sum
                            run_record["non_stage_overhead"] = non_stage_overhead
                            print(
                                f"[{i + 1}/{args.num_runs}] Non-stage overhead "
                                f"(e2e - stage sum): {non_stage_overhead:.3f}s"
                            )
                            if i >= measured_start_idx:
                                non_stage_overhead_times.append(non_stage_overhead)

                        sr_forward_total, sr_stage_entries, stage_names = extract_sr_forward_latency(result)
                        run_record["sr_stage_entries"] = sr_stage_entries
                        if sr_forward_total is None:
                            print(f"[{i + 1}/{args.num_runs}] SR forward latency unavailable")
                            if stage_names:
                                print(f"    Available stage keys: {', '.join(stage_names)}")
                                print(
                                    "    Tip: set FASTVIDEO_SR_LATENCY_STAGE_SUBSTR="
                                    "<substring> to match your SR stage key."
                                )
                        else:
                            run_record["sr_forward_latency"] = sr_forward_total
                            print(f"[{i + 1}/{args.num_runs}] SR forward latency: {sr_forward_total:.3f}s")
                            for sr_stage_name, sr_exec_time in sr_stage_entries:
                                print(f"    - {sr_stage_name}: {sr_exec_time:.3f}s")
                            if i >= measured_start_idx:
                                sr_forward_times.append(sr_forward_total)

                        if i >= measured_start_idx:
                            collect_stage_times(result, stage_times, stage_order)

                    run_records.append(run_record)
                    write_json(run_dir / "profile_runs_partial.json", {"runs": run_records})

                measured_times = run_times[measured_start_idx:]
                avg_time = sum(measured_times) / len(measured_times)
                print(
                    f"Average video generation time over {len(measured_times)} runs "
                    f"(runs {measured_start_idx + 1}-{len(run_times)}, skipping first {args.warmup_runs} warmup runs): "
                    f"{avg_time:.2f}s"
                )

                measured_e2e_times = e2e_times[measured_start_idx:]
                avg_e2e_time = sum(measured_e2e_times) / len(measured_e2e_times)
                print(
                    f"Average end-to-end latency over {len(measured_e2e_times)} runs "
                    f"(runs {measured_start_idx + 1}-{len(e2e_times)}, skipping first {args.warmup_runs} warmup runs): "
                    f"{avg_e2e_time:.2f}s"
                )

                avg_sr_forward = None
                if sr_forward_times:
                    avg_sr_forward = sum(sr_forward_times) / len(sr_forward_times)
                    print(f"Average SR forward latency over {len(sr_forward_times)} runs: {avg_sr_forward:.3f}s")
                else:
                    print("Average SR forward latency unavailable (no SR stages matched).")

                stage_averages = print_stage_averages(stage_times, stage_order, len(measured_times))

                avg_non_stage_overhead = None
                if non_stage_overhead_times:
                    avg_non_stage_overhead = sum(non_stage_overhead_times) / len(non_stage_overhead_times)
                    print(
                        "Average non-stage overhead over "
                        f"{len(non_stage_overhead_times)} measured runs: {avg_non_stage_overhead:.3f}s"
                    )
                else:
                    print("Average non-stage overhead unavailable (no stage timings).")

                summary = {
                    "averages": {
                        "video_generation_time": avg_time,
                        "e2e_latency": avg_e2e_time,
                        "sr_forward_latency": avg_sr_forward,
                        "non_stage_overhead": avg_non_stage_overhead,
                        "stage_times": stage_averages,
                    },
                    "measured_start_run": measured_start_idx + 1,
                    "measured_runs": len(measured_times),
                    "runs": run_records,
                    "profile_config_path": str(run_dir / "profile_config.json"),
                    "profile_log_path": str(log_path),
                }
                write_json(run_dir / "profile_summary.json", summary)
                print(f"Saved profile config: {run_dir / 'profile_config.json'}")
                print(f"Saved profile summary: {run_dir / 'profile_summary.json'}")
                print(f"Saved profile log: {log_path}")
            finally:
                generator.shutdown()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch._inductor.config
from fastvideo import VideoGenerator
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
from fastvideo.utils import maybe_download_model

VALIDATION_JSON = (
    Path(__file__).resolve().parents[2] / "training" / "finetune" / "ltx2" / "validation.json"
)
MODEL_ID = os.path.expandvars(
    os.path.expanduser(os.getenv("LTX2_MODEL_PATH", "FastVideo/LTX2-Distilled-Diffusers"))
)
OUTPUT_ROOT = Path("outputs_video/ltx2_generation_speed_sweep")

NUM_RUNS = 12
WARMUP_RUNS = 2
TP_SIZE = 1
DISTRIBUTED_EXECUTOR_BACKEND = "mp"
ATTENTION_BACKEND = "FLASH_ATTN"
NUM_FRAMES = 121
FPS = 24
SEED = 10
GUIDANCE_SCALE = 1.0
REFINE_GUIDANCE_SCALE = 1.0
REFINE_ADD_NOISE = True
LTX2_VAE_TILING = False
SAVE_VIDEO = False
RETURN_FRAMES = True
NVFP4_FA4 = False
TORCH_COMPILE = True
COMPILE_TEXT_ENCODER = True
COMPILE_VAE = True
COMPILE_BACKEND = "inductor"
COMPILE_DYNAMIC = False
STAGE_LOGGING = True
DIT_CPU_OFFLOAD = False
TEXT_ENCODER_CPU_OFFLOAD = False
VAE_CPU_OFFLOAD = False


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
        description="Run one LTX-2 generation speed sweep cell.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-gpus", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-inference-steps", type=int, choices=(5, 8), default=8)
    parser.add_argument("--refine-num-inference-steps", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--fp4-linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LTX-2 NVFP4 quantization for selected linear layers.",
    )
    parser.add_argument(
        "--compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile with fullgraph=True for the DiT/text encoder/VAE compile kwargs.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    schedule = (args.num_inference_steps, args.refine_num_inference_steps)
    if schedule not in {(5, 2), (8, 3)}:
        raise ValueError(
            "Unsupported LTX-2 sweep schedule. Expected 5+2 or 8+3, "
            f"got {args.num_inference_steps}+{args.refine_num_inference_steps}."
        )


def run_name(args: argparse.Namespace) -> str:
    fp4 = "fp4on" if args.fp4_linear else "fp4off"
    fullgraph = "fgon" if args.compile_fullgraph else "fgoff"
    return (
        f"ltx2_speed_s{args.num_inference_steps}p{args.refine_num_inference_steps}"
        f"_g{args.num_gpus}_{fp4}_{fullgraph}"
    )


def make_run_dir(args: argparse.Namespace) -> Path:
    run_dir = OUTPUT_ROOT / run_name(args)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    return run_dir


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
    sr_stage_entries: list[tuple[str, float]] = []
    for stage_name, exec_time in stage_times.items():
        stage_name_l = stage_name.lower()
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


def configure_environment() -> None:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1" if STAGE_LOGGING else "0"
    os.environ["FASTVIDEO_NVFP4_FA4"] = "1" if NVFP4_FA4 else "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_inductor() -> None:
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False


def build_torch_compile_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": COMPILE_BACKEND,
        "fullgraph": args.compile_fullgraph,
        "dynamic": COMPILE_DYNAMIC,
    }


def fixed_settings() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "validation_json": str(VALIDATION_JSON),
        "output_root": str(OUTPUT_ROOT),
        "num_runs": NUM_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "tp_size": TP_SIZE,
        "distributed_executor_backend": DISTRIBUTED_EXECUTOR_BACKEND,
        "attention_backend": ATTENTION_BACKEND,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "seed": SEED,
        "guidance_scale": GUIDANCE_SCALE,
        "refine_guidance_scale": REFINE_GUIDANCE_SCALE,
        "refine_add_noise": REFINE_ADD_NOISE,
        "ltx2_vae_tiling": LTX2_VAE_TILING,
        "save_video": SAVE_VIDEO,
        "return_frames": RETURN_FRAMES,
        "nvfp4_fa4": NVFP4_FA4,
        "torch_compile": TORCH_COMPILE,
        "compile_text_encoder": COMPILE_TEXT_ENCODER,
        "compile_vae": COMPILE_VAE,
        "compile_backend": COMPILE_BACKEND,
        "compile_dynamic": COMPILE_DYNAMIC,
        "stage_logging": STAGE_LOGGING,
        "dit_cpu_offload": DIT_CPU_OFFLOAD,
        "text_encoder_cpu_offload": TEXT_ENCODER_CPU_OFFLOAD,
        "vae_cpu_offload": VAE_CPU_OFFLOAD,
    }


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
    return {
        "tuned": {
            "num_inference_steps": args.num_inference_steps,
            "refine_num_inference_steps": args.refine_num_inference_steps,
            "num_gpus": args.num_gpus,
            "sp_size": args.num_gpus,
            "fp4_linear": args.fp4_linear,
            "compile_fullgraph": args.compile_fullgraph,
        },
        "fixed": fixed_settings(),
        "resolved": {
            "model_root": model_root,
            "refine_upsampler_path": str(refine_upsampler_path),
            "prompt": prompt,
            "height": benchmark_entry.get("height", 1088),
            "width": benchmark_entry.get("width", 1920),
            "torch_compile_kwargs": torch_compile_kwargs,
        },
        "pipeline_config": pipeline_config,
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
            configure_environment()
            configure_inductor()

            if not VALIDATION_JSON.exists():
                raise FileNotFoundError(f"Validation file not found: {VALIDATION_JSON}")

            validation_entries = load_validation_entries(VALIDATION_JSON)
            if not validation_entries:
                raise ValueError(f"No validation entries found in {VALIDATION_JSON}")

            benchmark_entry = validation_entries[0]
            prompt = benchmark_entry.get("caption")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("First validation entry is missing a usable caption")

            measured_start_idx = WARMUP_RUNS

            model_root = maybe_download_model(MODEL_ID)
            refine_upsampler_path = resolve_refine_upsampler_path(model_root)
            print(f"Profile run directory: {run_dir}")
            print(f"Using model root: {model_root}")
            print(f"Using refine upsampler: {refine_upsampler_path}")
            print(
                "Sweep cell: "
                f"steps={args.num_inference_steps}+{args.refine_num_inference_steps}, "
                f"gpus/sp={args.num_gpus}, fp4_linear={args.fp4_linear}, "
                f"fullgraph={args.compile_fullgraph}"
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
                tp_size=TP_SIZE,
                sp_size=args.num_gpus,
                distributed_executor_backend=DISTRIBUTED_EXECUTOR_BACKEND,
                nvfp4_fa4=NVFP4_FA4,
                ltx2_refine_enabled=True,
                ltx2_refine_upsampler_path=str(refine_upsampler_path),
                refine_lora_path="",
                ltx2_refine_lora_path="",
                ltx2_refine_num_inference_steps=args.refine_num_inference_steps,
                ltx2_refine_guidance_scale=REFINE_GUIDANCE_SCALE,
                ltx2_refine_add_noise=REFINE_ADD_NOISE,
                pipeline_config=pipeline_config,
                enable_torch_compile=TORCH_COMPILE,
                enable_torch_compile_text_encoder=COMPILE_TEXT_ENCODER,
                enable_torch_compile_vae=COMPILE_VAE,
                torch_compile_kwargs=torch_compile_kwargs,
                torch_compile_kwargs_vae=torch_compile_kwargs,
                dit_cpu_offload=DIT_CPU_OFFLOAD,
                text_encoder_cpu_offload=TEXT_ENCODER_CPU_OFFLOAD,
                vae_cpu_offload=VAE_CPU_OFFLOAD,
                ltx2_vae_tiling=LTX2_VAE_TILING,
            )

            run_times: list[float] = []
            e2e_times: list[float] = []
            sr_forward_times: list[float] = []
            non_stage_overhead_times: list[float] = []
            stage_times: dict[str, list[float]] = {}
            stage_order: OrderedDict[str, None] = OrderedDict()
            run_records: list[dict[str, Any]] = []

            try:
                for i in range(NUM_RUNS):
                    output_path = video_dir / f"output_ltx2_basic_t2v_run_{i + 1}.mp4"
                    print(f"[{i + 1}/{NUM_RUNS}] Generating: {output_path}")

                    start = time.perf_counter()
                    result = generator.generate_video(
                        prompt=prompt,
                        output_path=str(output_path),
                        fps=FPS,
                        seed=SEED,
                        save_video=SAVE_VIDEO,
                        return_frames=RETURN_FRAMES,
                        guidance_scale=GUIDANCE_SCALE,
                        height=benchmark_entry.get("height", 1088),
                        width=benchmark_entry.get("width", 1920),
                        num_frames=NUM_FRAMES,
                        num_inference_steps=args.num_inference_steps,
                    )

                    elapsed = result.get("generation_time") if isinstance(result, dict) else None
                    e2e_elapsed = result.get("e2e_latency") if isinstance(result, dict) else None
                    if elapsed is None:
                        elapsed = time.perf_counter() - start
                    if e2e_elapsed is None:
                        e2e_elapsed = time.perf_counter() - start

                    run_times.append(elapsed)
                    e2e_times.append(e2e_elapsed)
                    print(f"[{i + 1}/{NUM_RUNS}] Generation time: {elapsed:.2f}s")
                    print(f"[{i + 1}/{NUM_RUNS}] End-to-end latency: {e2e_elapsed:.2f}s")

                    run_record: dict[str, Any] = {
                        "run_index": i + 1,
                        "measured": i >= measured_start_idx,
                        "generation_time": elapsed,
                        "e2e_latency": e2e_elapsed,
                        "output_path": str(output_path),
                    }

                    if isinstance(result, dict):
                        stage_sum = print_stage_breakdown(result, i + 1, NUM_RUNS)
                        run_record["stage_times"] = extract_stage_times(result)
                        if stage_sum is not None:
                            non_stage_overhead = e2e_elapsed - stage_sum
                            run_record["stage_sum"] = stage_sum
                            run_record["non_stage_overhead"] = non_stage_overhead
                            print(
                                f"[{i + 1}/{NUM_RUNS}] Non-stage overhead "
                                f"(e2e - stage sum): {non_stage_overhead:.3f}s"
                            )
                            if i >= measured_start_idx:
                                non_stage_overhead_times.append(non_stage_overhead)

                        sr_forward_total, sr_stage_entries, stage_names = extract_sr_forward_latency(result)
                        run_record["sr_stage_entries"] = sr_stage_entries
                        if sr_forward_total is None:
                            print(f"[{i + 1}/{NUM_RUNS}] SR forward latency unavailable")
                            if stage_names:
                                print(f"    Available stage keys: {', '.join(stage_names)}")
                        else:
                            run_record["sr_forward_latency"] = sr_forward_total
                            print(f"[{i + 1}/{NUM_RUNS}] SR forward latency: {sr_forward_total:.3f}s")
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
                    f"(runs {measured_start_idx + 1}-{len(run_times)}, skipping first {WARMUP_RUNS} warmup runs): "
                    f"{avg_time:.2f}s"
                )

                measured_e2e_times = e2e_times[measured_start_idx:]
                avg_e2e_time = sum(measured_e2e_times) / len(measured_e2e_times)
                print(
                    f"Average end-to-end latency over {len(measured_e2e_times)} runs "
                    f"(runs {measured_start_idx + 1}-{len(e2e_times)}, skipping first {WARMUP_RUNS} warmup runs): "
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

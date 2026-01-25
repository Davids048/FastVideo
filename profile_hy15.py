import os
from fastvideo import VideoGenerator
from fastvideo.profiler import get_global_controller

prompt = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    "wide with interest. The playful yet serene atmosphere is complemented by soft "
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
)

output_path = "outputs_video/hy15/output_hy15_t2v.mp4"

def main():
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    # Torch profiler for model loading (keep traces small; no shapes/stack/memory)
    trace_dir = os.path.abspath("profile_hy15")
    os.makedirs(trace_dir, exist_ok=True)
    # os.environ["FASTVIDEO_TORCH_PROFILER_DIR"] = trace_dir
    # os.environ["FASTVIDEO_TORCH_PROFILE_REGIONS"] = "profiler_region_model_loading"
    os.environ["FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES"] = "0"
    os.environ["FASTVIDEO_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "0"
    os.environ["FASTVIDEO_TORCH_PROFILER_WITH_STACK"] = "0"
    os.environ["FASTVIDEO_TORCH_PROFILER_WITH_FLOPS"] = "0"
    os.environ["FASTVIDEO_TORCH_PROFILER_WAIT_STEPS"] = "1"
    os.environ["FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS"] = "0"
    os.environ["FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS"] = "1"

    generator = VideoGenerator.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        # FastVideo will automatically handle distributed setup
        num_gpus=4,
        tp_size=1,
        sp_size=4,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        dit_layerwise_offload=False,
    )

    
    for i in range(5):
        result = generator.generate_video(
            prompt,
            output_path=output_path,
            save_video=True,
            negative_prompt="",
            ##########
            num_frames=121,
            fps=24,
            # Res: 1080P (1088 * 1920) | 720P (736 * 1280)
            height=720,
            width=1280,
            guidance_scale=1.0,
            num_inference_steps=1, # Default 50.
        )
        print(f"==================OUTPUT LOGGING INFO {i=}=====================")
        logging_info = result.get("logging_info") if isinstance(result, dict) else None
        if logging_info is None:
            print("No logging_info returned; enable FASTVIDEO_STAGE_LOGGING=1.")
            return
        stage_names = logging_info.get_execution_order()
        stage_execution_times = [
            logging_info.get_stage_info(stage_name).get("execution_time", 0.0)
            for stage_name in stage_names
        ]
        for name, exec_time in zip(stage_names, stage_execution_times):
            print(f"{name} | {exec_time * 1e3:.5f} ms")
        print(f"=====================End LOGGING INFO {i=}=====================")


if __name__ == "__main__":
    main()

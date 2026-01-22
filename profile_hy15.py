import os
from fastvideo import VideoGenerator
from fastvideo.profiler import get_global_controller

OUTPUT_PATH = "video_samples_hy15"
def main():
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    # Torch profiler for VAE decode (keep traces small; no shapes/stack/memory)
    trace_dir = os.path.abspath("profile")
    os.makedirs(trace_dir, exist_ok=True)
    os.environ["FASTVIDEO_TORCH_PROFILER_DIR"] = trace_dir
    os.environ["FASTVIDEO_TORCH_PROFILE_REGIONS"] = "profiler_region_inference_decoding"
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
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        # image_encoder_cpu_offload=False,
        tp_size=1,
        sp_size=4,
        dit_layerwise_offload=False,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )

    
    for i in range(5):
        result = generator.generate_video(
            prompt,
            output_path=OUTPUT_PATH,
            save_video=True,
            negative_prompt="",
            ##########
            num_frames=121,
            fps=24,
            height=720,
            width=1280,
            guidance_scale=1.0,
            num_inference_steps=1,
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

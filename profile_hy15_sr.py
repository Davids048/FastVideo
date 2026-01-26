import os
from fastvideo import VideoGenerator

prompt = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    "wide with interest. The playful yet serene atmosphere is complemented by soft "
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
)

output_path = "outputs_video/hy15_sr/output_hy15_sr_t2v.mp4"


def main():
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    generator = VideoGenerator.from_pretrained(
        "weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR",
        # FastVideo will automatically handle distributed setup
        num_gpus=8,
        tp_size=1,
        sp_size=8,
        use_fsdp_inference=False,  # set to True if GPU is out of memory
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=
        True,  # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        dit_layerwise_offload=True,
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
            # height=720,
            # width=1280,
            guidance_scale=1.0,
            # num_inference_steps=1, # Default 50.
        )
        print(
            f"==================OUTPUT LOGGING INFO {i=}=====================")
        logging_info = result.get("logging_info") if isinstance(result,
                                                                dict) else None
        if logging_info is None:
            print("No logging_info returned; enable FASTVIDEO_STAGE_LOGGING=1.")
            return
        stage_names = logging_info.get_execution_order()
        stage_execution_times = [
            logging_info.get_stage_info(stage_name).get("execution_time", 0.0)
            for stage_name in stage_names
        ]
        # for name, exec_time in zip(stage_names, stage_execution_times):
        #     print(f"{name}, {exec_time * 1e3:.5f} ms")
        print(",".join(stage_names))
        print(",".join(f"{t * 1e3:.5f}" for t in stage_execution_times))
        print(
            f"=====================End LOGGING INFO {i=}=====================")


if __name__ == "__main__":
    main()

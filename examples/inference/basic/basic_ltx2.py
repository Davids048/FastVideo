from fastvideo import VideoGenerator

PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances aside, "
    "then mutters defensively, \"He's just having fun.\" The camera slowly "
    "pans right, revealing the grandfather in the garden wearing enormous "
    "butterfly wings, waving his arms in the air like he's trying to take "
    "off. He shouts, \"Wheeeew!\" as he flaps his wings with full commitment. "
    "The woman covers her face, on the verge of tears. The tone is deadpan, "
    "absurd, and quietly tragic."
)


import os
def main() -> None:
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    generator = VideoGenerator.from_pretrained(
        "FastVideo/LTX2-Distilled-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        # TODO: cpu offload off.
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,

        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        dit_layerwise_offload=False,
    )

    output_path = "outputs_video/ltx2_basic/output_ltx2_distilled_t2v.mp4"
    for i in range(5):
        result = generator.generate_video(
            prompt=PROMPT,
            output_path=output_path,
            save_video=True,
            ##################
            num_frames=121,
            fps=24,
            height = 1088,
            width = 1920,
            guidance_scale=1.0,
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
            print(f"{name}, {exec_time * 1e3:.5f}")
        print(f"=====================End LOGGING INFO {i=}=====================")

    generator.shutdown()


if __name__ == "__main__":
    main()

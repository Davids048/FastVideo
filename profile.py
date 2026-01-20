def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    load_start_time = time.perf_counter()
    model_name = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name,
        # FastVideo will automatically handle distributed setup
        num_gpus=8,
        use_fsdp_inference=True,
        # Adjust these offload parameters if you have < 32GB of VRAM
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        VSA_sparsity=0.8,
        override_pipeline_cls_name="WanDMDPipeline"
        # pipeline_config=PipelineConfig(
        # override_transformer_cls_name="WanTransformer3DModel",
    )
    load_end_time = time.perf_counter()
    load_time = load_end_time - load_start_time


    sampling_param = SamplingParam.from_pretrained(model_name)
    sampling_param.num_frames = 81
    sampling_param.height = 1088
    sampling_param.width = 1920
    sampling_param.num_inference_steps = 3
    sampling_param.guidance_scale = 1.0

    prompt = (
        "A neon-lit alley in futuristic Tokyo during a heavy rainstorm at night. The puddles reflect glowing signs in kanji, advertising ramen, karaoke, and VR arcades. A woman in a translucent raincoat walks briskly with an LED umbrella. Steam rises from a street food cart, and a cat darts across the screen. Raindrops are visible on the camera lens, creating a cinematic bokeh effect."
    )
    start_time = time.perf_counter()
    result = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=False, sampling_param=sampling_param, return_frames=False)
    logging_info = result.get("logging_info", None)
    stage_names = logging_info.get_execution_order()
    stage_execution_times = [
        logging_info.get_stage_info(stage_name).get("execution_time", 0.0) 
        for stage_name in stage_names
    ]
    print(f"Stage names: {stage_names}")
    print(f"Stage execution times: {stage_execution_times}")

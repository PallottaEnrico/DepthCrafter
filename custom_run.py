import gc
import torch
from run import DepthCrafterDemo
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    # running configs
    # the most important arguments for memory saving are `cpu_offload`, `enable_xformers`, `max_res`, and `window_size`
    # the most important arguments for trade-off between quality and speed are
    # `num_inference_steps`, `guidance_scale`, and `max_res`
    parser = argparse.ArgumentParser(description="DepthCrafter")
    parser.add_argument(
        "--video-folder", type=str, required=True, help="Path to the input video file(s)"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Number of denoising steps"
    )

    args = parser.parse_args()

    depthcrafter_demo = DepthCrafterDemo(
        unet_path="tencent/DepthCrafter",
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload="model",
    )

    # process the videos, the video paths are separated by comma
    video_paths = []
    for subfolder in os.listdir(args.video_folder):
        subfolder_path = os.path.join(args.video_folder, subfolder)
        if os.path.isdir(subfolder_path):
            video_paths.extend(
                [
                    os.path.join(subfolder_path, video)
                    for video in os.listdir(subfolder_path)
                    if video.endswith(".avi")
                ]
            )

    for video in tqdm(video_paths):
        depthcrafter_demo.infer(
            video,
            num_denoising_steps=args.steps,
            guidance_scale=1.0,
            window_size=300,
            process_length=1000,
            overlap=25,
            dataset="open",
            target_fps=25,
            seed=42,
            track_time=True,
            save_npz=False,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()

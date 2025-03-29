import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import os

def generate_video_sequence(input_panorama, output_folder, num_frames=24, prompt="", width=1024, height=512):
    """
    Generate a sequence of panoramic images using Stable Diffusion Video.
    
    Args:
        input_panorama: Path to the input panorama image
        output_folder: Folder to save the generated frames
        num_frames: Number of frames to generate
        prompt: Text prompt to guide the generation
        width: Width of output frames
        height: Height of output frames
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the panorama image
    init_image = Image.open(input_panorama).convert("RGB")
    init_image = init_image.resize((width, height))
    
    # Load the Stable Diffusion Video pipeline
    # Note: You'll need to have the model downloaded or accessible
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Default prompt if none provided
    if not prompt:
        prompt = "A detailed, photorealistic 360-degree panoramic environment"
    
    # Generate video frames
    print(f"Generating {num_frames} frames with prompt: '{prompt}'")
    
    # Set the conditioning strength (0.0 to be closer to the original image)
    conditioning_scale = 0.5  # Adjust as needed
    
    # Generate the frames
    frames = pipe(
        prompt,
        image=init_image,
        num_frames=num_frames,
        num_inference_steps=50,
        guidance_scale=7.5,
        conditioning_scale=conditioning_scale,
    ).frames
    
    # Save each frame
    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_folder, f"frame_{i:04d}.png")
        frame.save(frame_path)
        print(f"Saved frame {i+1}/{num_frames} to {frame_path}")
    
    print(f"Video sequence generation complete. {num_frames} frames saved to {output_folder}")
    return frames

if __name__ == "__main__":
    input_panorama = "panorama.png"  # From the previous step
    output_folder = "panorama_frames"
    
    # Example prompt to guide the generation
    prompt = "A detailed 360-degree view of a realistic environment with natural lighting, as someone slowly walks through the space"
    
    generate_video_sequence(
        input_panorama=input_panorama,
        output_folder=output_folder,
        num_frames=24,  # 1 second at 24fps
        prompt=prompt
    )

import argparse
import os
import sys
import importlib.util
from datetime import datetime

def import_from_file(file_path, module_name):
    """Dynamically import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import functions from our scripts using their actual filenames
current_dir = os.path.dirname(os.path.abspath(__file__))
ply_to_panorama = import_from_file(os.path.join(current_dir, "ply-to-panorama.py"), "ply_to_panorama")
sd_video_generation = import_from_file(os.path.join(current_dir, "sd-video-generation.py"), "sd_video_generation")
panorama_to_3d = import_from_file(os.path.join(current_dir, "panorama-to-3d.py"), "panorama_to_3d")

# Get the functions from our imported modules
process_ply_to_panorama = ply_to_panorama.process_ply_to_panorama
generate_video_sequence = sd_video_generation.generate_video_sequence
reconstruct_3d_scene = panorama_to_3d.reconstruct_3d_scene

def create_pipeline(input_ply, output_folder, prompt, num_frames=24, panorama_width=2048, panorama_height=1024):
    """
    Run the complete pipeline from PLY to 2D panorama to video sequence to 3D reconstruction.
    
    Args:
        input_ply: Path to the input PLY file
        output_folder: Base folder for all outputs
        prompt: Text prompt for Stable Diffusion
        num_frames: Number of frames to generate
        panorama_width: Width of the panorama image
        panorama_height: Height of the panorama image
    """
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{output_folder}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Convert PLY to panorama
    print("\n=== STEP 1: Converting PLY to Panorama ===")
    panorama_path = os.path.join(output_folder, "initial_panorama.png")
    process_ply_to_panorama(input_ply, panorama_path, panorama_width, panorama_height)
    
    # Step 2: Generate video sequence
    print("\n=== STEP 2: Generating Video Sequence ===")
    frames_folder = os.path.join(output_folder, "frames")
    frames = generate_video_sequence(
        input_panorama=panorama_path,
        output_folder=frames_folder,
        num_frames=num_frames,
        prompt=prompt,
        width=panorama_width,
        height=panorama_height
    )
    
    # Step 3: Reconstruct 3D scene
    print("\n=== STEP 3: Reconstructing 3D Scene ===")
    output_ply = os.path.join(output_folder, "reconstructed_scene.ply")
    reconstructed_pcd = reconstruct_3d_scene(frames_folder, output_ply)
    
    print(f"\n=== Pipeline Complete! ===")
    print(f"All outputs saved to: {output_folder}")
    print(f"- Initial panorama: {panorama_path}")
    print(f"- Generated frames: {frames_folder}")
    print(f"- Reconstructed 3D model: {output_ply}")
    
    return {
        "output_folder": output_folder,
        "panorama_path": panorama_path,
        "frames_folder": frames_folder,
        "output_ply": output_ply
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D-to-2D-to-3D Pipeline")
    parser.add_argument("--input", default='/mbz/users/zhengqing.yuan/code/LucidDreamer/examples/christmas.ply', help="Input PLY file path")
    parser.add_argument("--output", default="output", help="Output folder base name")
    parser.add_argument("--prompt", default="A detailed 360-degree view of a realistic environment with natural lighting", 
                        help="Prompt for Stable Diffusion")
    parser.add_argument("--frames", type=int, default=24, help="Number of frames to generate")
    parser.add_argument("--width", type=int, default=2048, help="Panorama width")
    parser.add_argument("--height", type=int, default=1024, help="Panorama height")
    
    args = parser.parse_args()
    
    create_pipeline(
        input_ply=args.input,
        output_folder=args.output,
        prompt=args.prompt,
        num_frames=args.frames,
        panorama_width=args.width,
        panorama_height=args.height
    )

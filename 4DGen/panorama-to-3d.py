import torch
import numpy as np
import os
import open3d as o3d
from PIL import Image
import glob
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

def estimate_depth(image_path, model, feature_extractor):
    """Estimate depth from a single image using DPT model."""
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Get depth prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Convert to numpy and normalize
    depth_map = predicted_depth.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return depth_map, np.array(image)

def panorama_to_point_cloud(rgb_image, depth_map, fov_h=360, fov_v=180):
    """Convert a panoramic image and its depth map to a point cloud."""
    height, width = depth_map.shape
    
    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to normalized coordinates in range [-1, 1]
    u_normalized = (u / width) * 2 * np.pi - np.pi  # [-π, π]
    v_normalized = (v / height) * np.pi - np.pi/2   # [-π/2, π/2]
    
    # Convert spherical to Cartesian coordinates
    x = np.cos(v_normalized) * np.cos(u_normalized)
    y = np.cos(v_normalized) * np.sin(u_normalized)
    z = np.sin(v_normalized)
    
    # Scale by depth
    depth_scaled = depth_map * 10.0  # Scale factor, adjust as needed
    x = x * depth_scaled
    y = y * depth_scaled
    z = z * depth_scaled
    
    # Stack coordinates
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    # Extract colors from the RGB image
    colors = rgb_image.reshape(-1, 3) / 255.0
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def reconstruct_3d_scene(frames_folder, output_ply, model_name="Intel/dpt-large"):
    """
    Reconstruct a 3D scene from a sequence of panoramic images.
    
    Args:
        frames_folder: Folder containing the panoramic frames
        output_ply: Path to save the output PLY file
        model_name: Name of the depth estimation model to use
    """
    # Load depth estimation model
    print(f"Loading depth estimation model: {model_name}")
    model = DPTForDepthEstimation.from_pretrained(model_name)
    feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get all frame paths
    frame_paths = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_folder}")
    
    print(f"Found {len(frame_paths)} frames")
    
    # Process each frame and create a combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    
    for i, frame_path in enumerate(frame_paths):
        print(f"Processing frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
        
        # Estimate depth
        depth_map, rgb_image = estimate_depth(frame_path, model, feature_extractor)
        
        # Convert to point cloud
        pcd = panorama_to_point_cloud(rgb_image, depth_map)
        
        # For frames after the first, transform the point cloud to simulate camera movement
        if i > 0:
            # Simple translation along x-axis for demonstration
            # In a real implementation, you'd calculate the camera transform between frames
            translation = np.array([i * 0.5, 0, 0])  # Adjust as needed
            pcd.translate(translation)
        
        # Combine with the main point cloud
        combined_pcd += pcd
    
    # Optional: Downsample to reduce point count
    print("Downsampling point cloud...")
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)
    
    # Optional: Estimate normals for better visualization
    print("Estimating normals...")
    combined_pcd.estimate_normals()
    
    # Save the combined point cloud
    print(f"Saving point cloud to {output_ply}")
    o3d.io.write_point_cloud(output_ply, combined_pcd)
    
    print("3D reconstruction complete!")
    return combined_pcd

if __name__ == "__main__":
    frames_folder = "panorama_frames"  # From the previous step
    output_ply = "reconstructed_scene.ply"
    
    # Run the reconstruction
    reconstruct_3d_scene(frames_folder, output_ply)
    
    # Visualize the result
    print("Visualizing the reconstructed scene...")
    pcd = o3d.io.read_point_cloud(output_ply)
    o3d.visualization.draw_geometries([pcd])

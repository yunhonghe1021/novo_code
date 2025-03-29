import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, asin, pi

def load_ply(file_path):
    """Load a PLY file as a point cloud."""
    print(f"Loading PLY file from {file_path}...")
    try:
        # First try to load as mesh
        mesh = o3d.io.read_triangle_mesh(file_path)
        if len(mesh.triangles) == 0:
            print("No triangles detected, loading as point cloud instead...")
            pcd = o3d.io.read_point_cloud(file_path)
            return pcd
        else:
            return mesh
    except Exception as e:
        print(f"Error loading as mesh: {e}. Trying as point cloud...")
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

def point_cloud_to_panorama(pcd, width=2048, height=1024):
    """Convert a 3D point cloud to a 2D panoramic image with gap filling."""
    # Extract points and colors
    points = np.asarray(pcd.points)
    
    # For coloring, we'll use either point colors if available, or default colors
    has_colors = pcd.has_colors()
    if has_colors:
        colors = np.asarray(pcd.colors)
        print(f"Point cloud has colors. Color shape: {colors.shape}")
    else:
        print("Point cloud has no colors. Using consistent coloring instead of position-based.")
        # Instead of using rainbow colors based on position, use more natural colors
        # Generate a consistent neutral color palette
        colors = np.zeros((len(points), 3), dtype=np.float32)
        # Use distance from center for shading
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        
        # Create a natural color palette (sky blue to tan)
        blue_component = np.clip(0.5 - distances * 0.3, 0, 1)
        green_component = np.clip(0.5 - distances * 0.1, 0, 1)
        red_component = np.clip(0.5 + distances * 0.2, 0, 1)
        
        colors[:, 0] = red_component
        colors[:, 1] = green_component
        colors[:, 2] = blue_component
    
    # Print stats about the point cloud
    print(f"Number of points: {len(points)}")
    print(f"Point cloud bounds: Min {np.min(points, axis=0)}, Max {np.max(points, axis=0)}")
    
    # Initialize the panorama
    panorama = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.ones((height, width)) * np.inf
    confidence = np.zeros((height, width), dtype=np.float32)
    
    # Calculate central point (viewpoint at the center of the point cloud)
    center = np.mean(points, axis=0)
    print(f"Center point: {center}")
    
    # Process each point
    for i, point in enumerate(points):
        # Calculate relative position from center
        relative_pos = point - center
        
        # Convert to spherical coordinates
        r = np.linalg.norm(relative_pos)
        if r < 1e-6:  # Skip points at the center
            continue
            
        # Calculate theta (azimuth) and phi (elevation)
        x, y, z = relative_pos
        theta = atan2(y, x)  # Azimuth angle (horizontal)
        phi = asin(z / r)    # Elevation angle (vertical)
        
        # Convert to panorama coordinates
        u = (theta + pi) / (2 * pi)  # Map [-π, π] to [0, 1]
        v = (phi + pi/2) / pi        # Map [-π/2, π/2] to [0, 1]
        
        # Convert to pixel coordinates
        col = int(u * width) % width
        row = int((1 - v) * height) % height
        
        # Splat the point with a Gaussian kernel for smoother result
        kernel_size = 3
        for ky in range(-kernel_size, kernel_size+1):
            for kx in range(-kernel_size, kernel_size+1):
                # Weight by distance from center of kernel
                weight = np.exp(-(kx**2 + ky**2) / (2 * (kernel_size/2)**2))
                
                # Calculate target position
                tc = (col + kx) % width
                tr = min(max(row + ky, 0), height-1)
                
                # Depth test
                if r < depth_buffer[tr, tc] or weight > confidence[tr, tc]:
                    panorama[tr, tc] = colors[i]
                    depth_buffer[tr, tc] = r
                    confidence[tr, tc] = weight
    
    # Fill remaining holes using inpainting
    from scipy import ndimage
    mask = (confidence > 0).astype(np.float32)
    
    # Use larger dilation and smoothing to fill bigger gaps
    dilated_mask = ndimage.binary_dilation(mask, iterations=10)
    
    # Apply a distance-weighted fill for empty regions
    for c in range(3):
        # Create a distance-weighted copy of the channel
        channel = panorama[:,:,c]
        dist = ndimage.distance_transform_edt(~mask.astype(bool))
        dist = np.clip(1.0 - dist/30.0, 0, 1)  # Adjust the divisor to control fill radius
        
        # Fill with a large-kernel Gaussian blur
        filled = ndimage.gaussian_filter(channel, sigma=10)
        
        # Combine original with fill based on the distance-weighted mask
        panorama[:,:,c] = channel * mask + filled * (dilated_mask - mask)
    
    # Apply a final smoothing pass for natural look
    panorama = ndimage.gaussian_filter(panorama, sigma=2)
    
    # Apply contrast enhancement
    p_min, p_max = np.percentile(panorama[panorama > 0], [2, 98])
    panorama = np.clip((panorama - p_min) / (p_max - p_min), 0, 1)
    
    # Print coverage statistics
    print(f"Initial panorama coverage: {np.count_nonzero(mask) / (width * height) * 100:.2f}%")
    print(f"After filling coverage: {np.count_nonzero(dilated_mask) / (width * height) * 100:.2f}%")
    
    return panorama

def mesh_to_panorama(mesh, width=2048, height=1024):
    """Convert a 3D mesh to a 2D panoramic image."""
    # Extract vertices
    vertices = np.asarray(mesh.vertices)
    
    # For coloring, we'll use either vertex colors if available, or default colors
    has_colors = mesh.has_vertex_colors()
    if has_colors:
        colors = np.asarray(mesh.vertex_colors)
    else:
        # Generate colors based on normalized position
        min_pos = np.min(vertices, axis=0)
        max_pos = np.max(vertices, axis=0)
        normalized_pos = (vertices - min_pos) / (max_pos - min_pos)
        colors = normalized_pos  # Using position as RGB
    
    # Initialize the panorama with zeros
    panorama = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.ones((height, width)) * np.inf
    
    # Calculate central point (assuming the center of the mesh is the viewpoint)
    center = np.mean(vertices, axis=0)
    
    for i, vertex in enumerate(vertices):
        # Calculate relative position from center
        relative_pos = vertex - center
        
        # Convert to spherical coordinates
        r = np.linalg.norm(relative_pos)
        if r < 1e-6:  # Skip points at the center
            continue
            
        # Calculate theta (azimuth) and phi (elevation)
        x, y, z = relative_pos
        theta = atan2(y, x)  # Azimuth angle (horizontal)
        phi = asin(z / r)    # Elevation angle (vertical)
        
        # Convert to panorama coordinates
        u = (theta + pi) / (2 * pi)  # Map [-π, π] to [0, 1]
        v = (phi + pi/2) / pi        # Map [-π/2, π/2] to [0, 1]
        
        # Convert to pixel coordinates
        col = int(u * width) % width
        row = int((1 - v) * height) % height
        
        # Handle depth (Z-buffer approach)
        if r < depth_buffer[row, col]:
            depth_buffer[row, col] = r
            panorama[row, col] = colors[i]
    
    # Normalize the image
    if np.max(panorama) > 0:
        panorama = panorama / np.max(panorama)
        
    return panorama

def save_panorama(panorama, output_path):
    """Save the panorama as an image."""
    plt.imsave(output_path, panorama)
    print(f"Panorama saved to {output_path}")

def process_ply_to_panorama(ply_file, output_file, width=2048, height=1024, render_style='realistic'):
    """
    Process a PLY file to a panoramic image.
    
    Args:
        ply_file: Path to the input PLY file
        output_file: Path to save the output panorama
        width: Width of the panorama in pixels
        height: Height of the panorama in pixels
        render_style: Style of rendering ('realistic' for filled panorama or 'raw' for point cloud)
    """
    geometry = load_ply(ply_file)
    
    # Check if it's a mesh or point cloud
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        print("Processing as mesh...")
        panorama = mesh_to_panorama(geometry, width, height)
    else:
        print("Processing as point cloud...")
        panorama = point_cloud_to_panorama(geometry, width, height)
    
    # Apply additional post-processing for more pleasing aesthetics
    if render_style == 'realistic':
        print("Applying final image enhancement...")
        from skimage import exposure
        # Enhance contrast
        panorama = exposure.equalize_adapthist(panorama)
        
        # Save both raw and enhanced versions
        raw_output = output_file.replace('.png', '_raw.png')
        plt.imsave(raw_output, panorama)
        print(f"Raw panorama saved to {raw_output}")
    
    save_panorama(panorama, output_file)
    
    # Also display the panorama
    plt.figure(figsize=(20, 10))
    plt.imshow(panorama)
    plt.axis('off')
    plt.title("Generated Panorama")
    plt.show()

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ply_file = sys.argv[1]
    else:
        ply_file = "your_model.ply"  # Replace with your PLY file path
        
    output_file = "panorama.png"
    process_ply_to_panorama(ply_file, output_file)
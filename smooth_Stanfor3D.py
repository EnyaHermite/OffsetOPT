import os
import open3d as o3d

input_dir = './results/Stanford3D/'
output_dir = './results/smoothed_Stanford3D/'

# Create output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith('.ply'):
        mesh_path = os.path.join(input_dir, fname)

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(mesh_path)

        # Apply Laplacian smoothing (no normals)
        smoothed = mesh.filter_smooth_laplacian(number_of_iterations=1) # multiple iterations reduces accuracy

        # Save to new folder
        out_path = os.path.join(output_dir, fname)
        o3d.io.write_triangle_mesh(out_path, smoothed, write_ascii=True)

        print(f"Smoothed: {fname} → {out_path}")

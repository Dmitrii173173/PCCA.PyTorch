import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import matplotlib.pyplot as plt

# Define directories
input_dir = '/hy-tmp/main/Pointnet2.PyTorch/tools/data/KITTI/object/training/velodyne/'
output_dir = '/hy-tmp/main/Pointnet2.PyTorch/tools/data/KITTI/object/training/cluster/'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_file(file_index):
    velodyne_file = os.path.join(input_dir, f'{file_index:06d}.bin')

    if not os.path.exists(velodyne_file):
        print(f"File {velodyne_file} does not exist.")
        return None, None
    
    # Load point cloud data from .bin file
    try:
        velo_points = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)
    except Exception as e:
        print(f"Error reading file {velodyne_file}: {e}")
        return None, None

    points_np = velo_points[:, :3]

    # Check if point cloud is empty
    if len(points_np) == 0:
        print(f"File {file_index:06d}.bin is empty or contains no valid data.")
        return None, None

    # Scale the data
    scaler = StandardScaler()
    scaled_points_np = scaler.fit_transform(points_np)

    # Cut points below a certain height
    cutting_height = -1.2
    cut_indices = np.where(points_np[:, 2] > cutting_height)[0]
    cut_points_np = points_np[cut_indices]
    cut_scaled_points_np = scaled_points_np[cut_indices]

    # Find nearest neighbors
    n_neighbors = 1
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(cut_scaled_points_np)
    distances, indices = nbrs.kneighbors(cut_scaled_points_np)

    # Filter points based on distance threshold
    avg_distances = np.mean(distances, axis=1)
    threshold_low = np.percentile(avg_distances, 1)
    filtered_indices = np.where(avg_distances <= threshold_low)[0]
    filtered_cut_points_np = cut_points_np[filtered_indices]

    # Perform K-means clustering
    num_clusters = 15  # Set number of clusters to 15
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(filtered_cut_points_np)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Save clustering results with bounding box, centroid, and point count
    output_file = os.path.join(output_dir, f'{file_index:06d}.txt')
    with open(output_file, 'w') as f:
        for cluster_idx in range(num_clusters):
            cluster_points = filtered_cut_points_np[labels == cluster_idx]
            bbox_min = cluster_points.min(axis=0)
            bbox_max = cluster_points.max(axis=0)
            centroid = cluster_centers[cluster_idx]
            point_count = len(cluster_points)
            f.write(f"Cluster {cluster_idx}:\n")
            f.write(f"  Bounding Box Min: {bbox_min.tolist()}\n")
            f.write(f"  Bounding Box Max: {bbox_max.tolist()}\n")
            f.write(f"  Centroid: {centroid.tolist()}\n")
            f.write(f"  Point Count: {point_count}\n")

    print(f"Processed file {file_index:06d}.bin and saved to {file_index:06d}.txt")

    return filtered_cut_points_np, labels, cluster_centers

def visualize_clusters(filtered_points_np, labels, cluster_centers):
    if filtered_points_np is None or labels is None:
        return
    
    # Compute center of mass
    center = np.mean(filtered_points_np, axis=0)

    # Create red point at the center of mass
    center_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    center_point.paint_uniform_color([1.0, 0.0, 0.0])
    center_point.translate(center)

    # Assign colors to clusters
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))

    # Create point cloud with cluster colors
    kmeans_pcd = o3d.geometry.PointCloud()
    kmeans_pcd.points = o3d.utility.Vector3dVector(filtered_points_np)
    kmeans_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Create bounding boxes for each cluster
    bbox_list = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_points = filtered_points_np[labels == label]
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cluster_points))
        bbox.color = (0, 0, 0)
        bbox_list.append(bbox)

    # Create lines from the center point to cluster centers
    lines = []
    for cluster_center in cluster_centers:
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([center, cluster_center])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[0, 0, 0]])
        lines.append(line)

    # Visualize
    o3d.visualization.draw_geometries([kmeans_pcd, center_point] + bbox_list + lines)

# Process all files in the input directory
for file_index in range(7481):
    filtered_cut_points_np, labels, cluster_centers = process_file(file_index)
    if filtered_cut_points_np is not None and labels is not None:
        visualize_clusters(filtered_cut_points_np, labels, cluster_centers)

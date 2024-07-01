import copy
import numpy as np
import open3d as o3d

data = np.load("./test_folder/pyramid_closejar.npy")
data = data.reshape(-1, 3)
print(data.shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)
# o3d.io.write_point_cloud("./test_folder/pcd.ply", pcd)

# pcd_load = o3d.io.read_point_cloud("./test_folder/pcd.ply")
o3d.visualization.draw_geometries([pcd])
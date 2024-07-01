import open3d as o3d
import numpy as np

point_cloud_data = np.load('./test_folder/pyramid_closejar.npy')
point_cloud_data = point_cloud_data[0]
print(point_cloud_data.shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud_data)
o3d.visualization.draw_geometries([pcd])

scale_factor = 0.3

scaled_points = np.asarray(pcd.points) * scale_factor
pcd.points = o3d.utility.Vector3dVector(scaled_points)

np.save('./test_folder/pyramid_closejar_scaled.npy', scaled_points)

o3d.visualization.draw_geometries([pcd])
import open3d as o3d

# 1. .ply 파일 경로
ply_path = "Data/WashMachine/Retrieve/point_cloud/pointcloud_1.ply"  # 예: "data/output.ply"

# 2. 파일 읽기
pcd = o3d.io.read_point_cloud(ply_path)

# 3. 시각화
o3d.visualization.draw_geometries([pcd],
    window_name="PLY Viewer",
    width=800,
    height=600,
    point_show_normal=False
)

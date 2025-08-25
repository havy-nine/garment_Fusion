import os
import sys
import time
import random
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.prims import delete_prim
from omni.physx import acquire_physx_interface
from pxr import Gf

from Env_Config.Config.basket_config import Config
from Env_Config.Garment.Garment import WrapGarment
from Env_Config.Camera.Basket_Point_Cloud_Camera import Point_Cloud_Camera
from Env_Config.Camera.Recording_Camera import Recording_Camera

class GarmentBasketEnv:
    def __init__(self, save_rgb=True, save_gif=True):
        self.save_rgb = save_rgb
        self.save_gif = save_gif
        self.config = Config()

        self.world = World(backend="torch", device="cpu")
        set_camera_view(
            eye=[-0.782, 1.914, 2.357],
            target=[5.60, -1.53, 0.5],
            camera_prim_path="/OmniverseKit_Persp",
        )
        self.stage = self.world.stage
        self.scene = self.world.get_physics_context()._physics_scene
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        self.scene.CreateGravityMagnitudeAttr().Set(9.8)

        # Light
        import omni.replicator.core as rep
        self.demo_light = rep.create.light(position=[0, 0, 10], light_type="dome")

        # Camera
        self.recording_camera = Recording_Camera(
            self.config.recording_camera_position,
            self.config.recording_camera_orientation,
        )
        self.point_cloud_camera = Point_Cloud_Camera(
            self.config.point_cloud_camera_position,
            self.config.point_cloud_camera_orientation,
            garment_num=self.config.garment_num,
        )

        # Basket
        delete_prim(self.config.basket_prim_path)
        add_reference_to_stage(
            usd_path=self.config.basket_usd_path,
            prim_path=self.config.basket_prim_path
        )

        # Garments
        for i in range(self.config.garment_num):
            delete_prim(f"/World/Garment/garment_{i}")

        self.garments = WrapGarment(
            self.stage,
            self.scene,
            self.config.garment_num,
            self.config.clothpath,
            self.config.garment_position,
            self.config.garment_orientation,
            self.config.garment_scale,
        )

        self.world.reset()

        print("[✓] Garment + Basket scene loaded.")
        self.point_cloud_camera.initialize(self.config.garment_num)
        self.recording_camera.initialize()

        if self.save_gif:
            import threading
            threading.Thread(target=self.recording_camera.get_rgb_graph).start()

    def capture(self):
        # 포인트 클라우드 데이터 받아오기
        pc_judge, color_judge = self.point_cloud_camera.get_point_cloud_data()

        # 포인트 클라우드가 비었는지 확인
        if pc_judge is None or color_judge is None:
            print("[!] Point cloud not available. Skipping capture.")
            return

        # 저장
        _, self.ply_count = self.point_cloud_camera.save_point_cloud(
            sample_flag=True,
            sample_num=4096,
            save_flag=True,
            save_path="/media/eric/T31/Data/Basket/Retrieve/point_cloud/pointcloud",
        )

        # RGB 이미지 저장
        if self.save_rgb:
            self.point_cloud_camera.get_rgb_graph(
                save_path="/media/eric/T31/Data/Basket/Retrieve/rgb/rgb"
            )

        print("[✓] Captured RGB and PointCloud")


if __name__ == "__main__":
    env = GarmentBasketEnv(save_rgb=True, save_gif=False)
    env.capture()
    env.world.clear_instance()
    simulation_app.close()

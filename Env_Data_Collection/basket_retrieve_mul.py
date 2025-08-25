import os
import sys
import time
import numpy as np
import threading
import random
from omni.isaac.kit import SimulationApp
sys.path.append(os.getcwd())
simulation_app = SimulationApp({"headless": False})

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import delete_prim
from pxr import Gf

from Env_Config.Config.basket_config import Config
from Env_Config.Garment.Garment import WrapGarment
from Env_Config.Camera.Basket_Point_Cloud_Camera import Point_Cloud_Camera
from Utils_Project.utils import write_ply_with_colors, write_rgb_image

class SceneUnit:
    def __init__(self, scene_id: int, config: Config):
        self.scene_id = scene_id
        self.prefix = f"/World/Scene{scene_id}"
        self.config = config

        self.stage = World().stage
        self.scene = World().get_physics_context()._physics_scene
        self.offset = np.array([scene_id * 1.0, 0, 0])

        # Light
        self.light = rep.create.light(position=[scene_id * 1.0, 0, 10], light_type="dome")

        # Basket
        self.basket_path = f"{self.prefix}/Basket"
        delete_prim(self.basket_path)
        add_reference_to_stage(
            usd_path=self.config.basket_usd_path,
            prim_path=self.basket_path
        )

        # Garments
        for i in range(self.config.garment_num):
            delete_prim(f"{self.prefix}/Garment/garment_{i}")

        self.garments = WrapGarment(
            stage=self.stage,
            scene=self.scene,
            garment_num=self.config.garment_num,
            garment_usd_path_dict=self.config.clothpath,
            garment_position=self.config.garment_position + self.offset,
            garment_orientation=self.config.garment_orientation,
            garment_scale=self.config.garment_scale,
        )

        # Camera
        self.camera_path = f"{self.prefix}/Camera"
        self.camera = Point_Cloud_Camera(
            camera_position=self.config.point_cloud_camera_position + self.offset,
            camera_orientation=self.config.point_cloud_camera_orientation,
            prim_path=self.camera_path,
            garment_num=self.config.garment_num,
        )
        self.camera.initialize(self.config.garment_num)

    def save(self):
        pc, color = self.camera.get_point_cloud_data()
        if pc is None or color is None:
            print(f"[!] Scene {self.scene_id}: point cloud not available.")
            return

        os.makedirs("./output", exist_ok=True)
        ply_path = f"./output/scene{self.scene_id:02d}_pc.ply"
        rgb_path = f"./output/scene{self.scene_id:02d}_rgb.png"

        write_ply_with_colors(pc, color, ply_path)
        rgb_data = self.camera.camera.get_rgb()
        write_rgb_image(rgb_data, rgb_path)

        print(f"[✓] Scene {self.scene_id} saved: {ply_path}, {rgb_path}")


if __name__ == "__main__":
    config = Config()
    world = World()
    world.scene.add_default_ground_plane()

    scene_units = []
    for scene_id in range(3):
        unit = SceneUnit(scene_id, config)
        scene_units.append(unit)

    world.reset()

    for unit in scene_units:
        unit.save()

    simulation_app.close()

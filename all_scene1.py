
# Open the Simulation App
import os
import sys
from omni.isaac.kit import SimulationApp
import time
sys.path.append(os.getcwd())
simulation_app = SimulationApp({"headless":False})


# ---------------------coding begin---------------------#
import numpy as np
import omni.replicator.core as rep
import threading
import time
import random
import copy
from termcolor import cprint
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from omni.physx import acquire_physx_interface
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.sensor import Camera
from omni.kit.viewport.utility import get_active_viewport
from omni.isaac.core.utils.prims import delete_prim, set_prim_visibility
from omni.isaac.core.utils.viewports import set_camera_view

from Env_Config.Config.basket_config import Config
from Env_Config.Robot.WrapFranka import WrapFranka
from Env_Config.Room.Room import (
    Wrap_room,
    Wrap_base,
    Wrap_basket,
    Wrap_chair,
    Wrap_wash_machine,
)
from Env_Config.Garment.Garment import WrapGarment, Garment
from Env_Config.Utils_Project.utils import (
    get_unique_filename,
    basket_judge_final_poses,
    record_success_failure,
    write_ply,
    write_ply_with_colors,
    load_basket_transport_helper,
)
from Env_Config.Utils_Project.Basket_Collision_Group import Collision_Group
from Env_Config.Utils_Project.AttachmentBlock import AttachmentBlock
from Env_Config.Camera.Basket_Point_Cloud_Camera import Point_Cloud_Camera
from Env_Config.Camera.Recording_Camera import Recording_Camera
import Env_Config.Utils_Project.utils as util

class BaseEnv:
    def __init__(
        self,
        random_flag=True,
        model_path=None,
        rgb_flag=False,
        gif_flag=False,
    ):
        self.random_flag = random_flag
        # save judge_rgb or not
        self.rgb_flag = rgb_flag
        # save gif or not
        self.gif_flag = gif_flag
        
        
        self.world = World(backend="torch", device="cpu")
        set_camera_view(
            eye=[-0.782, 1.914, 2.357],
            target=[5.60, -1.53, 0.5],
            camera_prim_path="/OmniverseKit_Persp",
        )
        
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1) 
        
        self.stage = self.world.stage
        
        self.scene = self.world.get_physics_context()._physics_scene
        
        # set gravity 
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr().Set(9.8)
        
        
        #set defaoult ground
        self.world.scene.add_default_ground_plane()
        
        # Set G_light
        self.demo_light = rep.create.light(position=[0, 0, 10], light_type="dome")

        # set default config
        self.config = Config()
        
        
        # load pc camera 
        self.point_cloud_camera = Point_Cloud_Camera(
                self.config.point_cloud_camera_position,
                self.config.point_cloud_camera_orientation,
                garment_num=self.config.garment_num,
        )
        
        self.point_cloud_camera.initialize(self.config.garment_num)
        
        
        # load r_camera
        self.recording_camera = Recording_Camera(
            self.config.recording_camera_position,
            self.config.recording_camera_orientation,
        )
        gif_generation_thread = threading.Thread(
            target=self.recording_camera.get_rgb_graph
        )
        
        self.recording_camera.initialize()
        
        
        
                # 바구니(Basket)
        delete_prim(f"/World/Basket")
        self.basket = Wrap_basket(
            self.config.basket_position,
            self.config.basket_orientation,
            self.config.basket_scale,
            self.config.basket_usd_path,
            self.config.basket_prim_path,
        )

        # # 세탁기(Washing_Machine) << 
        # delete_prim(f"/World/Washing_Machine")
        # self.wm = Wrap_wash_machine(
        #     self.config.wm_position,
        #     self.config.wm_orientation,
        #     self.config.wm_scale,
        #     self.config.wm_usd_path,
        #     self.config.wm_prim_path,
        # )      
        #self.config.garment = random.random(0,6)
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
        self.garment_index = [True] * self.config.garment_num
        
        load_basket_transport_helper(self.world)
        
        self.collision = Collision_Group(self.stage)
        
        if not self.random_flag:
            self.point_cloud_camera = Point_Cloud_Camera(
                self.config.point_cloud_camera_position,
                self.config.point_cloud_camera_orientation,
                garment_num=self.config.garment_num,
                retrieve_model_path=self.model_path, 
            )
        else:
            self.point_cloud_camera = Point_Cloud_Camera(
                self.config.point_cloud_camera_position,
                self.config.point_cloud_camera_orientation,
                garment_num=self.config.garment_num,
            )
    def save_point_cloud(self):
        
        _, self.ply_count = self.point_cloud_camera.save_point_cloud(
            sample_flag =True,
            sample_num = 4096,
            save_flag=True,
            save_path="/media/eric/T31/Data/Basket/Retrieve/point_cloud/pointcloud",
        )
        
    def save_rgb(self):
        if self.rgb_flag==True:
            self.point_cloud_camera.get_rgb_graph(
                save_path="/media/eric/T31/Data/Basket/Retrieve/rgb/rbg"
            )
                
        
        
if __name__ == "__main__":
    s_time = time.time()
    random_flag = sys.argv[1] == "True"
    if not random_flag:
        model_path = sys.argv[2]
    else:
        model_path = None
    rgb_flag = sys.argv[3] == "True"
    gif_flag = sys.argv[4] == "True"

    env = BaseEnv(
        random_flag=random_flag,
        model_path=model_path,
        rgb_flag=rgb_flag,
        gif_flag=gif_flag,
    )
    env.save_point_cloud()
    env.save_rgb()
    
    print(f'saving time : {time.time() - s_time}')
    
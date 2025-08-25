"""
Create Garment_WashMachine Environment
Include:
    -All components (wash_machine, franka, garment, camera, other helpful parts)
    -Whole Procedure of Project
"""

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
        # random select retrieve point or not
        self.random_flag = random_flag
        # if not random, load model
        if not self.random_flag:
            self.model_path = model_path
        else:
            self.model_path = None
        # save judge_rgb or not
        self.rgb_flag = rgb_flag
        # save gif or not
        self.gif_flag = gif_flag

        # define the world
        self.world = World(backend="torch", device="cpu")
        set_camera_view(
            eye=[-0.782, 1.914, 2.357],
            target=[5.60, -1.53, 0.5],
            camera_prim_path="/OmniverseKit_Persp",
        )
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1)  # garment render request

        # define the stage
        self.stage = self.world.stage

        # define the physics context
        self.scene = self.world.get_physics_context()._physics_scene
        # set gravity
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr().Set(9.8)

        

        # get environment config
        self.config = Config()

        # set global light
        self.demo_light = rep.create.light(position=[0, 0, 10], light_type="dome")

        # load recording_camera (use this camera to generate gif)
        self.recording_camera = Recording_Camera(
            self.config.recording_camera_position,
            self.config.recording_camera_orientation,
        )

        # load floating_camera (use this camera to generate point_cloud graph)
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


        delete_prim(f"/World/Basket")
        self.basket = Wrap_basket(
            self.config.basket_position,
            self.config.basket_orientation,
            self.config.basket_scale,
            self.config.basket_usd_path,
            self.config.basket_prim_path,
        )



        delete_prim(f"/World/chair")
        self.chair = Wrap_chair(
            self.config.chair_position,
            self.config.chair_orientation,
            self.config.chair_scale,
            self.config.chair_usd_path,
            self.config.chair_prim_path,
        )

        delete_prim(f"/World/Base_Layer")
        # load base_layer
        self.base_layer = Wrap_base(
            self.config.base_layer_position,
            self.config.base_layer_orientation,
            self.config.base_layer_scale,
            self.config.base_layer_usd_path,
            self.config.base_layer_prim_path,
        )

        delete_prim(f"/World/Room")
  

        for i in range(self.config.garment_num):
            delete_prim(f"/World/Garment/garment_{i}")

        self.config.garment_num = random.choices([3, 4, 5], [0.1, 0.45, 0.45])[0]
        # self.config.garment_num = 1
        print(f"garment_num: {self.config.garment_num}")

        # load garment
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

        # # create collision group
        self.collision = Collision_Group(self.stage)

 

        self.world.reset()

        print("--------------------------------------")
        print("ramdom pick point:", self.random_flag)
        if not self.random_flag:
            print("model path:", self.model_path)
        print("save rgb:", self.rgb_flag)
        print("save gif:", self.gif_flag)

        cprint("world load successfully", "green", on_color="on_green")

        # -------------------initialize world------------------- #

        # initialize camera
        self.point_cloud_camera.initialize(self.config.garment_num)

        self.recording_camera.initialize()

        cprint("camera initialize successfully", "green")

        # begin to record gif
        gif_generation_thread = threading.Thread(
            target=self.recording_camera.get_rgb_graph
        )
        if self.gif_flag:
            gif_generation_thread.start()

        # transport garment
        self.garment_transportation()

        cprint("garment transportation finish!", "green")

        for garment in self.garments.garment_group:
            garment.particle_material.set_friction(0.25)
            garment.particle_material.set_damping(10.0)

        # delete helper
        delete_prim(f"/World/transport_helper/transport_helper_1")
        delete_prim(f"/World/transport_helper/transport_helper_2")
        delete_prim(f"/World/transport_helper/transport_helper_3")
        delete_prim(f"/World/transport_helper/transport_helper_4")

   

        cprint("world ready!", "green", on_color="on_green")

    def garment_transportation(self):
        """
        Let the clothes float into the washing machine
        by changing the direction of gravity.
        """
        # change gravity direction
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1))
        self.scene.CreateGravityMagnitudeAttr().Set(15)
        for i in range(150):
            if not simulation_app.is_running():
                simulation_app.close()
            simulation_app.update()


    def save_data(self):
        while True:
            # flush record flag and make .txt file to be writable

            pc_judge, color_judge = self.point_cloud_camera.get_point_cloud_data()
            # print(pc_judge)
            if pc_judge is None:
                cprint("Finish picking all garments", "green", on_color="on_green")
                break

            # save pc and rgb graph
            _, self.ply_count = self.point_cloud_camera.save_point_cloud(
                sample_flag=True,
                sample_num=4096,
                save_flag=True,
                save_path="/media/eric/T31/Data/Basket/Retrieve/point_cloud/pointcloud",
            )
            if self.rgb_flag:
                self.point_cloud_camera.get_rgb_graph(
                    save_path="/media/eric/T31/Data/Basket/Retrieve/rgb/rgb"
                )

            for _ in range(125):
                self.world.step(
                    render=True
                )  # render the world to wait the garment fall down

            garment_cur_index = int(self.cur[8:])
            print(f"garment_cur_index: {garment_cur_index}")


            for i in range(10):
                self.world.step(
                    render=True
                )  # render the world to wait the garment disappear

      
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

    # 시뮬레이션 업데이트(카메라가 제대로 찍히도록)
    for _ in range(10):
        simulation_app.update()

    # 포인트 클라우드 저장
    env.point_cloud_camera.save_point_cloud(
        sample_flag=True,
        sample_num=4096,
        save_flag=True,
        save_path="/media/eric/T31/Data/Basket/Retrieve/point_cloud/pointcloud",
    )

    # RGB 이미지 저장 (옵션)
    if env.rgb_flag:
        env.point_cloud_camera.get_rgb_graph(
            save_path="/media/eric/T31/Data/Basket/Retrieve/rgb/rgb"
        )

    print(f"총 실행 시간 = {time.time() - s_time}s")

    env.world.clear_instance()
    simulation_app.close()


# ---------------------coding ending---------------------#

# Close the Simulation App
simulation_app.close()

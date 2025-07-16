"""
Isaac Sim 환경에서 로봇 움직임 없이 scene(환경)만 구성하는 코드
Env_Eval/washmachine.py 참고
"""

import sys
import os
sys.path.append(os.getcwd())
import random
from Env_Config.Config.wash_machine_config import Config
import torch

# from isaacsim import SimulationApp
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

# ---------------------coding begin---------------------#
import numpy as np
import omni.replicator.core as rep
import threading
import time
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

from Env_Config.Robot.WrapFranka import WrapFranka
from Env_Config.Garment.Garment import WrapGarment, Garment
from Env_Config.Utils_Project.utils import (
    add_wm_door,
    change_door_pos,
    compare_position_before_and_after,
    get_unique_filename,
    wm_judge_final_poses,
    load_conveyor_belt,
    load_washmachine_model,
    record_success_failure,
    write_ply,
    write_ply_with_colors,
)
from Env_Config.Utils_Project.WM_Collision_Group import Collision_Group
from Env_Config.Utils_Project.AttachmentBlock import AttachmentBlock
from Env_Config.Camera.WashMachine_Point_Cloud_Camera import Point_Cloud_Camera
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Model.pointnet2_Retrieve_Model import Retrieve_Model
from Env_Config.Model.pointnet2_Place_Model import Place_Model
from Env_Config.Model.pointnet2_Pick_Model import Pick_Model
import Env_Config.Utils_Project.utils as util
from Env_Config.Room.Room import Wrap_base, Wrap_room, Wrap_basket, Wrap_wash_machine


# Isaac Sim 실행
simulation_app = SimulationApp({"headless": False})

# 월드 생성
world = World(backend="torch", device="cpu")

# 환경 설정 불러오기
config = Config()

# 카메라 시점 세팅
set_camera_view(
    eye=[-1.5, -1.5, 1.5],
    target=[0.01, 0.01, 0.01],
    camera_prim_path="/OmniverseKit_Persp",
)

# 카메라 생성
point_cloud_camera = Point_Cloud_Camera(
    camera_position=[0.0, 0.0, 1.5],
    camera_orientation=[0.0, 0.0, 0.0],
    frequency=20,
    resolution=(512, 512),
    camera_prim_path="/World/point_cloud_camera",
    garment_model_pth_path="Env_Config/Model/wm_retrieve_model_finetuned.pth"
)

# 주요 오브젝트(USD) 불러오기
franka = WrapFranka(
    world,
    config.robot_position,
    config.robot_orientation,
    prim_path="/World/Franka",
    robot_name="franka_robot",
    recording_camera=None,
)
wash_machine = Wrap_wash_machine(
    config.wm_position,
    config.wm_orientation,
    config.wm_scale,
    config.wm_usd_path,
    config.wm_prim_path,
)
room = Wrap_room(
    config.room_position,
    config.room_orientation,
    config.room_scale,
    config.room_usd_path,
    config.room_prim_path,
)
basket = Wrap_basket(
    config.basket_position,
    config.basket_orientation,
    config.basket_scale,
    config.basket_usd_path,
    config.basket_prim_path,
)
base = Wrap_base(
    config.base_position,
    config.base_orientation,
    config.base_scale,
    config.base_usd_path,
    config.base_prim_path,
)
garments = WrapGarment(
    world.stage,
    world.get_physics_context()._physics_scene,
    config.garment_num,
    config.clothpath,
    config.garment_position,
    config.garment_orientation,
    config.garment_scale,
)
door = FixedCuboid(
            name="wm_door",
            position=[-0.39497, 0.0, 0.225],
            prim_path="/World/wm_door",
            scale=np.array([0.05, 1, 0.55]),
            orientation=euler_angles_to_quat([0, 0, 0], degrees=True),
            size=1.0,
            color=np.array([180, 180, 180]),
            visible=False,
        )
        # create collision group
    # self.collision = Collision_Group(self.stage)

    # # add obstacle to franka's rmpflow motion, in case that franka can avoid collision with washing machine smartly
    # for obstacle in obstacle_list:
    #     self.franka.add_obstacle(obstacle)
    # print("collision add successfully")

    # # load retrieval model
    # self.garment_model = Retrieve_Model(normal_channel=False).cuda()
    # self.garment_model.load_state_dict(
    #     torch.load("Env_Config/Model/wm_retrieve_model_finetuned.pth")
    # )
    # self.garment_model.eval()

    # # load place model
    # self.place_model = Place_Model(normal_channel=False).cuda()
    # self.place_model.load_state_dict(
    #     torch.load("Env_Config/Model/wm_place_model_finetuned.pth")
    # )
    # self.place_model.eval()

    # # load pick model
    # self.pick_model = Pick_Model(normal_channel=False).cuda()
    # self.pick_model.load_state_dict(
    #     torch.load("Env_Config/Model/wm_pick_model_finetuned.pth")
    # )
    # self.pick_model.eval()

from omni.isaac.core.utils.prims import set_parent
#set camera attach to robot. 
set_parent("/World/point_cloud_camera", "/World/Franka/panda_link7")

# 카메라 초기화
point_cloud_camera.initialize()

# 월드 렌더링만 100번 반복 (동작 없음)
world.reset()

world.point_cloud_camera.get_rgb()



for i in range(100):
    world.step(render=True)

# Isaac Sim 종료
simulation_app.close() 
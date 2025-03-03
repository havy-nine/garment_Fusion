"""
Create Garment_WashMachine Environment
Include:
    -All components (wash_machine, franka, garment, camera, other helpful parts)
    -Whole Procedure of Project
"""

# Open the Simulation App
import random
import sys
import os

sys.path.append(os.getcwd())
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
from Env_Config.Wash_Machine.Wash_Machine import Wrap_Wash_Machine
from Env_Config.Garment.Garment import WrapGarment, Garment
from Env_Config.Utils_Project.utils import (
    add_wm_door,
    change_door_pos,
    compare_position_before_and_after,
    get_unique_filename,
    judge_final_poses,
    load_conveyor_belt,
    load_washmachine_model,
    record_success_failure,
    write_ply,
    write_ply_with_colors,
)
from Env_Config.Utils_Project.Collision_group import Collision_Group
from Env_Config.Utils_Project.AttachmentBlock import AttachmentBlock
from Env_Config.Camera.Point_Cloud_Camera import Point_Cloud_Camera
from Env_Config.Camera.Recording_Camera import Recording_Camera
from Env_Config.Model.pointnet2_Retrieve_Model import Retrieve_Model
from Env_Config.Model.pointnet2_Place_Model import Place_Model
from Env_Config.Model.pointnet2_Pick_Model import Pick_Model
import Env_Config.Utils_Project.utils as util
from Env_Config.Room.Room import Wrap_base, Wrap_room, Wrap_basket


class washmachineEnv:
    def __init__(self):
        # define the world
        self.world = World(backend="torch", device="cpu")
        set_camera_view(
            eye=[-1.5, -1.5, 1.5],
            target=[0.01, 0.01, 0.01],
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

        # set default_ground
        self.world.scene.add_default_ground_plane(z_position=-0.05)

        # get environment config
        self.config = Config()

        # set global light
        self.demo_light = rep.create.light(position=[0, 0, 0], light_type="dome")

        self.disk_light_inside = rep.create.light(
            position=[0.13459, -0.07683, 0.83689], light_type="disk", intensity=3000
        )

        self.disk_light_inside2 = rep.create.light(
            position=[0.12834, 0.17148, 0.74336], light_type="disk", intensity=3000
        )

        # set local light in washing machine
        self.disk_light = rep.create.light(
            position=[0.2184, 0, 1.11234],
            light_type="rect",
            intensity=15000,
            rotation=[0, -42, 0],
        )

        # load recording_camera (use this camera to generate gif)
        self.recording_camera = Recording_Camera(
            self.config.recording_camera_position,
            self.config.recording_camera_orientation,
        )

        # load point_cloud_camera (use this camera to generate point_cloud graph)
        self.point_cloud_camera = Point_Cloud_Camera(
            self.config.point_cloud_camera_position,
            self.config.point_cloud_camera_orientation,
            garment_num=self.config.garment_num,
        )

        # load franka
        self.franka = WrapFranka(
            self.world,
            self.config.robot_position,
            self.config.robot_orientation,
            prim_path="/World/Franka",
            robot_name="franka_robot",
            recording_camera=self.recording_camera,
        )

        # load wash_machine
        self.wash_machine = Wrap_Wash_Machine(
            self.config.wm_position,
            self.config.wm_orientation,
            self.config.wm_scale,
            self.config.wm_usd_path,
            self.config.wm_prim_path,
        )

        # load room
        # self.room = Wrap_room(
        #     self.config.room_position,
        #     self.config.room_orientation,
        #     self.config.room_scale,
        #     self.config.room_usd_path,
        #     self.config.room_prim_path,
        # )

        # load basket
        # self.basket = Wrap_basket(
        #     self.config.basket_position,
        #     self.config.basket_orientation,
        #     self.config.basket_scale,
        #     self.config.basket_usd_path,
        #     self.config.basket_prim_path,
        # )

        # load base
        # self.base = Wrap_base(
        #     self.config.base_position,
        #     self.config.base_orientation,
        #     self.config.base_scale,
        #     self.config.base_usd_path,
        #     self.config.base_prim_path,
        # )

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

        # load conveyor_belt
        load_conveyor_belt(self.world)

        # load washmachine_model
        obstacle_list = load_washmachine_model(self.world)
        # add_wm_door(self.world)

        self.door = FixedCuboid(
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
        self.collision = Collision_Group(self.stage)

        # add obstacle to franka's rmpflow motion, in case that franka can avoid collision with washing machine smartly
        for obstacle in obstacle_list:
            self.franka.add_obstacle(obstacle)
        print("collision add successfully")

        # load retrieval model
        self.garment_model = Retrieve_Model(normal_channel=False).cuda()
        self.garment_model.load_state_dict(
            torch.load("Env_Config/Model/wm_retrieve_model_finetuned.pth")
        )
        self.garment_model.eval()

        # load place model
        self.place_model = Place_Model(normal_channel=False).cuda()
        self.place_model.load_state_dict(
            torch.load("Env_Config/Model/wm_place_model_finetuned.pth")
        )
        self.place_model.eval()

        # load pick model
        self.pick_model = Pick_Model(normal_channel=False).cuda()
        self.pick_model.load_state_dict(
            torch.load("Env_Config/Model/wm_pick_model_finetuned.pth")
        )
        self.pick_model.eval()

    def garment_into_machine(self):
        """
        Let the clothes float into the washing machine
        by changing the direction of gravity.
        """
        # change gravity direction
        self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(1.5, 0.0, -1.2))
        self.scene.CreateGravityMagnitudeAttr().Set(8.0)
        for i in range(650):
            if not simulation_app.is_running():
                simulation_app.close()
            simulation_app.update()
            if i == 250:
                self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(1.5, 0.0, -0.3))
                self.scene.CreateGravityMagnitudeAttr().Set(8.0)
            if i == 400:
                self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(1.5, 0.0, 0.05))
                self.scene.CreateGravityMagnitudeAttr().Set(9.8)
            # print(i)
            if i == 550:
                print("ready to change")
                # return to the normal gravity direction
                self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1))
                self.scene.CreateGravityMagnitudeAttr().Set(9.8)

    def remove_conveyor_belt(self):
        """
        remove conveyor belt and its collision group
        """

        delete_prim("/World/Conveyor_belt")

        change_door_pos(self.door)
        self.world.step(render=True)
        # self.world.play(
        print("Conveyor Belt Removed!")

    def create_attach_block(self, init_position=np.array([0.0, 0.0, 1.0])):
        """
        Create attachment block and update the collision group at the same time.
        """
        # create attach block and finish attach
        self.attach = AttachmentBlock(
            self.world,
            self.stage,
            "/World/AttachmentBlock",
            self.garments.garment_mesh_path,
        )
        self.attach.create_block(
            block_name="attach", block_position=init_position, block_visible=True
        )
        print("attach finish!")
        # update attach collision group
        self.collision.update_after_attach()
        for i in range(100):
            simulation_app.update()
        print("Update collision group successfully!")

    def set_attach_to_garment(self, attach_position):
        """
        push attach_block to new grasp point and attach to the garment
        """
        # set the position of block
        self.attach.set_block_position(attach_position)
        # create attach
        self.attach.attach()
        # render the world
        self.world.step(render=True)

    def get_point_cloud_data(self, filename=None):
        """
        make franka unvisible before taking point_cloud graph
        when finish taking point_cloud_graph, make franka visible
        """
        # make franka unvisible then get point cloud data
        # set_prim_visibility(self.franka._robot.prim,False)
        for i in range(30):
            self.world.step(render=True)

        self.point_cloud, self.colors = self.point_cloud_camera.get_point_cloud_data()

        if filename is not None:
            write_ply_with_colors(
                points=self.point_cloud, colors=self.colors, filename=filename
            )
            print("write into ply file")

    def pick_point(self, random=True):
        """
        select random point from point_cloud graph and pick
        record corresponding data into .txt file
        """
        # get pick point
        if random:
            # select pick point randomly
            pick_point = self.point_cloud_camera.get_random_point()
            print(pick_point)
        else:
            # use model to select pick point
            pick_point, max_value, output = self.point_cloud_camera.get_model_point()

            count_below_threshold = (output > 0.93).sum().item()
            total_elements = output.numel()
            percentage_below_threshold = count_below_threshold / total_elements

            count = sum(self.garment_index)

            if percentage_below_threshold < 0.044 and count > 3:
                print("start to stir")

                pick_point_stir, index = self.stir(self.point_cloud, max_value)

                self.cur = self.point_cloud_camera.get_cloth_picking()
                self.id = self.point_cloud_camera.id
                print(f"picking {self.cur}")
                print("get position:", pick_point_stir)

                self.set_attach_to_garment(pick_point_stir)

                return pick_point_stir

        print("no stir")

        self.cur = self.point_cloud_camera.get_cloth_picking()
        self.id = self.point_cloud_camera.id
        print(f"picking {self.cur}")
        print("get position:", pick_point)

        # attach block to pick point
        self.set_attach_to_garment(pick_point)
        # return True
        return pick_point

    def pick_multiple_times(self):
        """
        use for multiple garments picking
        """
        # ---------if you want to create gif, keep following codes---------- #
        # thread_rgb = threading.Thread(target= self.recording_camera.get_rgb_graph)
        # thread_rgb.daemon=True
        # thread_rgb.start()
        # ------------------------------------------------------------------ #

        while True:
            success = False
            # pass invalid cases
            if len(self.point_cloud_camera.get_point_cloud_data()) == 0:
                break
            pointcloud, rgb = self.point_cloud_camera.get_point_cloud_data()
            if not isinstance(pointcloud, np.ndarray):
                pointcloud = np.array(pointcloud)
            if len(pointcloud) == 0 or pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
                break

            util.flag_record = True

            self.recording_camera.judge = True

            self.get_point_cloud_data()

            set_prim_visibility(self.franka._robot.prim, True)

            pick_point = self.pick_point(random=False)

            thread_judge = threading.Thread(
                target=self.recording_camera.judge_contact_with_ground
            )
            thread_judge.daemon = True
            thread_judge.start()
            self.franka.fetch_garment_from_washing_machine(
                self.config.target_positions, self.attach
            )

            self.recording_camera.judge = False
            garment_cur_index = int(self.cur[8:])

            self.franka.open()
            self.attach.detach()
            for i in range(100):
                self.world.step(render=True)

            garment_cur_poses = self.garments.get_cur_poses(self.garment_index)

            # will record the success or failure in Env_Eval/washmachine_record.txt
            self.garment_index, success = judge_final_poses(
                garment_cur_poses, garment_cur_index, self.garment_index
            )
            print("success flag", success)
        # ----------------for creating gif-------------------- #
        # self.recording_camera.capture=False
        # self.recording_camera.create_gif()

    def stir(self, point_cloud, max_value, percentage_below_threshold=None):
        index = 0
        continue_stir = True
        while continue_stir:

            point_cloud = self.point_cloud

            garment_model_input = point_cloud.reshape(1, -1, 3)
            garment_model_input = torch.Tensor(garment_model_input).to("cuda:0")
            garment_model_output = self.garment_model(garment_model_input)

            max_value, indices = torch.max(garment_model_output, dim=1, keepdim=False)
            pick_max_value = max_value.item()

            count_below_threshold = (garment_model_output > 0.9).sum().item()
            total_elements = garment_model_output.numel()
            pick_percentage_below_threshold = count_below_threshold / total_elements

            print("garment_max_value", pick_max_value)
            print("garment_percentage_below_threshold", pick_percentage_below_threshold)

            # find pick point
            pick_model_input = point_cloud.reshape(1, -1, 3)
            pick_model_input = torch.Tensor(pick_model_input).to("cuda:0")
            pick_model_output = self.pick_model(pick_model_input.transpose(2, 1))

            max_value, indices = torch.max(pick_model_output, dim=1, keepdim=False)
            print("pick max_value", max_value)
            pick = indices
            print("pick indices", pick)
            print("pick point", point_cloud[pick])

            pick_pc = point_cloud
            pick_pc[0] = point_cloud[pick]

            # find place point
            place_pc = point_cloud

            pick_pc = torch.Tensor(pick_pc.reshape(1, -1, 3)).to("cuda:0")
            place_pc = torch.Tensor(place_pc.reshape(1, -1, 3)).to("cuda:0")

            place_model_output = self.place_model(
                pick_pc.transpose(2, 1), place_pc.transpose(2, 1)
            )

            max_value, indices = torch.max(place_model_output, dim=1, keepdim=False)

            place = indices
            print("place point", point_cloud[place])

            self.set_attach_to_garment(point_cloud[pick])

            self.franka.pick(self.config.target_positions, self.attach)
            self.franka.place(point_cloud[place], self.attach)

            self.franka.open()
            self.attach.detach()
            for i in range(30):
                self.world.step(render=True)

            # check the new retrieval affordance
            self.franka.adjust_after_stir()
            self.get_point_cloud_data()

            pick_point, max_value, output = self.point_cloud_camera.get_model_point()

            place_max_value = max_value.item()

            count_below_threshold = (output > 0.9).sum().item()
            total_elements = output.numel()
            place_percentage_below_threshold = count_below_threshold / total_elements

            print("place_percentage_below_threshold", place_percentage_below_threshold)

            if (
                place_percentage_below_threshold > 0.05
                or place_max_value - pick_max_value > 0.1
                or place_percentage_below_threshold - pick_percentage_below_threshold
                > 0.02
            ):
                continue_stir = False

            index += 1
            if index >= 2:
                continue_stir = False

        return pick_point, index


if __name__ == "__main__":

    env = washmachineEnv()
    set_prim_visibility(env.franka._robot.prim, False)

    env.world.reset()

    env.point_cloud_camera.initialize(env.config.garment_num)

    env.recording_camera.initialize()

    env.garment_into_machine()

    for garment in env.garments.garment_group:
        garment.particle_material.set_particle_friction_scale(3.5)
        garment.particle_material.set_particle_adhesion_scale(1.0)
        garment.particle_material.set_friction(0.2)

    env.remove_conveyor_belt()

    env.create_attach_block()

    env.pick_multiple_times()

    for i in range(100):
        env.world.step(render=True)


# #---------------------coding ending---------------------#

# Close the Simulation App
simulation_app.close()

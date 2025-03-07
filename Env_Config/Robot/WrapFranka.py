import os
import sys

sys.path.append("Env_Config/")

from Utils_Project.utils import delete_wm_door, record_success_failure
from omni.isaac.franka import Franka
from omni.isaac.core import objects
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core import World
from omni.isaac.franka.controllers.pick_place_controller import PickPlaceController
from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.motion_generation.lula import RmpFlow
from pxr import UsdGeom, UsdLux, Sdf, Gf, Vt, Usd, UsdPhysics, PhysxSchema

from omni.isaac.motion_generation.articulation_motion_policy import (
    ArticulationMotionPolicy,
)
from omni.isaac.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
import numpy as np
import torch
import time
from termcolor import cprint


class WrapFranka:
    def __init__(
        self,
        world: World,
        position=torch.tensor,
        orientation=[0.0, 0.0, 0.0],
        prim_path: str = None,
        robot_name: str = None,
        recording_camera=None,
    ):
        # define world
        self.world = world
        self.scene = self.world.get_physics_context()._physics_scene
        self.recording_camera = recording_camera
        # set robot parameters
        if prim_path is None:
            self._franka_prim_path = find_unique_string_name(
                initial_name="World/Franka",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
        else:
            self._franka_prim_path = prim_path
        if robot_name is None:
            self._franka_robot_name = find_unique_string_name(
                initial_name="my_franka",
                is_unique_fn=lambda x: not self.world.scene.object_exists(x),
            )
        else:
            self._franka_robot_name = robot_name
        self._initial_position = position
        self._initial_orientation = euler_angles_to_quat(orientation, degrees=True)

        # set robot
        self._robot = Franka(
            prim_path=self._franka_prim_path,
            name=self._franka_robot_name,
            position=self._initial_position,
            orientation=self._initial_orientation,
        )

        self.rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
        self.rmpflow = RmpFlow(**self.rmp_config)
        self.rmpflow.set_robot_base_pose(
            self._initial_position, self._initial_orientation
        )
        self.articulation_rmpflow = ArticulationMotionPolicy(
            self._robot, self.rmpflow, default_physics_dt=1 / 60.0
        )
        self._articulation_controller = self._robot.get_articulation_controller()

        # add franka to the world
        self.world.scene.add(self._robot)

        # check whether point is reachable or not
        self.pre_error = 0
        self.error_nochange_epoch = 0

    def initialize(self):
        """
        initialize robot
        """
        self._robot.initialize()

    def get_cur_ee_pos(self):
        """
        get current end_effector_position and end_effector orientation
        """
        position, orientation = self.rmpflow.get_end_effector_as_prim().get_world_pose()
        return position, orientation

    def get_cur_grip_pos(self):
        """
        get current gripper position and orientation
        """
        position, orientation = self._robot.gripper.get_world_pose()
        return position, orientation

    def open(self):
        """
        open the gripper of franka
        """
        self._robot.gripper.open()

    def close(self):
        """
        close the gripper of franka
        """
        self._robot.gripper.close()

    def Rotation(self, quaternion, vector):
        q0 = quaternion[0].item()
        q1 = quaternion[1].item()
        q2 = quaternion[2].item()
        q3 = quaternion[3].item()
        R = torch.tensor(
            [
                [
                    1 - 2 * q2**2 - 2 * q3**2,
                    2 * q1 * q2 - 2 * q0 * q3,
                    2 * q1 * q3 + 2 * q0 * q2,
                ],
                [
                    2 * q1 * q2 + 2 * q0 * q3,
                    1 - 2 * q1**2 - 2 * q3**2,
                    2 * q2 * q3 - 2 * q0 * q1,
                ],
                [
                    2 * q1 * q3 - 2 * q0 * q2,
                    2 * q2 * q3 + 2 * q0 * q1,
                    1 - 2 * q1**2 - 2 * q2**2,
                ],
            ]
        )
        vector = torch.mm(vector.unsqueeze(0), R.transpose(1, 0))
        return vector.squeeze(0)

    def add_obstacle(self, obstacle):
        """
        add obstacle to franka motion
        make franka avoid potential collision smartly
        """
        self.rmpflow.add_obstacle(obstacle, False)
        for i in range(10):
            self.world.step(render=True)
        return

    def RMPflow_Move(self, position, orientation=None):
        """
        Use RMPflow_controller to move the Franka
        """
        self.world.step(render=True)
        # obtain target position
        position = position.cpu().numpy().reshape(-1)
        # obtain target orientation(from euler to quartenion)
        if orientation is not None:
            orientation = euler_angles_to_quat(orientation, degrees=True)
        # set end effector target
        self.rmpflow.set_end_effector_target(
            target_position=position, target_orientation=orientation
        )
        # update obstacle position and get target action
        self.rmpflow.update_world()
        actions = self.articulation_rmpflow.get_next_articulation_action()
        # apply actions
        self._articulation_controller.apply_action(actions)

    def wm_check_gripper_arrive(
        self,
        target_position,
        stir=False,
        error_record_file: str = "Env_Eval/washmachine_record.txt",
    ) -> bool:
        """
        check whether gripper has arrived at the attach block position
        if arrived, return True; else return False.
        """
        import torch.nn.functional as F

        # get current position and calculate the gap between current position and target position
        gripper_position, gripper_orientation = self.get_cur_grip_pos()
        current_position = (
            gripper_position
            + self.Rotation(gripper_orientation, torch.Tensor([0.0, 0.0, 0.05]))
        ).cpu()
        error = F.mse_loss(
            current_position.clone().detach(), target_position.clone().detach()
        ).item()

        error_gap = error - self.pre_error
        self.pre_error = error

        # Judge whether gripper has arrived or not according to error
        if abs(error_gap) < 1e-5:
            self.error_nochange_epoch += 1
        # print("error_epoch:", self.error_nochange_epoch)
        if self.error_nochange_epoch >= 200:
            self.world.stop()
            if stir:
                with open(error_record_file, "a") as file:
                    file.write(f"0 point unreachable" + "\n")
            else:
                record_success_failure(
                    False,
                    error_record_file,
                    str="pick_point is unreachable",
                )
        if error >= 0.00065:
            return False
        elif np.isnan(error):
            self.world.stop()
            record_success_failure(False, error_record_file, str="franka fly")
        else:
            return True

    def check_gripper_arrive(
        self,
        target_position,
        error_record_file: str = "Env_Eval/washmachine_record.txt",
    ) -> bool:
        """
        check whether gripper has arrived at the attach block position
        if arrived, return True; else return False.
        """
        import torch.nn.functional as F

        # get current position and calculate the gap between current position and target position
        gripper_position, gripper_orientation = self.get_cur_grip_pos()
        current_position = (
            gripper_position
            + self.Rotation(gripper_orientation, torch.Tensor([0.0, 0.0, 0.05]))
        ).cpu()
        error = F.mse_loss(
            current_position.clone().detach(), target_position.clone().detach()
        ).item()
        # print("current error:", error)
        error_gap = error - self.pre_error
        self.pre_error = error
        # print("error gap:", error_gap)
        # Judge whether gripper has arrived or not according to error
        if abs(error_gap) < 1e-5:
            self.error_nochange_epoch += 1
        # print("error_epoch:", self.error_nochange_epoch)
        if error >= 0.00065:
            return False
        elif np.isnan(error):
            self.world.stop()
            record_success_failure(False, "Env_Eval/sofa_record.txt", str="franka fly")

        else:
            return True

    def move_block_follow_gripper(
        self, attach_block, target_position, target_orientation=None
    ):
        """
        make attach_block follow the franka's gripper during the movement of franka.
        """
        # control franka to move
        self.RMPflow_Move(position=target_position, orientation=target_orientation)
        # get the position and orientation of franka's gripper
        gripper_position, gripper_orientation = self.get_cur_grip_pos()
        # get block position
        block_position = gripper_position + self.Rotation(
            gripper_orientation, torch.Tensor([0.0, 0.0, 0.05])
        )
        # let block follow the gripper
        attach_block.block.set_world_pose(block_position, gripper_orientation)
        # Render World to see the block move
        self.world.step(render=True)

    def fetch_garment_from_washing_machine(
        self,
        target_positions,
        attach_block,
        error_record_file: str = "Env_Eval/washmachine_record.txt",
    ):
        """
        whole procedure of bringing out the garment from wash_machine
        """
        # initialize franka
        self.initialize()
        # render world
        self.world.step(render=True)
        # reach attach block position
        self.open()  # open franka's gripper
        # enter washing machine

        position = torch.tensor(target_positions[0])
        print("start to enter washing machine")
        while not self.wm_check_gripper_arrive(
            position, error_record_file=error_record_file
        ):
            self.RMPflow_Move(position)

        # catch the block
        reach_position = attach_block.get_block_position().cpu()
        reach_position = reach_position[0]
        print(f"start to reach the fetch point : {reach_position}")
        while not self.wm_check_gripper_arrive(
            reach_position, error_record_file=error_record_file
        ):
            self.RMPflow_Move(reach_position)
        # close franka's gripper
        self.close()
        for i in range(30):
            self.world.step(render=True)
        print("start to go to the target push_garment position")
        # move franka according to the pre-set position

        delete_wm_door()
        # self.scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1))
        # self.scene.CreateGravityMagnitudeAttr().Set(15)
        for i in range(len(target_positions)):
            # if i==0: i=1
            # elif i==1: i=0
            position = torch.Tensor(target_positions[i])
            while not self.wm_check_gripper_arrive(
                position, error_record_file=error_record_file
            ):
                self.move_block_follow_gripper(attach_block, position)

    def fetch_garment_from_sofa(
        self,
        target_positions,
        attach_block,
        error_record_file: str = "Env_Eval/sofa_record.txt",
    ):
        """
        whole procedure of bringing out the garment from sofa
        """
        # render world
        self.world.step(render=True)
        # reach attach block position
        self.open()  # open franka's gripper

        position = torch.Tensor(target_positions[0])
        cprint("start to enter the initial point above sofa", "magenta")
        while not self.check_gripper_arrive(position):
            self.RMPflow_Move(position)

        # catch the block
        reach_position = attach_block.get_block_position().cpu()
        reach_position = reach_position[0]
        cprint(f"start to reach the fetch point : {reach_position}", "magenta")
        while not self.check_gripper_arrive(reach_position):
            if self.error_nochange_epoch >= 100:
                record_success_failure(
                    False, "Env_Eval/sofa_record.txt", str="pick_point is unreachable"
                )
                self.error_nochange_epoch = 0
                return False
            self.RMPflow_Move(reach_position)

        # close franka's gripper
        self.close()
        for i in range(30):
            self.world.step(render=True)
        cprint(
            f"start to go to the target push_garment position {target_positions[1]}",
            "magenta",
        )

        for i in range(len(target_positions)):
            position = torch.Tensor(target_positions[i])
            if i == 0:
                while not self.check_gripper_arrive(position):
                    self.move_block_follow_gripper(attach_block, position)
            else:
                while not self.check_gripper_arrive(position):
                    self.move_block_follow_gripper(
                        attach_block, position, target_orientation=[0, 90, 0]
                    )

        return True

    def fetch_garment_from_basket(
        self,
        target_positions,
        attach_block,
        error_record_file: str = "Env_Eval/basket_record.txt",
    ):
        """
        whole procedure of bringing out the garment from basket
        """
        # render world
        self.world.step(render=True)
        # reach attach block position
        self.open()  # open franka's gripper

        position = torch.Tensor(target_positions[1])
        cprint("start to enter the initial point above basket", "magenta")
        while not self.check_gripper_arrive(position):
            self.RMPflow_Move(position, orientation=[0.0, 90.0, 90.0])

        position = torch.Tensor(target_positions[0])
        cprint("start to enter the initial point above basket", "magenta")
        while not self.check_gripper_arrive(position):
            self.RMPflow_Move(position)

        # catch the block
        reach_position = attach_block.get_block_position().cpu()
        reach_position = reach_position[0]
        cprint(f"start to reach the fetch point : {reach_position}", "magenta")
        while not self.check_gripper_arrive(reach_position):
            if self.error_nochange_epoch >= 200:
                record_success_failure(
                    False, "Env_Eval/basket_record.txt", str="pick_point is unreachable"
                )
                self.error_nochange_epoch = 0
                return False
            self.RMPflow_Move(reach_position)

        # close franka's gripper
        self.close()
        for i in range(30):
            self.world.step(render=True)
        cprint("start to go to the target push_garment position", "magenta")

        for i in range(len(target_positions)):
            position = torch.Tensor(target_positions[i])
            if i == 0:
                while not self.check_gripper_arrive(position):
                    self.move_block_follow_gripper(attach_block, position)
            elif i == 1:
                continue
                while not self.check_gripper_arrive(position):
                    self.move_block_follow_gripper(
                        attach_block, position, target_orientation=[0, 90, 90]
                    )
            else:
                while not self.check_gripper_arrive(position):
                    if self.error_nochange_epoch >= 200:
                        record_success_failure(
                            False,
                            "Env_Eval/basket_record.txt",
                            str="pick_point is unreachable",
                        )
                        self.error_nochange_epoch = 0
                        return False
                    self.move_block_follow_gripper(
                        attach_block, position, target_orientation=[0, 90, 0]
                    )

        return True

    def pick(
        self,
        target_positions,
        attach_block,
        error_record_file: str = "Env_Eval/washmachine_record.txt",
    ):
        """
        whole procedure of bringing out the garment from wash_machine
        """
        # initialize franka
        self.initialize()
        # render world
        self.world.step(render=True)
        # reach attach block position
        self.open()  # open franka's gripper
        # enter washing machine

        position = torch.tensor(target_positions[0])
        print("start to enter washing machine")
        while not self.wm_check_gripper_arrive(
            position, stir=True, error_record_file=error_record_file
        ):
            self.RMPflow_Move(position)

        # catch the block
        reach_position = attach_block.get_block_position().cpu()
        reach_position = reach_position[0]
        print(f"start to reach the fetch point : {reach_position}")
        while not self.wm_check_gripper_arrive(
            reach_position, stir=True, error_record_file=error_record_file
        ):
            self.RMPflow_Move(reach_position)
        # close franka's gripper
        self.close()
        for i in range(30):
            self.world.step(render=True)
        print("start to go to the target push_garment position")
        # move franka according to the pre-set position

    def place(
        self,
        place_pos,
        attach_block,
        error_record_file: str = "Env_Eval/washmachine_record.txt",
    ):

        position = torch.tensor([0.0, 0.0, 0.75])
        while not self.wm_check_gripper_arrive(
            position, stir=True, error_record_file=error_record_file
        ):
            self.move_block_follow_gripper(attach_block, position)

        position = torch.Tensor(place_pos)
        while not self.wm_check_gripper_arrive(
            position, stir=True, error_record_file=error_record_file
        ):
            self.move_block_follow_gripper(attach_block, position)

    def adjust_after_stir(self, target_pos=[-1.2, -0.55, 0.9]):
        print("start to adjust after stir")
        position = torch.tensor(target_pos)

        while not self.wm_check_gripper_arrive(position, stir=True):
            self.RMPflow_Move(position)

    def return_to_initial_position(self, initial_position, initial_orientation=None):
        """
        return to initial position
        """
        initial_position = torch.Tensor(initial_position)
        while not self.check_gripper_arrive(initial_position):
            self.RMPflow_Move(initial_position, initial_orientation)
        self.open()
        print("return to initial position")

    def sofa_pick_place_procedure(self, target_positions, attach_block):
        """
        whole procedure of pick and place
        """
        # render world
        self.world.step(render=True)
        # reach attach block position
        self.open()  # open franka's gripper

        position = torch.Tensor(target_positions[0])
        cprint("start to enter the initial point above sofa", "magenta")
        while not self.check_gripper_arrive(position):
            self.RMPflow_Move(position)

        # catch the block
        reach_position = attach_block.get_block_position().cpu()
        reach_position = reach_position[0]
        cprint(f"start to reach the fetch point : {reach_position}", "magenta")
        while not self.check_gripper_arrive(reach_position):
            if self.error_nochange_epoch >= 200:
                # record_success_failure(False,"data/Record.txt",str="pick_point is unreachable")
                self.error_nochange_epoch = 0
                return False
            self.RMPflow_Move(reach_position)

        # close franka's gripper
        self.close()
        for i in range(30):
            self.world.step(render=True)
        cprint("start to go to the target push_garment position", "magenta")

        for i in range(len(target_positions)):
            position = torch.Tensor(target_positions[i])
            if i == 0:
                while not self.check_gripper_arrive(position):
                    self.move_block_follow_gripper(attach_block, position)
            else:
                while not self.check_gripper_arrive(position):
                    if self.error_nochange_epoch >= 200:
                        # record_success_failure(False,"data/Record.txt",str="pick_point is unreachable")
                        self.error_nochange_epoch = 0
                        return False
                    self.move_block_follow_gripper(attach_block, position)

        return True

    def basket_pick_place_procedure(self, target_positions, attach_block):
        """
        whole procedure of bringing out the garment from basket
        """
        # render world
        self.world.step(render=True)
        # reach attach block position
        self.open()  # open franka's gripper

        position = torch.Tensor(target_positions[1])
        cprint("start to enter the initial point above basket", "magenta")
        while not self.check_gripper_arrive(position):
            self.RMPflow_Move(position, orientation=[0.0, 90.0, 90.0])

        position = torch.Tensor(target_positions[0])
        cprint("start to enter the initial point above basket", "magenta")
        while not self.check_gripper_arrive(position):
            self.RMPflow_Move(position)

        # catch the block
        reach_position = attach_block.get_block_position().cpu()
        reach_position = reach_position[0]
        cprint(f"start to reach the fetch point : {reach_position}", "magenta")
        while not self.check_gripper_arrive(reach_position):
            if self.error_nochange_epoch >= 200:
                record_success_failure(
                    False, "Env_Eval/basket_record.txt", str="pick_point is unreachable"
                )
                self.error_nochange_epoch = 0
                return False
            self.RMPflow_Move(reach_position)

        # close franka's gripper
        self.close()
        for i in range(30):
            self.world.step(render=True)
        cprint("start to go to the target push_garment position", "magenta")

        for i in range(len(target_positions)):
            position = torch.Tensor(target_positions[i])
            if i == 0:
                while not self.check_gripper_arrive(position):
                    self.move_block_follow_gripper(attach_block, position)
            elif i == 1:
                continue
                while not self.check_gripper_arrive(position):
                    self.move_block_follow_gripper(
                        attach_block, position, target_orientation=[0, 90, 90]
                    )
            else:
                while not self.check_gripper_arrive(position):
                    if self.error_nochange_epoch >= 200:
                        record_success_failure(
                            False,
                            "Env_Eval/basket_record.txt",
                            str="pick_point is unreachable",
                        )
                        self.error_nochange_epoch = 0
                        return False
                    self.move_block_follow_gripper(attach_block, position)

        return True

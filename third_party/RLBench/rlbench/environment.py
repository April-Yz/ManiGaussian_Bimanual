import importlib
from os.path import exists, dirname, abspath, join
from typing import Type, List
import pickle

from pyrep import PyRep
from pyrep.objects import VisionSensor
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.robots.arms.dual_panda import PandaLeft
from pyrep.robots.arms.dual_panda import PandaRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperRight
from pyrep.robots.end_effectors.dual_panda_gripper import PandaGripperLeft

from rlbench import utils
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.const import *
from rlbench.backend.robot import Robot
from rlbench.backend.robot import UnimanualRobot
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.scene import Scene
from rlbench.backend.task import Task
from rlbench.const import SUPPORTED_ROBOTS
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig
from rlbench.sim2real.domain_randomization import RandomizeEvery, \
    VisualRandomizationConfig, DynamicsRandomizationConfig
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene
from rlbench.task_environment import TaskEnvironment

import logging

DIR_PATH = dirname(abspath(__file__))


class Environment(object):
    """Each environment has a scene."""

    def __init__(self,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 obs_config: ObservationConfig = ObservationConfig(),
                 headless: bool = False,
                 static_positions: bool = False,
                 robot_setup: str = 'panda',
                 randomize_every: RandomizeEvery = None,
                 frequency: int = 1,
                 visual_randomization_config: VisualRandomizationConfig = None,
                 dynamics_randomization_config: DynamicsRandomizationConfig = None,
                 attach_grasped_objects: bool = True,
                 shaped_rewards: bool = False
                 ):

        self._dataset_root = dataset_root
        self._action_mode = action_mode
        self._obs_config = obs_config
        self._headless = headless
        self._static_positions = static_positions
        self._robot_setup = robot_setup.lower()

        self._randomize_every = randomize_every
        self._frequency = frequency
        self._visual_randomization_config = visual_randomization_config
        self._dynamics_randomization_config = dynamics_randomization_config
        self._attach_grasped_objects = attach_grasped_objects
        self._shaped_rewards = shaped_rewards

        if robot_setup not in SUPPORTED_ROBOTS.keys():
            raise ValueError('robot_configuration must be one of %s' %
                             str(SUPPORTED_ROBOTS.keys()))

        if (randomize_every is not None and
                visual_randomization_config is None and
                dynamics_randomization_config is None):
            raise ValueError(
                'If domain randomization is enabled, must supply either '
                'visual_randomization_config or dynamics_randomization_config')

        self._check_dataset_structure()
        self._pyrep = None
        self._robot = None
        self._scene = None
        self._prev_task = None

    def _check_dataset_structure(self):
        if len(self._dataset_root) > 0 and not exists(self._dataset_root):
            raise RuntimeError(
                'Data set root does not exists: %s' % self._dataset_root)

    def _string_to_task(self, task_name: str):
        task_name = task_name.replace('.py', '')
        try:
            class_name = ''.join(
                [w[0].upper() + w[1:] for w in task_name.split('_')])
            mod = importlib.import_module("rlbench.tasks.%s" % task_name)
        except Exception as e:
            raise RuntimeError(
                'Tried to interpret %s as a task, but failed. Only valid tasks '
                'should belong in the tasks/ folder' % task_name) from e
        return getattr(mod, class_name)

    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError('Already called launch!')
        self._pyrep = PyRep()
        if self._robot_setup == 'dual_panda':
            self._pyrep.launch(join(DIR_PATH, BIMANUAL_TTT_FILE), headless=self._headless)
        else:
            self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=self._headless)

        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
            self._robot_setup]


        if self._robot_setup == 'dual_panda':

            logging.info("Using dual panda robot")
           
            #panda_arm = Panda()
            #panda_pos = panda_arm.get_position()
            #panda_arm.remove()

            right_arm = PandaRight()
            left_arm = PandaLeft()
            right_gripper = PandaGripperRight()
            left_gripper = PandaGripperLeft()

            # ..not updating position as we assume that the scene already contains two pandas which are placed correctly     
            #relative_left_position = left_arm.get_position(relative_to=right_arm)            
            #right_arm.set_position(panda_pos)
            #left_arm.set_position(relative_left_position, relative_to=right_arm)

            self._robot = BimanualRobot(right_arm, right_gripper, left_arm, left_gripper)

        # We assume the panda is already loaded in the scene.
        elif self._robot_setup != 'panda':
            # Remove the panda from the scene
            panda_arm = Panda()
            panda_pos = panda_arm.get_position()
            panda_arm.remove()
            arm_path = join(DIR_PATH, 'robot_ttms', self._robot_setup + '.ttm')
            self._pyrep.import_model(arm_path)
            arm, gripper = arm_class(), gripper_class()
            arm.set_position(panda_pos)
            self._robot = UnimanualRobot(arm, gripper)
        else:
            arm, gripper = arm_class(), gripper_class()
            self._robot = UnimanualRobot(arm, gripper)


        if self._randomize_every is None:
            self._scene = Scene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup)
        else:
            self._scene = DomainRandomizationScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup,
                self._randomize_every, self._frequency,
                self._visual_randomization_config,
                self._dynamics_randomization_config)

        self._action_mode.arm_action_mode.set_control_mode(self._robot)

    def shutdown(self):
        if self._pyrep is not None:
            self._pyrep.shutdown()
        self._pyrep = None

    def get_task(self, task_class: Type[Task]) -> TaskEnvironment:

        # If user hasn't called launch, implicitly call it.  
        if self._pyrep is None:
            self.launch()
        self._scene.unload()
        task = task_class(self._pyrep, self._robot)
        self._prev_task = task
        return TaskEnvironment(
            self._pyrep, self._robot, self._scene, task,
            self._action_mode, self._dataset_root, self._obs_config,
            self._static_positions, self._attach_grasped_objects,
            self._shaped_rewards 
            )

    @property
    def action_shape(self):
        return self._action_mode.action_shape(self._scene),

    def get_demos(self, task_name: str, amount: int,
                  variation_number=0,
                  image_paths=False,
                  random_selection: bool = True,
                  from_episode_number: int = 0) -> List[Demo]:
        if self._dataset_root is None or len(self._dataset_root) == 0:
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")
        demos = utils.get_stored_demos(
            amount, image_paths, self._dataset_root, variation_number,
            task_name, self._obs_config, random_selection, from_episode_number)
        return demos

    def get_task_descriptions_with_episode(self, task_name: str,
                                           episode_number: int) -> List[str]:
        episode_description_pkl_file = join(self._dataset_root,
                                            f'{task_name}',
                                            VARIATIONS_ALL_FOLDER,
                                            EPISODES_FOLDER,
                                            EPISODE_FOLDER % episode_number,
                                            VARIATION_DESCRIPTIONS)
        with open(episode_description_pkl_file, 'rb') as f:
            episode_description = pickle.load(f)

        return episode_description

    def get_scene_data(self) -> dict:
        """Get the data of various scene/camera information.

        This temporarily starts the simulator in headless mode.

        :return: A dictionary containing scene data.
        """

        def _get_cam_info(cam: VisionSensor):
            if not cam.still_exists():
                logging.warning("Camera no longer exists")
                return None
            intrinsics = cam.get_intrinsic_matrix()
            return dict(
                intrinsics=intrinsics,
                near_plane=cam.get_near_clipping_plane(),
                far_plane=cam.get_far_clipping_plane(),
                extrinsics=cam.get_matrix())

        headless = self._headless
        self._headless = True
        self.launch()

        d = {camera_name: _get_cam_info(sensor) for camera_name, sensor in self._scene.camera_sensors}

        self.shutdown()
        self._headless = headless
        return d

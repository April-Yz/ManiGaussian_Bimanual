from abc import abstractmethod

import numpy as np
from pyquaternion import Quaternion

from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.const import ConfigurationPathAlgorithms as ObjectType
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.const import ObjectType
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper

from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.robot import Robot
from rlbench.backend.robot import UnimanualRobot
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.scene import Scene
from rlbench.const import SUPPORTED_ROBOTS

import logging

from abc import ABC



def assert_action_shape(action: np.ndarray, expected_shape: tuple):
    if np.shape(action) != expected_shape:
        raise InvalidActionError(
            'Expected the action shape to be: %s, but was shape: %s' % (
                str(expected_shape), str(np.shape(action))))


def assert_unit_quaternion(quat):
    if not np.isclose(np.linalg.norm(quat), 1.0):
        raise InvalidActionError('Action contained non unit quaternion!')


def calculate_delta_pose(robot: Robot, action: np.ndarray):
    a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
    x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
    new_rot = Quaternion(
        a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
    qw, qx, qy, qz = list(new_rot)
    pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
    return pose


class ArmActionMode(ABC):
    
    _callable_each_step = None

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene)
        self.action_post_step(scene, action)    

    def action_step(self, scene: Scene):
        scene.step()
        if self._callable_each_step is not None:
            self._callable_each_step(scene.get_observation())

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        pass

    def action_post_step(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass

    def set_control_mode(self, robot: Robot):
        if isinstance(robot, UnimanualRobot):
            robot.arm.set_control_loop_enabled(True)
        elif isinstance(robot, BimanualRobot):
            logging.info("Setting control mode for both robots")
            robot.right_arm.set_control_loop_enabled(True)
            robot.left_arm.set_control_loop_enabled(True)

    def record_end(self, scene, steps=60, step_scene=True):
        if self._callable_each_step is not None:
            for _ in range(steps):
                if step_scene:
                    scene.step()
                self._callable_each_step(scene.get_observation())

    def set_callable_each_step(self, callable_each_step):
        self._callable_each_step = callable_each_step

class JointVelocity(ArmActionMode):
    """Control the joint velocities of the arm.

    Similar to the action space in many continious control OpenAI Gym envs.
    """

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        scene.robot.arm.set_joint_target_velocities(action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.arm.set_control_loop_enabled(False)
        robot.arm.set_motor_locked_at_zero_velocity(True)


class BimanualJointVelocity(ArmActionMode): 

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        right_action = action[:7]
        left_action = action[7:]
        scene.robot.right_arm.set_joint_target_velocities(right_action)
        scene.robot.left_arm.set_joint_target_velocities(left_action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))
        right_action = action[:7]
        left_action = action[7:]
        scene.robot.right_arm.set_joint_target_velocities(np.zeros_like(right_action))
        scene.robot.left_arm.set_joint_target_velocities(np.zeros_like(left_action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def set_control_mode(self, robot: Robot):
        robot.right_arm.set_control_loop_enabled(False)
        robot.right_arm.set_motor_locked_at_zero_velocity(True)
        robot.left_arm.set_control_loop_enabled(False)
        robot.left_arm.set_motor_locked_at_zero_velocity(True)
        

class BimanualJointPosition(ArmActionMode):

    def __init__(self, absolute_mode: bool = True):
        self._absolute_mode = absolute_mode

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        
        right_action = action[:7]
        left_action = action[7:]

        if not self._absolute_mode:
            right_action = np.array(scene.robot.right_arm.get_joint_positions()) + right_action
            left_action = np.array(scene.robot.left_arm.get_joint_positions()) + left_action
            
        scene.robot.right_arm.set_joint_target_positions(right_action)
        scene.robot.left_arm.set_joint_target_positions(left_action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.right_arm.set_joint_target_positions(
            scene.robot.right_arm.get_joint_positions())
        scene.robot.left_arm.set_joint_target_positions(
            scene.robot.left_arm.get_joint_positions())
    
    def action_shape(self, scene: Scene) -> tuple:
        return (14, )
        #return SUPPORTED_ROBOTS[scene.robot_setup][2],

    def unimanual_action_shape(self, scene: Scene) -> tuple:
        return (7, )



class JointPosition(ArmActionMode):
    """Control the target joint positions (absolute or delta) of the arm.

    The action mode opoerates in absolute mode or delta mode, where delta
    mode takes the current joint positions and adds the new joint positions
    to get a set of target joint positions. The robot uses a simple control
    loop to execute until the desired poses have been reached.
    It os the users responsibility to ensure that the action lies within
    a usuable range.
    """

    def __init__(self, absolute_mode: bool = True):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
        """
        self._absolute_mode = absolute_mode


    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        if not  self._absolute_mode :
            action = np.array(scene.robot.arm.get_joint_positions()) + action
        scene.robot.arm.set_joint_target_positions(action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.arm.set_joint_target_positions(
            scene.robot.arm.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],



class JointTorque(ArmActionMode):
    """Control the joint torques of the arm.
    """

    TORQUE_MAX_VEL = 9999

    def _torque_action(self, robot, action):
        tml = JointTorque.TORQUE_MAX_VEL
        robot.arm.set_joint_target_velocities(
            [(tml if t < 0 else -tml) for t in action])
        robot.arm.set_joint_forces(np.abs(action))

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, self.action_shape(scene))
        self._torque_action(scene.robot, action)

    def action_post_step(self, scene: Scene, action: np.ndarray):
        self._torque_action(scene.robot, scene.robot.arm.get_joint_forces())
        scene.robot.arm.set_joint_target_velocities(np.zeros_like(action))

    def action_shape(self, scene: Scene) -> tuple:
        return SUPPORTED_ROBOTS[scene.robot_setup][2],


class EndEffectorPoseViaPlanning(ArmActionMode):
    """High-level action where target pose is given and reached via planning.

    Given a target pose, a linear path is first planned (via IK). If that fails,
    sample-based planning will be used. The decision to apply collision
    checking is a crucial trade off! With collision checking enabled, you
    are guaranteed collision free paths, but this may not be applicable for task
    that do require some collision. E.g. using this mode on pushing object will
    mean that the generated path will actively avoid not pushing the object.

    Note that path planning can be slow, often taking a few seconds in the worst
    case.

    This was the action mode used in:
    James, Stephen, and Andrew J. Davison. "Q-attention: Enabling Efficient
    Learning for Vision-based Robotic Manipulation."
    arXiv preprint arXiv:2105.14829 (2021).
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False):
        """
        If collision check is enbled, and an object is grasped, then we

        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._callable_each_step = None
        self._robot_shapes = None

        if frame not in ['world', 'end effector']:
            raise ValueError("Expected frame to one of: 'world, 'end effector'")

    def _quick_boundary_check(self, scene: Scene, action: np.ndarray):
        pos_to_check = action[:3]
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()
        if relative_to is not None:
            scene.target_workspace_check.set_position(pos_to_check, relative_to)
            pos_to_check = scene.target_workspace_check.get_position()
        if not scene.check_target_in_workspace(pos_to_check):
            raise InvalidActionError('A path could not be found because the '
                                     'target is outside of workspace.')

    def _pose_in_end_effector_frame(self, robot: Robot, action: np.ndarray):
        a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = action
        x, y, z, qx, qy, qz, qw = robot.arm.get_tip().get_pose()
        new_rot = Quaternion(
            a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx, qy, qz)
        qw, qx, qy, qz = list(new_rot)
        pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
        return pose
    
    def set_callable_each_step(self, callable_each_step):
        self._callable_each_step = callable_each_step


    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        path = self.get_path(scene, action, ignore_collisions, scene.robot.arm, scene.robot.gripper)
        done = False
        while not done:
            done = path.step()
            scene.step()
            if self._callable_each_step is not None:
                self._callable_each_step(scene.get_observation())
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break    

    def get_path(self, scene: Scene, action: np.ndarray, ignore_collisions: bool, arm: Arm, gripper: Gripper):
        # logging.info("######## 0 in get_path")
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else arm.get_tip()
        self._quick_boundary_check(scene, action)
        # logging.info("2 in get_path")

        colliding_shapes = []
        if not ignore_collisions:
            if self._robot_shapes is None:
                self._robot_shapes = arm.get_objects_in_tree(
                    object_type=ObjectType.SHAPE)
            # logging.info("3 in get_path")
            # First check if we are colliding with anything
            colliding = arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = gripper.get_grasped_objects()
                colliding_shapes = [
                    s for s in scene.pyrep.get_objects_in_tree(
                        object_type = ObjectType.SHAPE) if (
                            s.is_collidable() and
                            s not in self._robot_shapes and
                            s not in grasped_objects and
                            arm.check_arm_collision(
                                s))]
                [s.set_collidable(False) for s in colliding_shapes]
                # logging.info("4 if in get_path")
            # logging.info("5 in get_path")
            
        try:
            # try once with collision checking (if ignore_collisions is true)
            try:
                # logging.info("6 try in get_path")
                path = arm.get_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=ignore_collisions,
                    relative_to=relative_to,
                    trials=200, #..TODO was 100
                    max_configs=10,
                    max_time_ms=20, #..TODO was 10
                    trials_per_goal=10, #..TODO was 5
                    algorithm=Algos.RRTConnect
                )
                # logging.info("######## FINALLY in get_path")
                return path
            except ConfigurationPathError as e:
                if ignore_collisions:
                    raise InvalidActionError(
                        'A path could not be found. Most likely due to the target '
                        'being inaccessible or a collison was detected.') from e
                else:
                    # try once more with collision checking disabled
                    path = arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=True,
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect
                    )
        except ConfigurationPathError as e:
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e


    def action_shape(self, scene: Scene) -> tuple:
        return 7,



class UnimanualEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False,
                 robot_name: str = ''):
        super().__init__(absolute_mode, frame, collision_checking)
        self.robot_name = robot_name

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if self.robot_name == 'right':
            path = self.get_path(scene, action, ignore_collisions, scene.robot.right_arm, scene.robot.right_gripper)
        elif self.robot_name == 'left':
            path = self.get_path(scene, action, ignore_collisions, scene.robot.left_arm, scene.robot.left_gripper)
        else:
            logging.error('Invalid robot name')

        if not path:
            logging.warning('No path found')
            return
        done = False
        while not done:
            done = path.step()
            scene.step()
            if self._callable_each_step is not None:
                # Record observations
                self._callable_each_step(scene.get_observation())
            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success and self._callable_each_step is None:
                break

class BimanualEndEffectorPoseViaPlanning(EndEffectorPoseViaPlanning):


    def action(self, scene: Scene, action: np.ndarray, ignore_collisions):

        assert_action_shape(action, self.action_shape(scene))
 
        right_action = action[:7]
        left_action = action[7:]

        right_ignore_collision = ignore_collisions[0]
        left_ignore_collison = ignore_collisions[1]

        assert_unit_quaternion(right_action[3:])
        assert_unit_quaternion(left_action[3:])

        right_done = True
        left_done = True
        try:
            right_path = self.get_path(scene, right_action, right_ignore_collision, scene.robot.right_arm, scene.robot.right_gripper)
            if right_path:
                right_done = False
            else:
                logging.warning("right path is none")
        except (ConfigurationPathError, InvalidActionError):
            pass
        
        try:
            left_path = self.get_path(scene, left_action, left_ignore_collison, scene.robot.left_arm, scene.robot.left_gripper)
            if left_path:
                left_done = False
            else:
                logging.warning("left path is none")
        except (ConfigurationPathError, InvalidActionError):
            pass
        

        done = False

        while not done:
            if not right_done and right_path:
                right_done = right_path.step()
            if not left_done and left_path:
                left_done = left_path.step()

            done = right_done and left_done
            scene.step()
            if self._callable_each_step is not None:
                self._callable_each_step(scene.get_observation())

            success, terminate = scene.task.success()
            # If the task succeeds while traversing path, then break early
            if success:
                break
    
    def action_shape(self, scene: Scene) -> tuple:
        return 14,

    def unimanual_action_shape(self, scene: Scene) -> tuple:
        return 7,


class EndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(self,
                 absolute_mode: bool = True,
                 frame: str = 'world',
                 collision_checking: bool = False):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either 'world' or 'end effector'.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        if frame not in ['world', 'end effector']:
            raise ValueError(
                "Expected frame to one of: 'world, 'end effector'")

    def action(self, scene: Scene, action: np.ndarray):
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])
        if not self._absolute_mode and self._frame != 'end effector':
            action = calculate_delta_pose(scene.robot, action)
        relative_to = None if self._frame == 'world' else scene.robot.arm.get_tip()

        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian; most likely due to current '
                'end-effector pose being too far from the given target pose. '
                'Try limiting/bounding your action space.') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            scene.step()
            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def action_shape(self, scene: Scene) -> tuple:
        return 7,

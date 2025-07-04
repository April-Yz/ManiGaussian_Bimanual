import numpy as np
import torch
from helpers import utils
from pytorch3d import transforms as torch3d_tf
from termcolor import cprint
import einops
from scipy.spatial.transform import Rotation as R

def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
    """Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # baatch bounds if necessary
    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        num_points = p_shape[-1] * p_shape[-2]

        action_trans_3x1 = (
            action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )
        trans_shift_3x1 = (
            trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(
            p_flat_4x1_action_origin.transpose(2, 1), rot_shift_4x4
        ).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(
            action_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
        )
        action_then_trans_3x1_y = torch.clamp(
            action_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
        )
        action_then_trans_3x1_z = torch.clamp(
            action_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
        )
        action_then_trans_3x1 = torch.stack(
            [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
            dim=1,
        )

        # shift back the origin
        perturbed_p_flat_3x1 = (
            perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
        )

        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)
    return perturbed_pcd



def perturb_se3_camera_pose(camera_pose,        
                trans_shift_4x4,               
                rot_shift_4x4,               
                action_gripper_4x4,          
                bounds):              
    """ 
    Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # batch bounds if necessary
    bs = camera_pose[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_camera_pose = []
    for cam_pose in camera_pose:
        
        cam_R, cam_T = cam_pose[:, :3, :3], cam_pose[:, :3, 3:]

        # action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(bs, 1, 1)
        # trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(bs, 1, 1)
        action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1)

        cam_T = cam_T - action_trans_3x1    # [bs, 3, 1]
        cam_T_4x1 = torch.ones(bs, 4, 1).to(cam_T.device)
        cam_T_4x1[:, :3, :] = cam_T
        cam_T_4x1 = torch.bmm(cam_T_4x1.transpose(2, 1), rot_shift_4x4).transpose(2, 1)

        cam_R = torch.bmm(cam_R.transpose(2, 1), rot_shift_4x4[:, :3, :3]).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(action_then_trans_3x1[:, 0],
                                              min=bounds_x_min, max=bounds_x_max)
        action_then_trans_3x1_y = torch.clamp(action_then_trans_3x1[:, 1],
                                              min=bounds_y_min, max=bounds_y_max)
        action_then_trans_3x1_z = torch.clamp(action_then_trans_3x1[:, 2],
                                              min=bounds_z_min, max=bounds_z_max)
        action_then_trans_3x1 = torch.stack([action_then_trans_3x1_x,
                                             action_then_trans_3x1_y,
                                             action_then_trans_3x1_z], dim=1)

        # shift back the origin 
        cam_T_4x1[:, :3]  = cam_T_4x1[:, :3] + action_then_trans_3x1

        cam_T = cam_T_4x1[:, :3, :]
        cam_pose[:, :3, :3], cam_pose[:, :3, 3:] = cam_R, cam_T
        perturbed_camera_pose.append(cam_pose)

    return perturbed_camera_pose



#### original
def peract2_bimanual_apply_se3_augmentation_with_camera_pose(
    pcd,
    camera_pose,
    right_action_gripper_pose,
    right_action_trans,
    right_action_rot_grip,
    left_action_gripper_pose,
    left_action_trans,
    left_action_rot_grip,
    bounds,
    layer,
    trans_aug_range,
    rot_aug_range,
    rot_aug_resolution,
    voxel_size,
    rot_resolution,
    device,
):
    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    right_action_gripper_trans = right_action_gripper_pose[:, :3]
    right_action_gripper_quat_wxyz = torch.cat(
        (
            right_action_gripper_pose[:, 6].unsqueeze(1),
            right_action_gripper_pose[:, 3:6],
        ),
        dim=1,
    )

    right_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        right_action_gripper_quat_wxyz
    )
    right_action_gripper_4x4 = identity_4x4.detach().clone()
    right_action_gripper_4x4[:, :3, :3] = right_action_gripper_rot
    right_action_gripper_4x4[:, 0:3, 3] = right_action_gripper_trans

    right_perturbed_trans = torch.full_like(right_action_trans, -1.0)
    right_perturbed_rot_grip = torch.full_like(right_action_rot_grip, -1.0)

    left_action_gripper_trans = left_action_gripper_pose[:, :3]
    left_action_gripper_quat_wxyz = torch.cat(
        (left_action_gripper_pose[:, 6].unsqueeze(1), left_action_gripper_pose[:, 3:6]),
        dim=1,
    )

    left_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        left_action_gripper_quat_wxyz
    )
    left_action_gripper_4x4 = identity_4x4.detach().clone()
    left_action_gripper_4x4[:, :3, :3] = left_action_gripper_rot
    left_action_gripper_4x4[:, 0:3, 3] = left_action_gripper_trans

    left_perturbed_trans = torch.full_like(left_action_trans, -1.0)
    left_perturbed_rot_grip = torch.full_like(left_action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(right_perturbed_trans < 0) and torch.any(left_perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            raise Exception("Failing to perturb action and keep it within bounds.")

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        # trans_shift = torch.tensor([0.0, 0.0, 0.0]).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete(
            (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        # yaw = torch.tensor([45.0]).repeat(bs, 1)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        )
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        right_perturbed_action_gripper_4x4 = torch.bmm(
            right_action_gripper_4x4, rot_shift_4x4
        )
        right_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        right_perturbed_action_trans = (
            right_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        right_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            right_perturbed_action_gripper_4x4[:, :3, :3]
        )
        right_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    right_perturbed_action_quat_wxyz[:, 1:],
                    right_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # rotate then translate the 4x4 keyframe action
        left_perturbed_action_gripper_4x4 = torch.bmm(
            left_action_gripper_4x4, rot_shift_4x4
        )
        left_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        left_perturbed_action_trans = (
            left_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        left_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            left_perturbed_action_gripper_4x4[:, :3, :3]
        )
        left_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    left_perturbed_action_quat_wxyz[:, 1:],
                    left_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        right_trans_indicies, right_rot_grip_indicies = [], []
        left_trans_indicies, left_rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            right_trans_idx = utils.point_to_voxel_index(
                right_perturbed_action_trans[b], voxel_size, bounds_np
            )
            right_trans_indicies.append(right_trans_idx.tolist())

            right_quat = right_perturbed_action_quat_xyzw[b]
            right_quat = utils.normalize_quaternion(right_perturbed_action_quat_xyzw[b])
            if right_quat[-1] < 0:
                right_quat = -right_quat
            right_disc_rot = utils.quaternion_to_discrete_euler(
                right_quat, rot_resolution
            )
            right_rot_grip_indicies.append(
                right_disc_rot.tolist()
                + [int(right_action_rot_grip[b, 3].cpu().numpy())]
            )

            left_trans_idx = utils.point_to_voxel_index(
                left_perturbed_action_trans[b], voxel_size, bounds_np
            )
            left_trans_indicies.append(left_trans_idx.tolist())

            left_quat = left_perturbed_action_quat_xyzw[b]
            left_quat = utils.normalize_quaternion(left_perturbed_action_quat_xyzw[b])
            if left_quat[-1] < 0:
                left_quat = -left_quat
            left_disc_rot = utils.quaternion_to_discrete_euler(
                left_quat, rot_resolution
            )
            left_rot_grip_indicies.append(
                left_disc_rot.tolist() + [int(left_action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        right_perturbed_trans = torch.from_numpy(np.array(right_trans_indicies)).to(
            device=device
        )
        right_perturbed_rot_grip = torch.from_numpy(
            np.array(right_rot_grip_indicies)
        ).to(device=device)

        left_perturbed_trans = torch.from_numpy(np.array(left_trans_indicies)).to(
            device=device
        )
        left_perturbed_rot_grip = torch.from_numpy(np.array(left_rot_grip_indicies)).to(
            device=device
        )

    right_action_trans = right_perturbed_trans
    right_action_rot_grip = right_perturbed_rot_grip

    left_action_trans = left_perturbed_trans
    left_action_rot_grip = left_perturbed_rot_grip

    # apply perturbation to pointclouds
    # pcd = bimanual_perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, left_action_gripper_4x4, bounds)

    pcd = perturb_se3(
        pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, bounds
    )
    camera_pose = perturb_se3_camera_pose(camera_pose, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, bounds)

    return (
        right_action_trans,
        right_action_rot_grip,
        left_action_trans,
        left_action_rot_grip,
        pcd,
        camera_pose,
    )

def bimanual_apply_se3_augmentation_with_camera_pose(
    pcd,
    camera_pose,
    right_action_gripper_pose,
    right_action_trans,
    right_action_rot_grip,
    left_action_gripper_pose,
    left_action_trans,
    left_action_rot_grip,
    bounds,
    layer,
    trans_aug_range,
    rot_aug_range,
    rot_aug_resolution,
    voxel_size,
    rot_resolution,
    device,
    ):
    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose

    # center_action = (right_action_gripper_pose[:, :3] + left_action_gripper_pose[:, :3]) / 2
    center_action = right_action_gripper_pose[:, :3]
    # center_action = left_action_gripper_pose[:, :3]
    # center_action = torch.tensor([[0.0, 0.0, 0.0]]).to(device=device)

    right_action_gripper_trans = right_action_gripper_pose[:, :3]
    right_action_gripper_quat_wxyz = torch.cat(
        (
            right_action_gripper_pose[:, 6].unsqueeze(1),
            right_action_gripper_pose[:, 3:6],
        ),
        dim=1,
    )

    right_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        right_action_gripper_quat_wxyz
    )
    right_action_gripper_4x4 = identity_4x4.detach().clone()
    right_action_gripper_4x4[:, :3, :3] = right_action_gripper_rot
    right_action_gripper_4x4[:, 0:3, 3] = right_action_gripper_trans

    right_perturbed_trans = torch.full_like(right_action_trans, -1.0)
    right_perturbed_rot_grip = torch.full_like(right_action_rot_grip, -1.0)

    left_action_gripper_trans = left_action_gripper_pose[:, :3]
    left_action_gripper_quat_wxyz = torch.cat(
        (left_action_gripper_pose[:, 6].unsqueeze(1), left_action_gripper_pose[:, 3:6]),
        dim=1,
    )

    left_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        left_action_gripper_quat_wxyz
    )
    left_action_gripper_4x4 = identity_4x4.detach().clone()
    left_action_gripper_4x4[:, :3, :3] = left_action_gripper_rot
    left_action_gripper_4x4[:, 0:3, 3] = left_action_gripper_trans

    left_perturbed_trans = torch.full_like(left_action_trans, -1.0)
    left_perturbed_rot_grip = torch.full_like(left_action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(right_perturbed_trans < 0) or torch.any(left_perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            # raise Exception("Failing to perturb action and keep it within bounds.")
            cprint("Failed to perturb action and keep it within bounds, return the original action.", "red")
            # trans_shift_4x4 = identity_4x4.detach().clone()
            # rot_shift_4x4 = identity_4x4.detach().clone()
            # left_perturbed_trans
            return (
                right_action_trans,
                right_action_rot_grip,
                left_action_trans,
                left_action_rot_grip,
                pcd,
                camera_pose,
            )
            
        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        # trans_shift = torch.tensor([0.0, 0.0, 0.0]).to(device=device)
        # for debugging

        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete(
            (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        # yaw = torch.tensor([45.0]).repeat(bs, 1)
        # for debugging

        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        ).to(device=device)
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        def transform(matrix, C, translation, rotation_matrix):
            """
            Rotate and translate a 4x4 gripper pose matrix around the center C.
            TBD.

            Args:
                matrix: 4x4 matrix to be transformed. tensor.
                C: 3 Center of rotation. tensor.
                translation: 3 Translation vector. tensor.
                rotation_matrix: 3x3 rotation matrix. tensor.
            """
            # print(matrix.shape)
            # print(C.shape)
            matrix[:, 0:3, 3] -= C
            matrix[:, 0:3, 3] = torch.matmul(matrix[:, 0:3, 3].unsqueeze(1), rotation_matrix).squeeze(1)
            matrix[:, 0:3, 3] += C
            matrix[:, 0:3, 3] += translation

            matrix[:, :3, :3] = torch.matmul(matrix[:, :3, :3], rotation_matrix)

            # print(matrix.shape)
            return matrix

        # rotate then translate the 4x4 keyframe action
        right_perturbed_action_gripper_4x4 = torch.bmm(
            right_action_gripper_4x4, rot_shift_4x4
        )
        right_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        right_perturbed_action_trans = (
            right_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        right_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            right_perturbed_action_gripper_4x4[:, :3, :3]
        )
        right_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    right_perturbed_action_quat_wxyz[:, 1:],
                    right_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        left_perturbed_action_gripper_4x4 = left_action_gripper_4x4.detach().clone()  # [4, 4]
        left_perturbed_action_gripper_4x4 = transform(
            left_perturbed_action_gripper_4x4,
            center_action,
            trans_shift,
            rot_shift_3x3
        )

        # convert transformation matrix to translation + quaternion
        # print(left_perturbed_action_gripper_4x4.shape)
        left_perturbed_action_trans = (
            left_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )

        # print(left_perturbed_action_gripper_4x4.shape)
        left_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            left_perturbed_action_gripper_4x4[:, :3, :3]
        )
        left_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    left_perturbed_action_quat_wxyz[:, 1:],
                    left_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        right_trans_indicies, right_rot_grip_indicies = [], []
        left_trans_indicies, left_rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            right_trans_idx = utils.point_to_voxel_index(
                right_perturbed_action_trans[b], voxel_size, bounds_np
            )
            right_trans_indicies.append(right_trans_idx.tolist())

            right_quat = right_perturbed_action_quat_xyzw[b]
            right_quat = utils.normalize_quaternion(right_perturbed_action_quat_xyzw[b])
            if right_quat[-1] < 0:
                right_quat = -right_quat
            right_disc_rot = utils.quaternion_to_discrete_euler(
                right_quat, rot_resolution
            )
            right_rot_grip_indicies.append(
                right_disc_rot.tolist()
                + [int(right_action_rot_grip[b, 3].cpu().numpy())]
            )

            left_trans_idx = utils.point_to_voxel_index(
                left_perturbed_action_trans[b], voxel_size, bounds_np
            )
            left_trans_indicies.append(left_trans_idx.tolist())

            left_quat = left_perturbed_action_quat_xyzw[b]
            left_quat = utils.normalize_quaternion(left_perturbed_action_quat_xyzw[b])
            if left_quat[-1] < 0:
                left_quat = -left_quat
            left_disc_rot = utils.quaternion_to_discrete_euler(
                left_quat, rot_resolution
            )
            left_rot_grip_indicies.append(
                left_disc_rot.tolist() + [int(left_action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        right_perturbed_trans = torch.from_numpy(np.array(right_trans_indicies)).to(
            device=device
        )
        right_perturbed_rot_grip = torch.from_numpy(
            np.array(right_rot_grip_indicies)
        ).to(device=device)

        left_perturbed_trans = torch.from_numpy(np.array(left_trans_indicies)).to(
            device=device
        )
        left_perturbed_rot_grip = torch.from_numpy(np.array(left_rot_grip_indicies)).to(
            device=device
        )

    right_action_trans = right_perturbed_trans
    right_action_rot_grip = right_perturbed_rot_grip

    left_action_trans = left_perturbed_trans
    left_action_rot_grip = left_perturbed_rot_grip

    center_action_gripper_4x4 = identity_4x4.detach().clone()
    # center_action_gripper_4x4 = torch.bmm(center_action_gripper_4x4, rot_shift_4x4)
    # center_action_gripper_4x4 = rot_shift_4x4
    center_action_gripper_4x4[:, 0:3, 3] += center_action
    pcd = perturb_se3(
        pcd, trans_shift_4x4, rot_shift_4x4, center_action_gripper_4x4, bounds
    )
    camera_pose = perturb_se3_camera_pose(camera_pose, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, bounds)
    return (
        right_action_trans,
        right_action_rot_grip,
        left_action_trans,
        left_action_rot_grip,
        pcd,
        camera_pose,
    )

import importlib
import pickle

from itertools import product            
import os
from os import listdir
from os.path import join, exists
from typing import List

import numpy as np
from PIL import Image
from natsort import natsorted
from pyrep.objects import VisionSensor

from rlbench.backend.const import *
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask, rgb_handles_to_mask_mani
from rlbench.demo import Demo
from rlbench.observation_config import ObservationConfig

import logging


class InvalidTaskName(Exception):
    pass


def name_to_task_class(task_file: str, bimanual=False):
    name = task_file.replace('.py', '')
    class_name = ''.join([w[0].upper() + w[1:] for w in name.split('_')])
    try:
        if bimanual:
            mod = importlib.import_module("rlbench.bimanual_tasks.%s" % name)
        else:
            mod = importlib.import_module("rlbench.tasks.%s" % name)
        mod = importlib.reload(mod)
    except ModuleNotFoundError as e:
        raise InvalidTaskName(
            "The task file '%s' does not exist or cannot be compiled."
            % name) from e
    try:
        task_class = getattr(mod, class_name)
    except AttributeError as e:
        raise InvalidTaskName(
            "Cannot find the class name '%s' in the file '%s'."
            % (class_name, name)) from e
    return task_class


def get_stored_demos(amount: int, image_paths: bool, dataset_root: str,
                     variation_number: int, task_name: str,
                     obs_config: ObservationConfig,
                     random_selection: bool = True,
                     from_episode_number: int = 0) -> List[Demo]:
    task_root = join(dataset_root, task_name)
    if not exists(task_root):
        raise RuntimeError("Can't find the demos for %s at: %s" % (
            task_name, task_root))

    if variation_number == -1:
        # All variations 
        examples_path = join(
            task_root, VARIATIONS_ALL_FOLDER,
            EPISODES_FOLDER)
        examples = listdir(examples_path)
    else:
        # Sample an amount of examples for the variation of this task
        examples_path = join(
            task_root, VARIATIONS_FOLDER % variation_number,
            EPISODES_FOLDER)
        examples = listdir(examples_path)

    # hack: ignore .DS_Store files from macOS zips
    examples = [e for e in examples if '.DS_Store' not in e]

    if amount == -1:
        amount = len(examples)
    if amount > len(examples):
        raise RuntimeError(
            'You asked for %d examples, but only %d were available.' % (
                amount, len(examples)))
    
    if amount > len(examples[from_episode_number:]):
        raise RuntimeError('You specified from_episode_number=%d, but only %d examples were available', from_episode_number,  len(examples))

    if random_selection:
        selected_examples = np.random.choice(examples, amount, replace=False)
    else:
        selected_examples = natsorted(
            examples)[from_episode_number:from_episode_number+amount]

    # Process these examples (e.g. loading observations)
    demos = []
    for example in selected_examples:
        example_path = join(examples_path, example)
        with open(join(example_path, LOW_DIM_PICKLE), 'rb') as f:
            obs = pickle.load(f)

        if variation_number == -1:
            with open(join(example_path, VARIATION_NUMBER), 'rb') as f:
                # print(obs)
                obs.variation_number = pickle.load(f)
        else:
            obs.variation_number = variation_number

        # language description
        episode_descriptions_f = join(example_path, VARIATION_DESCRIPTIONS)
        if exists(episode_descriptions_f):
            with open(episode_descriptions_f, 'rb') as f:
                descriptions = pickle.load(f)
        else:
            descriptions = ["unknown task description"]

        # l_sh_rgb_f = join(example_path, LEFT_SHOULDER_RGB_FOLDER)
        # l_sh_depth_f = join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
        # l_sh_mask_f = join(example_path, LEFT_SHOULDER_MASK_FOLDER)
        # r_sh_rgb_f = join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
        # r_sh_depth_f = join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
        # r_sh_mask_f = join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
        # oh_rgb_f = join(example_path, OVERHEAD_RGB_FOLDER)
        # oh_depth_f = join(example_path, OVERHEAD_DEPTH_FOLDER)
        # oh_mask_f = join(example_path, OVERHEAD_MASK_FOLDER)
        # wrist_rgb_f = join(example_path, WRIST_RGB_FOLDER)
        # wrist_depth_f = join(example_path, WRIST_DEPTH_FOLDER)
        # wrist_mask_f = join(example_path, WRIST_MASK_FOLDER)
        # front_rgb_f = join(example_path, FRONT_RGB_FOLDER)
        # front_depth_f = join(example_path, FRONT_DEPTH_FOLDER)
        # front_mask_f = join(example_path, FRONT_MASK_FOLDER)
        # # for nerf
        nerf_multi_view_f = join(example_path, NERF_MULTI_VIEW_FOLDER)

        num_steps = len(obs)

        camera_names = obs_config.camera_configs.keys()

        # if not (num_steps == len(listdir(l_sh_rgb_f)) == len(
        #         listdir(l_sh_depth_f)) == len(listdir(r_sh_rgb_f)) == len(
        #         listdir(r_sh_depth_f)) == len(listdir(oh_rgb_f)) == len(
        #         listdir(oh_depth_f)) == len(listdir(wrist_rgb_f)) == len(
        #         listdir(wrist_depth_f)) == len(listdir(front_rgb_f)) == len(
        #         listdir(front_depth_f))):
        #     print('num_steps: %d' % num_steps)
        #     print('l_sh_rgb_f: %d' % len(listdir(l_sh_rgb_f)))
        #     print('l_sh_depth_f: %d' % len(listdir(l_sh_depth_f)))
        #     print('r_sh_rgb_f: %d' % len(listdir(r_sh_rgb_f)))
        #     print('r_sh_depth_f: %d' % len(listdir(r_sh_depth_f)))
        #     print('oh_rgb_f: %d' % len(listdir(oh_rgb_f)))
        #     print('oh_depth_f: %d' % len(listdir(oh_depth_f)))
        #     print('wrist_rgb_f: %d' % len(listdir(wrist_rgb_f)))
        #     print('wrist_depth_f: %d' % len(listdir(wrist_depth_f)))
        #     print('front_rgb_f: %d' % len(listdir(front_rgb_f)))
        #     print('front_depth_f: %d' % len(listdir(front_depth_f)))
        #     print('Broken dataset assumption. continuing anyway :/')

        data_types = ["rgb", "depth", "mask"]
        full_camera_names = map(lambda x: ('_'.join(x), x[-1]), product(camera_names, data_types))
        
        for camera_name, _ in full_camera_names:
            data_path = os.path.join(example_path, camera_name)
            if num_steps != len(os.listdir(data_path)):
                print(f"not sufficent data points {data_path} expected {num_steps} was {len(os.listdir(data_path))}")
                #raise RuntimeError('Broken dataset assumption')

        for i in range(num_steps):
            # descriptions
            obs[i].misc['descriptions'] = descriptions
           
            for camera_name, camera_config in obs_config.camera_configs.items(): 
                # if  not obs_config.nerf_multi_view:               
                if i>0:
                    obs[i-1].misc[f'{camera_name}_next_camera_extrinsics'] = obs[i].misc[f'{camera_name}_camera_extrinsics']
                    obs[i-1].misc[f'{camera_name}_next_camera_intrinsics'] = obs[i].misc[f'{camera_name}_camera_intrinsics']
                if i==num_steps-1:
                    obs[i].misc[f'{camera_name}_next_camera_extrinsics'] = obs[i].misc[f'{camera_name}_camera_extrinsics']
                    obs[i].misc[f'{camera_name}_next_camera_intrinsics'] = obs[i].misc[f'{camera_name}_camera_intrinsics']    
                                
                if camera_config.rgb:
                    data_path = os.path.join(example_path, f"{camera_name}_rgb")
                    image_name = f"rgb_{i:04d}.png"
                    image_path = os.path.join(data_path, image_name)
                    image = np.array(_resize_if_needed(Image.open(image_path), camera_config.image_size))
                    obs[i].perception_data[f"{camera_name}_rgb"] = image
                    if i>0:
                        obs[i-1].perception_data[f"{camera_name}_next_rgb"] = image
                    if i==num_steps-1:
                        obs[i].perception_data[f"{camera_name}_next_rgb"] = image

                if camera_config.depth or camera_config.point_cloud:
                    data_path = os.path.join(example_path, f"{camera_name}_depth")
                    image_name = f"depth_{i:04d}.png"
                    image_path = os.path.join(data_path, image_name)
                    image = image_to_float_array( _resize_if_needed(Image.open(image_path), camera_config.image_size),DEPTH_SCALE)
                    if camera_config.depth_in_meters:
                        near = obs[i].misc[f'{camera_name}_camera_near']
                        far = obs[i].misc[f'{camera_name}_camera_far']
                        depth_image_m = near + image * (far - near)
                        obs[i].perception_data[f"{camera_name}_depth"] = camera_config.depth_noise.apply(depth_image_m)
                        if i>0:
                            obs[i-1].perception_data[f"{camera_name}_next_depth"] = camera_config.depth_noise.apply(depth_image_m)
                        if i==num_steps-1:
                            obs[i].perception_data[f"{camera_name}_next_depth"] = camera_config.depth_noise.apply(depth_image_m)
                    else:                    
                        obs[i].perception_data[f"{camera_name}_depth"] = camera_config.depth_noise.apply(image)
                        if i>0:
                            obs[i-1].perception_data[f"{camera_name}_next_depth"] = camera_config.depth_noise.apply(image)
                        if i==num_steps-1:
                            obs[i].perception_data[f"{camera_name}_next_depth"] = camera_config.depth_noise.apply(image)

                    near = obs[i].misc[f'{camera_name}_camera_near']
                    far = obs[i].misc[f'{camera_name}_camera_far']
                    depth_image_m = near + image * (far - near)

                    obs[i].perception_data[f"{camera_name}_point_cloud"] = VisionSensor.pointcloud_from_depth_and_camera_params(
                    depth_image_m,
                    obs[i].misc[f'{camera_name}_camera_extrinsics'],
                    obs[i].misc[f'{camera_name}_camera_intrinsics'])

                if camera_config.mask:
                    data_path = os.path.join(example_path, f"{camera_name}_mask")
                    image_name = f"mask_{i:04d}.png"
                    image_path = os.path.join(data_path, image_name)
                    image = rgb_handles_to_mask_mani(np.array(_resize_if_needed(Image.open(image_path), camera_config.image_size)))
                    obs[i].perception_data[f"{camera_name}_mask"] = image
                    if i>0:
                        obs[i-1].perception_data[f"{camera_name}_next_mask"] = image
                    if i==num_steps-1:
                        obs[i].perception_data[f"{camera_name}_next_mask"] = image

            if obs_config.nerf_multi_view:
                obs[i].nerf_multi_view = join(nerf_multi_view_f, str(i))

            # Remove low dim info if necessary
            if not obs_config.joint_velocities:
                obs[i].joint_velocities = None
            if not obs_config.joint_positions:
                obs[i].joint_positions = None
            if not obs_config.joint_forces:
                obs[i].joint_forces = None
            if not obs_config.gripper_open:
                obs[i].gripper_open = None
            if not obs_config.gripper_pose:
                obs[i].gripper_pose = None
            if not obs_config.gripper_joint_positions:
                obs[i].gripper_joint_positions = None
            if not obs_config.gripper_touch_forces:
                obs[i].gripper_touch_forces = None
            if not obs_config.task_low_dim_state:
                obs[i].task_low_dim_state = None

        if not image_paths:
            for i in range(num_steps):
                
                # for nerf, need to load multi-view images
                if obs_config.nerf_multi_view:
                    nerf_img_dir = join(obs[i].nerf_multi_view, 'images')
                    nerf_depth_dir = join(obs[i].nerf_multi_view, 'depths')
                    nerf_camera_dir = join(obs[i].nerf_multi_view, 'poses')
                    nerf_mask_dir = join(obs[i].nerf_multi_view,'masks')
                    if not exists(nerf_img_dir):
                        obs[i].nerf_multi_view_rgb = None
                        obs[i].nerf_multi_view_depth = None
                        obs[i].nerf_multi_view_camera = None
                        obs[i].nerf_multi_view_mask = None
                        # print(colored(f'nerf multi-view images not found in {nerf_img_dir}', 'red'))
                    elif not exists(nerf_camera_dir):
                        obs[i].nerf_multi_view_rgb = None
                        obs[i].nerf_multi_view_depth = None
                        obs[i].nerf_multi_view_camera = None
                        obs[i].nerf_multi_view_mask = None
                        # print(colored(f'nerf multi-view cameras not found {nerf_camera_dir}', 'red'))
                    else:
                        all_img_files = natsorted(listdir(nerf_img_dir))
                        all_depth_files = natsorted(listdir(nerf_depth_dir))
                        all_camera_files = natsorted(listdir(nerf_camera_dir))
                        all_mask_files = natsorted(listdir(nerf_mask_dir))

                        all_img_files = [join(nerf_img_dir, f) for f in all_img_files]
                        all_depth_files = [join(nerf_depth_dir, f) for f in all_depth_files]
                        all_camera_files = [join(nerf_camera_dir, f) for f in all_camera_files]
                        all_mask_files = [join(nerf_mask_dir, f) for f in all_mask_files]

                        # here we do not load the images, but load the paths. during training, we will load the images
                        obs[i].nerf_multi_view_rgb = np.array(all_img_files)
                        obs[i].nerf_multi_view_depth = np.array(all_depth_files)
                        obs[i].nerf_multi_view_camera = np.array(all_camera_files)
                        obs[i].nerf_multi_view_mask = np.array(all_mask_files)
                    

        #         if obs_config.left_shoulder_camera.rgb:
        #             obs[i].left_shoulder_rgb = np.array(_resize_if_needed(Image.open(obs[i].left_shoulder_rgb),obs_config.left_shoulder_camera.image_size))
        #         if obs_config.right_shoulder_camera.rgb:
        #             obs[i].right_shoulder_rgb = np.array(_resize_if_needed(Image.open(obs[i].right_shoulder_rgb),obs_config.right_shoulder_camera.image_size))
        #         if obs_config.overhead_camera.rgb:
        #             obs[i].overhead_rgb = np.array(_resize_if_needed(Image.open(obs[i].overhead_rgb),obs_config.overhead_camera.image_size))
        #         if obs_config.wrist_camera.rgb:
        #             obs[i].wrist_rgb = np.array(_resize_if_needed(Image.open(obs[i].wrist_rgb),obs_config.wrist_camera.image_size))
        #         if obs_config.front_camera.rgb: # in
        #             obs[i].front_rgb = np.array(_resize_if_needed(Image.open(obs[i].front_rgb), obs_config.front_camera.image_size))

        #         if obs_config.left_shoulder_camera.depth or obs_config.left_shoulder_camera.point_cloud:
        #             l_sh_depth = image_to_float_array(_resize_if_needed(Image.open(obs[i].left_shoulder_depth),obs_config.left_shoulder_camera.image_size),DEPTH_SCALE)
        #             near = obs[i].misc['left_shoulder_camera_near']
        #             far = obs[i].misc['left_shoulder_camera_far']
        #             l_sh_depth_m = near + l_sh_depth * (far - near)
        #             if obs_config.left_shoulder_camera.depth:
        #                 d = l_sh_depth_m if obs_config.left_shoulder_camera.depth_in_meters else l_sh_depth
        #                 obs[i].left_shoulder_depth = obs_config.left_shoulder_camera.depth_noise.apply(d)
        #             else:
        #                 obs[i].left_shoulder_depth = None

        #         if obs_config.right_shoulder_camera.depth or obs_config.right_shoulder_camera.point_cloud:
        #             r_sh_depth = image_to_float_array(_resize_if_needed(Image.open(obs[i].right_shoulder_depth),obs_config.right_shoulder_camera.image_size),DEPTH_SCALE)
        #             near = obs[i].misc['right_shoulder_camera_near']
        #             far = obs[i].misc['right_shoulder_camera_far']
        #             r_sh_depth_m = near + r_sh_depth * (far - near)
        #             if obs_config.right_shoulder_camera.depth:
        #                 d = r_sh_depth_m if obs_config.right_shoulder_camera.depth_in_meters else r_sh_depth
        #                 obs[i].right_shoulder_depth = obs_config.right_shoulder_camera.depth_noise.apply(d)
        #             else:
        #                 obs[i].right_shoulder_depth = None

        #         if obs_config.overhead_camera.depth or obs_config.overhead_camera.point_cloud:
        #             oh_depth = image_to_float_array(
        #                 _resize_if_needed(
        #                     Image.open(obs[i].overhead_depth),
        #                     obs_config.overhead_camera.image_size),
        #                 DEPTH_SCALE)
        #             near = obs[i].misc['overhead_camera_near']
        #             far = obs[i].misc['overhead_camera_far']
        #             oh_depth_m = near + oh_depth * (far - near)
        #             if obs_config.overhead_camera.depth:
        #                 d = oh_depth_m if obs_config.overhead_camera.depth_in_meters else oh_depth
        #                 obs[i].overhead_depth = obs_config.overhead_camera.depth_noise.apply(d)
        #             else:
        #                 obs[i].overhead_depth = None

        #         if obs_config.wrist_camera.depth or obs_config.wrist_camera.point_cloud:
        #             wrist_depth = image_to_float_array(
        #                 _resize_if_needed(
        #                     Image.open(obs[i].wrist_depth),
        #                     obs_config.wrist_camera.image_size),
        #                 DEPTH_SCALE)
        #             near = obs[i].misc['wrist_camera_near']
        #             far = obs[i].misc['wrist_camera_far']
        #             wrist_depth_m = near + wrist_depth * (far - near)
        #             if obs_config.wrist_camera.depth:
        #                 d = wrist_depth_m if obs_config.wrist_camera.depth_in_meters else wrist_depth
        #                 obs[i].wrist_depth = obs_config.wrist_camera.depth_noise.apply(d)
        #             else:
        #                 obs[i].wrist_depth = None

        #         if obs_config.front_camera.depth or obs_config.front_camera.point_cloud:
        #             front_depth = image_to_float_array(
        #                 _resize_if_needed(
        #                     Image.open(obs[i].front_depth),
        #                     obs_config.front_camera.image_size),
        #                 DEPTH_SCALE)
        #             near = obs[i].misc['front_camera_near'] # 0
        #             far = obs[i].misc['front_camera_far'] # 4.5
        #             front_depth_m = near + front_depth * (far - near)
        #             if obs_config.front_camera.depth:
        #                 d = front_depth_m if obs_config.front_camera.depth_in_meters else front_depth
        #                 obs[i].front_depth = obs_config.front_camera.depth_noise.apply(d)
        #             else:
        #                 obs[i].front_depth = None
                
                
        #         if obs_config.left_shoulder_camera.point_cloud:
        #             obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        #                 l_sh_depth_m,
        #                 obs[i].misc['left_shoulder_camera_extrinsics'],
        #                 obs[i].misc['left_shoulder_camera_intrinsics'])

        #         if obs_config.right_shoulder_camera.point_cloud:
        #             obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        #                 r_sh_depth_m,
        #                 obs[i].misc['right_shoulder_camera_extrinsics'],
        #                 obs[i].misc['right_shoulder_camera_intrinsics'])


        #         if obs_config.overhead_camera.point_cloud:
        #             obs[i].overhead_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        #                 oh_depth_m,
        #                 obs[i].misc['overhead_camera_extrinsics'],
        #                 obs[i].misc['overhead_camera_intrinsics'])

        #         if obs_config.wrist_camera.point_cloud:
        #             obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        #                 wrist_depth_m,
        #                 obs[i].misc['wrist_camera_extrinsics'],
        #                 obs[i].misc['wrist_camera_intrinsics'])
        #         if obs_config.front_camera.point_cloud:
        #             obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(
        #                 front_depth_m,
        #                 obs[i].misc['front_camera_extrinsics'],
        #                 obs[i].misc['front_camera_intrinsics'])


        #         # Masks are stored as coded RGB images.
        #         # Here we transform them into 1 channel handles.
        #         if obs_config.left_shoulder_camera.mask:
        #             obs[i].left_shoulder_mask = rgb_handles_to_mask(
        #                 np.array(_resize_if_needed(Image.open(
        #                     obs[i].left_shoulder_mask),
        #                     obs_config.left_shoulder_camera.image_size)))
        #         if obs_config.right_shoulder_camera.mask:
        #             obs[i].right_shoulder_mask = rgb_handles_to_mask(
        #                 np.array(_resize_if_needed(Image.open(
        #                     obs[i].right_shoulder_mask),
        #                     obs_config.right_shoulder_camera.image_size)))
        #         if obs_config.overhead_camera.mask:
        #             obs[i].overhead_mask = rgb_handles_to_mask(
        #                 np.array(_resize_if_needed(Image.open(
        #                     obs[i].overhead_mask),
        #                     obs_config.overhead_camera.image_size)))
        #         if obs_config.wrist_camera.mask:
        #             obs[i].wrist_mask = rgb_handles_to_mask(np.array(
        #                 _resize_if_needed(Image.open(
        #                     obs[i].wrist_mask),
        #                     obs_config.wrist_camera.image_size)))
        #         if obs_config.front_camera.mask:
        #             obs[i].front_mask = rgb_handles_to_mask(np.array(
        #                 _resize_if_needed(Image.open(
        #                     obs[i].front_mask),
        #                     obs_config.front_camera.image_size)))
        demos.append(obs)
    return demos


def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image

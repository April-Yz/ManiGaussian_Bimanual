import os
import torch

import torch.nn.functional as F
import torchvision.transforms.functional as func


from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from PIL import Image

def image_transform(image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image_transformed


def grounding_dino_prompt(image, text):
    
    image_tensor = image_transform(Image.fromarray(image))
    model_root = 'path to Grounded-Segment-Anything/'
    model = load_model(os.path.join(model_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"), \
                       os.path.join(model_root, "weights/groundingdino_swint_ogc.pth"))
    
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    h, w, _ = image.shape
    print("boxes device", boxes.device)
    boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    print(xyxy)
    return xyxy

def porject_to_2d(viewpoint_camera, points3D):
    full_matrix = viewpoint_camera.full_proj_transform  # w2c @ K 
    # project to image plane
    points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

## assume obtain 2d convariance matrx: N, 2, 2
def compute_ratios(conv_2d, points_xy, indices_mask, sam_mask, h, w):
    means = points_xy[indices_mask]
    eigvals, eigvecs = torch.linalg.eigh(conv_2d)
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(eigvecs, dim=1, 
                        index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,2)) 
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    vertex1 = means + 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex2 = means - 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex1 = torch.clip(vertex1, torch.tensor([0, 0]).to(points_xy.device), torch.tensor([w-1, h-1]).to(points_xy.device))
    vertex2 = torch.clip(vertex2, torch.tensor([0, 0]).to(points_xy.device), torch.tensor([w-1, h-1]).to(points_xy.device))
    vertex1_xy = torch.round(vertex1).long()
    vertex2_xy = torch.round(vertex2).long()
    vertex1_label = sam_mask[vertex1_xy[:, 1], vertex1_xy[:, 0]]
    vertex2_label = sam_mask[vertex2_xy[:, 1], vertex2_xy[:, 0]]
    index = torch.nonzero(vertex1_label ^ vertex2_label, as_tuple=True)[0]
    # special_index = (vertex1_label == 0) & (vertex2_label == 0)
    # index = torch.cat((index, special_index), dim=0)
    selected_vertex1_xy = vertex1_xy[index]
    selected_vertex2_xy = vertex2_xy[index]
    sign_direction = vertex1_label[index] - vertex2_label[index]
    direction_vector = max_eigvec[index] * sign_direction.unsqueeze(-1)

    ratios = []
    update_index = []
    for k in range(len(index)):
        x1, y1 = selected_vertex1_xy[k]
        x2, y2 = selected_vertex2_xy[k]
        # print(k, x1, x2)
        if x1 < x2:
            x_point = torch.arange(x1, x2+1).to(points_xy.device)
            y_point = y1 + (y2- y1) / (x2- x1) * (x_point - x1)
        elif x1 < x2:
            x_point = torch.arange(x2, x1+1).to(points_xy.device)
            y_point = y1 + (y2- y1) / (x2- x1) * (x_point - x1)
        else:
            if y1 < y2:
                y_point = torch.arange(y1, y2+1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1
            else:
                y_point = torch.arange(y2, y1+1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1
        
        x_point = torch.round(torch.clip(x_point, 0, w-1)).long()
        y_point = torch.round(torch.clip(y_point, 0, h-1)).long()
        # print(x_point.max(), y_point.max())
        in_mask = sam_mask[y_point, x_point]
        ratios.append(sum(in_mask) / len(in_mask))

    ratios = torch.tensor(ratios)
    index_in_all = indices_mask[index]

    return index_in_all, ratios, direction_vector

import math

def compute_conv3d(conv3d):
    complete_conv3d = torch.zeros((conv3d.shape[0], 3, 3))
    complete_conv3d[:, 0, 0] = conv3d[:, 0]
    complete_conv3d[:, 1, 0] = conv3d[:, 1]
    complete_conv3d[:, 0, 1] = conv3d[:, 1]
    complete_conv3d[:, 2, 0] = conv3d[:, 2]
    complete_conv3d[:, 0, 2] = conv3d[:, 2]
    complete_conv3d[:, 1, 1] = conv3d[:, 3]
    complete_conv3d[:, 2, 1] = conv3d[:, 4]
    complete_conv3d[:, 1, 2] = conv3d[:, 4]
    complete_conv3d[:, 2, 2] = conv3d[:, 5]

    return complete_conv3d

def conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device):
    # 3d convariance matrix
    conv3d = gaussians.get_covariance(scaling_modifier=1)[indices_mask]
    conv3d_matrix = compute_conv3d(conv3d).to(device)

    w2c = viewpoint_camera.world_view_transform
    mask_xyz = gaussians.get_xyz[indices_mask]
    pad_mask_xyz = F.pad(input=mask_xyz, pad=(0, 1), mode='constant', value=1)
    t = pad_mask_xyz @ w2c[:, :3]   # N, 3
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_x = width / (2.0 * tanfovx)
    focal_y = height / (2.0 * tanfovy)
    lim_xy = torch.tensor([1.3 * tanfovx, 1.3 * tanfovy]).to(device)
    t[:, :2] = torch.clip(t[:, :2] / t[:, 2, None], -1. * lim_xy, lim_xy) * t[:, 2, None]
    J_matrix = torch.zeros((mask_xyz.shape[0], 3, 3)).to(device)
    J_matrix[:, 0, 0] = focal_x / t[:, 2]
    J_matrix[:, 0, 2] = -1 * (focal_x * t[:, 0]) / (t[:, 2] * t[:, 2])
    J_matrix[:, 1, 1] = focal_y / t[:, 2]
    J_matrix[:, 1, 2] = -1 * (focal_y * t[:, 1]) / (t[:, 2] * t[:, 2])
    W_matrix = w2c[:3, :3]  # 3,3
    T_matrix = (W_matrix @ J_matrix.permute(1, 2, 0)).permute(2, 0, 1) # N,3,3

    conv2d_matrix = torch.bmm(T_matrix.permute(0, 2, 1), torch.bmm(conv3d_matrix, T_matrix))[:, :2, :2]

    return conv2d_matrix


def update(gaussians, view, selected_index, ratios, dir_vector):

    ratios = ratios.unsqueeze(-1).to("cuda")
    selected_xyz = gaussians.get_xyz[selected_index]
    selected_scaling = gaussians.get_scaling[selected_index]
    conv3d = gaussians.get_covariance(scaling_modifier=1)[selected_index]
    conv3d_matrix = compute_conv3d(conv3d).to("cuda")

    eigvals, eigvecs = torch.linalg.eigh(conv3d_matrix)
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(eigvecs, dim=1, 
                        index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,3)) # (N, 1, 3)
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    new_scaling = selected_scaling * ratios * 0.8
    # new_scaling = selected_scaling
    
    max_eigvec_2d = porject_to_2d(view, max_eigvec)
    sign_direction = torch.sum(max_eigvec_2d * dir_vector, dim=1).unsqueeze(-1)
    sign_direction = torch.where(sign_direction > 0, 1, -1)
    new_xyz = selected_xyz + 0.5 * (1 - ratios) * long_axis.unsqueeze(1) * max_eigvec * sign_direction

    gaussians._xyz = gaussians._xyz.detach().clone().requires_grad_(False)
    gaussians._scaling =  gaussians._scaling.detach().clone().requires_grad_(False)
    gaussians._xyz[selected_index] = new_xyz
    gaussians._scaling[selected_index] = gaussians.scaling_inverse_activation(new_scaling)

    return gaussians



import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
from seg_utils import grounding_dino_prompt

SAM_ARCH = 'vit_h'
SAM_CKPT_PATH = 'path to sam_vit_h_4b8939.pth'

model_type = SAM_ARCH
sam = sam_model_registry[model_type](checkpoint=SAM_CKPT_PATH).to('cuda')
predictor = SamPredictor(sam)

# text guided 
def text_prompting_ID(image, text, id):
    input_boxes = grounding_dino_prompt(image, text)

    boxes = torch.tensor(input_boxes)[0:1].cuda()

    predictor.set_image(image) 

    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    masks,  _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    masks = masks[0].cpu().numpy()
    return_mask = (masks[id, :, :, None]*255).astype(np.uint8)
    return return_mask / 255

def text_prompting(image, text, ID):
    input_boxes = grounding_dino_prompt(image, text)
    boxes = torch.tensor(input_boxes)[ID:1].cuda()
    predictor.set_image(image)  
    print("boxes",boxes.shape)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
    masks,  _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    return masks 


def self_prompt(point_prompts, sam_feature, id):
    input_point = point_prompts.detach().cpu().numpy()
    
    input_label = np.ones(len(input_point))

    predictor.features = sam_feature
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    return_mask = (masks[id, :, :, None]*255).astype(np.uint8)

    return return_mask / 255


def main():
    # task_name = #'bimanual_sweep_to_dustpan' #'bimanual_straighten_rope' # 'bimanual_pick_plate' # 'bimanual_pick_laptop'
    # task_name = #'coordinated_put_item_in_drawer' #'coordinated_put_bottle_in_fridge' # 'coordinated_push_box' #'coordinated_lift_tray'
    task_name='handover_item_easy' # 'handover_item' #'dual_push_buttons' #'coordinated_take_tray_out_of_oven'
    rgb_id='0000'
    eposide_id='episode0'
    camera_id='overhead_rgb' # 'overhead_mask' #'over_shoulder_left_mask' #'front_mask'
    mask_path = f" path to train_data/{task_name}/all_variations/episodes/{eposide_id}/{camera_id}/rgb_{rgb_id}.png"
    output_name = f"{task_name}_{eposide_id}_{camera_id}_{rgb_id}.png"
    output_dir = f"path to output" 

    image_path = mask_path  
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    text = "object" 
    mask_id = 0 

    mask_from_text = text_prompting_ID(image, text, mask_id)

    output_path_text = os.path.join(output_dir, f"mask_id_{mask_id}.png")  
    cv2.imwrite(output_path_text, (mask_from_text * 255).astype(np.uint8))
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Mask from text prompting saved to {output_path_text} at {current_time}")

    mask_id = 0
    mask_list = text_prompting(image, text, mask_id)
    value = 0  # 0 for background
    import matplotlib.pyplot as plt
    for idx, mask in enumerate(mask_list):
        mask_img = torch.zeros(mask.shape[-2:])
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_img.numpy(), cmap='gray') 
        plt.axis('off')  
        plt.savefig(os.path.join(output_dir, f'mask_{value + idx + 1}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
        plt.close() 
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   
    print(f"all images saved at {current_time}")


if __name__ == "__main__":
    main()
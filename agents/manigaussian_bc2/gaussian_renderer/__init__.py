#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import numpy as np

# def mask_to_rgb(mask):
#     """Convert a single channel mask to an RGB image."""
#     # mask is a single channel encoded as R + G * 256 + B * 256 * 256
#     r = mask % 256
#     g = (mask // 256) % 256
#     b = (mask // (256 * 256)) % 256
#     print("mask",mask)
#     print("mask.shape",mask.shape) # [655366, 3]
#     print(r,g,b)
#     # Stack R, G, B channels to get RGB image
#     rgb_image = np.stack([r, g, b], axis=-1)
#     return rgb_image

def render(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None):
    """
    Render the scene.     
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data['novel_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['novel_view']['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['novel_view']['height'][idx]),
        image_width=int(data['novel_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['novel_view']['world_view_transform'][idx],  
        projmatrix=data['novel_view']['full_proj_transform'][idx],   
        sh_degree=3 if features_color is None else 1,                
        campos=data['novel_view']['camera_center'][idx],             
        prefiltered=False,                                           
        debug=False,
        include_feature=(features_language is not None),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if features_color is not None:  # default: None, use SHs
        shs = features_color
        # print("shs.shape",shs.shape) # [65536, 4, 3]
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb
        # print("pts_rgb.shape",pts_rgb.shape)

    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }
def render1(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None):
    """
    Render the scene.     
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data['mask_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['mask_view']['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['mask_view']['height'][idx]),
        image_width=int(data['mask_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['mask_view']['world_view_transform'][idx], 
        projmatrix=data['mask_view']['full_proj_transform'][idx],  
        sh_degree=3 if features_color is None else 1,               
        campos=data['mask_view']['camera_center'][idx],             
        prefiltered=False,                                           
        debug=False,
        include_feature=(features_language is not None),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if features_color is not None:  # default: None, use SHs
        shs = features_color
        # print("shs.shape",shs.shape) # [65536, 4, 3]
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb
        # print("pts_rgb.shape",pts_rgb.shape)

    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # language_feature_precomp = language_feature_precomp/2 + 0.5
    print("language_feature_precomp = ",language_feature_precomp.shape,language_feature_precomp)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }



# new from mask  precomputed_mask = None
def render_mask(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None,precomputed_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    new add mask 
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    # screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.long, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data['mask_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['mask_view']['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['mask_view']['height'][idx]),
        image_width=int(data['mask_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0, 
        viewmatrix=data['mask_view']['world_view_transform'][idx], 
        projmatrix=data['mask_view']['full_proj_transform'][idx],  
        sh_degree=3 if features_color is None else 1,              
        campos=data['mask_view']['camera_center'][idx],            
        prefiltered=False,                                         
        debug=False,                                                 
        include_feature=(features_language is not None),              
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ############################for mask################################
    mask = precomputed_mask # [65536, 3]
    # mask = pc.get_mask if precomputed_mask is None else precomputed_mask

    # target:[h,w,3]
    mask = torch.tensor(mask).cuda()
    # if len(mask.shape) == 2:  
    mask = mask.reshape(256,256,3)
        # mask =mask.unsqueeze(1)
    # print("in gen mask.shape",mask.shape)
    # print("1 mask.shape",mask.shape)mask = mask.reshape(256,256,3)print("2 mask.shape",mask.shape)   mask = torch.tensor(mask).cuda()
    # ############################for mask################################
    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if mask is not None:
        colors_precomp = mask  # / 2 + 0.5

    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # rendered_image, language_feature_image, radii = rasterizer(
    rendered_mask, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,                    
        means2D=screenspace_points,         
        shs=None, # shs,
        colors_precomp=colors_precomp,   
        language_feature_precomp=language_feature_precomp,
        opacities=opacity, 
        scales=scales, # * rescale_factor,
        rotations=rotations,
        cov3D_precomp=None,
        )
    # print("3 rendered_mask.shape",rendered_mask.shape) # [3, 128, 128]
    return {
        "mask": rendered_mask,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }

def render_mask_gen(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None,precomputed_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    new add mask 

    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    # screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.long, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data['novel_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['novel_view']['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['novel_view']['height'][idx]),
        image_width=int(data['novel_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0, 
        viewmatrix=data['novel_view']['world_view_transform'][idx],  
        projmatrix=data['novel_view']['full_proj_transform'][idx],   
        sh_degree=3 if features_color is None else 1,             
        campos=data['novel_view']['camera_center'][idx],          
        prefiltered=False,                                       
        debug=False,                                                
        include_feature=(features_language is not None),           
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # ############################for mask################################

    mask = precomputed_mask
    # mask = pc.get_mask if precomputed_mask is None else precomputed_mask

    # if len(mask.shape) == 1 or mask.shape[-1] == 1: # [W, H, 1] -> [W, H, 3] repeat([1,3])
        # mask = mask.squeeze().unsqueeze(-1).repeat([1,3]).cuda()
    # if len(mask.shape) == 2 or (len(mask.shape) == 3 and mask.shape[-1] == 1):
    #     mask = mask.squeeze() 
    #     mask = mask_to_rgb(mask)  

    mask = mask.reshape(256,256,3)
    # mask =mask.unsqueeze(1)
    # print("in gen mask.shape",mask.shape)
    # mask = mask.reshape(128,128,3)
    # print("in gen mask.shape",mask.shape)

    # mask = torch.tensor(mask).cuda()
    mask = mask.clone().detach().cuda()
    # ############################for mask################################

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if mask is not None:
        colors_precomp = mask
    elif features_color is not None:  
        shs = features_color
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb
  
    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_image, language_feature_image, radii = rasterizer(
    rendered_mask, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,   #?
        language_feature_precomp=language_feature_precomp,
        opacities=opacity, 
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "mask": rendered_mask,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }

def render_rgb(data, idx, pts_xyz, rotations, scales, opacity, bg_color, pts_rgb=None, features_color=None, features_language=None):
    """
    Render the scene.     
    Background tensor (bg_color) must be on GPU!
    features_language: the newly-added feature (Nxd)
    """
    device = pts_xyz.device
    bg_color = torch.tensor(bg_color, dtype=torch.float32, device=device)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pts_xyz, dtype=torch.float32, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(data['mask_view']['FovX'][idx] * 0.5)
    tanfovy = math.tan(data['mask_view']['FovY'][idx] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['mask_view']['height'][idx]),
        image_width=int(data['mask_view']['width'][idx]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=data['mask_view']['world_view_transform'][idx],  
        projmatrix=data['mask_view']['full_proj_transform'][idx],  
        sh_degree=3 if features_color is None else 1,                
        campos=data['mask_view']['camera_center'][idx],             
        prefiltered=False,                                         
        debug=False,
        include_feature=(features_language is not None),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed colors are provided, use them. Otherwise, SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if features_color is not None:  # default: None, use SHs
        shs = features_color
        # print("shs.shape",shs.shape) # [65536, 4, 3]
    else:
        assert pts_rgb is not None
        colors_precomp = pts_rgb
        # print("pts_rgb.shape",pts_rgb.shape)
  
    if features_language is not None:
        MIN_DENOMINATOR = 1e-12
        language_feature_precomp = features_language    # [N, 3]
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + MIN_DENOMINATOR)
    else:
        # FIXME: other dimension choices may cause "illegal memory access"
        language_feature_precomp = torch.zeros((opacity.shape[0], 3), dtype=opacity.dtype, device=opacity.device)   

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, language_feature_image, radii = rasterizer(
        means3D=pts_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors_precomp,
        language_feature_precomp=language_feature_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,   # cuda
        "render_embed": language_feature_image,
        "viewspace_points": screenspace_points,
        "radii": radii,
        }
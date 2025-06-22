
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import torch.autograd.profiler as profiler

import os
import os.path as osp
import warnings
from termcolor import colored, cprint

from agents.manigaussian_bc2.utils import PositionalEncoding, visualize_pcd
from agents.manigaussian_bc2.resnetfc import ResnetFC


from typing import List
import numpy as np
import visdom


class GSPointCloudRegresser(nn.Module):
    def __init__(self, cfg, out_channels, bias, scale):
        '''
        for weight initialization
        '''
        super().__init__()
        self.out_channels = out_channels
        self.cfg = cfg
        self.activation = torch.nn.functional.softplus
        self.out = nn.Linear(
            in_features=sum(out_channels),
            out_features=sum(out_channels),
        )
    def forward(self, x):
        return self.out(self.activation(x, beta=100))

# def fc_block(in_f, out_f):
#     return torch.nn.Sequential(
#         torch.nn.Linear(in_f, out_f),
#         torch.nn.ReLU(out_f)
#     )


class GeneralizableGSEmbedNet(nn.Module):
    def __init__(self, cfg, with_gs_render=True):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render

        self.coordinate_bounds = cfg.coordinate_bounds # default: [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        # print(colored(f"[GeneralizableNeRFEmbedNet] coordinate_bounds: {self.coordinate_bounds}", "red"))
    
        self.use_xyz = cfg.use_xyz
        d_in = 3 if self.use_xyz else 1

        self.use_code = cfg.use_code
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z
            self.code = PositionalEncoding.from_conf(cfg["code"], d_in=d_in)
            d_in = self.code.d_out  # 39

        self.d_in = d_in

        self.image_shape = (cfg.image_height, cfg.image_width)
        self.num_objs = 0
        self.num_views_per_obj = 1

        split_dimensions, scale_inits, bias_inits = self._get_splits_and_inits(cfg)

        # backbone
        self.d_latent = d_latent = cfg.d_latent # 128
        self.d_lang = d_lang = cfg.d_lang   # 128
        self.d_out = sum(split_dimensions)
        print(colored(f"self.d_in {self.d_in}", "red"))  # 39
        print(colored(f"self.d_out {self.d_out}", "red"))  # 26-> 29(mask)

        self.encoder = ResnetFC(
                d_in=d_in, # xyz                    # 39
                d_latent=d_latent,  # volumetric representation
                d_lang=d_lang, 
                d_out=self.d_out,                   # 26 + 3
                d_hidden=cfg.mlp.d_hidden,          #  512 
                n_blocks=cfg.mlp.n_blocks,          # 5  
                combine_layer=cfg.mlp.combine_layer,    # 3
                beta=cfg.mlp.beta,  # 0.0
                use_spade=cfg.mlp.use_spade, # False
            )
        
        self.gs_parm_regresser = GSPointCloudRegresser(
            cfg,
            split_dimensions, 
            scale=scale_inits, # scale_inits
            bias=bias_inits,
            )
        self.scaling_activation = torch.exp
        # self.scaling_activation = torch.nn.functional.softplus
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize    # [B, N, 4]
        self.max_sh_degree = cfg.mlp.max_sh_degree
        # self.mask_activation = torch.sigmoid

        # we move xyz, rot
        self.use_dynamic_field = cfg.use_dynamic_field
        self.mask_gen =cfg.mask_gen
        self.field_type = cfg.field_type
        self.use_nerf_picture = cfg.use_nerf_picture
        self.warm_up = cfg.next_mlp.warm_up
        self.use_action = cfg.next_mlp.use_action
        self.use_mask = ((self.field_type =='LF' and self.mask_gen== 'pre') or (not self.use_nerf_picture))
        # # if self.use_mask:
        # W = 20
        # self.semantic_linear = nn.Sequential(fc_block(W, W // 2), nn.Linear(W // 2, 3))
        cprint(f"[GeneralizableGSEmbedNet] Using dynamic field: {self.use_dynamic_field}", "red")
        if self.use_dynamic_field:
            cprint(f"[GeneralizableGSEmbedNet] field_type: {self.field_type}", "red")
            if self.field_type =='bimanual':
                self.use_semantic_feature = (cfg.foundation_model_name == 'diffusion')
                cprint(f"[GeneralizableGSEmbedNet] Using action input: {self.use_action}", "red")
                cprint(f"[GeneralizableGSEmbedNet] Using semantic feature: {self.use_semantic_feature}", "red")
                # if self.leader:
                next_d_in = self.d_out + self.d_in      # (26 + 3(mask) ) + 39 = (65 + 3) = 68      
                next_d_in = next_d_in + 16 if self.use_action else next_d_in  # 73 -> 81 +3(mask) -> 84  action: 8->16 dim # new theta_right=8(16)    
                next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3   # 70->78
                next_d_in = next_d_in if self.use_mask  else next_d_in - 3          
                self.gs_deformation_field = ResnetFC(
                        d_in=next_d_in, # all things despite volumetric representation (26 + 39 + 8 -3 = 70) 
                        d_latent=self.d_latent,
                        d_lang=self.d_lang,
                        d_out=3 + 4,    # xyz, rot
                        d_hidden=cfg.next_mlp.d_hidden, 
                        n_blocks=cfg.next_mlp.n_blocks, 
                        combine_layer=cfg.next_mlp.combine_layer,
                        beta=cfg.next_mlp.beta, use_spade=cfg.next_mlp.use_spade,
                    )

                # -------------------------------for leader - Follower ------------------------------
                # if self.field_type =='LF':
            else:
                self.use_semantic_feature = (cfg.foundation_model_name == 'diffusion')
                cprint(f"[GeneralizableGSEmbedNet] Using action input: {self.use_action}", "red")
                cprint(f"[GeneralizableGSEmbedNet] Using semantic feature: {self.use_semantic_feature}", "red")
                # if self.leader:
                next_d_in = self.d_out + self.d_in      # 26( +3 mask ) + 39 = 65 + 3 = 68   
                next_d_in = next_d_in + 8 if self.use_action else next_d_in  # 73 +3(mask) = 76 action: 8 dim # new theta_right=8(16)
                next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3   # 73-3 -> 70 +3(mask) = 73
                next_d_in = next_d_in if self.use_mask else next_d_in - 3      # mask    # cprint(f"70 next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3 {next_d_in}", "green")
                #         next_mlp out of memory 
                #         d_hidden=cfg.next_mlp.d_hidden, 
                #         n_blocks=cfg.next_mlp.n_blocks, 

                self.gs_deformation_field_leader_smaller = ResnetFC(
                        d_in=next_d_in, # all things despite volumetric representation (26 + 39 + 8 -3 = 70) 
                        d_latent=self.d_latent,
                        d_lang=self.d_lang,
                        d_out=3 + 4,    # xyz, rot
                        d_hidden=cfg.next_mlp_small.d_hidden, 
                        n_blocks=cfg.next_mlp_small.n_blocks, 
                       combine_layer=cfg.next_mlp_small.combine_layer,
                        beta=cfg.next_mlp_small.beta, use_spade=cfg.next_mlp_small.use_spade,
                    )
                # else:
                # follower
                next_d_in = self.d_out + self.d_in # 39+26=65
                next_d_in = next_d_in + 8 + 8 if self.use_action else next_d_in  # action: 8 +8 dim # new theta_right=8(16)
                next_d_in = next_d_in if self.use_semantic_feature else next_d_in - 3
                next_d_in = next_d_in if self.use_mask else next_d_in - 3
                self.gs_deformation_field_follower = ResnetFC(
                        d_in=next_d_in, # all things despite volumetric representation (26 + 39 + 8 -3 = 70) 
                        d_latent=self.d_latent,
                        d_lang=self.d_lang,
                        d_out=3 + 4,    # xyz, rot   #
                        d_hidden=cfg.next_mlp.d_hidden, 
                        n_blocks=cfg.next_mlp.n_blocks, 
                        combine_layer=cfg.next_mlp.combine_layer,
                        beta=cfg.next_mlp.beta, use_spade=cfg.next_mlp.use_spade,
                    )
            # -------------------------------for leader - Follower ------------------------------

    def _get_splits_and_inits(self, cfg):
        '''Gets channel split dimensions and last layer initialization
        Credit: https://github.com/szymanowiczs/splatter-image/blob/main/scene/gaussian_predictor.py
        '''
        split_dimensions = []
        scale_inits = []
        bias_inits = []
        # split_dimensions = split_dimensions + [3, 1, 3, 4, 3, 3] # 17
        split_dimensions = split_dimensions + [3, 1, 3, 4, 3, 3, 3] # 17->20 last 3->mask
        # print("1 split_dimensions: ", split_dimensions, len(split_dimensions))
        scale_inits = scale_inits + [              
            cfg.mlp.xyz_scale,
            cfg.mlp.opacity_scale,
            cfg.mlp.scale_scale,
            1.0,    # rotation
            5.0,    # feature_dc
            1.0,    # feature
            1.0     # mask
            ]
        bias_inits = [                              
            cfg.mlp.xyz_bias, 
            cfg.mlp.opacity_bias,
            np.log(cfg.mlp.scale_bias),
            0.0,
            0.0,
            0.0,
            0.0 # mask
            ]
        if cfg.mlp.max_sh_degree != 0:    # default: 1
            sh_num = (self.cfg.mlp.max_sh_degree + 1) ** 2 - 1   
            sh_num_rgb = sh_num * 3                                    
            split_dimensions.append(sh_num_rgb)                       # [3, 1, 3, 4, 3, 3, 3, 9] 20 + 9 = 29( 26 + 3(mask) ) 
            scale_inits.append(0.0)                                   
            bias_inits.append(0.0)
        self.split_dimensions_with_offset = split_dimensions          
        return split_dimensions, scale_inits, bias_inits  

    @torch.no_grad()
    def world_to_canonical(self, xyz):
        """
        :param xyz (B, N, 3) or (B, 3, N)                                     
        :return (B, N, 3) or (B, 3, N)
        transform world coordinate to canonical coordinate with bounding box [0, 1]
        """
        xyz = xyz.clone()
        bb_min = self.coordinate_bounds[:3] 
        bb_max = self.coordinate_bounds[3:]
        bb_min = torch.tensor(bb_min, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
            else torch.tensor(bb_min, device=xyz.device).unsqueeze(-1).unsqueeze(0)
        bb_max = torch.tensor(bb_max, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
            else torch.tensor(bb_max, device=xyz.device).unsqueeze(-1).unsqueeze(0)
        xyz -= bb_min
        xyz /= (bb_max - bb_min)

        return xyz

    def sample_in_canonical_voxel(self, xyz, voxel_feat):   # USED
        """
        :param xyz (B, 3)
        :param self.voxel_feat: [B, 128, 20, 20, 20] 
        :return (B, Feat)
        """
        xyz_voxel_space = xyz.clone()
        xyz_voxel_space = xyz_voxel_space * 2 - 1.0 # [0,1]->[-1,1]

        # unsqueeze the point cloud to also have 5 dim
        xyz_voxel_space = xyz_voxel_space.unsqueeze(1).unsqueeze(1)   
        point_feature = F.grid_sample(voxel_feat, xyz_voxel_space, align_corners=True, mode='bilinear')
        point_feature = point_feature.squeeze(2).squeeze(2).permute(0, 2, 1)                                                            
        return point_feature

    def forward(self, data):
        """
        SB is batch size               
        N is batch of points         
        NS is number of input views
        Predict gaussian parameter maps
        """
        SB, N, _ = data['xyz'].shape # 1 65536
        NS = self.num_views_per_obj # 1
        # print("SB=",SB,", N=",N,", NS=",NS)

        canon_xyz = self.world_to_canonical(data['xyz'])    
        data['canon_xyz'] = canon_xyz # gt mask
        # volumetric sampling 
        point_latent = self.sample_in_canonical_voxel(canon_xyz, data['dec_fts'])  
        point_latent = point_latent.reshape(-1, self.d_latent)  # (SB * NS * B, latent)  
        if self.use_xyz:    # True
            z_feature = canon_xyz.reshape(-1, 3)  # (SB*B, 3)   
        if self.use_code:    # True
            z_feature = self.code(z_feature)    
        
        latent = torch.cat((point_latent, z_feature), dim=-1) # [N, 128+39] [65536,167]

        # Camera frustum culling stuff, currently disabled
        combine_index = None
        dim_size = None
        # backbone
        latent, _ = self.encoder(
            latent, # [65536,167]   d_latent + d_in
            combine_inner_dims=(self.num_views_per_obj, N),
            combine_index=combine_index,
            dim_size=dim_size,
            language_embed=data['lang'],
            batch_size=SB,
            )   # 26

        latent = latent.reshape(-1, N, self.d_out)  # [1, N, d_out] [1,65536,26] 

        ## regress gaussian parms 
        split_network_outputs = self.gs_parm_regresser(latent) # [1, N, (3, 1, 3, 4, 3, 9)]  [3, 1, 3, 4, 3, 3, 3, 9] 20 + 9 = 29( 26 + 3(mask) )
        split_network_outputs = split_network_outputs.split(self.split_dimensions_with_offset, dim=-1)
        xyz_maps, opacity_maps, scale_maps, rot_maps, features_dc_maps, feature_maps, mask_maps = split_network_outputs[:7] # 6]
        # xyz_maps, opacity_maps, scale_maps, rot_maps, features_dc_maps, feature_maps,  precomputed_mask = split_network_outputs[:7]
        if self.max_sh_degree > 0:
            features_rest_maps = split_network_outputs[7] # [6]

        # spherical function head 
        features_dc_maps = features_dc_maps.unsqueeze(2) #.transpose(2, 1).contiguous().unsqueeze(2) # [B, H*W, 1, 3] [1, 65536, 1, 3]
        features_rest_maps = features_rest_maps.reshape(*features_rest_maps.shape[:2], -1, 3) # [B, H*W, 3, 3] [1, 65536, 3, 3]
        sh_out = torch.cat([features_dc_maps, features_rest_maps], dim=2)  # [B, H*W, 4, 3]   [1, 65536, 1, 3]+ [1, 65536, 3, 3] = [1, 65536, 4, 3]

        scale_maps = self.scaling_activation(scale_maps)    # exp   [1, 65536, 3]
        scale_maps = torch.clamp_max(scale_maps, 0.05) # [1, 65536, 3] 

        data['xyz_maps'] = data['xyz'] + xyz_maps   # [B, N, 3]           [1, 65536, 3]
        data['sh_maps'] = sh_out    # [B, N, 4, 3]                        [1, 65536, 4, 3]        
        data['rot_maps'] = self.rotation_activation(rot_maps, dim=-1)   # [1, 65536, 4]
        data['scale_maps'] = scale_maps                                 # [1, 65536, 3]
        data['opacity_maps'] = self.opacity_activation(opacity_maps)    # [1, 65536, 1]    
        data['feature_maps'] = feature_maps # [B, N, 3]                   [1, 65536, 3]
        data['mask_maps'] = mask_maps #self.mask_activation(mask_maps) #10.24  [B, N, 3]  [1, 65536, 3]

        # Dynamic Modeling: predict next gaussian maps
        # print("self.field_type = ",self.field_type)
        if self.use_dynamic_field: #and data['step'] >= self.warm_up:

            if self.use_mask: 
                if not self.use_semantic_feature :
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        data['mask_maps'].detach().reshape(N, 3),
                        # d_in:
                        z_feature,  
                    ), dim=-1) # no batch dim    
                else:
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        data['mask_maps'].detach().reshape(N, 3),
                        data['feature_maps'].detach().reshape(N, 3), 
                        # d_in:
                        z_feature,  
                    ), dim=-1) # no batch dim      
            #  without mask
            elif self.field_type == 'bimanual' or (self.field_type =='LF' and self.mask_gen != 'pre'):
                if not self.use_semantic_feature :
                    # dyna_input: (d_latent, d_in)
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        # d_in:
                        z_feature,
                    ), dim=-1) # no batch dim
                else:
                    # use feature_maps
                    dyna_input = torch.cat((
                        point_latent,   # [N, 128]
                        data['xyz_maps'].detach().reshape(N, 3), 
                        features_dc_maps.detach().reshape(N, 3),
                        features_rest_maps.detach().reshape(N, 9),
                        data['rot_maps'].detach().reshape(N, 4),
                        data['scale_maps'].detach().reshape(N, 3),
                        data['opacity_maps'].detach().reshape(N, 1),
                        data['feature_maps'].detach().reshape(N, 3),
                        # d_in:
                        z_feature,  
                    ), dim=-1) # no batch dim
          
            else:
                print("error in models_embed.py")

            # voxel embedding, stop gradient (gaussian xyz), (128+39)+3=170
            if self.field_type =='bimanual':
                if self.use_action:
                    dyna_input = torch.cat((dyna_input, data['action'].repeat(N, 1)), dim=-1)   # action detach
                    # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") 

                next_split_network_outputs, _ = self.gs_deformation_field(
                    dyna_input,
                    combine_inner_dims=(self.num_views_per_obj, N),
                    combine_index=combine_index,
                    dim_size=dim_size,
                    language_embed=data['lang'],
                    batch_size=SB,
                    )
                next_xyz_maps, next_rot_maps = next_split_network_outputs.split([3, 4], dim=-1)
                
                data['next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                data['next']['sh_maps'] = data['sh_maps'].detach()
                data['next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                data['next']['scale_maps'] = data['scale_maps'].detach()
                data['next']['opacity_maps'] = data['opacity_maps'].detach()
                data['next']['feature_maps'] = data['feature_maps'].detach()
            else:
                # -------------------------------------------------------------------------------------------------
                if self.mask_gen == 'pre':
                    if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['right_action'].repeat(N, 1)), dim=-1)   # action detach [65536, 206]
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") # [65536, 206]

                        # with torch.no_grad():
                        next_split_network_outputs_leader, _ = self.gs_deformation_field_leader_smaller(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                        )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_leader.split([3, 4], dim=-1)   
                        data['right_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps    # [1, 65536, 3]?
                        data['right_next']['sh_maps'] = data['sh_maps'].detach()
                        data['right_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)  # [1, 65536, 4]
                        data['right_next']['scale_maps'] = data['scale_maps'].detach()
                        data['right_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['right_next']['feature_maps'] = data['feature_maps'].detach()
                        data['right_next']['mask_maps'] = data['mask_maps'].detach()
                
                        # if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['left_action'].repeat(N, 1)), dim=-1)   # action detach
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red")

                        next_split_network_outputs_follower, _ = self.gs_deformation_field_follower(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                            )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_follower.split([3, 4], dim=-1)   
                        data['left_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                        data['left_next']['sh_maps'] = data['sh_maps'].detach()
                        data['left_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                        data['left_next']['scale_maps'] = data['scale_maps'].detach()
                        data['left_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['left_next']['feature_maps'] = data['feature_maps'].detach()
                        data['left_next']['mask_maps'] = data['mask_maps'].detach()
                elif self.mask_gen == 'nonerf':
                    if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['right_action'].repeat(N, 1)), dim=-1)   # action detach [65536, 206]
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") # [65536, 206]

                        # with torch.no_grad():
                        next_split_network_outputs_leader, _ = self.gs_deformation_field_leader_smaller(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                        )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_leader.split([3, 4], dim=-1)   
                        data['right_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps    # [1, 65536, 3]?
                        data['right_next']['sh_maps'] = data['sh_maps'].detach()
                        data['right_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)  # [1, 65536, 4]
                        data['right_next']['scale_maps'] = data['scale_maps'].detach()
                        data['right_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['right_next']['feature_maps'] = data['feature_maps'].detach()
                        data['right_next']['mask_maps'] = data['mask_maps'].detach()
                
                        # if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['left_action'].repeat(N, 1)), dim=-1)   # action detach
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red")

                        next_split_network_outputs_follower, _ = self.gs_deformation_field_follower(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                            )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_follower.split([3, 4], dim=-1)   
                        data['left_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                        data['left_next']['sh_maps'] = data['sh_maps'].detach()
                        data['left_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                        data['left_next']['scale_maps'] = data['scale_maps'].detach()
                        data['left_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['left_next']['feature_maps'] = data['feature_maps'].detach()
                        data['left_next']['mask_maps'] = data['mask_maps'].detach()
                
                elif self.mask_gen == 'gt':
                    if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['right_action'].repeat(N, 1)), dim=-1)   # action detach [65536, 206]
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red") # [65536, 206]

                        # with torch.no_grad():
                        next_split_network_outputs_leader, _ = self.gs_deformation_field_leader_smaller(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                        )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_leader.split([3, 4], dim=-1)   
                        data['right_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps    # [1, 65536, 3]?
                        data['right_next']['sh_maps'] = data['sh_maps'].detach()
                        data['right_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)  # [1, 65536, 4]
                        data['right_next']['scale_maps'] = data['scale_maps'].detach()
                        data['right_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['right_next']['feature_maps'] = data['feature_maps'].detach()
                
                        # if self.use_action:
                        dyna_input = torch.cat((dyna_input, data['left_action'].repeat(N, 1)), dim=-1)   # action detach
                        # cprint(f"dyna_input.shape: {dyna_input.shape}", "red")

                        next_split_network_outputs_follower, _ = self.gs_deformation_field_follower(
                            dyna_input,
                            combine_inner_dims=(self.num_views_per_obj, N),
                            combine_index=combine_index,
                            dim_size=dim_size,
                            language_embed=data['lang'],
                            batch_size=SB,
                            )
                        next_xyz_maps, next_rot_maps = next_split_network_outputs_follower.split([3, 4], dim=-1)   
                        data['left_next']['xyz_maps'] = data['xyz_maps'].detach() + next_xyz_maps
                        data['left_next']['sh_maps'] = data['sh_maps'].detach()
                        data['left_next']['rot_maps'] = self.rotation_activation(data['rot_maps'].detach() + next_rot_maps, dim=-1)
                        data['left_next']['scale_maps'] = data['scale_maps'].detach()
                        data['left_next']['opacity_maps'] = data['opacity_maps'].detach()
                        data['left_next']['feature_maps'] = data['feature_maps'].detach()

                # -------------------------------------------------------------------------------------------------
        return data
    

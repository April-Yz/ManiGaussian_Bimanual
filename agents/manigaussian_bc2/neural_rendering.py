import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import torchvision.transforms as T

from termcolor import colored, cprint
from dotmap import DotMap

import agents.manigaussian_bc2.utils as utils
from agents.manigaussian_bc2.models_embed import GeneralizableGSEmbedNet
from agents.manigaussian_bc2.loss import l1_loss, l2_loss, cosine_loss, ssim
from agents.manigaussian_bc2.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from agents.manigaussian_bc2.gaussian_renderer import render,render_mask, render_mask_gen,render1,render_rgb
from agents.manigaussian_bc2.project_hull import label_point_cloud, points_inside_convex_hull, \
    depth_mask_to_3d, project_3d_to_2d, create_2d_mask_from_convex_hull, merge_arrays, merge_tensors
import visdom
import logging
import einops
import time
import random

# for debugging 
# from PIL import Image
# import cv2 

def PSNR_torch(img1, img2, max_val=1, mask=None):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0).to(img1.device)
    PIXEL_MAX = max_val
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class NeuralRenderer(nn.Module):
    """
    take a voxel, camera pose, and camera intrinsics as input,
    and output a rendered image
    """
    def __init__(self, cfg):
        super(NeuralRenderer, self).__init__()

        self.cfg = cfg
        self.coordinate_bounds = cfg.coordinate_bounds # bounds of voxel grid
        self.W = cfg.image_width
        self.H = cfg.image_height
        self.bg_color = cfg.dataset.bg_color
        self.bg_mask = [-1,-1,-1] #[1,0,0] #[1,0,0]

        self.znear = cfg.dataset.znear
        self.zfar = cfg.dataset.zfar
        self.trans = cfg.dataset.trans # default: [0, 0, 0]
        self.scale = cfg.dataset.scale
        self.num_classes = 3


        self.use_CEloss = 21 # 0 # 21 # 0(L1) #21 # 7 #21 #0/7/1  2: 7:3-ignore0 

        # gs regressor 
        self.gs_model = GeneralizableGSEmbedNet(cfg, with_gs_render=True)
        print(colored("[NeuralRenderer] GeneralizableGSEmbedNet is build", "cyan"))
        print(colored(f"[NeuralRenderer] Use Mask Loss:{self.use_CEloss},1:Lce-softmax 2:Lce", "cyan"))

        self.model_name = cfg.foundation_model_name
        self.d_embed = cfg.d_embed
        self.loss_embed_fn = cfg.loss_embed_fn
        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean') 
        self.criterion_nll = nn.NLLLoss()

        if self.model_name == "diffusion":
            from odise.modeling.meta_arch.ldm import LdmFeatureExtractor
            import torchvision.transforms as T
            self.feature_extractor = LdmFeatureExtractor(
                            encoder_block_indices=(5, 7),
                            unet_block_indices=(2, 5, 8, 11),
                            decoder_block_indices=(2, 5),
                            steps=(0,),
                            captioner=None,
                        )
            self.diffusion_preprocess = T.Resize(512, antialias=True)
            cprint("diffusion feature dims: "+str(self.feature_extractor.feature_dims), "yellow")
        elif self.model_name == "dinov2":
            from agents.manigaussian_bc2.dino_extractor import VitExtractor
            import torchvision.transforms as T
            self.feature_extractor = VitExtractor(
                model_name='dinov2_vitl14',
            )
            self.dino_preprocess = T.Compose([
                T.Resize(224 * 8, antialias=True),  # must be a multiple of 14
            ])
            cprint("dinov2 feature dims: "+str(self.feature_extractor.feature_dims), "yellow")
        else:
            cprint(f"foundation model {self.model_name} is not implemented", "yellow")

        self.lambda_embed = cfg.lambda_embed
        print(colored(f"[NeuralRenderer] foundation model {self.model_name} is build. loss weight: {self.lambda_embed}", "cyan"))

        self.lambda_rgb = 1.0 if cfg.lambda_rgb is None else cfg.lambda_rgb
        print(colored(f"[NeuralRenderer] rgb loss weight: {self.lambda_rgb}", "cyan"))

        self.use_dynamic_field = cfg.use_dynamic_field
        self.field_type = cfg.field_type
        self.mask_gen = cfg.mask_gen
        self.hierarchical = cfg.hierarchical
        self.use_nerf_picture = cfg.use_nerf_picture

    def _embed_loss_fn(self, render_embed, gt_embed):
        """
        render_embed: [bs, h, w, 3]
        gt_embed: [bs, h, w, 3]
        """
        if self.loss_embed_fn == "l2_norm":
            # label normalization
            MIN_DENOMINATOR = 1e-12
            gt_embed = (gt_embed - gt_embed.min()) / (gt_embed.max() - gt_embed.min() + MIN_DENOMINATOR)
            loss_embed = l2_loss(render_embed, gt_embed)
        elif self.loss_embed_fn == "l2":
            loss_embed = l2_loss(render_embed, gt_embed)
        elif self.loss_embed_fn == "cosine":
            loss_embed = cosine_loss(render_embed, gt_embed)
        else:
            cprint(f"loss_embed_fn {self.loss_embed_fn} is not implemented", "yellow")
        return loss_embed

    def L1_loss_bg(self,pred, target): 
        """
        L1 deal one hot
        Args:
            pred: (B, C, H, W) 
            target: (B, H, W) 
        """
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        total_loss = l1_loss(pred, target)    
        return total_loss

    def dice_L1_loss(self,pred, target, smooth=1e-8): 
        """
        Args:
            pred: (B, C, H, W) 
            target: (B, H, W) 
        """
        # pred = pred/2 +0.5
        # celoss = nn.NLLLoss(ignore_index=0)
        # pred1 = torch.log(pred/2 +0.5)
        pred = pred/2 +0.5 # new
        # ce_loss = celoss(pred1, target) 
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        loss_l1 = l1_loss(pred, target)  

        pred = F.softmax(pred, dim=1)
        intersection = torch.sum(pred * target)
        pred_sum = torch.sum(pred)
        target_sum = torch.sum(target)        
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice = torch.clamp(dice, max=1.0)
        dice_loss = -torch.log(dice)
        total_loss = dice_loss + loss_l1
        return total_loss

    def dice_loss_bg(self,pred, target, smooth=1e-8): 
        """
        Args:
            pred: (B, C, H, W) 
            target: (B, H, W)
            smooth: 
        Returns:
            scalar tensor
        """
        celoss = nn.CrossEntropyLoss()
        ce_loss = celoss(pred, target) 
        #    pred1 = torch.log(pred/ 2 + 0.5)
        #    ce_loss = self.criterion_nll (pred1, target)
        pred = F.softmax(pred, dim=1)
        # target -> one-hot
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        intersection = torch.sum(pred * target)
        pred_sum = torch.sum(pred)
        target_sum = torch.sum(target)  
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)  # (B, C)
        dice = torch.clamp(dice, max=1.0)
        dice_loss = -torch.log(dice)
        total_loss = dice_loss + ce_loss
        return total_loss

    def dice_loss_ignorebg1(self,pred, target, smooth=1e-8): # 1.0gpt?
        """
        Dice Loss + bg    CE-log - bg
        Args:
            pred: (B, C, H, W) 
            target: (B, H, W) 
            smooth: 
        Returns:    scalar tensor
        """
        # pred = pred/2 +0.5
        celoss = nn.NLLLoss(ignore_index=0)
        pred1 = torch.log(pred/2 +0.5)
        ce_loss = celoss(pred1, target) 
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        intersection = torch.sum(pred * target)
        pred_sum = torch.sum(pred)
        target_sum = torch.sum(target)        
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice = torch.clamp(dice, max=1.0)
        dice_loss = -torch.log(dice)
        total_loss = dice_loss + ce_loss
        return total_loss

    def dice_loss(self,pred, target, smooth=1e-8): # 1.0gpt?
        """
        Dice Loss ce dice - bg
        Args:
            pred: (B, C, H, W) 
            target: (B, H, W) 
            smooth: 
        Returns:    scalar tensor
        """
        celoss = nn.CrossEntropyLoss(ignore_index=0)
        ce_loss = celoss(pred, target) 
        pred = F.softmax(pred, dim=1)
        
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)

        pred_foreground = pred[:, 1:]  # (B, C-1, H, W)
        target_foreground = target[:, 1:].float()  # (B, C-1, H, W)
        intersection = torch.sum(pred_foreground * target_foreground)
        pred_sum = torch.sum(pred_foreground)
        target_sum = torch.sum(target_foreground)        
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice = torch.clamp(dice, max=1.0)
        
        dice_loss = -torch.log(dice)
        total_loss = dice_loss + ce_loss
        return total_loss

    def focal_loss(self,pred, target, smooth=1e-8,alpha=0.25, gamma=2.0):
        """focal+dice loss"""
        pred = F.softmax(pred, dim=1)
        
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)

        pt = (pred * target).sum(1)  
        focal_loss = -alpha * (1 - pt) ** gamma * torch.log(pt + smooth)
        focal_loss = focal_loss.mean()

        pred_foreground = pred[:, 1:]  # (B, C-1, H, W)
        target_foreground = target[:, 1:].float()  # (B, C-1, H, W)

        intersection = torch.sum(pred_foreground * target_foreground)
        pred_sum = torch.sum(pred_foreground)
        target_sum = torch.sum(target_foreground)        
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        dice = torch.clamp(dice, max=1.0)
        
        dice_loss = -torch.log(dice)
        total_loss = dice_loss + focal_loss
        return total_loss

    def ce_weight_ignore(self, pred, target):
        """CrossEntropyLoss ignore BG + weight"""
        # weights = torch.tensor([0.1, 0.45, 0.45])
        weights = torch.tensor([1.0, 3.0, 3.0])
        device = pred.device
        weights = weights.to(device)
        celoss = nn.CrossEntropyLoss(weight=weights) # , ignore_index=0)
        loss = celoss(pred, target)
        return loss

    def ce_weight_ignore_l1(self, pred, target):
        """CrossEntropyLoss ignore BG + weight + L1"""
        weights = torch.tensor([0.1, 0.45, 0.45])
        device = pred.device
        weights = weights.to(device)
        celoss = nn.CrossEntropyLoss(weight=weights, ignore_index=0)
        ce_loss = celoss(pred, target)
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        # pred = pred/2 +0.5
        loss_l1 = l1_loss(pred, target)  
        total_loss = loss_l1 * 0.5 + ce_loss * 0.5
        # total_loss = loss_l1 * 0.7 + ce_loss * 0.3
        return total_loss

    def ce_weight_l1(self, pred, target):
        """CrossEntropyLoss ignore BG + weight + L1"""
        weights = torch.tensor([1.0, 3.0, 3.0])
        device = pred.device
        weights = weights.to(device)
        celoss = nn.CrossEntropyLoss(weight=weights)
        ce_loss = celoss(pred, target)
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        loss_l1 = l1_loss(pred, target)  
        total_loss = loss_l1 * 0.5 + ce_loss * 0.5
        return total_loss

    def ce_weight_ignore_l1_nosoft(self, pred, target, smooth=1e-8):
        pred = pred / 2 + 0.5 + smooth
        # weights = torch.tensor([0.2, 0.4, 0.4])
        weights = torch.tensor([0.1, 0.45, 0.45])
        device = pred.device
        weights = weights.to(device)
        celoss = nn.NLLLoss(weight=weights, ignore_index=0)
        pred1 = torch.log(pred)
        ce_loss = celoss(pred1, target)

        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        loss_l1 = l1_loss(pred, target)  
        total_loss = loss_l1 * 0.7 + ce_loss * 0.3 
        return total_loss
    
    def ce_weight_l1_nosoft(self, pred, target):
        weights = torch.tensor([0.1, 0.45, 0.45])
        device = pred.device
        weights = weights.to(device)
        celoss = nn.NLLLoss(weight=weights)
        pred = pred/2 + 0.5
        pred1 = torch.log(pred)
        ce_loss = celoss(pred1, target)

        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        loss_l1 = l1_loss(pred, target)  
        total_loss = loss_l1 * 0.5 + ce_loss * 0.5
        return total_loss

    def dice_loss_bg_weight(self,pred, target, smooth=1e-8): 
        weights = torch.tensor([0.2, 0.4, 0.4])
        device = pred.device
        weights = weights.to(device)
        celoss = nn.CrossEntropyLoss(weight=weights) #, ignore_index=0)
        ce_loss = celoss(pred, target) 
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)  # (B, C, H, W)
        intersection = torch.sum(pred * target)
        pred_sum = torch.sum(pred)
        target_sum = torch.sum(target)  
        dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)  # (B, C)
        dice = torch.clamp(dice, max=1.0)
        dice_loss = -dice #-torch.log(dice)
        total_loss = dice_loss *0.2 + ce_loss *0.8
        return total_loss

    def _mask_loss_fn(self, render_mask, gt_mask):
        
        if self.use_CEloss == 1:   # (ceLoss-Softmax)
            render_mask = torch.log(render_mask/ 2 + 0.5)
            loss = self.criterion_nll (render_mask, gt_mask) # (ceLoss-Softmax)
        elif self.use_CEloss == 2:  # (ceLoss-bg)
            celoss = nn.CrossEntropyLoss(ignore_index=-1)
            loss = celoss(render_mask, gt_mask-1)  
        elif self.use_CEloss == 21:
            # 2-1
            # celoss = nn.CrossEntropyLoss(reduction='mean')
            # loss = celoss(render_mask, gt_mask)  
            # 1 ce + dice
            # loss = self.dice_loss_bg(render_mask, gt_mask)
            # 2 ce + dice ignore bg
            # loss = self.dice_loss(render_mask, gt_mask)
            # 3 focal+ dice 
            # loss = self.focal_loss(render_mask, gt_mask)
            # 4 L1 + one hot 100 010 001 
            # loss = self.L1_loss_bg(render_mask, gt_mask)
            # 5 Ldice Lce-soft
            # loss = self.dice_loss_ignorebg1(render_mask, gt_mask)
            # 6 dice+L1 mask
            # loss = self.dice_L1_loss(render_mask, gt_mask)
            # 7 ce_weight_ignore 
            # loss = self.ce_weight_ignore(render_mask, gt_mask) #
            # loss = self.ce_weight_l1(render_mask, gt_mask)
            # 8 ce_weight_ignore_l1 
            # loss = self.ce_weight_ignore_l1(render_mask, gt_mask)
            # 9 ce_weight_ignore_l1_nosoft 0.5:0.5Nan
            # loss = self.ce_weight_ignore_l1_nosoft(render_mask, gt_mask)
            # 10 ce_weight_l1_nosoft     not useful
            # loss = self.ce_weight_l1_nosoft(render_mask, gt_mask)
            # 11 dice_loss_bg_weight
            # loss = self.dice_loss_bg_weight(render_mask, gt_mask)
            # 21 Lce
            # print("pre render_mask.shape= [1,3,128,128]",render_mask.shape,"  gt_mask.shape=[1,128,128]",gt_mask.shape)
            # print("render_mask.shape= [1,3,128,128]",render_mask.shape,"  gt_mask.shape=[1,128,128]",gt_mask.shape)
            loss = self.CrossEntropyLoss(render_mask, gt_mask)

        elif self.use_CEloss == 7:
            celoss = nn.CrossEntropyLoss(ignore_index=0)
            loss = celoss(render_mask, gt_mask)  
        elif self.use_CEloss == 0:
            # render_mask = (render_mask.permute(0, 3, 1, 2)/2 + 0.5) * (self.num_classes-1)
            # render_mask = render_mask/2+0.5
            # gt_mask =gt_mask.permute(0, 3, 1, 2)
            loss = l1_loss(render_mask, gt_mask) # + 0.2 * (1.0 - ssim(render_mask, gt_mask))
            # loss = l1_loss(render_mask, gt_mask) # + 0.5 * l2_loss(render_mask, gt_mask) #0.2 * (1.0 - ssim(render_mask, gt_mask))
            # loss = l2_loss(render_mask, gt_mask) #0.2 * (1.0 - ssim(render_mask, gt_mask))
        elif self.use_CEloss == 4: # like rgb loss
            render_mask = render_mask.permute(0, 3, 1, 2) * (self.num_classes-1)
            gt_mask =gt_mask.permute(0, 3, 1, 2)
            loss = 0.8 * l1_loss(render_mask, gt_mask) + 0.2 * (1.0 - ssim(render_mask, gt_mask))

        return loss

    def _mask_ce_loss_fn(self, render_mask, gt_mask):
        """
        render_embed: [bs, h, w, 3]
        gt_embed: [bs, h, w, 3]
        """
        MIN_DENOMINATOR = 1e-12
        render_mask = (render_mask - render_mask.min()) / (render_mask.max() - render_mask.min() + MIN_DENOMINATOR)
        loss_mask = self.CrossEntropyLoss(render_mask, gt_mask)
        return loss_mask

    def _save_gradient(self, name):
        """
        for debugging language feature rendering
        """
        def hook(grad):
            print(f"name={name}, grad={grad}")
            return grad
        return hook

    def extract_foundation_model_feature(self, gt_rgb, lang_goal):
        """
        we use the last layer of the diffusion feature extractor  
        since we reshape 128x128 img to 512x512, the last layer's feature is just 128x128
        thus, no need to resize the feature map    
        lang_goal: numpy.ndarray, [bs, 1, 1]
        """
        
        if self.model_name == "diffusion":
            """
            we support multiple captions for batched input here
            """
            if lang_goal.shape[0] > 1:
                caption = ['a robot arm ' + cap.item() for cap in lang_goal]
            else:
                caption = "a robot arm " + lang_goal.item()
            batched_input = {'img': self.diffusion_preprocess(gt_rgb.permute(0, 3, 1, 2)), 'caption': caption}
            feature_list, lang_embed = self.feature_extractor(batched_input) # list of visual features, and 77x768 language embedding
            used_feature_idx = -1  
            gt_embed = feature_list[used_feature_idx]   # [bs,512,128,128]

            # NOTE: dimensionality reduction with PCA, which is used to satisfy the output dimension of the Gaussian Renderer
            bs = gt_rgb.shape[0]
            A = gt_embed.reshape(bs, 512, -1).permute(0, 2, 1)  # [bs, 128*128, 512]
            gt_embed_list = []
            for i in range(bs):
                U, S, V = torch.pca_lowrank(A[i], q=np.maximum(6, self.d_embed))
                reconstructed_embed = torch.matmul(A[i], V[:, :self.d_embed])
                gt_embed_list.append(reconstructed_embed)

            gt_embed = torch.stack(gt_embed_list, dim=0).permute(0, 2, 1).reshape(bs, self.d_embed, 128, 128)
            return gt_embed
        
        elif self.model_name == "dinov2":
            batched_input = self.dino_preprocess(gt_rgb.permute(0, 3, 1, 2))    # resize
            feature = self.feature_extractor(batched_input)
            gt_embed = F.interpolate(feature, size=(128, 128), mode='bilinear', align_corners=False)    # [b, 1024, 128, 128]

            # NOTE: dimensionality reduction with PCA, which is used to satisfy the output dimension of the Gaussian Renderer
            bs = gt_rgb.shape[0]
            A = gt_embed.reshape(bs, 1024, -1).permute(0, 2, 1)  # [bs, 128*128, 1024]
            gt_embed_list = []
            for i in range(bs):
                U, S, V = torch.pca_lowrank(A[i], q=np.maximum(6, self.d_embed))
                reconstructed_embed = torch.matmul(A[i], V[:, :self.d_embed])
                gt_embed_list.append(reconstructed_embed)
            gt_embed = torch.stack(gt_embed_list, dim=0).permute(0, 2, 1).reshape(bs, self.d_embed, 128, 128)
            return gt_embed
        else:
            return None

    def encode_data(self, pcd, dec_fts, lang, 
                    rgb=None, depth=None, mask=None, focal=None, c=None, lang_goal=None, tgt_pose=None, tgt_intrinsic=None,
                    next_tgt_pose=None, next_tgt_intrinsic=None, action=None, step=None, 
                    gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, indx=None,
                    next_gt_mask_camera_extrinsic=None, next_gt_mask_camera_intrinsic=None,
                    gt_mask=None,next_gt_mask = None,
                    ): 
        '''prepare data dict'''
        bs = pcd.shape[0]
        data = {}
        # format input
        data['img'] = rgb
        data['dec_fts'] = dec_fts
        data['depth'] = depth
        data['lang'] = lang
        data['action'] = action
        right_action, left_action = torch.split(action, split_size_or_sections=8, dim=1)
        data['right_action'] = right_action
        data['left_action'] = left_action
        data['step'] = step

        if self.mask_gen == 'pre':
            data['mask_view'] = {}
            data['mask_view']['intr'] = gt_mask_camera_intrinsic # [indx] 
            data['mask_view']['extr'] = gt_mask_camera_extrinsic #  [indx]         
            if data['mask_view']['intr'] is not None:
                data_novel = self.get_novel_calib(data['mask_view'], True)
                data['mask_view'].update(data_novel) 
        elif self.mask_gen == 'nonerf':
            data['mask'] = mask

        # novel pose
        data['novel_view'] = {}
        data['intr'] = tgt_intrinsic 
        data['extr'] = tgt_pose     
        data['xyz'] = einops.rearrange(pcd, 'b c h w -> b (h w) c') # bs,256*256,(xyz)

        # use extrinsic pose to generate gaussain parameters
        if data['intr'] is not None:
            data_novel = self.get_novel_calib(data, False)
            data['novel_view'].update(data_novel)

        if self.use_dynamic_field:
            if self.field_type =='bimanual':      # ManiGaussian*2
                data['next'] = {
                    'extr': next_tgt_pose,
                    'intr': next_tgt_intrinsic,
                    'novel_view': {},
                }
                if data['next']['intr'] is not None:
                    data_novel = self.get_novel_calib(data['next'], False)
                    data['next']['novel_view'].update(data_novel)
            elif self.field_type =='LF':            # leader follower Mode
                if self.mask_gen == 'MASK_IN_NERF':          # mask in nerf
                    data['right_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['right_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['right_next'], False)
                        data['right_next']['novel_view'].update(data_novel)

                    data['left_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['left_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['left_next'], False)
                        data['left_next']['novel_view'].update(data_novel)

                if self.mask_gen == 'pre':
                    data['right_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['right_next']['intr'] is not None:
                        # to gen leftmask
                        data_novel = self.get_novel_calib(data['right_next'], False)
                        data['right_next']['novel_view'].update(data_novel)
                        data['right_next']['mask_view'] = {}
                        data['right_next']['mask_view']['intr'] = next_gt_mask_camera_intrinsic # [indx] 
                        data['right_next']['mask_view']['extr'] = next_gt_mask_camera_extrinsic # [indx]   
                        if data['right_next']['mask_view']['intr'] is not None:
                            data_novel_test = self.get_novel_calib(data['right_next']['mask_view'], True)
                            data['right_next']['mask_view'].update(data_novel_test) 

                    data['left_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['left_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['left_next'], False)
                        data['left_next']['novel_view'].update(data_novel)
                        data['left_next']['mask_view'] = {}
                        data['left_next']['mask_view']['intr'] = next_gt_mask_camera_intrinsic # [indx] 
                        data['left_next']['mask_view']['extr'] = next_gt_mask_camera_extrinsic # [indx]   
                        if data['left_next']['mask_view']['intr'] is not None:
                            data_novel_test = self.get_novel_calib(data['left_next']['mask_view'], True)
                            data['left_next']['mask_view'].update(data_novel_test) 
                
                elif self.mask_gen == 'gt':    # gt mask
                    data['right_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['right_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['right_next'],False)
                        data['right_next']['novel_view'].update(data_novel)
                    data['left_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['left_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['left_next'], False)
                        data['left_next']['novel_view'].update(data_novel)
                elif self.mask_gen == 'nonerf':          # use 6 camera train nerf
                    data['right_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['right_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['right_next'], False)
                        data['right_next']['novel_view'].update(data_novel)

                    data['left_next'] = {
                        'extr': next_tgt_pose,
                        'intr': next_tgt_intrinsic,
                        'novel_view': {},
                    }
                    if data['left_next']['intr'] is not None:
                        data_novel = self.get_novel_calib(data['left_next'], False)
                        data['left_next']['novel_view'].update(data_novel)

        return data

    def get_novel_calib(self, data, mask=False):
        """
        True :mask          False: rgb
        get readable camera state for gaussian renderer from gt_pose
        :param data: dict
        :param data['intr']: intrinsic matrix
        :param data['extr']: c2w matrix

        :return: dict
        """
        bs = data['intr'].shape[0]
        device = data['intr'].device
        fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
        for i in range(bs):
            intr = data['intr'][i, ...].cpu().numpy()
            if mask: # mask need to change 256 256 -> 128 128
                intr = intr / 2

            extr = data['extr'][i, ...].cpu().numpy()
            extr = np.linalg.inv(extr)  # the saved extrinsic is actually cam2world matrix, so turn it to world2cam matrix 

            width, height = self.W, self.H
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)    # inverse
            T = np.array(extr[:3, 3], np.float32)                                   
            FovX = focal2fov(intr[0, 0], width)     
            FovY = focal2fov(intr[1, 1], height)
            projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=intr, h=height, w=width).transpose(0, 1)
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1) # [4, 4], w2c
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)    # [4, 4]
            camera_center = world_view_transform.inverse()[3, :3]   # inverse is c2w

            fovx_list.append(FovX)
            fovy_list.append(FovY)
            world_view_transform_list.append(world_view_transform.unsqueeze(0))
            full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
            camera_center_list.append(camera_center.unsqueeze(0))

        novel_view_data = {
            'FovX': torch.FloatTensor(np.array(fovx_list)).to(device),
            'FovY': torch.FloatTensor(np.array(fovy_list)).to(device),
            'width': torch.tensor([width] * bs).to(device),
            'height': torch.tensor([height] * bs).to(device),
            'world_view_transform': torch.concat(world_view_transform_list).to(device),
            'full_proj_transform': torch.concat(full_proj_transform_list).to(device),
            'camera_center': torch.concat(camera_center_list).to(device),
        }

        return novel_view_data

    def forward(self, pcd, dec_fts, language, mask=None, gt_rgb=None, gt_pose=None, gt_intrinsic=None, rgb=None, depth=None, camera_intrinsics=None, camera_extrinsics=None, 
                focal=None, c=None, lang_goal=None, gt_depth=None,
                next_gt_pose=None, next_gt_intrinsic=None, next_gt_rgb=None, step=None, action=None,
                training=True, gt_mask=None, gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, next_gt_mask = None,
                next_gt_mask_camera_extrinsic=None, next_gt_mask_camera_intrinsic=None,
                gt_maskdepth=None,next_gt_maskdepth=None,):
        '''
        main forward function
        Return:
        :loss_dict: dict, loss values
        :ret_dict: dict, rendered images
        '''
        bs = rgb.shape[0]
        indx = random.randint(0, 1) # 0 front or 1 overhead
        if self.mask_gen == 'nonerf':
            data = self.encode_data(
                rgb=rgb, depth=depth, pcd=pcd, focal=focal, c=c, lang_goal=None, tgt_pose=gt_pose, tgt_intrinsic=gt_intrinsic,
                dec_fts=dec_fts, lang=language, next_tgt_pose=next_gt_pose, next_tgt_intrinsic=next_gt_intrinsic, 
                action=action, step=step, gt_mask=gt_mask, gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic,  
                next_gt_mask=next_gt_mask, indx = indx, 
            )
        else: # pre | MASK_IN_NERF
            data = self.encode_data(
                rgb=rgb, depth=depth, pcd=pcd,mask=mask, focal=focal, c=c, lang_goal=None, 
                tgt_pose=gt_pose, tgt_intrinsic=gt_intrinsic,next_tgt_pose=next_gt_pose, next_tgt_intrinsic=next_gt_intrinsic,   
                dec_fts=dec_fts, lang=language, action=action, step=step, 
                gt_mask=gt_mask,next_gt_mask=next_gt_mask, 
                gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic, 
                next_gt_mask_camera_extrinsic=next_gt_mask_camera_extrinsic, next_gt_mask_camera_intrinsic=next_gt_mask_camera_intrinsic, 
                            )

        render_novel, next_render_novel = None, None
        render_embed = None
        gt_embed = None
        render_mask_gtrgb = None
        render_mask_novel = None
        next_render_mask_right = None
        next_render_mask = None
        next_render_rgb_right = None
        next_left_mask_gen, exclude_left_mask = None, None
        gt_mask_vis,next_gt_mask_vis = None, None

        # create gt feature from foundation models 
        with torch.no_grad():
            # Diffusion or dinov2
            gt_embed = self.extract_foundation_model_feature(gt_rgb, lang_goal)
        
        # change the mask
        
        if self.mask_gen == 'MASK_IN_NERF':
            # ID[1,128,128,3]->ID[1,128,128]->target[1,128,128][0bg,1l,2r]
            # gt_mask_label = gt_mask[:, :, :, 0].long() # [10,0,0]  [1,3,128,128]->[1,128,128]
            gt_mask_label = self.mask_label_onehot(gt_mask[:, :, :, 0].long())#[1 128 128](9...)->label
            device = gt_mask.device
            gt_mask_label = gt_mask_label.to(device)
            if self.use_dynamic_field:
                    next_gt_mask_label = self.mask_label_onehot(next_gt_mask[:, :, :, 0].long())#[1 128 128](9...)->label
                    device = next_gt_mask.device
                    next_gt_mask_label = next_gt_mask_label.to(device)
        elif self.mask_gen == 'pre': # pre mask in 6 camera
            gt_mask = F.interpolate(gt_mask, size=(128, 128), mode='bilinear', align_corners=False)
            gt_mask =gt_mask.permute(0,2,3,1) # [1, 1, 256, 256] ->[1, 256, 256, 1]
            if (self.use_CEloss >=1 and self.use_CEloss <= 3) or (self.use_CEloss == 7) or (self.use_CEloss == 21):
                gt_mask_label = self.mask_label_onehot(gt_mask) # 1 128 128 [target 0 1 2]  
            elif self.use_CEloss == 0:
                gt_mask_label = self.mask_label(gt_mask) # [1 128 128 3]   
            elif self.use_CEloss == 4:
                gt_mask_label = self.mask_label(gt_mask) 
                gt_mask = gt_mask_label
            # 6 only rgb
            if self.use_dynamic_field:
                next_gt_mask = F.interpolate(next_gt_mask, size=(128, 128), mode='bilinear', align_corners=False)
                next_gt_mask =next_gt_mask.permute(0,2,3,1) # [1, 1, 256, 256] ->[1, 256, 256, 1]
                if (self.use_CEloss >=1 and self.use_CEloss <= 3) or (self.use_CEloss == 7) or (self.use_CEloss == 21):
                    next_gt_mask_label = self.mask_label_onehot(next_gt_mask) # 1 128 128 [target 0 1 2]   7(L_ce)
                elif self.use_CEloss == 0: # L1
                    next_gt_mask_label = self.mask_label(next_gt_mask) # [1 128 128 3]   
                elif self.use_CEloss == 4:
                    next_gt_mask_label = self.mask_label(next_gt_mask) 
                    next_gt_mask = next_gt_mask_label
        elif self.mask_gen =="nonerf":
                gt_rgb = gt_rgb.permute(0,2,3,1) # [1, 3, 256, 256] ->[1, 256, 256, 3]
                gt_mask =gt_mask.permute(0,2,3,1) # [1, 1, 256, 256] ->[1, 256, 256, 1]
                if self.use_CEloss == 0:
                    gt_mask_label = self.mask_label(gt_mask) # [1 128 128 3]   
                else:
                    gt_mask_label = self.mask_label_onehot(gt_mask) 
                if self.use_dynamic_field:
                    next_gt_rgb = next_gt_rgb.permute(0,2,3,1)
                    next_gt_mask = next_gt_mask.permute(0,2,3,1)
                    next_gt_mask_label = self.mask_label_onehot(next_gt_mask) 
        if training:
            # Gaussian Generator 
            data = self.gs_model(data) # GeneralizableGSEmbedNet(cfg, with_gs_render=True)

            # Gaussian Render
            data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]

            # Loss L(GEO)  Current Scence Consistency Loss
            render_novel = data['novel_view']['img_pred'].permute(0, 2, 3, 1)   # [1, 128, 128, 3] 

            # visdom
            if self.cfg.visdom: # False
                vis = visdom.Visdom()
                rgb_vis = data['img'][0].detach().cpu().numpy() * 0.5 + 0.5
                vis.image(rgb_vis, win='front_rgb', opts=dict(title='front_rgb'))

                depth_vis = data['depth'][0].detach().cpu().numpy()#/255.0
                # convert 128x128 0-255 depth map to 3x128x128 0-1 colored map 
                vis.image(depth_vis, win='front_depth', opts=dict(title='front_depth'))
                vis.image(render_novel[0].permute(2, 0, 1).detach().cpu().numpy(), win='render_novel', opts=dict(title='render_novel'))
                vis.image(gt_rgb[0].permute(2, 0, 1).detach().cpu().numpy(), win='gt_novel', opts=dict(title='gt_novel'))
            
            loss = 0.

            # Ll1 = l1_loss(render_novel, gt_rgb) 
            loss_dyna_mask = torch.tensor(0.) 
            if self.mask_gen == 'nonerf':
                if True:
                    # # 1 mask now  loss_dyna_mask_novel
                    data =self.pts2render_mask_gen(data, bg_color=self.bg_mask)
                    if self.use_CEloss==21:
                        render_mask_novel = data['novel_view']['mask_gen']  # 1 3 256 256                 
                    elif self.use_CEloss==0:
                        render_mask_novel = data['novel_view']['mask_gen'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]                           
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                if not self.use_dynamic_field :
                    loss_dyna_mask = loss_dyna_mask_novel
                    lambda_mask = 0 # 1   #if step >= 1000 else 0 # 3000
                    # lambda_mask = min(0.1 * step / 200, 1)
                    loss += loss_dyna_mask * lambda_mask # * 0.001
                elif self.use_dynamic_field:
                    lambda_mask = self.cfg.lambda_mask if step >= self.cfg.mask_warm_up else 0. 
                    loss += loss_dyna_mask_novel * lambda_mask # * 0.001     

            elif self.mask_gen == 'pre':
                data =self.pts2render_mask(data, bg_color=self.bg_mask)
                if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                    render_mask_novel = data['novel_view']['mask_pred']
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)
                elif self.use_CEloss==2:
                    render_mask_novel = data['novel_view']['mask_pred'] # 1 3 128 128 
                    render_mask_novel = render_mask_novel[:, [0, 1], :, :]
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)
                elif self.use_CEloss == 0:
                    render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                elif self.use_CEloss == 4:
                    render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 3 128 128 
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)                    

                if not self.use_dynamic_field and self.use_CEloss != 6:
                    loss_dyna_mask = loss_dyna_mask_novel
                    lambda_mask = 0 #1   if step >= 1000 else 0 # 3000
                    # lambda_mask = min(0.1 * step / 300, 1)
                    loss += loss_dyna_mask * lambda_mask # * 0.001
                elif self.use_CEloss != 6:
                    # loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)
                    # lambda_mask = self.cfg.lambda_mask * self.cfg.lambda_dyna if step >= self.cfg.mask_warm_up else 0. # 0.4 2000
                    lambda_mask = 0
                    loss += loss_dyna_mask_novel * lambda_mask # * 0.001
            elif self.mask_gen == 'MASK_IN_NERF':
                data =self.pts2render_for_MASK_IN_NERF(data, bg_color=self.bg_mask)
                if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                    render_mask_novel = data['novel_view']['mask_pred']
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)
                elif self.use_CEloss==2:
                    render_mask_novel = data['novel_view']['mask_pred'] # 1 3 128 128 
                    render_mask_novel = render_mask_novel[:, [0, 1], :, :]
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)
                elif self.use_CEloss == 0:
                    render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label)     
                elif self.use_CEloss == 4:
                    render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 3 128 128 
                    loss_dyna_mask_novel = self._mask_loss_fn(render_mask_novel, gt_mask_label) 
                    render_mask_novel = render_mask_novel.permute(0, 2, 3, 1)                    
       
                if not self.use_dynamic_field and self.use_CEloss != 6:
                    loss_dyna_mask = loss_dyna_mask_novel
                    lambda_mask = 0.005 #1   if step >= 1000 else 0 # 3000
                    # lambda_mask = min(0.1 * step / 300, 1)
                    loss += loss_dyna_mask * lambda_mask # * 0.001
                elif self.use_CEloss != 6:
                    loss_dyna_mask = loss_dyna_mask_novel # for vis
                    loss += loss_dyna_mask_novel * lambda_mask # * 0.001
            
            Ll1 = l2_loss(render_novel, gt_rgb) # loss_now_rgb
            # Lssim = 1.0 - ssim(render_novel, gt_rgb)
            Lssim = 0.
            psnr = PSNR_torch(render_novel, gt_rgb)

            # loss_rgb = self.cfg.lambda_l1 * Ll1 + self.cfg.lambda_ssim * Lssim
            loss_rgb = Ll1
            # 1 LGeo?
            # lambda_rgb = max(1-(0.1 * step / 300), 0.1)
            loss += loss_rgb #* 0 #* lambda_rgb

            # optional
            if gt_embed is not None:
                gt_embed = gt_embed.permute(0, 2, 3, 1) # channel last
                render_embed = data['novel_view']['embed_pred'].permute(0, 2, 3, 1)
                # DEBUG gradient    
                # render_embed_grad = render_embed.register_hook(self._save_gradient('render_embed'))
                loss_embed = self._embed_loss_fn(render_embed, gt_embed)
                # 2 loss = loss_rgb + self.cfg.lambda_embed * loss_embed
                loss += self.cfg.lambda_embed * loss_embed
            else:
                loss_embed = torch.tensor(0.)

            # next frame prediction Ldyna(optional)
            if self.field_type == 'bimanual':
                if self.use_dynamic_field and (next_gt_rgb is not None) and ('xyz_maps' in data['next']):
                    data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                    next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)
                    loss_dyna = l2_loss(next_render_novel, next_gt_rgb)
                    lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.
                    loss += lambda_dyna * loss_dyna

                    loss_reg = torch.tensor(0.)
                    loss_LF = torch.tensor(0.)
                    loss_dyna_mask = torch.tensor(0.)
                else:
                    loss_dyna = torch.tensor(0.)
                    loss_LF = torch.tensor(0.)
                    loss_dyna_mask = torch.tensor(0.)
                    loss_reg = torch.tensor(0.)
            elif self.field_type == 'LF':    # Leader Follower condition
                elif self.mask_gen == 'MASK_IN_NERF':  
                    if self.use_dynamic_field and (next_gt_rgb is not None and next_gt_mask is not None):
                        # 2 GRB total(Follower)   loss_dyna_follower  LEFT RGB Loss   
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) 

                        data['left_next'] =self.pts2render_for_MASK_IN_NERF(data['left_next'], bg_color=self.bg_mask)
                        # data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            next_render_mask = data['left_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) 
                            # gen exclude
                            if self.hierarchical:
                                next_left_mask_gen = next_loss_dyna_mask_left.permute(0, 2, 3, 1)  # 1 128 128 3   
                                exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen)
                                exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                        if self.hierarchical:   
                            result_right_image = next_gt_rgb * exclude_left_mask # + background_color * (~exclude_left_mask) # [1, 128, 128, 3] #
                        else:
                            result_right_image = next_gt_rgb
                            
                        #  4 RGB loss_dyna_leader leader 
                        if ('xyz_maps' in data['right_next']):
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            if self.hierarchical:
                                next_render_novel_mask = next_render_rgb_right * exclude_left_mask  
                            else:
                                next_render_novel_mask = next_render_rgb_right #* exclude_left_mask
                            loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)
                        
                        # 5 Mask loss_dyna_mask_next_right 
                        data['right_next'] =self.pts2render_for_MASK_IN_NERF(data['right_next'], bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            next_loss_dyna_mask_right = self._mask_loss_fn(next_render_mask_right, next_gt_mask_label)
                            next_render_mask_right = next_render_mask_right.permute(0, 2, 3, 1) 

                        # next mask = right +left    
                        next_loss_dyna_mask = next_loss_dyna_mask_left * ( 1 - self.cfg.lambda_mask_right ) + next_loss_dyna_mask_right * self.cfg.lambda_mask_right  
                        
                        # MASK = now +pre
                        # loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)  + next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        # loss_dyna_mask += next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        loss_dyna_mask = next_loss_dyna_mask


                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)
                        # print('loss_LF = ', loss_LF, loss_dyna_leader, loss_dyna_follower)

                        lambda_mask = self.cfg.lambda_mask if step >= self.cfg.next_mlp.warm_up + self.cfg.mask_warm_up else 0. # 5000
                        loss_dyna = loss_LF * (1 - lambda_mask) + loss_dyna_mask * lambda_mask

                        # print('loss_dyna = ', loss_dyna,loss_LF,loss_dyna_mask)
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
                        
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna

                        loss_reg = torch.tensor(0.)
                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        # loss_dyna_mask = torch.tensor(0.) 
            
                if self.mask_gen == 'gt':   
                    if self.use_dynamic_field and next_gt_rgb is not None:
                        mask_3d, next_mask_3d = self.createby_gt_mask(data=data, gt_mask=gt_mask, 
                            gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic,  
                            next_gt_mask=next_gt_mask,gt_maskdepth=gt_maskdepth, next_gt_maskdepth=next_gt_maskdepth)

                        # ->2D
                        projected_points = project_3d_to_2d(next_mask_3d, next_gt_intrinsic)
                        
                        mask_shape = (128, 128)
                        exclude_left_mask = create_2d_mask_from_convex_hull(projected_points, mask_shape)
                        exclude_left_mask = exclude_left_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3) # [1,256,256,3]

                        device = next_gt_rgb.device 
                        exclude_left_mask = exclude_left_mask.to(device)
                        next_render_mask = exclude_left_mask
                        result_right_image = next_gt_rgb * exclude_left_mask
                        render_mask_novel = result_right_image

                        # 2 GRB total(Follower)  Left RGB   
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) 
                        
                        #  4 RGB loss_dyna_leader leader  Right RGB  -mask
                        if ('xyz_maps' in data['right_next']):
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            next_render_novel_mask = next_render_rgb_right * exclude_left_mask  # next_gt_rgb -> next_render_rgb_right
                            loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)

                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)

                        loss_dyna_mask = torch.tensor(0.) 
                        loss_reg = torch.tensor(0.) 
                        loss_dyna = loss_LF    # * (1-self.cfg.lambda_mask) + loss_dyna_mask * self.cfg.lambda_mask 
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
                        
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna
                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        loss_dyna_mask = torch.tensor(0.)
                elif self.mask_gen == 'pre':  
                    if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                        # 2 GRB total(Follower)  
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) 

                        data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_mask)
                        data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            next_render_mask = data['left_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label)
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1) 
                            # gen exclude
                            # next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)  # 1 128 128 3   
                            # exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen)
                            # exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                        elif self.use_CEloss == 0:     # Ll1
                            next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3(rgb label mean-> 0 1 2)
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) 

                            next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1) # 1 128 128 3
                            next_left_mask_gen = (next_left_mask_gen / 2 + 0.5  )*(self.num_classes-1)           # 1 128 128 3 [0,2]
                            exclude_left_mask = self.generate_final_class_labels_L1(next_left_mask_gen)
                        elif self.use_CEloss == 4: 
                            next_render_mask = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 3 128 128 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) 
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)                    
                        elif self.use_CEloss==2: 
                            next_render_mask = data['novel_view']['mask_pred'] # 1 3 128 128 
                            next_render_mask = next_render_mask[:, [0, 1], :, :]
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label)
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)

                        result_right_image = next_gt_rgb #* exclude_left_mask # + background_color * (~exclude_left_mask) # [1, 128, 128, 3] #

                        #  4 RGB loss_dyna_leader leader  
                        if ('xyz_maps' in data['right_next']):
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            next_render_novel_mask = next_render_rgb_right #* exclude_left_mask 
                            loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)

                        data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'] # 1 3 128 128 
                            next_loss_dyna_mask_right = self._mask_loss_fn(next_render_mask_right, next_gt_mask_label) 
                            next_render_mask_right = next_render_mask_right.permute(0, 2, 3, 1) 
                        elif self.use_CEloss == 0:     # Ll1
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3(rgb label mean-> 0 1 2)
                            next_loss_dyna_mask_right = self._mask_loss_fn(next_render_mask_right, next_gt_mask_label) 
                        elif self.use_CEloss == 4: 
                            next_render_mask = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 3 128 128 
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label) 
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)                    
                        elif self.use_CEloss==2: 
                            next_render_mask = data['novel_view']['mask_pred'] # 1 3 128 128 
                            next_render_mask = next_render_mask[:, [0, 1], :, :]
                            next_loss_dyna_mask_left = self._mask_loss_fn(next_render_mask, next_gt_mask_label)
                            next_render_mask = next_render_mask.permute(0, 2, 3, 1)

                        # next mask = right +left    
                        next_loss_dyna_mask = next_loss_dyna_mask_left * ( 1 - self.cfg.lambda_mask_right ) + next_loss_dyna_mask_right * self.cfg.lambda_mask_right  
                        
                        # MASK = now +pre
                        # loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)  + next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        # loss_dyna_mask += next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        loss_dyna_mask = next_loss_dyna_mask


                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)
                        # print('loss_LF = ', loss_LF, loss_dyna_leader, loss_dyna_follower)

                        lambda_mask = self.cfg.lambda_mask if step >= self.cfg.next_mlp.warm_up + self.cfg.mask_warm_up else 0. # 5000
                        loss_dyna = loss_LF * (1 - lambda_mask) + loss_dyna_mask * lambda_mask

                        # print('loss_dyna = ', loss_dyna,loss_LF,loss_dyna_mask)
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
                        
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna

                        loss_reg = torch.tensor(0.)
                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        # loss_dyna_mask = torch.tensor(0.) 

                elif self.mask_gen == 'None':
                    if self.use_dynamic_field and (next_gt_rgb is not None):
                        # 2 GRB total(Follower)  
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb)

                        #  4 RGB loss_dyna_leader leader  
                        if ('xyz_maps' in data['right_next']):
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            loss_dyna_leader = l2_loss(next_render_rgb_right, next_gt_rgb)

                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)
                        loss_dyna = loss_LF 
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
    
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna
                        loss_dyna_mask = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)

                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        loss_dyna_mask = torch.tensor(0.)
                elif self.mask_gen == 'nonerf': 
                    if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                        # next_gt_mask_label = self.mask_label(next_gt_mask)                        
                        # 2 GRB total(Follower)   
                        if ('xyz_maps' in data['left_next']):
                            data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                            next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb) 

                        # 3 next mask train (pre - left(mask) ) next_loss_dyna_mask_left  LEFT Mask Loss
                        data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                        next_render_mask = data['left_next']['novel_view']['mask_gen'] # .permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                        next_loss_dyna_mask_left = torch.tensor(0.) #self._mask_loss_fn(next_render_mask, next_gt_mask_label) # -left mask
                        next_render_mask = next_render_mask.permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]

                        result_right_image = next_gt_rgb  # * exclude_left_mask 

                        #  4 RGB loss_dyna_leader leader 
                        if ('xyz_maps' in data['right_next']):
                            data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                            next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            next_render_novel_mask = next_render_rgb_right # * exclude_left_mask 
                            loss_dyna_leader = l2_loss(next_render_novel_mask, result_right_image)
                        
                        # 5 Mask loss_dyna_mask_next_right 
                        data['right_next'] =self.pts2render_mask_gen(data['right_next'], bg_color=self.bg_mask)
                        next_render_mask_right = data['right_next']['novel_view']['mask_gen'] # .permute(0, 2, 3, 1)
                        # CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
                        next_loss_dyna_mask_right = torch.tensor(0.) # self._mask_loss_fn(next_render_mask_right, next_gt_mask_label)
                        next_render_mask_right = next_render_mask_right.permute(0, 2, 3, 1) # [1,128, 128, 3]

                        # pre mask = right +left    
                        next_loss_dyna_mask = next_loss_dyna_mask_left * ( 1 - self.cfg.lambda_mask_right ) + next_loss_dyna_mask_right * self.cfg.lambda_mask_right 
                        
                        # MASK = now +pre
                        # loss_dyna_mask = loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)  + next_loss_dyna_mask * self.cfg.lambda_next_loss_mask
                        loss_dyna_mask = next_loss_dyna_mask # loss_dyna_mask_novel * (1 - self.cfg.lambda_next_loss_mask)  + next_loss_dyna_mask * self.cfg.lambda_next_loss_mask


                        # RGB pre = leader( right ) + follower
                        loss_LF = loss_dyna_leader * self.cfg.lambda_dyna_leader + loss_dyna_follower * (1-self.cfg.lambda_dyna_leader)
                        # print('loss_LF = ', loss_LF, loss_dyna_leader, loss_dyna_follower)
                        lambda_mask = self.cfg.lambda_mask if step >= self.cfg.next_mlp.warm_up + self.cfg.mask_warm_up else 0. # 5000
                        loss_dyna = loss_LF * (1-lambda_mask) + loss_dyna_mask * lambda_mask 
                        lambda_dyna = self.cfg.lambda_dyna if step >= self.cfg.next_mlp.warm_up else 0.                    
                        
                        # Step 3 Loss(LGeo? + L embed/L sem + L dyna) = loss_rgb + self.cfg.lambda_embed * loss_embed + lambda_dyna * loss_dyna
                        loss += lambda_dyna * loss_dyna

                        loss_reg = torch.tensor(0.)
                    else:
                        loss_dyna = torch.tensor(0.)
                        loss_reg = torch.tensor(0.)      
                        loss_LF = torch.tensor(0.)
                        # loss_dyna_mask = torch.tensor(0.) 

            loss_dict = {
                'loss': loss,
                'loss_rgb': loss_rgb.item(),
                'loss_embed': loss_embed.item(),
                'loss_dyna': loss_dyna.item(),
                'loss_LF': loss_LF.item(),
                'loss_dyna_mask': loss_dyna_mask.item(),
                'loss_reg': loss_reg.item(),
                'l1': Ll1.item(),
                'psnr': psnr.item(),
                }
        else: # not training 
            # no ground-truth given, rendering (inference) 
            print("not training")
            with torch.no_grad():
                # Gaussian Generator
                data = self.gs_model(data)
                # Gaussian Render
                data = self.pts2render(data, bg_color=self.bg_color) # default: [0, 0, 0]
                render_novel = data['novel_view']['img_pred'].permute(0, 2, 3, 1) # channel last
                render_embed = data['novel_view']['embed_pred'].permute(0, 2, 3, 1)
                
                # DYN
                if self.field_type == 'bimanual':
                    if self.use_dynamic_field and 'xyz_maps' in data['next']:
                        data['next'] = self.pts2render(data['next'], bg_color=self.bg_color)
                        next_render_novel = data['next']['novel_view']['img_pred'].permute(0, 2, 3, 1)
                else:
                    if self.mask_gen == 'gt':
                        if self.use_dynamic_field:
                            mask_3d, next_mask_3d = self.createby_gt_mask(data=data, 
                                gt_mask=gt_mask,next_gt_mask=next_gt_mask, 
                                gt_mask_camera_extrinsic=gt_mask_camera_extrinsic, gt_mask_camera_intrinsic=gt_mask_camera_intrinsic,  
                                gt_maskdepth=gt_maskdepth, next_gt_maskdepth=next_gt_maskdepth)
                            projected_points = project_3d_to_2d(next_mask_3d, next_gt_intrinsic)
                            mask_shape = (128,128) # (256, 256) 
                            exclude_left_mask = create_2d_mask_from_convex_hull(projected_points, mask_shape)
                            exclude_left_mask = exclude_left_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3) # [1,256,256,3]
                            if next_gt_rgb is not None:
                                device = next_gt_rgb.device 
                                exclude_left_mask = exclude_left_mask.to(device)
                                next_render_mask = exclude_left_mask
                                result_right_image = next_gt_rgb * exclude_left_mask  
                                render_mask_novel = result_right_image

                            # 2 GRB total(Follower)  
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            
                            #  4 RGB loss_dyna_leader leader 
                            if ('xyz_maps' in data['right_next']):
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]                                
                    elif self.mask_gen == 'MASK_IN_NERF':   # LF +  train mask   
                        data =self.pts2render(data, bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss==3 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                            ## !! render_mask_novel = render_novel * render_mask_gtrgb
                            render_mask_gtrgb = self.generate_final_class_labels(render_mask_novel).unsqueeze(3).repeat(1, 1, 1, 3)     # 1 128 128 3 [TTT/FFF]                        
                            # rgb1 = F.interpolate(rgb, size=(128, 128), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) / 2 + 0.5 # [1,128, 128, 3]
                            rgb1 = gt_rgb
                            # print("rgb1.shape=torch.Size([1, 128, 128, 3])",rgb1.shape)
                            render_mask_gtrgb = rgb1 * render_mask_gtrgb
                            # vis render
                            render_mask_novel = self.vis_labels(render_mask_novel)
                            # print("render_mask_novel.shape=torch.Size([1, 128, 128, 3]) = ",render_mask_novel.shape)
                            gt_mask1 = self.mask_onehot(gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                            # print("2 gt_mask1.shape=",gt_mask1.shape,gt_mask1)
                            gt_mask_vis =  self.vis_labels(gt_mask1)
                            # print("3 gt_mask_vis.shape=",gt_mask_vis.shape,gt_mask_vis)
                            # data =self.pts2render_rgb(data, bg_color=self.bg_color)
                            # next_render_mask = data['mask_view']['rgb_pred'].permute(0, 2, 3, 1)

                        if self.use_CEloss==2: 
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)  # 1 128 128 3
                            render_mask_novel1 = render_mask_novel
                            print("render_mask_novel",render_mask_novel.shape) # [1 128 128 3]
                            render_mask_novel = self.generate_final_class_labels_ce1(render_mask_novel) # [1 128 128] 0/1
                            next_render_mask_right = self.vis_labels_ce1(render_mask_novel1)

                            # render_mask_novel = self.generate_final_class_labels(render_mask_novel1) # 1 128 128
                            render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)   # 1 128 128 3
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            render_mask_novel = render_novel * render_mask_novel

                            # gt
                            gt_mask1 = self.mask_onehot(gt_mask)  # 1 128 128 3
                            # gt_mask1 =gt_mask1.permute(0, 3, 1, 2)
                            print("gt_mask1",gt_mask1.shape,gt_mask1) # 1 128 128 3 
                            next_render_rgb_right =  self.vis_labels(gt_mask1)
                        elif self.use_CEloss == 0: # l1
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3
                                # render_mask_novel = (render_mask_novel / 2 + 0.5)*(self.num_classes-1)  # [0 , 2]
                            print("render_mask_novel",render_mask_novel.shape,render_mask_novel) # 1 128 128 3
                            next_render_mask_right = self.vis_labelsL1(render_mask_novel)           #
                            render_mask_novel = self.generate_final_class_labels_L1(render_mask_novel)

                            # vis gt left 
                            ## !! 
                            gt_mask_vis = self.vis_labelsL1(gt_mask_label)
                            next_gt_mask_vis = self.vis_labelsL1_1(gt_mask_label)
                            exclude_left_mask = self.vis_labelsL1_2(gt_mask_label)
                            next_render_mask = self.vis_labelsL1_3(gt_mask_label)
                            
                            next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            next_render_rgb_right = gt_rgb * next_render_rgb_right
                            
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            render_mask_novel = render_novel * render_mask_novel
                        elif self.use_CEloss == 4:
                            render_mask_novel = render_novel
                            render_mask_novel = render_mask_novel*(self.num_classes-1)
                            next_render_mask_right = self.vis_labelsL1(render_mask_novel) 
                            # render_mask_novel = self.generate_final_class_labels(render_mask_novel)
                            # render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)
                            render_mask_novel = self.generate_final_class_labels_L1(render_mask_novel)

                            # gt left
                            next_render_mask = self.vis_labelsL1(gt_mask_label)
                            next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            next_render_rgb_right = gt_rgb * next_render_rgb_right
                            
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            print("render_mask_novel = ",render_mask_novel.shape,render_mask_novel)
                            render_mask_novel = render_novel * render_mask_novel                           
                        elif self.use_CEloss == 6:
                            print("only rgb")                          
                        if self.use_dynamic_field:
                            # 2 next Left RGB     
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            
                            # 3 next Left mask
                            if self.hierarchical:
                                # data['left_next'] =self.pts2render_for_MASK_IN_NERF(data['left_next'], bg_color=self.bg_mask)
                                # next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                                # next_render_mask = self.generate_final_class_labels(next_render_mask)
                                # next_render_mask = next_render_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                                # next_render_mask = next_render_novel * next_render_mask
                            
                            data['left_next'] =self.pts2render_for_MASK_IN_NERF(data['left_next'], bg_color=self.bg_mask)
                            # data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                            if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                                next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) 
                                next_render_mask = self.vis_labels(next_render_mask)
                                # gen exclude
                                next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)  # 1 128 128 3   
                                exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen).unsqueeze(3).repeat(1, 1, 1, 3)
                                next_left_mask_gen = self.vis_labels(next_left_mask_gen)
                                next_gt_mask1 = self.mask_onehot(next_gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                                next_gt_mask_vis =  self.vis_labels(next_gt_mask1)
                            elif self.use_CEloss == 0:     # Ll1
                                next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3(rgb label mean-> 0 1 2)
                                next_render_mask = (next_render_mask / 2 + 0.5  )*(self.num_classes-1)    
                                next_render_mask = self.vis_labelsL1(next_render_mask)
                                next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1) # 1 128 128 3
                                next_left_mask_gen = (next_left_mask_gen / 2 + 0.5  )*(self.num_classes-1)           # 1 128 128 3 [0,2]
                                exclude_left_mask = self.generate_final_class_labels_L1(next_left_mask_gen)
                                next_left_mask_gen = self.vis_labelsL1(next_left_mask_gen)
                                next_gt_mask_vis = self.vis_labelsL1(next_gt_mask_label)
                            exclude_left_mask = exclude_left_mask * next_render_novel


                            # left gen mask
                                # data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                                # next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)
                                # exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen).unsqueeze(3).repeat(1, 1, 1, 3) 
                                # next_left_mask_gen = self.vis_labels(next_left_mask_gen)

                            #  4 RGB loss_dyna_leader leader  
                            if ('xyz_maps' in data['right_next']):
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                                # 1
                            # 5 Mask loss_dyna_mask_next_right 
                                # next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                            data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_mask)
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) 
                            if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                                next_render_mask_right = self.vis_labels(next_render_mask_right)
                            elif self.use_CEloss == 0:     # Ll1
                                next_render_mask_right = (next_render_mask_right / 2 + 0.5  )*(self.num_classes-1) 
                                next_render_mask_right = self.vis_labelsL1(next_render_mask_right)

                            ## test
                                # next_render_mask_right = self.vis_labels(next_render_mask_right)
                            # next_render_mask_right = self.vis_labels(next_render_mask_right).unsqueeze(3).repeat(1, 1, 1, 3)
                            # next_render_mask_right = next_render_rgb_right * next_render_mask_right

                    elif self.mask_gen == 'pre':   # LF + train mask   
                        data =self.pts2render_mask(data, bg_color=self.bg_mask)
                        if self.use_CEloss==1 or self.use_CEloss==3 or self.use_CEloss == 7 or self.use_CEloss == 21:
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                            ## !! render_mask_novel = render_novel * render_mask_gtrgb
                            render_mask_gtrgb = self.generate_final_class_labels(render_mask_novel).unsqueeze(3).repeat(1, 1, 1, 3)     # 1 128 128 3 [TTT/FFF]                        
                            rgb1 = F.interpolate(rgb, size=(128, 128), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) / 2 + 0.5 # [1,128, 128, 3]
                            render_mask_gtrgb = rgb1 * render_mask_gtrgb
                            # vis render
                            render_mask_novel = self.vis_labels(render_mask_novel)
                            
                            gt_mask1 = self.mask_onehot(gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                            gt_mask_vis =  self.vis_labels(gt_mask1)
                            # vis rgb render in mask camera 
                            data =self.pts2render_rgb(data, bg_color=self.bg_color)
                            next_render_mask = data['mask_view']['rgb_pred'].permute(0, 2, 3, 1)

                        if self.use_CEloss==2: 
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)  # 1 128 128 3
                            render_mask_novel1 = render_mask_novel
                            print("render_mask_novel",render_mask_novel.shape) # [1 128 128 3]
                            render_mask_novel = self.generate_final_class_labels_ce1(render_mask_novel) # [1 128 128] 0/1
                            next_render_mask_right = self.vis_labels_ce1(render_mask_novel1)

                            # render_mask_novel = self.generate_final_class_labels(render_mask_novel1) # 1 128 128
                            render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)   # 1 128 128 3
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            render_mask_novel = render_novel * render_mask_novel

                            # gt
                            gt_mask1 = self.mask_onehot(gt_mask)  # 1 128 128 3
                            # gt_mask1 =gt_mask1.permute(0, 3, 1, 2)
                            print("gt_mask1",gt_mask1.shape,gt_mask1) # 1 128 128 3 
                            next_render_rgb_right =  self.vis_labels(gt_mask1)
                        elif self.use_CEloss == 0: # l1
                            render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3
                                # render_mask_novel = (render_mask_novel / 2 + 0.5)*(self.num_classes-1)  # [0 , 2]
                            ## !! 
                            print("render_mask_novel",render_mask_novel.shape,render_mask_novel) # 1 128 128 3
                            next_render_mask_right = self.vis_labelsL1(render_mask_novel)           #
                            render_mask_novel = self.generate_final_class_labels_L1(render_mask_novel)

                            # vis gt left 
                            ## !! 
                            gt_mask_vis = self.vis_labelsL1(gt_mask_label)
                            next_gt_mask_vis = self.vis_labelsL1_1(gt_mask_label)
                            exclude_left_mask = self.vis_labelsL1_2(gt_mask_label)
                            next_render_mask = self.vis_labelsL1_3(gt_mask_label)
                            
                            next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            next_render_rgb_right = gt_rgb * next_render_rgb_right
                            
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            render_mask_novel = render_novel * render_mask_novel
                        elif self.use_CEloss == 4:
                            render_mask_novel = render_novel
                            render_mask_novel = render_mask_novel*(self.num_classes-1)
                            next_render_mask_right = self.vis_labelsL1(render_mask_novel)
                            # render_mask_novel = self.generate_final_class_labels(render_mask_novel)
                            # render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)
                            render_mask_novel = self.generate_final_class_labels_L1(render_mask_novel)

                            # gt left 
                            next_render_mask = self.vis_labelsL1(gt_mask_label)
                            next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            next_render_rgb_right = gt_rgb * next_render_rgb_right
                            
                            render_mask_gtrgb = gt_rgb * render_mask_novel
                            print("render_mask_novel = ",render_mask_novel.shape,render_mask_novel)
                            render_mask_novel = render_novel * render_mask_novel                           
                        elif self.use_CEloss == 6:
                            print("only rgb")                          
                        if self.use_dynamic_field:
                            # 2 next Left RGB     
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            
                            # 3 next Left mask                          
                            data['left_next'] =self.pts2render_mask(data['left_next'], bg_color=self.bg_mask)
                            data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                            if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                                next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) 
                                next_render_mask = self.vis_labels(next_render_mask)
                                # gen exclude
                                next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)  # 1 128 128 3   
                                exclude_left_mask = self.generate_final_class_labels(next_left_mask_gen).unsqueeze(3).repeat(1, 1, 1, 3)
                                next_left_mask_gen = self.vis_labels(next_left_mask_gen)
                                next_gt_mask1 = self.mask_onehot(next_gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                                next_gt_mask_vis =  self.vis_labels(next_gt_mask1)
                            elif self.use_CEloss == 0:     # Ll1
                                next_render_mask = data['left_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) # 1 128 128 3(rgb label mean-> 0 1 2)
                                next_render_mask = (next_render_mask / 2 + 0.5  )*(self.num_classes-1)    
                                next_render_mask = self.vis_labelsL1(next_render_mask)
                                next_left_mask_gen = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1) # 1 128 128 3
                                next_left_mask_gen = (next_left_mask_gen / 2 + 0.5  )*(self.num_classes-1)           # 1 128 128 3 [0,2]
                                exclude_left_mask = self.generate_final_class_labels_L1(next_left_mask_gen)
                                next_left_mask_gen = self.vis_labelsL1(next_left_mask_gen)
                                next_gt_mask_vis = self.vis_labelsL1(next_gt_mask_label)
                            exclude_left_mask = exclude_left_mask * next_render_novel

                            # left gen mask
                            #  4 RGB loss_dyna_leader leader  
                            if ('xyz_maps' in data['right_next']):
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            # 5 Mask loss_dyna_mask_next_right 
                                # next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                            data['right_next'] =self.pts2render_mask(data['right_next'], bg_color=self.bg_mask)
                            next_render_mask_right = data['right_next']['novel_view']['mask_pred'].permute(0, 2, 3, 1) 
                            if self.use_CEloss==1 or self.use_CEloss == 7 or self.use_CEloss == 21:
                                next_render_mask_right = self.vis_labels(next_render_mask_right)
                            elif self.use_CEloss == 0:     # Ll1
                                next_render_mask_right = (next_render_mask_right / 2 + 0.5  )*(self.num_classes-1) 
                                next_render_mask_right = self.vis_labelsL1(next_render_mask_right)

                    elif self.mask_gen == 'None':
                        if self.use_dynamic_field :
                            # 2 GRB total(Follower)   loss_dyna_follower  RGB Loss   
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]

                            #  4 RGB loss_dyna_leader leader  
                            if ('xyz_maps' in data['right_next']):
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                    elif self.mask_gen == 'nonerf':
                        data =self.pts2render_mask_gen(data, bg_color=self.bg_mask)
                        if  self.use_CEloss == 0: # l1
                            render_mask_novel = data['novel_view']['mask_gen'].permute(0, 2, 3, 1) # 1 128 128 3
                                # render_mask_novel = (render_mask_novel / 2 + 0.5)*(self.num_classes-1)  # [0 , 2]
                            # render_mask_novel = render_mask_novel / 2 + 0.5
                            print("render_mask_novel",render_mask_novel.shape,render_mask_novel) # 1 128 128 3
                            render_mask_gtrgb = self.generate_final_class_labels_L1(render_mask_novel)
                            next_render_mask_right =render_mask_novel/2+0.5
                            next_gt_mask_vis = self.vis_labelsL1_1(render_mask_novel)
                            exclude_left_mask = self.vis_labelsL1_2(render_mask_novel)
                            next_render_mask = self.vis_labelsL1_3(render_mask_novel)
                            next_render_rgb_right = render_mask_novel.mean(dim=-1)/2+0.5
                            render_mask_novel = self.vis_labelsL1(render_mask_novel)       
                            
                            # vis gt left 
                            gt_mask_vis = self.vis_labelsL1(gt_mask_label)

                            # next_render_rgb_right  = self.generate_final_class_labels_L1(gt_mask_label)
                            render_mask_gtrgb = gt_rgb * render_mask_gtrgb
                            
                            # render_mask_gtrgb = gt_rgb * render_mask_novel
                            # render_mask_novel = render_novel * render_mask_novel
                        else:
                            # gt_mask_label = self.mask_onehot(gt_mask)
                            render_mask_novel = data['novel_view']['mask_gen'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]                           
                            # # print("1",render_mask_novel.shape,render_mask_novel) # [1,256,256,3]
                            # render_mask_novel = self.generate_final_class_labels(render_mask_novel)
                            # # print("2",render_mask_novel.shape,render_mask_novel) # [1,3, 256,256]
                            # render_mask_novel = render_mask_novel.unsqueeze(3).repeat(1, 1, 1, 3)
                            # # render_mask_gtrgb = gt_rgb * render_mask_novel
                            # render_mask_novel = render_novel * render_mask_novel 

                                # render_mask_novel = data['novel_view']['mask_pred'].permute(0, 2, 3, 1)
                                ## !! render_mask_novel = render_novel * render_mask_gtrgb
                            render_mask_gtrgb = self.generate_final_class_labels(render_mask_novel).unsqueeze(3).repeat(1, 1, 1, 3)     # 1 128 128 3 [TTT/FFF]                        
                            # print("gt_rgb",gt_rgb.shape,render_mask_gtrgb.shape) # [1,128,128,3]
                            render_mask_gtrgb = gt_rgb * render_mask_gtrgb # 1 256 256 3
                            # vis render
                            next_gt_mask_vis = render_mask_novel[:,:,:,0]
                            exclude_left_mask = render_mask_novel[:,:,:,1]
                            next_render_mask = render_mask_novel[:,:,:,2]
                            next_render_rgb_right = render_mask_novel.mean(dim=-1)/2+0.5
                            render_mask_novel = self.vis_labels(render_mask_novel)
                                
                            gt_mask1 = self.mask_onehot(gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                            gt_mask_vis =  self.vis_labels(gt_mask1)
                            # data =self.pts2render_rgb(data, bg_color=self.bg_color)
                            # next_render_mask = data['mask_view']['rgb_pred'].permute(0, 2, 3, 1)

                        if self.use_dynamic_field and (next_gt_rgb is not None and gt_mask is not None):
                                # next_gt_rgb = next_gt_rgb.permute(0,2,3,1)
                                # next_gt_mask = next_gt_mask.permute(0,2,3,1)
                                # next_gt_mask_label = self.mask_onehot(next_gt_mask)
                            # device = gt_mask.device  
                            # gt_mask_label = gt_mask_label.to(device)
                            # next_gt_mask_label = next_gt_mask_label.to(device)
                            # 1  loss_dyna_mask_novel

                            # 2 GRB total(Follower)   
                            if ('xyz_maps' in data['left_next']):
                                data['left_next'] = self.pts2render(data['left_next'], bg_color=self.bg_color)
                                next_render_novel = data['left_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                                # loss_dyna_follower = l2_loss(next_render_novel, next_gt_rgb)

                            # 3 next mask train (pre - left(mask) ) next_loss_dyna_mask_left   Mask Loss
                            data['left_next'] =self.pts2render_mask_gen(data['left_next'], bg_color=self.bg_mask)
                            next_render_mask = data['left_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1) # [1,3,128, 128] -> [1,128, 128, 3]
                            next_render_mask = self.vis_labels(next_render_mask)
                            exclude_left_mask = self.generate_final_class_labels(next_render_mask)
                            exclude_left_mask = exclude_left_mask.unsqueeze(3).repeat(1, 1, 1, 3)
                            next_gt_mask1 = self.mask_onehot(next_gt_mask) # 1 128 128 1[1,200] -> 1 128 128 3[100 010 001] 
                            next_gt_mask_vis =  self.vis_labels(next_gt_mask1)

                            #  4 RGB loss_dyna_leader leader  
                            if ('xyz_maps' in data['right_next']):
                                data['right_next'] = self.pts2render(data['right_next'], bg_color=self.bg_color)
                                next_render_rgb_right = data['right_next']['novel_view']['img_pred'].permute(0, 2, 3, 1) # [1,128, 128, 3]
                            
                            # 5 Mask loss_dyna_mask_next_right 
                            data['right_next'] =self.pts2render_mask_gen(data['right_next'], bg_color=self.bg_color)
                            next_render_mask_right = data['right_next']['novel_view']['mask_gen'].permute(0, 2, 3, 1)
                            next_render_mask_right = self.vis_labels(next_render_mask_right)

                loss_dict = {
                    'loss': 0.,
                    'loss_rgb': 0.,
                    'loss_embed': 0.,
                    'loss_dyna': 0.,
                    'loss_LF':  0.,
                    'loss_dyna_mask':  0.,
                    'loss_reg': 0.,
                    'l1': 0.,
                    'psnr': 0.,
                }

        # get Gaussian embedding 
        ret_dict = DotMap(render_novel=render_novel, next_render_novel=next_render_novel,
                          render_embed=render_embed, gt_embed=gt_embed, 
                          render_mask_novel = render_mask_novel,           # now mask * render_rgb
                          render_mask_gtrgb = render_mask_gtrgb,           # now mask * gt_rgb
                        next_render_mask = next_render_mask,               # left mask * next_render_novel
                        next_render_mask_right = next_render_mask_right,   # no use rightmask 
                        next_render_rgb_right = next_render_rgb_right,            # rightnext rgb
                        next_left_mask_gen = next_left_mask_gen, 
                        exclude_left_mask =exclude_left_mask,                      # gen left mask
                        gt_mask_vis =gt_mask_vis,
                        next_gt_mask_vis =next_gt_mask_vis,
                        )             
        return loss_dict, ret_dict
    
    def pts2render(self, data: dict, bg_color=[0,0,0]):
        '''use render function in GSZ '''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        i = 0
        xyz_i = data['xyz_maps'][i, :, :]
        feature_i = data['sh_maps'][i, :, :, :] # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :]
        scale_i = data['scale_maps'][i, :, :]
        opacity_i = data['opacity_maps'][i, :, :]
        feature_language_i = data['feature_maps'][i, :, :]  # [B, N, 3]   [1, 65536, 3]  

        #  from agents.manigaussian_bc2.gaussian_renderer import render
        render_return_dict = render(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i
            )

        data['novel_view']['img_pred'] = render_return_dict['render'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data

    def pts2render1(self, data: dict, bg_color=[0,0,0]):
        '''(MASK) feature_language_i is mask'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        i = 0
        xyz_i = data['xyz_maps'][i, :, :].detach()
        feature_i = data['sh_maps'][i, :, :, :].detach() # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :].detach()
        scale_i = data['scale_maps'][i, :, :].detach()
        opacity_i = data['opacity_maps'][i, :, :].detach()
        feature_language_i = data['mask_maps'][i, :, :]  # [B, N, 3]   [1, 65536, 3]  

        render_return_dict = render1(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i
            )

        data['novel_view']['img_pred'] = render_return_dict['render'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data

    def pts2render_mask(self, data: dict, bg_color=[0,0,0]):
        '''precomputed_mask_i  MASK'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        i = 0
        xyz_i = data['xyz_maps'][i, :, :].detach() 
        feature_i = data['sh_maps'][i, :, :, :].detach()  # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :].detach() 
        scale_i = data['scale_maps'][i, :, :].detach() 
        opacity_i = data['opacity_maps'][i, :, :].detach() 
        precomputed_mask_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        feature_language_i = data['feature_maps'][i, :, :].detach()   # [B, N, 3]   [1, 65536, 3]  
        
        render_return_dict = render_mask(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i,
            precomputed_mask = precomputed_mask_i,
            )

        data['novel_view']['mask_pred'] = render_return_dict['mask'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data

    def pts2render_for_MASK_IN_NERF(self, data: dict, bg_color=[0,0,0]):
        '''precomputed_mask_i  for MASK_IN_NERF'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        i = 0
        xyz_i = data['xyz_maps'][i, :, :] #.detach() 
        feature_i = data['sh_maps'][i, :, :, :] #.detach()  # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :] #.detach() 
        scale_i = data['scale_maps'][i, :, :] #.detach() 
        opacity_i = data['opacity_maps'][i, :, :] #.detach() 
        # precomputed_mask_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        feature_language_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        # feature_language_i = data['feature_maps'][i, :, :].detach()   # [B, N, 3]   [1, 65536, 3]  
        
        render_return_dict = render(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i,
            # precomputed_mask = precomputed_mask_i,
            )

        data['novel_view']['mask_pred'] = render_return_dict['render'].unsqueeze(0)
        # data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data


    def pts2render_rgb(self, data: dict, bg_color=[0,0,0]):
        """RGB"""
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        i = 0
        xyz_i = data['xyz_maps'][i, :, :].detach() 
        feature_i = data['sh_maps'][i, :, :, :].detach()  # [16384, 4, 3]
        rot_i = data['rot_maps'][i, :, :].detach() 
        scale_i = data['scale_maps'][i, :, :].detach() 
        opacity_i = data['opacity_maps'][i, :, :].detach() 
        feature_language_i = data['feature_maps'][i, :, :].detach()   # [B, N, 3]   [1, 65536, 3]  
        
        render_return_dict = render_rgb(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i,
            )

        data['mask_view']['rgb_pred'] = render_return_dict['render'].unsqueeze(0)
        return data

    def pts2render_mask_gen(self, data: dict, bg_color=[0,0,0]):
        '''GEN mask | in order to exclude LEFT HAND'''
        bs = data['intr'].shape[0]
        assert bs == 1, "batch size should be 1"
        i = 0
        xyz_i = data['xyz_maps'][i, :, :]#.detach()       # [65536, 3]
        feature_i = data['sh_maps'][i, :, :, :]#.detach()  # [16384(256 * 256), 4, 3]
        rot_i = data['rot_maps'][i, :, :]#.detach() 
        scale_i = data['scale_maps'][i, :, :]#.detach() 
        opacity_i = data['opacity_maps'][i, :, :]#.detach() 
        precomputed_mask_i = data['mask_maps'][i, :, :] # mask  [1, 65536, 3]
        feature_language_i = data['feature_maps'][i, :, :]#.detach()   # [B, N, 3]   [1, 65536, 3]  
          
        render_return_dict = render_mask_gen(
            data, i, xyz_i, rot_i, scale_i, opacity_i, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_i, features_language=feature_language_i,
            precomputed_mask = precomputed_mask_i,
            )

        data['novel_view']['mask_gen'] = render_return_dict['mask'].unsqueeze(0)
        data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return data

    def createby_gt_mask(self, data: dict, gt_mask=None, gt_mask_camera_extrinsic=None, gt_mask_camera_intrinsic=None, next_gt_mask = None,
                gt_maskdepth=None,next_gt_maskdepth=None):
        front_intrinsic = gt_mask_camera_intrinsic[0] 
        overhead_intrinsic = gt_mask_camera_intrinsic[1]

        mask_3d = None  
        next_front_mask = next_gt_mask[0]
        next_overhead_mask = next_gt_mask[1]
        next_front_depth = next_gt_maskdepth[0]
        next_overhead_depth = next_gt_maskdepth[1]

        next_leftxyz_front = depth_mask_to_3d(next_front_depth,next_front_mask,front_intrinsic)
        
        next_leftxyz_overhead = depth_mask_to_3d(next_overhead_depth,next_overhead_mask,overhead_intrinsic)

        # GPU 
        next_leftxyz = merge_tensors(next_leftxyz_front, next_leftxyz_overhead).cpu()

        if len(next_leftxyz) > 0:
            next_mask_3d = points_inside_convex_hull( data['xyz'][0].detach(), next_leftxyz)

        return mask_3d, next_mask_3d

    def generate_final_class_labels(self,next_left_mask_gen):
        """
        exclude left 1 128 128 3 -> T/F
        """
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        exclude_left_mask = class_indices != 2 
        return exclude_left_mask

    def generate_final_class_labels_ce1(self,next_left_mask_gen):
        """
        exclude bg loss(label-1)  1 128 128 3 -> 1 128 128 2 -> output: 1 128 128 [T:1 / F:0]
        """
        # exclude_left_mask = (next_left_mask_gen > 2.5) | (next_left_mask_gen < 1.5) 
        next_left_mask_gen = next_left_mask_gen[:,:,:,[0,1]]
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        exclude_left_mask = class_indices != 1  

        return exclude_left_mask

    def generate_final_class_labels_L1(self,next_left_mask_gen):
        """Is Left? | [1 128 128 3] -> [1 128 128 3] [True False]       """
        class_indices = next_left_mask_gen.mean(dim=-1)
        print("class_indices = ", class_indices.shape,class_indices)
        next_left_mask_gen = next_left_mask_gen.squeeze(-1)
        exclude_left_mask = torch.ones_like(next_left_mask_gen, dtype=torch.float32)
        exclude_left_mask = (class_indices < 1.5) 
        exclude_left_mask = exclude_left_mask.unsqueeze(-1).repeat(1,1,1,3)   
        print("exclude_left_mask",exclude_left_mask)
        # exclude_left_mask = class_indices != 2 
        return exclude_left_mask
    
    def mask_onehot(self,mask):
        """1 128 128 1 -> [1 128 128 3] 100 010 001"""
        right_min, right_max, left_min, left_max = 53, 73, 94, 114
        # gt_mask1 = mask.squeeze(-1)           # [1,256,256,1] -> [1,256,256] 
        gt_mask1 = mask[:,:,:,0].long() 
        print("def mask_onehot mask origin shape",mask.shape,"->",gt_mask1.shape)
        # init onhot bg:[0,0,0]   right:[0,1,0]  left:[0,0,1]
        gt_mask_label = torch.zeros((*gt_mask1.shape, self.num_classes), dtype=torch.float32) 
        bg_mask = (gt_mask1 < right_min) | ((gt_mask1 > right_max) & (gt_mask1 < left_min)) | (gt_mask1 > left_max)
        gt_mask_label[bg_mask] = torch.tensor([1, 0, 0], dtype=torch.float32)
        right_mask = (gt_mask1 > right_min - 1) & (gt_mask1 < right_max + 1)
        gt_mask_label[right_mask] = torch.tensor([0, 1, 0], dtype=torch.float32)                   
        left_mask = (gt_mask1 > left_min - 1) & (gt_mask1 < left_max + 1)
        gt_mask_label[left_mask] = torch.tensor([0, 0, 1], dtype=torch.float32) 

        return gt_mask_label

    def mask_id_to_color(self, mask):
        """
        mask [1, H, W, 1] ——ID mask
        RGB[1, H, W, 3]
        """
        right_min, right_max, left_min, left_max = 53, 73, 94, 114

        mask = mask[0, :, :, 0].long()

        h, w = mask.shape
        color_mask = torch.zeros((h, w, 3), dtype=torch.uint8)

        right_mask = (mask >= right_min) & (mask <= right_max)
        color_mask[right_mask] = torch.tensor([255, 0, 0], dtype=torch.uint8)

        left_mask = (mask >= left_min) & (mask <= left_max)
        color_mask[left_mask] = torch.tensor([0, 0, 255], dtype=torch.uint8)

        # [H, W, 3] -> [1, H, W, 3]
        color_mask = color_mask.unsqueeze(0)
        return color_mask


    def mask_label_onehot(self,gt_mask):
        """[1 128 128 1] -> [1 128 128]  mask 0:bg    1:ritght    2:left"""
        right_min, right_max, left_min, left_max = 53, 73, 94, 114

        gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.long)
        gt_mask_label[(gt_mask > right_min-1) & (gt_mask < right_max+1)] = 1
        gt_mask_label[(gt_mask > left_min-1) & (gt_mask < left_max+1)] = 2
        gt_mask_label = gt_mask_label.squeeze(-1)

        
        return gt_mask_label

    def mask_label(self,gt_mask):
        """[1 128 128 1] -> [1 128 128 3] [000 / 111 / 222]"""
        right_min, right_max, left_min, left_max = 53, 73, 94, 114
        #0:bg    1:ritght    2:left
        gt_mask = gt_mask.squeeze(-1)
        gt_mask_label = torch.zeros_like(gt_mask, dtype=torch.float32) - 1.0
        gt_mask_label[(gt_mask > right_min-1) & (gt_mask < right_max+1)] = 0.0
        gt_mask_label[(gt_mask > left_min-1) & (gt_mask < left_max+1)] = 1.0 
        gt_mask_label = gt_mask_label.unsqueeze(-1).repeat(1,1,1,3)         
        return gt_mask_label

    def vis_labels(self,next_left_mask_gen):
        """
        vis [1 128 128 3] - > rgb  (max)
        """
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        print("class_indices = ", class_indices.shape,class_indices)

        color_image = torch.zeros((*class_indices.shape, 3), dtype=torch.uint8)
        color_image[class_indices == 0] = torch.tensor([0, 0, 0], dtype=torch.uint8)  
        color_image[class_indices == 1] = torch.tensor([255, 0, 0], dtype=torch.uint8)  
        color_image[class_indices == 2] = torch.tensor([0, 0, 255], dtype=torch.uint8) 

        return color_image

    def vis_labels_ce1(self,next_left_mask_gen):
        """
        (vis_labels - bg | -bgloss) vis [1 128 128 3] -> [1 128 128 2] - > rgb
        """
        next_left_mask_gen = next_left_mask_gen[:,:,:,[0,1]]
        class_indices = torch.argmax(next_left_mask_gen, dim=-1)
        print("class_indices1 128 128 2 = ", class_indices.shape,class_indices)

        color_image = torch.zeros((*class_indices.shape, 3), dtype=torch.uint8)
        # color_image[class_indices == 0] = torch.tensor([0, 0, 0], dtype=torch.uint8)  # BG 
        color_image[class_indices == 0] = torch.tensor([255, 0, 0], dtype=torch.uint8)  # RIGHT
        color_image[class_indices == 1] = torch.tensor([0, 0, 255], dtype=torch.uint8)  # LEFT
        return color_image

    def vis_labelsL1(self,mask):
        """
        vis [1 128 128 3] (ave)- > rgb
        """
        mask_mean = mask.mean(dim=-1)   # average
        # mask_mean = mask[:,:,:,0]     
        print("vis_labelsL1 mask_mean =",mask_mean.shape,mask_mean)
        mask_rgb = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=torch.uint8)
        # [0, 2]
        # mask_rgb[(mask_mean >= 0) & (mask_mean < 0.5)] = torch.tensor([0, 0, 0], dtype=torch.uint8)    
        # mask_rgb[(mask_mean >= 0.7) & (mask_mean < 1.3)] = torch.tensor([255, 0, 0], dtype=torch.uint8) 
        # mask_rgb[mask_mean >= 1.3] = torch.tensor([0, 0, 255], dtype=torch.uint8)                       # Green 
        # [-1, 1]
        mask_rgb[(mask_mean >= -0.4) & (mask_mean < 0.4)] = torch.tensor([255, 0, 0], dtype=torch.uint8) 
        mask_rgb[mask_mean >= 0.4] = torch.tensor([0, 0, 255], dtype=torch.uint8)                       # Green 

        return mask_rgb

    def vis_labelsL1_1(self,mask):
        """
        vis [1 128 128 3] - > rgb
        """
        # mask_mean = mask.mean(dim=-1)
        mask_mean = mask[:,:,:,0]
        print("vis_labelsL1 mask_mean =",mask_mean.shape,mask_mean)
        mask_rgb = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=torch.uint8)
        mask_rgb[(mask_mean >= -0.4) & (mask_mean < 0.4)] = torch.tensor([255, 0, 0], dtype=torch.uint8) 
        mask_rgb[mask_mean >= 0.4] = torch.tensor([0, 0, 255], dtype=torch.uint8)                       # Green 

        return mask_rgb

    def vis_labelsL1_2(self,mask):
        """
        vis [1 128 128 3]  - > rgb
        """
        # mask_mean = mask.mean(dim=-1)
        mask_mean = mask[:,:,:,1]
        print("vis_labelsL1 mask_mean =",mask_mean.shape,mask_mean)
        mask_rgb = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=torch.uint8)
        mask_rgb[(mask_mean >= -0.4) & (mask_mean < 0.4)] = torch.tensor([255, 0, 0], dtype=torch.uint8) 
        mask_rgb[mask_mean >= 0.4] = torch.tensor([0, 0, 255], dtype=torch.uint8)                       # Green 

        return mask_rgb

    def vis_labelsL1_3(self,mask):
        """
        vis [1 128 128 3] - > rgb
        """
        mask_mean = mask[:,:,:,2]
        print("vis_labelsL1 mask_mean =",mask_mean.shape,mask_mean)
        mask_rgb = torch.zeros((mask.shape[0], mask.shape[1], mask.shape[2], 3), dtype=torch.uint8)
        mask_rgb[(mask_mean >= -0.4) & (mask_mean < 0.4)] = torch.tensor([255, 0, 0], dtype=torch.uint8) # red
        mask_rgb[mask_mean >= 0.4] = torch.tensor([0, 0, 255], dtype=torch.uint8)                       # Green 

        return mask_rgb

    def one_hot_encode(self,mask, num_classes):
        "NO USE b h w -> b c h w c=(100 010 001)"
        one_hot_mask = torch.nn.functional.one_hot(mask, num_classes=num_classes)    # (batch_size, num_classes, height, width)
        one_hot_mask = one_hot_mask.permute(0, 3, 1, 2)  
        return one_hot_mask.float()  
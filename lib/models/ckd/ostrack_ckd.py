"""
OSTrack with Cross-modal Knowledge Distillation (CKD) - RealHigh Version

RealHigh Version: Uses real high-resolution input for teachers,
then down-samples to low-resolution for students in vit_ce_realhigh.forward().
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.layers.head_dfl import build_box_head_dfl  # 新增：DFL head
from .vit import vit_base_patch16_224, resize_pos_embed
from .vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce, vit_small_patch16_224_ce, vit_tiny_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from greenlet import greenlet


def convert_lora_weights(state_dict, use_lora=False):
    """
    Convert old checkpoint weights to LoRA-compatible format
    
    Old format: conv5_ctr.weight, conv5_ctr.bias
    New LoRA format: conv5_ctr.conv.weight, conv5_ctr.conv.bias, conv5_ctr.lora_A, conv5_ctr.lora_B
    
    Args:
        state_dict: checkpoint state dict
        use_lora: whether current model uses LoRA
    
    Returns:
        converted state dict
    """
    if not use_lora:
        return state_dict
    
    converted = {}
    lora_keys = ['conv5_ctr', 'conv5_offset', 'conv5_size']
    
    for key, value in state_dict.items():
        converted_key = key
        
        # Check if this is a head conv5 layer that needs conversion
        for lora_key in lora_keys:
            # box_head.conv5_ctr.weight -> box_head.conv5_ctr.conv.weight
            if f'.{lora_key}.weight' in key and '.conv.' not in key:
                converted_key = key.replace(f'.{lora_key}.weight', f'.{lora_key}.conv.weight')
                print(f"  [LoRA Weight Conversion] {key} -> {converted_key}")
                break
            elif f'.{lora_key}.bias' in key and '.conv.' not in key:
                converted_key = key.replace(f'.{lora_key}.bias', f'.{lora_key}.conv.bias')
                print(f"  [LoRA Weight Conversion] {key} -> {converted_key}")
                break
        
        converted[converted_key] = value
    
    return converted


class _DummyGreenlet:
    """Simple stand-in for greenlet.switch calls when teachers are absent.

    Its switch() accepts any args and returns an empty list which the
    vit_ce code treats as an empty attention list.
    """
    def switch(self, *args, **kwargs):
        return []


class OSTrack_CKD(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, rgb_branch, tir_branch, teacher_rgb, teacher_tir, box_head, box_head_v, box_head_i, \
                 aux_loss=False, head_type="CORNER",mask_probability=0.0, mask_ratio=0.0, training=True, has_teacher=True):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.rgb_branch = rgb_branch
        self.tir_branch = tir_branch
        self.rgb_branch.mask_probability = mask_probability
        self.tir_branch.mask_probability = mask_probability
        self.rgb_branch.mask_ratio = mask_ratio
        self.tir_branch.mask_ratio = mask_ratio
        self.box_head = box_head
        if training:
            self.teacher_rgb = teacher_rgb
            self.teacher_tir = teacher_tir
            self.box_head_v = box_head_v
            self.box_head_i = box_head_i
        else:
            self.teacher_rgb = None
            self.teacher_tir = None
            self.box_head_v = None
            self.box_head_i = None

        self.mask_probability = mask_probability
        self.mask_ratio = mask_ratio

        self.aux_loss = aux_loss
        self.head_type = head_type
        self.has_teacher = has_teacher
        if head_type == "CORNER" or head_type == "CENTER" or head_type == "CENTER_DFL":
            # 学生网络的特征尺寸
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
            
            # RealHigh: 教师网络的特征尺寸（高分辨率）
            # 教师: search_size=256, patch_size=16 -> 256/16=16
            # 学生: search_size=128, patch_size=16 -> 128/16=8
            if training and has_teacher:
                self.feat_sz_t = int(box_head_v.feat_sz)  # 教师的特征尺寸
                self.feat_len_t = int(box_head_v.feat_sz ** 2)
            else:
                self.feat_sz_t = self.feat_sz_s
                self.feat_len_t = self.feat_len_s

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)



    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        """
        RealHigh版本forward:
        - 输入是真实高分辨率 (128x128 template, 256x256 search)
        - 教师网络直接使用高分辨率
        - 在vit_ce_realhigh的forward中,学生网络会自动下采样到低分辨率
        """

        if self.training:
            if self.has_teacher:
                # original greenlet-based teacher-student chain
                teacher_rgb_gr = greenlet(self.teacher_rgb)
                teacher_tir_gr = greenlet(self.teacher_tir)
                rgb_branch_gr = greenlet(self.rgb_branch)
                tir_branch_gr = greenlet(self.tir_branch)
                self.teacher_rgb.next_gr[0] = teacher_tir_gr
                self.teacher_tir.next_gr[0] = rgb_branch_gr
                self.rgb_branch.next_gr[0] = tir_branch_gr
                self.tir_branch.next_gr[0] = teacher_rgb_gr
                
                # RealHigh: 直接传入真实高分辨率
                # template: (4, B, C, 128, 128) - 真实高分辨率
                # search: (4, B, C, 256, 256) - 真实高分辨率
                # 学生网络在vit_ce_realhigh.forward中会自动下采样
                t_x_rgb, t_aux_dict_rgb = teacher_rgb_gr.switch(z_li=template, x_li=search,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, )

                t_x_tir, t_aux_dict_tir = teacher_tir_gr.switch()

                x_rgb, aux_dict_rgb = rgb_branch_gr.switch()

                x_tir, aux_dict_tir = tir_branch_gr.switch()

                aux_dict = {
                    'x_rgb': x_rgb,
                    'x_tir': x_tir,
                    't_x_rgb': t_x_rgb,
                    't_x_tir': t_x_tir,
                    'aux_dict_rgb': aux_dict_rgb,
                    'aux_dict_tir': aux_dict_tir,
                    'aux_dict_t_rgb': t_aux_dict_rgb,
                    'aux_dict_t_tir': t_aux_dict_tir,
                }
                x = torch.cat([x_rgb, x_tir], 2)
                # Forward head
                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]

                out = self.forward_head(feat_last, None)
                out_t_tir = self.forward_head(t_x_tir, None, head=self.box_head_i)
                out['out_t_tir'] = out_t_tir
                out_t_rgb = self.forward_head(t_x_rgb, None, head=self.box_head_v)
                out['out_t_rgb'] = out_t_rgb

                out.update(aux_dict)
                out['backbone_feat'] = x
                return out
            else:
                # No-teacher (nokd) simplified path: call student branches directly.
                # Make sure their next_gr is a dummy object so internal switch() calls
                # inside the backbone won't raise AttributeError.
                if getattr(self.rgb_branch, 'next_gr', None) is None:
                    self.rgb_branch.next_gr = [None]
                if getattr(self.tir_branch, 'next_gr', None) is None:
                    self.tir_branch.next_gr = [None]
                if self.rgb_branch.next_gr[0] is None:
                    self.rgb_branch.next_gr[0] = _DummyGreenlet()
                if self.tir_branch.next_gr[0] is None:
                    self.tir_branch.next_gr[0] = _DummyGreenlet()

                x_rgb, aux_dict_rgb = self.rgb_branch.forward(z_li=template, x_li=search,
                                                            ce_template_mask=ce_template_mask,
                                                            ce_keep_rate=ce_keep_rate,
                                                            return_last_attn=return_last_attn)
                x_tir, aux_dict_tir = self.tir_branch.forward(z_li=template, x_li=search,
                                                              ce_template_mask=ce_template_mask,
                                                              ce_keep_rate=ce_keep_rate,
                                                              return_last_attn=return_last_attn)

                # Treat student outputs as teacher outputs for compatibility with downstream code
                t_x_rgb, t_aux_dict_rgb = x_rgb, aux_dict_rgb
                t_x_tir, t_aux_dict_tir = x_tir, aux_dict_tir

                aux_dict = {
                    'x_rgb': x_rgb,
                    'x_tir': x_tir,
                    't_x_rgb': t_x_rgb,
                    't_x_tir': t_x_tir,
                    'aux_dict_rgb': aux_dict_rgb,
                    'aux_dict_tir': aux_dict_tir,
                    'aux_dict_t_rgb': t_aux_dict_rgb,
                    'aux_dict_t_tir': t_aux_dict_tir,
                }

                x = torch.cat([x_rgb, x_tir], 2)
                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]

                out = self.forward_head(feat_last, None)
                out_t_tir = self.forward_head(t_x_tir, None, head=self.box_head_i)
                out['out_t_tir'] = out_t_tir
                out_t_rgb = self.forward_head(t_x_rgb, None, head=self.box_head_v)
                out['out_t_rgb'] = out_t_rgb

                out.update(aux_dict)
                out['backbone_feat'] = x
                return out
        
        else:
            if self.has_teacher:
                rgb_branch_gr = greenlet(self.rgb_branch)
                tir_branch_gr = greenlet(self.tir_branch)
                self.rgb_branch.next_gr[0] = tir_branch_gr
                self.tir_branch.next_gr[0] = rgb_branch_gr
                x_rgb, aux_dict_rgb = rgb_branch_gr.switch(z_li=template[:2], x_li=search[:2],
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, )

                x_tir, aux_dict_tir = tir_branch_gr.switch()

                aux_dict = {
                    'aux_dict_rgb': aux_dict_rgb,
                    'aux_dict_tir': aux_dict_tir}
                x = torch.cat([x_rgb, x_tir], 2)
                # Forward head
                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]
                out = self.forward_head(feat_last, None)

                out.update(aux_dict)
                out['backbone_feat'] = x
                return out
            else:
                if getattr(self.rgb_branch, 'next_gr', None) is None:
                    self.rgb_branch.next_gr = [None]
                if getattr(self.tir_branch, 'next_gr', None) is None:
                    self.tir_branch.next_gr = [None]
                if self.rgb_branch.next_gr[0] is None:
                    self.rgb_branch.next_gr[0] = _DummyGreenlet()
                if self.tir_branch.next_gr[0] is None:
                    self.tir_branch.next_gr[0] = _DummyGreenlet()

                x_rgb, aux_dict_rgb = self.rgb_branch.forward(z_li=template[:2], x_li=search[:2],
                                                              ce_template_mask=ce_template_mask,
                                                              ce_keep_rate=ce_keep_rate,
                                                              return_last_attn=return_last_attn)
                x_tir, aux_dict_tir = self.tir_branch.forward(z_li=template[:2], x_li=search[:2],
                                                              ce_template_mask=ce_template_mask,
                                                              ce_keep_rate=ce_keep_rate,
                                                              return_last_attn=return_last_attn)

                aux_dict = {'aux_dict_rgb': aux_dict_rgb, 'aux_dict_tir': aux_dict_tir}
                x = torch.cat([x_rgb, x_tir], 2)
                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]
                out = self.forward_head(feat_last, None)
                out.update(aux_dict)
                out['backbone_feat'] = x
                return out

    def forward_head(self, cat_feature, gt_score_map=None, head=None):
        """
        RealHigh版本: 根据是否为教师网络head，使用不同的特征尺寸
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if head==None:
            box_head = self.box_head
            # 学生融合head，使用学生尺寸
            feat_len = self.feat_len_s
            feat_sz = self.feat_sz_s
        else:
            box_head = head
            # 教师head，使用教师尺寸（高分辨率）
            feat_len = self.feat_len_t
            feat_sz = self.feat_sz_t
            
        enc_opt = cat_feature[:, -feat_len:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, feat_sz, feat_sz)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER" or self.head_type == "CENTER_DFL":
            # run the center head (支持 DFL)
            head_output = box_head(opt_feat, gt_score_map)
            
            # DFL head 返回 5 个值，原始 head 返回 4 个值
            if len(head_output) == 5:
                # DFL head: (score_map_ctr, bbox, size_map, offset_map, offset_dist)
                score_map_ctr, bbox, size_map, offset_map, offset_dist = head_output
            else:
                # 原始 head: (score_map_ctr, bbox, size_map, offset_map)
                score_map_ctr, bbox, size_map, offset_map = head_output
                offset_dist = None
            
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            
            # 如果有 offset_dist，添加到输出（DFL loss 需要）
            if offset_dist is not None:
                out['offset_dist'] = offset_dist
            
            return out
        else:
            raise NotImplementedError


def build_ostrack_ckd(cfg, training=True):
    from lib.config.ckd.config import clone_config
    patch_start_index = 1
    
    # 创建教师网络使用的高分辨率配置
    teacher_cfg = clone_config(cfg)
    # import pdb;pdb.set_trace()
    # 设置教师网络使用高分辨率
    teacher_cfg.DATA.TEMPLATE.SIZE = cfg.TEACHER_SIZE.TEMPLATE.SIZE  # 从配置读取教师分辨率
    teacher_cfg.DATA.SEARCH.SIZE = cfg.TEACHER_SIZE.SEARCH.SIZE     # 从配置读取教师分辨率
    
    # 教师网络始终使用原始 CENTER head（不使用 DFL）
    teacher_cfg.MODEL.HEAD.TYPE = 'CENTER'
    teacher_cfg.MODEL.HEAD.USE_DFL = False
    
    # 根据配置选择模型架构
    model_constructors = {
        'vit_base_patch16_224_ce': vit_base_patch16_224_ce,
        'vit_small_patch16_224_ce': vit_small_patch16_224_ce,
        'vit_tiny_patch16_224_ce': vit_tiny_patch16_224_ce,
        'vit_large_patch16_224_ce': vit_large_patch16_224_ce,
    }
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    if backbone_type not in model_constructors:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Available: {list(model_constructors.keys())}")
    
    model_constructor = model_constructors[backbone_type]
    
    # 学生网络直接使用cfg（已经是学生分辨率）
    # RGB学生    
    rgb_branch = model_constructor(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                   ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                   ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                   is_teacher=False)  # 明确指定为学生模型
    rgb_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # TIR学生
    if cfg.MODEL.SHARE_STUDENT:    
        tir_branch = rgb_branch
    else:
        tir_branch = model_constructor(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                       ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                       ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                       is_teacher=False)  # 明确指定为学生模型
        tir_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # RGB教师 (always use base model for teacher)
    if cfg.MODEL.RGB_TEACHER:
        teacher_rgb = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            is_teacher=True)  # 明确指定为教师模型
        teacher_rgb.finetune_track(cfg=teacher_cfg, patch_start_index=patch_start_index)  # 使用高分辨率配置
        # 教师始终使用原始 head（不用 DFL）
        box_head_v = build_box_head(teacher_cfg, teacher_rgb.embed_dim)  # RealHigh: 使用teacher_cfg以获取正确的feat_sz
    else:
        teacher_rgb = None
        # 在nokd模式下,使用学生网络的维度来构建box head
        box_head_v = build_box_head(cfg, rgb_branch.embed_dim)
    
    # TIR教师 (always use base model for teacher)
    if cfg.MODEL.TIR_TEACHER:
        teacher_tir = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            is_teacher=True)  # 明确指定为教师模型
        teacher_tir.finetune_track(cfg=teacher_cfg, patch_start_index=patch_start_index)  # 使用高分辨率配置
        # 教师始终使用原始 head（不用 DFL）
        box_head_i = build_box_head(teacher_cfg, teacher_tir.embed_dim)  # RealHigh: 使用teacher_cfg以获取正确的feat_sz
    else:
        teacher_tir = None
        # 在nokd模式下,使用学生网络的维度来构建box head
        box_head_i = build_box_head(cfg, tir_branch.embed_dim)
    
    # 融合的跟踪头 - 使用学生配置（cfg已经是学生分辨率）
    # teacher_tir or teacher_rgb may be None (e.g. in nokd configs). Compute fused
    # embedding dimension from available branches (prefer teacher dims if present,
    # otherwise fall back to student branch dims).
    try:
        embed_v = teacher_rgb.embed_dim if teacher_rgb is not None else rgb_branch.embed_dim
    except Exception:
        embed_v = rgb_branch.embed_dim
    try:
        embed_i = teacher_tir.embed_dim if teacher_tir is not None else tir_branch.embed_dim
    except Exception:
        embed_i = tir_branch.embed_dim
    
    # 根据 HEAD TYPE 选择构建函数
    head_type = getattr(cfg.MODEL.HEAD, 'TYPE', 'CENTER')
    if head_type == 'CENTER_DFL':
        # 使用 DFL head
        print("[OSTrack_CKD] Using DFL head for student")
        box_head = build_box_head_dfl(cfg, embed_v + embed_i)
    else:
        # 使用原始 head
        box_head = build_box_head(cfg, embed_v + embed_i)

    backbone_weight_filter = lambda param_dict : {k.replace("backbone.",""):v for k,v in param_dict.items() if 'backbone' in k}
    boxhead_weight_filter = lambda param_dict : {k.replace("box_head.",""):v for k,v in param_dict.items() if 'box_head' in k}
    def pos_embed_filter(param, is_teacher=False):
        if is_teacher:
            # 对于教师网络，把144->64和576->256的位置编码裁剪出中心区域
            if param['backbone.pos_embed_z'].shape[1]==144:  # 12x12
                param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'].reshape(1, 12, 12, 768)
                param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'][:, 2:-2, 2:-2, :].reshape(1, 64, 768)  # 中心8x8=64区域
            if param['backbone.pos_embed_x'].shape[1]==576:  # 24x24
                param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'].reshape(1, 24, 24, 768)
                param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'][:, 4:-4, 4:-4, :].reshape(1, 256, 768)  # 中心16x16=256区域
            
        else:
            # 对于学生网络，先裁剪到教师尺寸，再resize到更小尺寸
            # 第一步：裁剪到教师尺寸
            if param['backbone.pos_embed_z'].shape[1]==144:  # 12x12
                param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'].reshape(1, 12, 12, 768)
                param['backbone.pos_embed_z'] = param['backbone.pos_embed_z'][:, 2:-2, 2:-2, :].reshape(1, 64, 768)  # 中心8x8=64区域
            if param['backbone.pos_embed_x'].shape[1]==576:  # 24x24
                param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'].reshape(1, 24, 24, 768)
                param['backbone.pos_embed_x'] = param['backbone.pos_embed_x'][:, 4:-4, 4:-4, :].reshape(1, 256, 768)  # 中心16x16=256区域

            # 第二步：从教师尺寸resize到学生尺寸
            template_size = cfg.DATA.TEMPLATE.SIZE // 16  # 16 is patch size
            search_size = cfg.DATA.SEARCH.SIZE // 16

            template_posemb = torch.zeros(1, template_size * template_size, 768)  # 768 is embed_dim
            search_posemb = torch.zeros(1, search_size * search_size, 768)

            param['backbone.pos_embed_z'] = resize_pos_embed(
                param['backbone.pos_embed_z'], 
                posemb_new=template_posemb,
                num_tokens=0,
                gs_new=(template_size, template_size)
            )
            param['backbone.pos_embed_x'] = resize_pos_embed(
                param['backbone.pos_embed_x'], 
                posemb_new=search_posemb,
                num_tokens=0,
                gs_new=(search_size, search_size)
            )
            
            # Add temporal position embeddings if they exist
            if 'backbone.temporal_pos_embed_z' in param:
                print("  [Student] Resizing temporal template position embedding...")
                if param['backbone.temporal_pos_embed_z'].shape[1]==144:
                    param['backbone.temporal_pos_embed_z'] = param['backbone.temporal_pos_embed_z'].reshape(1, 12, 12, 768)
                    param['backbone.temporal_pos_embed_z'] = param['backbone.temporal_pos_embed_z'][:, 2:-2, 2:-2, :].reshape(1, 64, 768)
                temp_z = resize_pos_embed(
                    param['backbone.temporal_pos_embed_z'],
                    posemb_new=template_posemb,
                    num_tokens=0,
                    gs_new=(template_size, template_size)
                )
                param['backbone.pos_embed_z'] += temp_z

            if 'backbone.temporal_pos_embed_x' in param:
                print("  [Student] Resizing temporal search position embedding...")
                if param['backbone.temporal_pos_embed_x'].shape[1]==576:
                    param['backbone.temporal_pos_embed_x'] = param['backbone.temporal_pos_embed_x'].reshape(1, 24, 24, 768)
                    param['backbone.temporal_pos_embed_x'] = param['backbone.temporal_pos_embed_x'][:, 4:-4, 4:-4, :].reshape(1, 256, 768)
                temp_x = resize_pos_embed(
                    param['backbone.temporal_pos_embed_x'],
                    posemb_new=search_posemb,
                    num_tokens=0,
                    gs_new=(search_size, search_size)
                )
                param['backbone.pos_embed_x'] += temp_x

        return param

    if training:
        print("\n=== Loading Student RGB Branch ===")
        print("load RGB parameters:", cfg.MODEL.RGB_BRANCH)
        rgb_param = torch.load(cfg.MODEL.RGB_BRANCH, map_location="cpu")['net']
        if "DropTrack" in cfg.MODEL.RGB_BRANCH:
            print("[Student RGB] Resizing position embeddings...")
            rgb_param = pos_embed_filter(rgb_param)
        m,n = rgb_branch.load_state_dict(backbone_weight_filter(rgb_param), strict=False)
        print("missing keys: ", m)
        
        if not cfg.MODEL.SHARE_STUDENT:
            print("\n=== Loading Student TIR Branch ===")
            print("load TIR parameters:", cfg.MODEL.TIR_BRANCH)
            tir_param = torch.load(cfg.MODEL.TIR_BRANCH, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_BRANCH:
                print("[Student TIR] Resizing position embeddings...")
                tir_param = pos_embed_filter(tir_param)
            m,n = tir_branch.load_state_dict(backbone_weight_filter(tir_param), strict=False)
            print("missing keys: ", m)

        print("Tracking head type: concat")
        head_param = boxhead_weight_filter(rgb_param)
        for k,v in list(head_param.items()):
            if k in ['conv1_ctr.0.weight','conv1_offset.0.weight','conv1_size.0.weight']:
                head_param[k] = torch.cat([v,v],1)
        
        # 如果使用 DFL，跳过 conv5_offset（形状不匹配：2 vs 16 通道）
        # 这部分会在后面从 Phase 1 checkpoint 加载时处理
        use_dfl = getattr(cfg.MODEL.HEAD, 'USE_DFL', False)
        if use_dfl:
            print("[DFL] Skipping conv5_offset from DropTrack checkpoint (shape mismatch)")
            print("      Will load and expand from Phase 1 checkpoint later")
            head_param.pop('conv5_offset.weight', None)
            head_param.pop('conv5_offset.bias', None)
        
        m,n = box_head.load_state_dict(head_param, strict=False)
        print("missing keys: ", m)

        if teacher_rgb!=None:
            print("\n=== Loading Teacher RGB Branch ===")
            print("load rgb teacher parameters:", cfg.MODEL.RGB_TEACHER)
            rgbTeacher_param = torch.load(cfg.MODEL.RGB_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.RGB_TEACHER:
                print("[Teacher RGB] Resizing position embeddings...")
                rgbTeacher_param = pos_embed_filter(rgbTeacher_param, is_teacher=True)  # 使用教师配置
            m,n = teacher_rgb.load_state_dict(backbone_weight_filter(rgbTeacher_param), strict=False)
            print("missing keys: ", m)
        if box_head_v!=None:
            head_param_v = boxhead_weight_filter(rgbTeacher_param)
            # Adjust the conv weights to match input channels
            for k,v in list(head_param_v.items()):
                if k in ['conv1_ctr.0.weight','conv1_offset.0.weight','conv1_size.0.weight']:
                    head_param_v[k] = v[:,:768,:,:]  # Trim to match RGB branch embed_dim
            m,n = box_head_v.load_state_dict(head_param_v, strict=False)
            print("missing keys: ", m)
        
        if teacher_tir!=None:
            print("\n=== Loading Teacher TIR Branch ===")
            print("load tir teacher parameters:", cfg.MODEL.TIR_TEACHER)
            tirTeacher_param = torch.load(cfg.MODEL.TIR_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_TEACHER:
                print("[Teacher TIR] Resizing position embeddings...")
                tirTeacher_param = pos_embed_filter(tirTeacher_param, is_teacher=True)  # 使用教师配置
            m,n = teacher_tir.load_state_dict(backbone_weight_filter(tirTeacher_param), strict=False)
            print("missing keys: ", m)
        if box_head_i!=None:
            head_param_i = boxhead_weight_filter(tirTeacher_param)
            # Adjust the conv weights to match input channels
            for k,v in list(head_param_i.items()):
                if k in ['conv1_ctr.0.weight','conv1_offset.0.weight','conv1_size.0.weight']:
                    head_param_i[k] = v[:,:768,:,:]  # Trim to match TIR branch embed_dim
            m,n = box_head_i.load_state_dict(head_param_i, strict=False)
            print("missing keys: ", m)



    # Build model. Pass has_teacher flag so OSTrack_CKD can choose the
    # appropriate forward path (greenlet-based when teachers exist, simpler
    # direct calls when not).
    has_teacher = bool(cfg.MODEL.RGB_TEACHER or cfg.MODEL.TIR_TEACHER)
    model = OSTrack_CKD(
        rgb_branch,
        tir_branch,
        teacher_rgb,
        teacher_tir,
        box_head = box_head,
        box_head_v = box_head_v,
        box_head_i = box_head_i,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        mask_ratio=cfg.TRAIN.INPUT_MASK_RATIO,
        mask_probability=cfg.TRAIN.MASK_PROBABILITY,
        training = training,
        has_teacher = has_teacher,
    )
    if cfg.MODEL.PRETRAIN_FILE!="" and training:
        print(f"\n=== Loading Pretrained Weights from Phase 1 ===")
        print(f"Checkpoint: {cfg.MODEL.PRETRAIN_FILE}")
        
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        pretrained_dict = checkpoint['net']
        
        # Convert weights if current model uses LoRA
        use_lora = getattr(cfg.MODEL.HEAD, 'USE_LORA', False)
        if use_lora:
            print("[LoRA] Converting Phase 1 weights to LoRA-compatible format...")
            pretrained_dict = convert_lora_weights(pretrained_dict, use_lora=True)
        
        # DFL head 特殊处理：Phase 1 是原始 CENTER head (2 通道 offset)
        # 需要扩展到 DFL head (2*K 通道 offset)
        # 注意：这里的扩展是**唯一**的扩展，不要在前面的加载中扩展！
        use_dfl = getattr(cfg.MODEL.HEAD, 'USE_DFL', False)
        if use_dfl:
            dfl_bins = getattr(cfg.MODEL.HEAD, 'DFL_BINS', 8)
            print(f"[DFL] Phase 1 checkpoint has CENTER head, expanding to CENTER_DFL")
            print(f"[DFL] Expanding box_head.conv5_offset: 2 channels -> {2*dfl_bins} channels")
            
            if 'box_head.conv5_offset.weight' in pretrained_dict:
                old_weight = pretrained_dict['box_head.conv5_offset.weight']  # [2, 32, 1, 1]
                old_bias = pretrained_dict['box_head.conv5_offset.bias']      # [2]
                
                # 确认是 Phase 1 的 2 通道权重
                if old_weight.shape[0] == 2:
                    # 扩展权重：每个原始通道复制 K 次
                    new_weight = torch.cat([
                        old_weight[0:1].repeat(dfl_bins, 1, 1, 1),  # x 的 K 个 bins
                        old_weight[1:2].repeat(dfl_bins, 1, 1, 1)   # y 的 K 个 bins
                    ], dim=0)  # [2*K, 32, 1, 1]
                    
                    # 添加小的确定性扰动（基于索引），避免所有 bin 完全相同
                    for i in range(dfl_bins):
                        new_weight[i] = new_weight[i] * (1.0 + 0.01 * i / dfl_bins)  # x bins
                        new_weight[dfl_bins + i] = new_weight[dfl_bins + i] * (1.0 + 0.01 * i / dfl_bins)  # y bins
                    
                    # 扩展 bias
                    new_bias = torch.cat([
                        old_bias[0:1].repeat(dfl_bins),  # x 的 K 个 bins
                        old_bias[1:2].repeat(dfl_bins)   # y 的 K 个 bins
                    ], dim=0)  # [2*K]
                    
                    pretrained_dict['box_head.conv5_offset.weight'] = new_weight
                    pretrained_dict['box_head.conv5_offset.bias'] = new_bias
                    print(f"  - Expanded: {old_weight.shape} -> {new_weight.shape}")
                else:
                    print(f"  - [Info] Checkpoint already has {old_weight.shape[0]} channels (DFL format), using directly")
        
        # Load student weights with strict=False to allow missing LoRA/DFL parameters
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
        
        if use_lora:
            # Filter out expected missing LoRA keys
            lora_missing = [k for k in missing_keys if 'lora_A' in k or 'lora_B' in k]
            other_missing = [k for k in missing_keys if k not in lora_missing]
            
            print(f"[LoRA] Initialized {len(lora_missing)} new LoRA parameters (lora_A, lora_B)")
            if other_missing:
                print(f"[Warning] Other missing keys: {other_missing}")
            if unexpected_keys:
                print(f"[Warning] Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10
        else:
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        
        print("[✓] Student weights loaded from Phase 1 checkpoint")
        
        # ========== 加载 Teacher 权重 ==========
        # Teacher 也需要从 Phase 1 checkpoint 加载，否则 teacher 特征是随机的，导致 style_loss 巨大！
        if teacher_rgb is not None:
            print("\n=== Loading Teacher RGB from Phase 1 Checkpoint ===")
            teacher_rgb_dict = {k.replace('t_rgb.', ''): v for k, v in pretrained_dict.items() if k.startswith('t_rgb.')}
            if teacher_rgb_dict:
                missing_t_rgb, unexpected_t_rgb = teacher_rgb.load_state_dict(teacher_rgb_dict, strict=False)
                print(f"[✓] Teacher RGB loaded from Phase 1 ({len(teacher_rgb_dict)} keys)")
                if missing_t_rgb:
                    print(f"    Missing: {missing_t_rgb}")
            else:
                print("[Warning] No teacher RGB weights in Phase 1 checkpoint, using ImageNet pretrained")
        
        if teacher_tir is not None:
            print("\n=== Loading Teacher TIR from Phase 1 Checkpoint ===")
            teacher_tir_dict = {k.replace('t_tir.', ''): v for k, v in pretrained_dict.items() if k.startswith('t_tir.')}
            if teacher_tir_dict:
                missing_t_tir, unexpected_t_tir = teacher_tir.load_state_dict(teacher_tir_dict, strict=False)
                print(f"[✓] Teacher TIR loaded from Phase 1 ({len(teacher_tir_dict)} keys)")
                if missing_t_tir:
                    print(f"    Missing: {missing_t_tir}")
            else:
                print("[Warning] No teacher TIR weights in Phase 1 checkpoint, using ImageNet pretrained")
        
        if box_head_v is not None and teacher_rgb is not None:
            print("\n=== Loading Teacher RGB Head from Phase 1 Checkpoint ===")
            head_v_dict = {k.replace('box_head_v.', ''): v for k, v in pretrained_dict.items() if k.startswith('box_head_v.')}
            if head_v_dict:
                missing_hv, unexpected_hv = box_head_v.load_state_dict(head_v_dict, strict=False)
                print(f"[✓] Teacher RGB head loaded from Phase 1 ({len(head_v_dict)} keys)")
            else:
                print("[Warning] No teacher RGB head weights in Phase 1 checkpoint")
        
        if box_head_i is not None and teacher_tir is not None:
            print("\n=== Loading Teacher TIR Head from Phase 1 Checkpoint ===")
            head_i_dict = {k.replace('box_head_i.', ''): v for k, v in pretrained_dict.items() if k.startswith('box_head_i.')}
            if head_i_dict:
                missing_hi, unexpected_hi = box_head_i.load_state_dict(head_i_dict, strict=False)
                print(f"[✓] Teacher TIR head loaded from Phase 1 ({len(head_i_dict)} keys)")
            else:
                print("[Warning] No teacher TIR head weights in Phase 1 checkpoint")
        
        print("\n[✓] All Phase 1 weights loaded successfully\n")
    
    return model
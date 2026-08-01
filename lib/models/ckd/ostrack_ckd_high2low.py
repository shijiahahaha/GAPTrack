"""

"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from .vit import vit_base_patch16_224, resize_pos_embed
from .vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from greenlet import greenlet


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
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)



    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):


        # # Debug: print incoming stacked template/search shapes to verify ordering
        # try:
        #     print("OSTrack_CKD.forward received template.shape:", tuple(template.shape), "search.shape:", tuple(search.shape))
        #     # If stacked along dim 0, print each version's HxW
        #     if template.dim() >= 5:
        #         for i in range(template.shape[0]):
        #             print(f"  template[{i}] image shape: {tuple(template[i].shape)}")
        #     if search.dim() >= 5:
        #         for i in range(search.shape[0]):
        #             print(f"  search[{i}] image shape: {tuple(search[i].shape)}")
        # except Exception:
        #     # don't crash on debug logging
        #     pass


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
                # ensure the teacher greenlet receives the high-resolution inputs first
                # template and search are stacked as (num_versions, B, C, H, W)
                # flip so the teacher sees the high-res item at index 0
                # t_x_rgb, t_aux_dict_rgb = teacher_rgb_gr.switch(z_li=template.flip(0), x_li=search.flip(0),
                # 现在约定：index0 为 teacher 分辨率，直接传递不翻转
                # import pdb; pdb.set_trace()
                t_x_rgb, t_aux_dict_rgb = teacher_rgb_gr.switch(z_li=template, x_li=search,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, )
                
                # import torch.nn.functional as F
                # # 为 teacher 分辨率上采样输入
                # template_teacher = F.interpolate(template.flip(0), size=(128, 128), mode='bilinear', align_corners=False)
                # search_teacher = F.interpolate(search.flip(0), size=(256, 256), mode='bilinear', align_corners=False)
                # t_x_rgb, t_aux_dict_rgb = teacher_rgb_gr.switch(
                #     z_li=template_teacher,
                #     x_li=search_teacher,
                #     ce_template_mask=ce_template_mask,
                #     ce_keep_rate=ce_keep_rate,
                #     return_last_attn=return_last_attn, )

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
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        if head==None:
            box_head = self.box_head
        else:
            box_head = head
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
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
    
    # 学生网络使用cfg中指定的低分辨率
    # RGB学生    
    rgb_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                        is_teacher=False)  # 明确指定为学生模型
    rgb_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # TIR学生
    if cfg.MODEL.SHARE_STUDENT:    
        tir_branch = rgb_branch
    else:
        tir_branch = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            is_teacher=False)  # 明确指定为学生模型
        tir_branch.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # RGB教师
    if cfg.MODEL.RGB_TEACHER:
        teacher_rgb = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            is_teacher=True)  # 明确指定为教师模型
        teacher_rgb.finetune_track(cfg=teacher_cfg, patch_start_index=patch_start_index)  # 使用高分辨率配置
        box_head_v = build_box_head(cfg, teacher_rgb.embed_dim)  # 使用教师网络的维度(1536)
    else:
        teacher_rgb = None
        # 在nokd模式下,使用学生网络的维度(768)来构建box head
        box_head_v = build_box_head(cfg, rgb_branch.embed_dim)
    
    # TIR教师
    if cfg.MODEL.TIR_TEACHER:
        teacher_tir = vit_base_patch16_224_ce(pretrained=False, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            is_teacher=True)  # 明确指定为教师模型
        teacher_tir.finetune_track(cfg=teacher_cfg, patch_start_index=patch_start_index)  # 使用高分辨率配置
        box_head_i = build_box_head(cfg, teacher_tir.embed_dim)  # 使用教师网络的维度(1536)
    else:
        teacher_tir = None
        # 在nokd模式下,使用学生网络的维度(768)来构建box head
        box_head_i = build_box_head(cfg, tir_branch.embed_dim)
    
    # 融合的跟踪头
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
        print("load RGB parameters:", cfg.MODEL.RGB_BRANCH)
        rgb_param = torch.load(cfg.MODEL.RGB_BRANCH, map_location="cpu")['net']
        if "DropTrack" in cfg.MODEL.RGB_BRANCH:
            rgb_param = pos_embed_filter(rgb_param)
        m,n = rgb_branch.load_state_dict(backbone_weight_filter(rgb_param), strict=False)
        print("missing keys: ", m)
        
        if not cfg.MODEL.SHARE_STUDENT:
            print("load TIR parameters:", cfg.MODEL.TIR_BRANCH)
            tir_param = torch.load(cfg.MODEL.TIR_BRANCH, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_BRANCH:
                tir_param = pos_embed_filter(tir_param)
            m,n = tir_branch.load_state_dict(backbone_weight_filter(tir_param), strict=False)
            print("missing keys: ", m)

        print("Tracking head type: concat")
        head_param = boxhead_weight_filter(rgb_param)
        for k,v in list(head_param.items()):
            if k in ['conv1_ctr.0.weight','conv1_offset.0.weight','conv1_size.0.weight']:
                head_param[k] = torch.cat([v,v],1)
        m,n = box_head.load_state_dict(head_param, strict=False)
        print("missing keys: ", m)

        if teacher_rgb!=None:
            print("load rgb teacher parameters:", cfg.MODEL.RGB_TEACHER)
            rgbTeacher_param = torch.load(cfg.MODEL.RGB_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.RGB_TEACHER:
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
            print("load tir teacher parameters:", cfg.MODEL.TIR_TEACHER)
            tirTeacher_param = torch.load(cfg.MODEL.TIR_TEACHER, map_location="cpu")['net']
            if "DropTrack" in cfg.MODEL.TIR_TEACHER:
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
        
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")['net'])
    return model
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens
from .vit import VisionTransformer
from ..layers.attn_blocks import MCEBlock, CEBlock
from greenlet import greenlet

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module
    
    RealHigh Version: Expects high-resolution input for teachers, 
    down-samples to low-resolution for students.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, is_teacher=False
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            num_classes=num_classes, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
            representation_size=representation_size, distilled=distilled,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate, embed_layer=embed_layer,
            norm_layer=norm_layer, act_layer=act_layer, weight_init=weight_init,
            is_teacher=is_teacher
        )
        self.next_gr = [None]
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                MCEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i, next_gr=self.next_gr, layer_idx=i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)
        

    def forward_features(self, z, x, mask_z=1., mask_x=1.,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, attn_li=[]
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        z = mask_z*z
        x = mask_x*x

        z += self.pos_embed_z
        x += self.pos_embed_x

        x = combine_tokens(z, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        x_list = {}
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn, attn_li = \
                blk(x, global_index_t, global_index_s, None, ce_template_mask, ce_keep_rate, attn_li)
            if self.training:
                x_list[i] = x.clone()

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
            "global_index_s": global_index_s,
            "x_list": x_list
        }
        self.next_gr[0].switch(attn_li)
        return x, aux_dict

    # ========== 以下为备用代码，当前未使用 ==========
    # RealHigh: 实际训练中所有网络都使用 forward_features（greenlet版本）
    # 以下 forward_features_student 保留作为参考，方便后续添加学生专用模块
    
    def forward_features_student(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False, mask_ratio=0.0, mask_probability=0.0
                         ):
        """
        [未使用] 学生网络专用的forward_features（非greenlet版本）
        
        保留此函数作为参考，展示如何在学生网络中：
        1. 应用token级别的mask
        2. 添加自定义增强模块（如TemplateGuidedEnhancer等）
        
        实际训练时不使用此函数，而是使用 forward_features（greenlet版本）
        """
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        # 学生网络的mask处理：在token级别应用
        B_tok, N_x, C = x.shape
        for i in range(B_tok):
            if torch.rand(1).item() < mask_probability:
                search_mask = (torch.rand(N_x) < mask_ratio)
                x[i][search_mask] = x[i][search_mask] * 0

        # attention mask handling (如果需要)
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        z += self.pos_embed_z
        x += self.pos_embed_x

        x = combine_tokens(z, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        # ========== 这里可以添加学生网络专用的模块 ==========
        # 例如: TemplateGuidedEnhancer, SoftResidualTADB 等
        # ====================================================

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        x_list = {}
        attn_li_student = (None, [], [])  # 学生网络的attn_li初始值
        for i, blk in enumerate(self.blocks):
            # MCEBlock总是返回6个值（包括attn_li），即使在学生网络中
            x, global_index_t, global_index_s, removed_index_s, attn, attn_li_student = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate, attn_li_student)
            if self.training:
                x_list[i] = x.clone()

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,
            "global_index_s": global_index_s,
            "x_list": x_list
        }

        return x, aux_dict


    def forward(self, z_li, x_li, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None, mask_probability=0.0, mask_ratio=0.0,
                return_last_attn=False, mask_x=1., mask_z=1.):
        """
        RealHigh版本的forward:
        - 教师网络: 直接使用 z_li[0], x_li[0] 的真实高分辨率输入
        - 学生网络: 将高分辨率下采样到低分辨率后使用
        """
        
        z = z_li[0]
        x = x_li[0]
        
        # 教师网络直接使用输入,不需要任何resize
        # 检查输入是否与position embedding匹配
        expected_t_side = int(math.sqrt(self.pos_embed_z.shape[1])) * self.patch_size if hasattr(self, 'pos_embed_z') and self.pos_embed_z is not None else None
        expected_s_side = int(math.sqrt(self.pos_embed_x.shape[1])) * self.patch_size if hasattr(self, 'pos_embed_x') and self.pos_embed_x is not None else None
        
        cur_t_side = z.shape[-1]
        cur_s_side = x.shape[-1]
        
        # RealHigh 逻辑:
        # - 数据加载器应该提供真实高分辨率图像 (128×128, 256×256)
        # - 教师网络: 直接使用高分辨率，不需要任何 resize
        # - 学生网络: 从高分辨率下采样到低分辨率
        
        if self.is_teacher:
            # 教师网络: 直接使用输入的真实高分辨率
            # 如果尺寸不匹配，说明数据加载器有问题
            if not hasattr(self, '_teacher_check_printed'):
                print(f"\n[RealHigh] Teacher forward - Input received:")
                print(f"  Template: {cur_t_side}x{cur_t_side} (expected {expected_t_side}x{expected_t_side})")
                print(f"  Search: {cur_s_side}x{cur_s_side} (expected {expected_s_side}x{expected_s_side})")
                if cur_t_side == expected_t_side and cur_s_side == expected_s_side:
                    print(f"  ✓ Perfect match! Using real high-resolution images")
                self._teacher_check_printed = True
            
            if expected_t_side is not None and cur_t_side != expected_t_side:
                print(f"WARNING: Teacher template size mismatch! Expected {expected_t_side}, got {cur_t_side}")
                print(f"  → Data loader should provide real high-resolution images!")
                print(f"  → Currently using interpolation (pseudo high-res), not ideal for RealHigh")
                z = F.interpolate(z, size=(expected_t_side, expected_t_side), mode='bilinear', align_corners=False)
            if expected_s_side is not None and cur_s_side != expected_s_side:
                print(f"WARNING: Teacher search size mismatch! Expected {expected_s_side}, got {cur_s_side}")
                print(f"  → Data loader should provide real high-resolution images!")
                print(f"  → Currently using interpolation (pseudo high-res), not ideal for RealHigh")
                x = F.interpolate(x, size=(expected_s_side, expected_s_side), mode='bilinear', align_corners=False)
        else:
            # 学生网络: 从真实高分辨率下采样到低分辨率
            # 输入应该是高分辨率 (128×128, 256×256)
            # 下采样到学生期望的低分辨率 (64×64, 128×128)
            if not hasattr(self, '_student_resize_printed'):
                print(f"\n[RealHigh] Student forward - Input received:")
                print(f"  Template: {cur_t_side}x{cur_t_side} → Downsampling to {expected_t_side}x{expected_t_side}")
                print(f"  Search: {cur_s_side}x{cur_s_side} → Downsampling to {expected_s_side}x{expected_s_side}")
                self._student_resize_printed = True
            
            if expected_t_side is not None and cur_t_side != expected_t_side:
                z = F.interpolate(z, size=(expected_t_side, expected_t_side), mode='bilinear', align_corners=False)
            if expected_s_side is not None and cur_s_side != expected_s_side:
                x = F.interpolate(x, size=(expected_s_side, expected_s_side), mode='bilinear', align_corners=False)

        # RealHigh: 在greenlet链中处理mask
        # 注意: 与原版vit_cecopy.py保持一致，mask会传递给所有网络
        # (虽然论文说只在学生中使用，但代码实现是所有网络都收到mask_x)
        if len(z_li) > 1:
            if len(z_li) == 2 and self.training:
                #计算mask只在学生网络使用
                # if not self.is_teacher:
                # 生成mask (与原版一致，不区分教师和学生)
                B = x.shape[0]
                H = x.shape[2]
                W = x.shape[3]
                ph = self.patch_size
                N_x = (H // ph) * (W // ph)
                
                if mask_probability < 1. and mask_probability > 0.:
                    mask_fun = lambda flag: torch.rand(N_x) < mask_ratio if flag else torch.ones(N_x)
                    mask_x = torch.stack([mask_fun(torch.rand(1).item() > mask_probability) for _ in range(B)]).unsqueeze(-1).to(x.device)
                elif mask_probability >= 1.:
                    mask_x = torch.stack([torch.rand(N_x) > mask_ratio for _ in range(B)], dim=0).unsqueeze(-1).to(x.device)
                else:
                    mask_x = 1.
            
            attn_li = self.next_gr[0].switch(z_li[1:], x_li[1:], ce_template_mask, ce_keep_rate, tnc_keep_rate, 
                                mask_probability, mask_ratio, return_last_attn, mask_x)
        else:
            attn_li = self.next_gr[0].switch((None, [], []))     # 传递初始值

        # RealHigh: 教师和学生都使用forward_features（greenlet版本）
        # mask_x通过greenlet链传递给所有网络（与原版一致）
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, 
                                            mask_x=mask_x, attn_li=attn_li, mask_z=mask_z)
        
        return x, aux_dict



def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            try:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            except:
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_tiny_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Tiny model for ultra-fast inference.
    Params: ~5.7M, embed_dim=192, heads=3
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_small_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Small model for fast inference with good accuracy.
    Params: ~22M, embed_dim=384, heads=6
    Target: 250+ FPS on RTX 3090
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model

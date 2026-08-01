"""
CKD Loss for RealHigh Version

RealHigh版本的损失函数：教师使用真实高分辨率，学生使用低分辨率
需要在计算蒸馏损失前对齐教师和学生的特征尺寸

主要变化：
- 教师特征: (B, N_t, C) 其中 N_t = template_tokens + search_tokens_high
  例如: 64 (template 8x8) + 256 (search 16x16) = 320 tokens
- 学生特征: (B, N_s, C) 其中 N_s = template_tokens + search_tokens_low
  例如: 64 (template 8x8) + 64 (search 8x8) = 128 tokens

对齐策略：使用 adaptive_avg_pool1d 将高分辨率特征下采样到低分辨率
"""

from torch import nn, Tensor
from torch.nn.functional import l1_loss, mse_loss
import torch.nn.functional as F
import torch


class BaseLoss():
    NAME = None
    def __init__(self, content_level="channel", style_level="channel") -> None:
        self.content_level = content_level
        self.style_level = style_level

    def content_distill(self):
        raise ImportError
    
    def style_distill(self):
        raise ImportError



class CKD_loss(BaseLoss):
    NAME = "CKD"
    def __init__(self, num_features=768, content_level="channel", style_level="channel"):
        super().__init__(content_level, style_level)
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss


    def content_distill(self, x:Tensor, y:Tensor, **arg_dict):
        """
        RealHigh内容蒸馏损失：
        - x: 教师特征 (B, N_teacher, C)
        - y: 学生特征 (B, N_student, C)
        
        对齐策略：
        方法1 (当前): adaptive_avg_pool1d - 自适应平均池化，保留全局信息
        方法2 (备选): interpolate - 双线性插值，可能保留更多细节但可能引入噪声
        
        约定：将高分辨率下采样到低分辨率（通常是教师->学生）
        
        ⚠️ 重要：此函数内的操作不会修改传入的原始tensor！
        - x, y 是局部变量，重新赋值不影响外部
        - 传入时通常已经 detach()，断开了计算图
        - 原始教师特征在网络中保持高分辨率不变
        """
        # 调试：记录原始尺寸（可选，训练稳定后删除）
        # original_x_shape = x.shape
        
        if self.content_level=="channel":
            x = x.transpose(-1,-2)  # B,C,N - 创建新tensor，不修改原始x
            y = y.transpose(-1,-2)
            # 对齐 token 维度（N）
            if x.shape[-1] != y.shape[-1]:
                # 约定：以较小的 token 数为目标，避免插值扩大噪声
                target_N = min(x.shape[-1], y.shape[-1])
                if x.shape[-1] != target_N:
                    # 改进版：保留空间结构的2D下采样
                    x = self._spatial_align(x, target_N)
                if y.shape[-1] != target_N:
                    y = F.adaptive_avg_pool1d(y, target_N)
            # InstanceNorm1d 作用在 (B,C,N)
            x = self.instanceNorm(x)  # 再次创建新tensor
            y = self.instanceNorm(y)
            
            # 调试：验证对齐（可选，训练稳定后删除）
            # if hasattr(self, '_debug_once') and not self._debug_once:
            #     print(f"[CKD Loss] Teacher: {original_x_shape} -> {x.shape}, Student: {y.shape}")
            #     self._debug_once = True
            
        elif self.content_level=="token":
            # token 模式下 InstanceNorm1d 视 C 为长度，这里仍需对齐 N 再转置
            if x.shape[1] != y.shape[1]:
                target_N = min(x.shape[1], y.shape[1])
                if x.shape[1] != target_N:
                    x = F.adaptive_avg_pool1d(x.transpose(-1,-2), target_N).transpose(-1,-2)
                if y.shape[1] != target_N:
                    y = F.adaptive_avg_pool1d(y.transpose(-1,-2), target_N).transpose(-1,-2)
            x = self.instanceNorm(x)
            y = self.instanceNorm(y)
        return self.dualDistill_loss(x,y)
    
    def _spatial_align(self, x:Tensor, target_N:int):
        """
        空间感知的特征对齐 (2D方式)
        x: (B, C, N) 教师特征，N=320 (64 template + 256 search)
        target_N: 80 (16 template + 64 search)
        
        关键改进：
        - 分别处理 template 和 search tokens
        - 使用 2D 双线性插值保留空间结构
        - 避免 1D 池化破坏空间邻接关系
        """
        B, C, N = x.shape
        
        # 假设 token 顺序: [template_tokens, search_tokens]
        # High-res: 64 template (8×8) + 256 search (16×16) = 320
        # Low-res:  16 template (4×4) + 64 search (8×8) = 80
        
        if N == 320 and target_N == 80:
            # 分离 template 和 search
            template_tokens = x[:, :, :64]   # (B, C, 64)
            search_tokens = x[:, :, 64:]     # (B, C, 256)
            # Debug: 仅首次打印对齐信息，帮助确认正在使用改进的 spatial 对齐逻辑
            if not hasattr(self, '_spatial_align_info_printed'):
                print('[CKD_loss_high2low] Using spatial 2D alignment: template 8x8->4x4, search 16x16->8x8')
                print(f'  Input teacher tokens = {N}, target student tokens = {target_N}')
                self._spatial_align_info_printed = True
            
            # Template: 8×8 → 4×4 (64 → 16)
            template_2d = template_tokens.view(B, C, 8, 8)  # (B, C, 8, 8)
            template_aligned = F.interpolate(template_2d, size=(4, 4), 
                                            mode='bilinear', align_corners=False)
            template_aligned = template_aligned.view(B, C, 16)  # (B, C, 16)
            
            # Search: 16×16 → 8×8 (256 → 64)
            search_2d = search_tokens.view(B, C, 16, 16)  # (B, C, 16, 16)
            search_aligned = F.interpolate(search_2d, size=(8, 8),
                                          mode='bilinear', align_corners=False)
            search_aligned = search_aligned.view(B, C, 64)  # (B, C, 64)
            
            # 合并
            x_aligned = torch.cat([template_aligned, search_aligned], dim=2)  # (B, C, 80)
            return x_aligned
        else:
            # 其他情况回退到 1D 池化
            return F.adaptive_avg_pool1d(x, target_N)
    
    def style_distill(self, x:Tensor, y:Tensor, **arg_dict):
        if self.style_level=="channel":
            # x: B,N,C -> B,C
            mx = x.mean(-2)
            my = y.mean(-2)
            stdx = x.std(-2)
            stdy = y.std(-2)
        elif self.style_level=="token":
            # x: B,N,C -> B,N
            mx = x.mean(-1)
            my = y.mean(-1)
            stdx = x.std(-1)
            stdy = y.std(-1)
        return ((mx-my)**2+(stdx-stdy)**2).mean()
    
    
def cov(input):
    # B,N
    b, c, h, w = input.size()
    x = input- torch.mean(input)
    x = x.view(b * c, h * w)
    cov_matrix = torch.matmul(x.T, x) / x.shape[0]

    return cov_matrix


class CKD_loss_Cov():
    NAME = "CKD_Cov"
    # channel-level align
    # 在原来的基础上加上了协方差,用在内容约束上
    def __init__(self, num_features=768):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss

    def content_distill(self, x:Tensor, y:Tensor, **arg_dict):
        # B,N,C -> B,C,N
        x = x.transpose(-1,-2)
        y = y.transpose(-1,-2)
        if x.shape[-1] != y.shape[-1]:
            target_N = min(x.shape[-1], y.shape[-1])
            if x.shape[-1] != target_N:
                x = F.adaptive_avg_pool1d(x, target_N)
            if y.shape[-1] != target_N:
                y = F.adaptive_avg_pool1d(y, target_N)
        x = self.instanceNorm(x)
        y = self.instanceNorm(y)
        return self.dualDistill_loss(x,y)
    
    def style_distill(self, x:Tensor, y:Tensor, **arg_dict):
        # x: B,N,C -> B,C
        mx = x.mean(-2)
        my = y.mean(-2)
        stdx = x.std(-2)
        stdy = y.std(-2)
        return ((mx-my)**2+(stdx-stdy)**2).mean()
    



class CKD_GlobalLocal_soft_loss(BaseLoss):
    NAME = "CKD_GlobalLocal_Soft"
    def __init__(self, num_features=768, **arg_dict):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss

    def content_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # B,N,C -> B,C,N 并对齐 token 数
        x = x.transpose(-1,-2)
        y = y.transpose(-1,-2)
        if x.shape[-1] != y.shape[-1]:
            target_N = min(x.shape[-1], y.shape[-1])
            if x.shape[-1] != target_N:
                x = F.adaptive_avg_pool1d(x, target_N)
            if y.shape[-1] != target_N:
                y = F.adaptive_avg_pool1d(y, target_N)
        x = self.instanceNorm(x)
        y = self.instanceNorm(y)

        B = x.shape[0]
        mask = torch.cat([torch.ones([B,1,64]).to(score_map_gt.device), 
                          score_map_gt.reshape(B,1,-1)+0.5], dim=-1)       # B,1,N
        loss = self.dualDistill_loss(x*mask, y*mask)
        return loss
    
    def style_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # x: B,N,C -> B,C
        B = x.shape[0]
        mask = torch.cat([torch.ones([B,64,1]).to(score_map_gt.device),
                          score_map_gt.reshape(B,-1,1)+0.5], dim=-2)       # B,N,1
        local_x = x*mask; local_y = y*mask
        local_mx = local_x.mean(-2)
        local_my = local_y.mean(-2)
        local_stdx = local_x.std(-2)
        local_stdy = local_y.std(-2)
        loss = ((local_mx-local_my)**2+(local_stdx-local_stdy)**2).mean()
        return loss
    

class CKD_GlobalLocal_hard_loss(BaseLoss):
    NAME = "CKD_GlobalLocal_Hard"
    def __init__(self, num_features=768, **arg_dict):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss

    def content_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # B,N,C -> B,C,N 并对齐 token 数
        x = x.transpose(-1,-2)
        y = y.transpose(-1,-2)
        if x.shape[-1] != y.shape[-1]:
            target_N = min(x.shape[-1], y.shape[-1])
            if x.shape[-1] != target_N:
                x = F.adaptive_avg_pool1d(x, target_N)
            if y.shape[-1] != target_N:
                y = F.adaptive_avg_pool1d(y, target_N)
        x = self.instanceNorm(x)
        y = self.instanceNorm(y)

        B = x.shape[0]
        mask = torch.cat([torch.zeros([B,1,64]).to(score_map_gt.device), 
                          score_map_gt.reshape(B,1,-1)], dim=-1)>0       # B,1,N
        loss = self.dualDistill_loss(x*mask, y*mask)+self.dualDistill_loss(x, y)
        return loss
    
    def style_distill(self, x:Tensor, y:Tensor, score_map_gt:Tensor, **arg_dict):
        # x: B,N,C -> B,C
        B = x.shape[0]
        mask = torch.cat([torch.zeros([B,64,1]).to(score_map_gt.device),
                          score_map_gt.reshape(B,-1,1)], dim=-2)>0       # B,N,1
        local_x = x*mask; local_y = y*mask
        local_mx = local_x.mean(-2)
        local_my = local_y.mean(-2)
        local_stdx = local_x.std(-2)
        local_stdy = local_y.std(-2)
        loss_local = ((local_mx-local_my)**2+(local_stdx-local_stdy)**2).mean()
        
        # x: B,N,C -> B,C
        mx = x.mean(-2)
        my = y.mean(-2)
        stdx = x.std(-2)
        stdy = y.std(-2)
        loss_global = ((mx-my)**2+(stdx-stdy)**2).mean()
        return loss_global+loss_local
    


def get_ckd_loss(cfg):
    name = cfg.TRAIN.CKD_LOSS
    content_level = cfg.TRAIN.CONTENT_LOSS_TYPE
    style_level = cfg.TRAIN.STYLE_LOSS_TYPE
    if name==None or name=="":
        name = CKD_loss.NAME

    if name==CKD_loss.NAME:
        return CKD_loss(content_level=content_level, style_level=style_level)
    elif name==CKD_loss_Cov.NAME:
        return CKD_loss_Cov(content_level=content_level, style_level=style_level)
    elif name==CKD_GlobalLocal_soft_loss.NAME:
        return CKD_GlobalLocal_soft_loss(content_level=content_level, style_level=style_level)
    elif name==CKD_GlobalLocal_hard_loss.NAME:
        return CKD_GlobalLocal_hard_loss(content_level=content_level, style_level=style_level)
    
    raise "error ckd loss type."
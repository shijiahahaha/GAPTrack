"""
CenterPredictor with Distribution Focal Loss (DFL) support

基于原始 head.py 的 CenterPredictor，添加 DFL 支持：
- offset 预测从 2 通道改为 2*K 通道（分布）
- size 保持原有的直接回归
- 推理时从分布解码出连续 offset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.layers.frozen_bn import FrozenBatchNorm2d
from lib.models.layers.lora import LoRAConv2d, add_lora_to_conv


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    """与原始 head.py 相同的 conv 函数"""
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class CenterPredictor_DFL(nn.Module):
    """
    Center-based Predictor with DFL (Distribution Focal Loss) support
    
    与原始 CenterPredictor 的区别：
    - offset 预测使用 DFL（输出 2*K 通道的分布）
    - size 保持直接回归（输出 2 通道）
    - 新增 offset_dist 输出用于计算 DFL loss
    
    Args:
        inplanes: 输入通道数（来自 backbone concat 后的特征）
        channel: 中间层通道数
        feat_sz: 特征图大小（例如 8 for 128×128 search）
        stride: 下采样步长（通常是 16）
        freeze_bn: 是否冻结 BN
        use_lora: 是否在最后一层使用 LoRA
        lora_r, lora_alpha, lora_dropout: LoRA 参数
        use_dfl: 是否启用 DFL
        dfl_bins: DFL 的 bin 数量（K）
    """
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.0,
                 use_dfl=False, dfl_bins=8):
        super(CenterPredictor_DFL, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        self.use_lora = use_lora
        self.use_dfl = use_dfl
        self.dfl_bins = dfl_bins

        # ===== Center branch (score map) =====
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # ===== Offset branch (DFL 支持) =====
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        
        if use_dfl:
            # DFL 模式：输出 2*K 通道（x 和 y 各 K 个 bin）
            self.conv5_offset = nn.Conv2d(channel // 8, 2 * dfl_bins, kernel_size=1)
        else:
            # 原始模式：输出 2 通道（x, y）
            self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # ===== Size branch (保持原始回归) =====
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 在初始化后添加 LoRA（如果启用）
        if use_lora:
            print(f"[DFL Head] Adding LoRA to final layers (r={lora_r}, alpha={lora_alpha})")
            self.conv5_ctr = add_lora_to_conv(self.conv5_ctr, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.conv5_offset = add_lora_to_conv(self.conv5_offset, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            self.conv5_size = add_lora_to_conv(self.conv5_size, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        # 预定义 bin 索引（用于解码分布）
        if use_dfl:
            self.register_buffer('bin_indices', torch.arange(dfl_bins, dtype=torch.float32))

    def get_score_map(self, x):
        """
        前向传播获取 score map, size map, offset map
        
        Returns:
            score_map_ctr: (B, 1, H, W) - 中心点热图
            size_map: (B, 2, H, W) - 尺寸预测 (w, h)
            offset_map: (B, 2, H, W) - offset 预测（如果 DFL，已解码）
            offset_dist: (B, 2, K, H, W) - offset 分布（仅 DFL 模式）
        """
        # Center branch
        score_map_ctr = self.conv5_ctr(self.conv4_ctr(self.conv3_ctr(self.conv2_ctr(self.conv1_ctr(x)))))
        
        # Size branch（保持原始）
        size_map = self.conv5_size(self.conv4_size(self.conv3_size(self.conv2_size(self.conv1_size(x)))))
        size_map = F.relu(size_map)
        
        # Offset branch（关键修改）
        offset_raw = self.conv5_offset(self.conv4_offset(self.conv3_offset(self.conv2_offset(self.conv1_offset(x)))))
        
        if self.use_dfl:
            # DFL 模式
            B, _, H, W = offset_raw.shape
            # 分离 x 和 y 的分布
            offset_x_dist = offset_raw[:, :self.dfl_bins, :, :]  # (B, K, H, W)
            offset_y_dist = offset_raw[:, self.dfl_bins:, :, :]  # (B, K, H, W)
            
            # Softmax 转为概率分布
            offset_x_prob = F.softmax(offset_x_dist, dim=1)  # (B, K, H, W)
            offset_y_prob = F.softmax(offset_y_dist, dim=1)
            
            # 计算期望值作为连续 offset
            bins = self.bin_indices.view(1, -1, 1, 1)  # (1, K, 1, 1)
            offset_x = (offset_x_prob * bins).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            offset_y = (offset_y_prob * bins).sum(dim=1, keepdim=True)
            
            # 归一化到 [-0.5, 0.5] 范围（中心点偏移）
            offset_x = (offset_x / (self.dfl_bins - 1)) - 0.5
            offset_y = (offset_y / (self.dfl_bins - 1)) - 0.5
            
            offset_map = torch.cat([offset_x, offset_y], dim=1)  # (B, 2, H, W)
            
            # 返回分布（用于计算 DFL loss）
            offset_dist = torch.stack([offset_x_dist, offset_y_dist], dim=1)  # (B, 2, K, H, W)
        else:
            # 原始模式
            offset_map = offset_raw
            offset_dist = None
        
        return score_map_ctr, size_map, offset_map, offset_dist

    def forward(self, x, gt_score_map=None):
        """
        前向传播
        
        Args:
            x: (B, C, H, W) - 输入特征
            gt_score_map: (B, H, W) - GT score map（可选，训练时使用）
        
        Returns:
            score_map_ctr: (B, 1, H, W)
            bbox: (B, 4) - 预测的 bbox (cx, cy, w, h)
            size_map: (B, 2, H, W)
            offset_map: (B, 2, H, W)
            offset_dist: (B, 2, K, H, W) or None - DFL 分布
        """
        score_map_ctr, size_map, offset_map, offset_dist = self.get_score_map(x)

        # 计算 bbox
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map, offset_dist

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        """
        从 score map, size map, offset map 计算最终 bbox
        
        与原始 CenterPredictor 完全相同
        """
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox: (cx, cy, w, h)
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        """
        获取预测结果（与原始相同）
        """
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        return size * self.feat_sz, offset


def build_box_head_dfl(cfg, hidden_dim):
    """
    构建支持 DFL 的 box head
    
    Args:
        cfg: 配置对象
        hidden_dim: 输入通道数
    
    Returns:
        box_head: CenterPredictor_DFL 实例
    """
    stride = cfg.MODEL.BACKBONE.STRIDE
    feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
    
    # 读取 DFL 配置
    use_dfl = getattr(cfg.MODEL.HEAD, 'USE_DFL', False)
    dfl_bins = getattr(cfg.MODEL.HEAD, 'DFL_BINS', 8)
    
    # 读取 LoRA 配置
    use_lora = getattr(cfg.MODEL.HEAD, 'USE_LORA', False)
    lora_r = getattr(cfg.MODEL.HEAD, 'LORA_R', 8)
    lora_alpha = getattr(cfg.MODEL.HEAD, 'LORA_ALPHA', 16)
    lora_dropout = getattr(cfg.MODEL.HEAD, 'LORA_DROPOUT', 0.0)
    
    # 读取通道数
    channel = getattr(cfg.MODEL.HEAD, 'NUM_CHANNELS', 256)
    
    print(f"[DFL Head] Building box head:")
    print(f"  - Input channels: {hidden_dim}")
    print(f"  - Feature size: {feat_sz}×{feat_sz}")
    print(f"  - Intermediate channels: {channel}")
    print(f"  - USE_DFL: {use_dfl}")
    if use_dfl:
        print(f"  - DFL bins: {dfl_bins}")
        print(f"  - Offset output: 2×{dfl_bins} = {2*dfl_bins} channels")
    print(f"  - USE_LORA: {use_lora}")
    
    box_head = CenterPredictor_DFL(
        inplanes=hidden_dim,
        channel=channel,
        feat_sz=feat_sz,
        stride=stride,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dfl=use_dfl,
        dfl_bins=dfl_bins
    )
    
    return box_head

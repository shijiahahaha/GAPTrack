"""
LoRA (Low-Rank Adaptation) layers for tracking head fine-tuning

适用于第二阶段训练：冻结 backbone，只在 head 上加入 LoRA 轻量适配层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRAConv2d(nn.Module):
    """
    LoRA adaptation for Conv2d layers
    
    对 1×1 卷积层添加低秩分解：W' = W + α/r * (B @ A)
    其中 A: (r, in_channels), B: (out_channels, r)
    只训练 A 和 B，冻结原始 W
    
    Args:
        conv_layer: 原始的 Conv2d 层（将被冻结）
        r: LoRA rank（秩）
        lora_alpha: 缩放系数
        lora_dropout: dropout 概率
    """
    def __init__(self, conv_layer, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        
        self.conv = conv_layer  # 原始卷积层，将被冻结
        self.r = r
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / r
        
        # 获取卷积层参数
        out_channels = conv_layer.out_channels
        in_channels = conv_layer.in_channels
        kernel_size = conv_layer.kernel_size
        
        # 只支持 1×1 卷积（tracking head 的最后一层）
        assert kernel_size == (1, 1), "LoRA currently only supports 1×1 convolutions"
        
        # LoRA 参数：A 和 B
        # A: (r, in_channels, 1, 1)
        # B: (out_channels, r, 1, 1)
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r, 1, 1))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 冻结原始卷积层参数
        for param in self.conv.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 原始卷积输出
        result = self.conv(x)
        
        # LoRA 分支：x → A → B
        # 先通过 A (r 个 1×1 卷积)
        lora_out = F.conv2d(
            self.lora_dropout(x),
            self.lora_A,
            stride=self.conv.stride,
            padding=self.conv.padding
        )
        
        # 再通过 B (out_channels 个 1×1 卷积，输入是 r 个通道)
        lora_out = F.conv2d(
            lora_out,
            self.lora_B,
            stride=1,
            padding=0
        )
        
        # 加权相加
        result = result + self.scale * lora_out
        
        return result
    
    def merge_weights(self):
        """
        合并 LoRA 权重到原始卷积层（部署时使用）
        W_merged = W + α/r * (B @ A)
        """
        with torch.no_grad():
            # A: (r, in_channels, 1, 1) → (r, in_channels)
            # B: (out_channels, r, 1, 1) → (out_channels, r)
            A = self.lora_A.squeeze(-1).squeeze(-1)  # (r, in_channels)
            B = self.lora_B.squeeze(-1).squeeze(-1)  # (out_channels, r)
            
            # Delta W = B @ A
            delta_w = B @ A  # (out_channels, in_channels)
            delta_w = delta_w.unsqueeze(-1).unsqueeze(-1)  # (out_channels, in_channels, 1, 1)
            
            # 更新原始权重
            self.conv.weight.data += self.scale * delta_w
            
            # 重置 LoRA 参数（可选）
            self.lora_A.zero_()
            self.lora_B.zero_()


class LoRALinear(nn.Module):
    """
    LoRA adaptation for Linear layers
    
    对全连接层添加低秩分解：W' = W + α/r * (B @ A)
    
    Args:
        linear_layer: 原始的 Linear 层（将被冻结）
        r: LoRA rank
        lora_alpha: 缩放系数
        lora_dropout: dropout 概率
    """
    def __init__(self, linear_layer, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        
        self.linear = linear_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / r
        
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # 冻结原始层
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        result = self.linear(x)
        
        # LoRA 分支
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # (*, r)
        lora_out = F.linear(lora_out, self.lora_B)  # (*, out_features)
        
        result = result + self.scale * lora_out
        
        return result
    
    def merge_weights(self):
        """合并 LoRA 权重到原始线性层"""
        with torch.no_grad():
            delta_w = self.lora_B @ self.lora_A  # (out_features, in_features)
            self.linear.weight.data += self.scale * delta_w
            self.lora_A.zero_()
            self.lora_B.zero_()


def add_lora_to_conv(conv_layer, r=8, lora_alpha=16, lora_dropout=0.0):
    """
    工厂函数：将普通 Conv2d 层包装为 LoRAConv2d
    
    Args:
        conv_layer: nn.Conv2d 层
        r: LoRA rank
        lora_alpha: 缩放系数
        lora_dropout: dropout 概率
    
    Returns:
        LoRAConv2d 实例
    """
    return LoRAConv2d(conv_layer, r, lora_alpha, lora_dropout)


def add_lora_to_linear(linear_layer, r=8, lora_alpha=16, lora_dropout=0.0):
    """
    工厂函数：将普通 Linear 层包装为 LoRALinear
    
    Args:
        linear_layer: nn.Linear 层
        r: LoRA rank
        lora_alpha: 缩放系数
        lora_dropout: dropout 概率
    
    Returns:
        LoRALinear 实例
    """
    return LoRALinear(linear_layer, r, lora_alpha, lora_dropout)


def get_lora_parameters(model):
    """
    提取模型中所有 LoRA 参数
    
    Args:
        model: PyTorch 模型
    
    Returns:
        list of LoRA parameters
    """
    lora_params = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRAConv2d, LoRALinear)):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def print_lora_info(model):
    """打印模型中 LoRA 层的信息"""
    lora_count = 0
    total_lora_params = 0
    
    print("\n" + "="*70)
    print("LoRA Layers Information:")
    print("="*70)
    
    for name, module in model.named_modules():
        if isinstance(module, (LoRAConv2d, LoRALinear)):
            lora_count += 1
            params = sum(p.numel() for p in [module.lora_A, module.lora_B])
            total_lora_params += params
            
            print(f"  [{lora_count}] {name}")
            print(f"      Type: {type(module).__name__}")
            print(f"      Rank: {module.r}")
            print(f"      Alpha: {module.lora_alpha}")
            print(f"      Scale: {module.scale:.4f}")
            print(f"      Params: {params:,}")
    
    print("="*70)
    print(f"Total LoRA layers: {lora_count}")
    print(f"Total LoRA parameters: {total_lora_params:,}")
    print("="*70 + "\n")

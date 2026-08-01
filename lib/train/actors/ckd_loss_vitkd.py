

import math
from torch import nn, Tensor
from torch.nn.functional import l1_loss, mse_loss
import torch.nn.functional as F
import torch

# 导入基础CKD_loss类（带完整的content_distill和style_distill实现）
from .ckd_loss import CKD_loss, CKD_loss_Cov as CKD_loss_Cov_Base


class BaseLoss(nn.Module):
    NAME = None
    def __init__(self, content_level="channel", style_level="channel") -> None:
        super().__init__()
        self.content_level = content_level
        self.style_level = style_level

    def content_distill(self):
        raise ImportError
    
    def style_distill(self):
        raise ImportError


# CKD_loss 现在从 ckd_loss.py 导入，保留这个注释以便了解历史
# class CKD_loss(BaseLoss):
#     NAME = "CKD"
#     def __init__(self, num_features=768, content_level="channel", style_level="channel"):
#         super().__init__(content_level, style_level)
#         self.instanceNorm = nn.InstanceNorm1d(num_features)
#         self.dualDistill_loss = mse_loss


class CKD_ViTKD_loss(BaseLoss):
    """
    完整的 ViTKD 风格损失类
    
    关键特性：
    1. 使用可学习的线性层进行特征维度对齐（而非双线性插值）
    2. 添加 CNN 生成器增强学生特征的语义信息（可指定应用于哪些层）
    3. 可选的 Mask 机制（类似 MAE）
    
    参考：D:\cls_KD-1.0\mmcls\models\dis_losses\vitkd.py
    """
    NAME = "CKD_vitkd"
    
    def __init__(self, num_features=768, content_level="channel", style_level="channel", 
                 student_channels=768, teacher_channels=768, use_generator=True, mask_ratio=0.0,
                 generator_layers=None, always_use_linear_align=True, use_learnable_upsample=True,
                 upsample_follows_generator=True):
        super().__init__(content_level, style_level)
        self.instanceNorm = nn.InstanceNorm1d(num_features)
        self.dualDistill_loss = mse_loss
        self.use_generator = use_generator
        self.mask_ratio = mask_ratio
        self.generator_layers = generator_layers  # 指定哪些层使用生成器，如 [8,9,10,11] 或 None(全部)
        self.current_layer_idx = 0  # 追踪当前处理的是第几层
        
        # ============================================================
        # 配置参数（现在从 YAML 文件读取）
        # ============================================================
        self.always_use_linear_align = always_use_linear_align
        self.use_learnable_upsample = use_learnable_upsample
        self.upsample_follows_generator = upsample_follows_generator
        
        # ===== ViTKD 核心组件 =====
        
        # 【组件1】线性层对齐
        # 作用：学习特征空间变换，即使维度相同也可保留（增强表达能力）
        # 配置：通过 VITKD_ALWAYS_USE_LINEAR_ALIGN 控制
        if self.always_use_linear_align or student_channels != teacher_channels:
            self.align_linear = nn.Linear(student_channels, teacher_channels, bias=True)
            print(f'[CKD_ViTKD] Linear alignment: {student_channels} -> {teacher_channels} (always_use={self.always_use_linear_align})')
        else:
            self.align_linear = None
            print(f'[CKD_ViTKD] Skip linear alignment (same dimensions: {student_channels})')
        
        # 【组件2】可学习的空间上采样模块（替代固定的双线性插值）
        # 策略：根据 VITKD_UPSAMPLE_FOLLOWS_GENERATOR 决定使用时机
        #   - True: 跟随生成器策略（浅层用固定插值，深层用可学习上采样）
        #   - False: 所有层都使用可学习上采样（更多参数，可能过拟合）
        # 配置：通过 VITKD_USE_LEARNABLE_UPSAMPLE 和 VITKD_UPSAMPLE_FOLLOWS_GENERATOR 控制
        if self.use_learnable_upsample:
            # Template: 4×4 → 8×8 (stride=2, 16→64 tokens)
            self.upsample_template = nn.ConvTranspose2d(
                in_channels=teacher_channels,
                out_channels=teacher_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True
            )
            # Search (2x): 8×8 → 16×16 (stride=2, 64→256 tokens, for 128-student case)
            self.upsample_search = nn.ConvTranspose2d(
                in_channels=teacher_channels,
                out_channels=teacher_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True
            )
            # Search (4x): 4×4 → 16×16 (两个stride=2串联, 16→256 tokens, for 64-student case)
            self.upsample_search_4x = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=teacher_channels,
                    out_channels=teacher_channels,
                    kernel_size=4, stride=2, padding=1, bias=True
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=teacher_channels,
                    out_channels=teacher_channels,
                    kernel_size=4, stride=2, padding=1, bias=True
                )
            )
            print(f'[CKD_ViTKD] Learnable spatial upsample enabled (ConvTranspose2d 2x+4x, follows_generator={self.upsample_follows_generator})')
        else:
            self.upsample_template = None
            self.upsample_search = None
            print('[CKD_ViTKD] Using fixed bilinear interpolation for all layers')
        print(f'[CKD_ViTKD] Learnable spatial upsampling: ConvTranspose2d')
        
        # 3. CNN 生成器（增强空间语义信息）
        # 原始 ViTKD 设计：通道数不变，使用 3×3 卷积在空间维度提取局部特征
        # 作用：让学生学习如何从局部邻域推断被 mask 的 token
        if use_generator:
            self.generator = nn.Sequential(
                nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1)
            )
            if generator_layers is None:
                print(f'[CKD_ViTKD] CNN Generator enabled for ALL layers')
            else:
                print(f'[CKD_ViTKD] CNN Generator enabled for layers: {generator_layers}')
        else:
            self.generator = None
            print(f'[CKD_ViTKD] CNN Generator disabled')
        
        # 4. Mask token（用于 masked generation）
        if mask_ratio > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_channels))
            print(f'[CKD_ViTKD] Mask ratio: {mask_ratio}')
        else:
            self.mask_token = None


    def content_distill(self, x:Tensor, y:Tensor, layer_idx=None, **arg_dict):
        """
        ViTKD 风格的内容蒸馏
        
        参数：
            x: 教师特征 (B, N_t, C_t)
            y: 学生特征 (B, N_s, C_s)
            layer_idx: 当前层索引（用于判断是否使用生成器/可学习上采样）
        
        处理流程（分层策略）：
            浅层 (0-7): 简单对齐（线性层 + 固定插值）
            深层 (8-11): 完整流程（线性层 + 可学习上采样 + 生成器 + Mask）
        
        步骤说明：
            1. 【通道对齐】使用线性层对齐通道维度 (C_s -> C_t)
            2. 【空间对齐】上采样对齐 token 数量 (N_s -> N_t)
               - 策略由 UPSAMPLE_FOLLOWS_GENERATOR 决定
            3. 【语义增强】根据层索引决定是否使用 CNN 生成器
            4. 【损失计算】MSE 蒸馏损失
        
        修改指南：
            - 禁用生成器：配置文件中设 VITKD_USE_GENERATOR=False
            - 全层使用生成器：配置文件中设 VITKD_GENERATOR_LAYERS=None
            - 改变上采样策略：修改类常量 UPSAMPLE_FOLLOWS_GENERATOR
        """
        # x,y: B,N,C ; teacher与学生 token 数(N)可能不同，需要先对齐
        if self.content_level=="channel":
            x = x.transpose(-1,-2)  # B,C,N (教师)
            y = y.transpose(-1,-2)  # B,C,N (学生)
            
            # ===== 第1步：通道对齐（使用线性层，而非简单插值） =====
            if self.align_linear is not None:
                # y: (B, C_s, N) -> (B, N, C_s) -> Linear -> (B, N, C_t) -> (B, C_t, N)
                y = y.transpose(-1,-2)  # B,N,C
                y = self.align_linear(y)  # B,N,C_t
                y = y.transpose(-1,-2)  # B,C_t,N
            
            # ===== 第2步：空间对齐（token 数量） =====
            # 策略：学生 80 tokens → 教师 320 tokens
            if x.shape[-1] != y.shape[-1]:
                target_N = max(x.shape[-1], y.shape[-1])
                
                # 【关键决策】是否使用可学习上采样
                # 根据配置决定：
                #   - upsample_follows_generator=True: 跟随生成器策略（浅层固定，深层可学习）
                #   - upsample_follows_generator=False: 所有层都使用可学习（如果模块已创建）
                if self.upsample_follows_generator:
                    use_learnable_upsample = self._should_use_generator(layer_idx)
                else:
                    use_learnable_upsample = (self.upsample_template is not None)
                
                if y.shape[-1] != target_N:
                    y = self._spatial_upsample_linear(y, target_N, learnable=use_learnable_upsample)
                if x.shape[-1] != target_N:
                    x = self._spatial_upsample_linear(x, target_N, learnable=use_learnable_upsample)
            
            # ===== 第3步：CNN 生成器增强（根据层索引决定） =====
            use_gen_for_this_layer = self._should_use_generator(layer_idx)
            
            if self.generator is not None and use_gen_for_this_layer:
                # 对使用生成器的层应用 mask（参照 ViTKD 原论文）
                B, C, N = y.shape
                
                if self.mask_ratio > 0:
                    # ===== Mask 操作（只对深层） =====
                    y_for_mask = y.transpose(-1, -2)  # B,N,C
                    y_keep, mask, ids_restore, ids_masked = self.random_masking(y_for_mask, self.mask_ratio)
                    
                    # 用 mask_token 填充被 mask 的位置（确保在同一设备上）
                    mask_tokens = self.mask_token.to(y.device).repeat(B, N - y_keep.shape[1], 1)  # B, N_masked, C
                    y_full = torch.cat([y_keep, mask_tokens], dim=1)  # B, N, C
                    
                    # 恢复原始顺序
                    y_full = torch.gather(y_full, dim=1, 
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, C))  # B, N, C
                    
                    y = y_full.transpose(-1, -2)  # B,C,N
                    
                    # mask 也要转为正确的形状用于后续损失计算
                    mask = mask.unsqueeze(1)  # B,1,N
                else:
                    mask = None
                
                # 将 token 序列转为 2D 特征图并应用生成器
                # 需要分别处理 template 和 search（因为 320 = 64 + 256 不是完全平方数）
                if N == 320:
                    # 分离 template 和 search
                    template_tokens = y[:, :, :64]   # (B, C, 64)
                    search_tokens = y[:, :, 64:]     # (B, C, 256)
                    
                    # Template: 8×8 → CNN 生成器 → 8×8
                    template_2d = template_tokens.view(B, C, 8, 8)
                    template_gen = self.generator(template_2d)
                    template_gen = template_gen.view(B, C, 64)
                    
                    # Search: 16×16 → CNN 生成器 → 16×16
                    search_2d = search_tokens.view(B, C, 16, 16)
                    search_gen = self.generator(search_2d)
                    search_gen = search_gen.view(B, C, 256)
                    
                    # 拼接回 320 tokens
                    y = torch.cat([template_gen, search_gen], dim=2)
                else:
                    # 通用情况：完全平方数
                    hw = int(N**0.5)
                    if hw * hw == N:
                        y_2d = y.view(B, C, hw, hw)
                        y_enhanced = self.generator(y_2d)
                        y = y_enhanced.view(B, C, N)
                    else:
                        # 无法重塑为 2D，跳过生成器
                        print(f'[Warning] Cannot reshape {N} tokens to 2D, skipping generator for this layer')
                        mask = None
            else:
                mask = None  # 浅层不使用 mask
            
            # ===== 第4步：归一化并计算损失 =====
            x = self.instanceNorm(x)
            y = self.instanceNorm(y)
            
            # 如果使用了 mask，损失只计算 masked 区域（参照 ViTKD）
            if mask is not None:
                x = x * mask  # 教师特征也应用相同的 mask
                y = y * mask  # 学生特征（生成器输出）应用 mask
            
        elif self.content_level=="token":
            # token 模式
            if self.align_linear is not None:
                y = self.align_linear(y)
            
            if x.shape[1] != y.shape[1]:
                target_N = max(x.shape[1], y.shape[1])
                if y.shape[1] != target_N:
                    y = self._spatial_upsample_linear(y.transpose(-1,-2), target_N).transpose(-1,-2)
                if x.shape[1] != target_N:
                    x = self._spatial_upsample_linear(x.transpose(-1,-2), target_N).transpose(-1,-2)
            
            x = self.instanceNorm(x)
            y = self.instanceNorm(y)
            
        return self.dualDistill_loss(x,y)
    
    def _spatial_align(self, x:Tensor, target_N:int):
        """
        【已废弃】原下采样对齐方法
        保留此函数以防代码其他地方调用，但建议使用 _spatial_upsample
        """
        B, C, N = x.shape
        
        if N == 320 and target_N == 80:
            # 分离 template 和 search
            template_tokens = x[:, :, :64]   # (B, C, 64)
            search_tokens = x[:, :, 64:]     # (B, C, 256)
            
            # Template: 8×8 → 4×4 (64 → 16)
            template_2d = template_tokens.view(B, C, 8, 8)
            template_aligned = F.interpolate(template_2d, size=(4, 4), 
                                            mode='bilinear', align_corners=False)
            template_aligned = template_aligned.view(B, C, 16)
            
            # Search: 16×16 → 8×8 (256 → 64)
            search_2d = search_tokens.view(B, C, 16, 16)
            search_aligned = F.interpolate(search_2d, size=(8, 8),
                                          mode='bilinear', align_corners=False)
            search_aligned = search_aligned.view(B, C, 64)
            
            x_aligned = torch.cat([template_aligned, search_aligned], dim=2)
            return x_aligned
        else:
            return F.adaptive_avg_pool1d(x, target_N)
    
    def _spatial_upsample_linear(self, x:Tensor, target_N:int, learnable:bool=False):
        """
        空间上采样（可选择固定插值或可学习转置卷积）
        
        Args:
            x: (B, C, N) 其中 C 已经通过 linear 对齐
            target_N: 目标 token 数量
            learnable: 是否使用可学习的转置卷积（深层）或固定插值（浅层）
        """
        B, C, N = x.shape
        
        if N == 80 and target_N == 320:
            # 分离 template 和 search
            template_tokens = x[:, :, :16]   # (B, C, 16)
            search_tokens = x[:, :, 16:]     # (B, C, 64)
            
            # 检查是否真的可以使用可学习上采样（模块必须存在）
            if learnable and self.upsample_template is not None:
                # 深层：使用可学习的转置卷积
                if not hasattr(self, '_learnable_upsample_printed'):
                    print('[CKD_ViTKD] Deep layers: Learnable spatial upsample (ConvTranspose2d)')
                    self._learnable_upsample_printed = True
                
                # Template: 4×4 → 8×8 (16 → 64)
                template_2d = template_tokens.view(B, C, 4, 4)
                template_upsampled = self.upsample_template(template_2d)  # (B, C, 8, 8)
                template_upsampled = template_upsampled.view(B, C, 64)
                
                # Search: 8×8 → 16×16 (64 → 256)
                search_2d = search_tokens.view(B, C, 8, 8)
                search_upsampled = self.upsample_search(search_2d)  # (B, C, 16, 16)
                search_upsampled = search_upsampled.view(B, C, 256)
            else:
                # 浅层：使用固定的双线性插值
                if not hasattr(self, '_fixed_upsample_printed'):
                    print('[CKD_ViTKD] Shallow layers: Fixed bilinear interpolation')
                    self._fixed_upsample_printed = True
                
                # Template: 4×4 → 8×8 (16 → 64)
                template_2d = template_tokens.view(B, C, 4, 4)
                template_upsampled = F.interpolate(template_2d, size=(8, 8), 
                                                  mode='bilinear', align_corners=False)
                template_upsampled = template_upsampled.view(B, C, 64)
                
                # Search: 8×8 → 16×16 (64 → 256)
                search_2d = search_tokens.view(B, C, 8, 8)
                search_upsampled = F.interpolate(search_2d, size=(16, 16),
                                                mode='bilinear', align_corners=False)
                search_upsampled = search_upsampled.view(B, C, 256)
            
            x_upsampled = torch.cat([template_upsampled, search_upsampled], dim=2)
            return x_upsampled
        elif N == 32 and target_N == 320:
            # 新配置: Student Template=64, Search=64 → Teacher Template=128, Search=256
            # Student: 16 + 16 = 32 tokens, Teacher: 64 + 256 = 320 tokens
            template_tokens = x[:, :, :16]   # (B, C, 16)
            search_tokens = x[:, :, 16:]     # (B, C, 16)
            
            if learnable and self.upsample_template is not None:
                # Template: 4×4 → 8×8 (16 → 64)，可学习2倍上采样
                template_2d = template_tokens.view(B, C, 4, 4)
                template_upsampled = self.upsample_template(template_2d)  # (B, C, 8, 8)
                template_upsampled = template_upsampled.view(B, C, 64)
                
                # Search: 4×4 → 16×16 (16 → 256)，可学习4倍上采样（两个stride=2串联）
                search_2d = search_tokens.view(B, C, 4, 4)
                search_upsampled = self.upsample_search_4x(search_2d)  # (B, C, 16, 16)
                search_upsampled = search_upsampled.view(B, C, 256)
            else:
                # 固定的双线性插值
                # Template: 4×4 → 8×8 (16 → 64)
                template_2d = template_tokens.view(B, C, 4, 4)
                template_upsampled = F.interpolate(template_2d, size=(8, 8), 
                                                  mode='bilinear', align_corners=False)
                template_upsampled = template_upsampled.view(B, C, 64)
                
                # Search: 4×4 → 16×16 (16 → 256)
                search_2d = search_tokens.view(B, C, 4, 4)
                search_upsampled = F.interpolate(search_2d, size=(16, 16),
                                                mode='bilinear', align_corners=False)
                search_upsampled = search_upsampled.view(B, C, 256)
            
            x_upsampled = torch.cat([template_upsampled, search_upsampled], dim=2)
            return x_upsampled
        else:
            # 回退到 2D 双线性插值（对于其他非标准尺寸）
            # 假设输入是正方形排列
            side = int(math.sqrt(N))
            target_side = int(math.sqrt(target_N))
            if side * side == N and target_side * target_side == target_N:
                x_2d = x.view(B, C, side, side)
                x_upsampled = F.interpolate(x_2d, size=(target_side, target_side), 
                                           mode='bilinear', align_corners=False)
                return x_upsampled.view(B, C, target_N)
            else:
                # 真正的回退：使用1D插值，但需要正确处理维度
                # x: (B, C, N) → (B*C, 1, N) → interpolate → (B*C, 1, target_N) → (B, C, target_N)
                x_flat = x.view(B * C, 1, N)
                x_interp = F.interpolate(x_flat, size=target_N, mode='linear', align_corners=False)
                return x_interp.view(B, C, target_N)
    
    def _spatial_upsample(self, x:Tensor, target_N:int):
        """
        【已废弃】保留向后兼容
        """
        return self._spatial_upsample_linear(x, target_N)
    
    def random_masking(self, x, mask_ratio):
        """
        对特征进行随机 mask（参照 ViTKD 原论文）
        每个样本独立随机打乱并保留 (1-mask_ratio) 的 tokens
        
        Args:
            x: [B, N, C] 特征序列
            mask_ratio: mask 比例，如 0.5 表示 mask 50% 的 tokens
            
        Returns:
            x_keep: [B, N*(1-mask_ratio), C] 保留的 tokens
            mask: [B, N] 二值 mask（0=保留，1=移除）
            ids_restore: [B, N] 用于恢复原始顺序的索引
            ids_masked: [B, N*mask_ratio] 被 mask 的 token 索引
        """
        B, N, C = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # 生成随机噪声并排序
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序：小的保留，大的移除
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前面的 tokens
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))
        
        # 生成二值 mask：0=保留，1=移除
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # 恢复原始顺序
        
        return x_keep, mask, ids_restore, ids_masked
    
    def _should_use_generator(self, layer_idx):
        """
        判断当前层是否应该使用 CNN 生成器（以及可学习上采样）
        
        Args:
            layer_idx: 当前层索引（从0开始），如果为 None 则自动递增
        
        Returns:
            bool: 是否使用生成器
        
        使用场景：
            - OSTrack-256 有 12 层 (0-11)
            - 推荐配置：generator_layers=[8,9,10,11]（深层使用）
            - 浅层策略：简单线性对齐 + 固定插值（快速，避免过拟合）
            - 深层策略：完整流程（可学习上采样 + 生成器 + Mask）
        
        修改方式（配置文件）：
            VITKD_GENERATOR_LAYERS: [8, 9, 10, 11]  # 指定层列表
            VITKD_GENERATOR_LAYERS: null            # 所有层都使用
        """
        if self.generator is None:
            return False
        
        if layer_idx is None:
            # 如果没有传入层索引，使用内部计数器
            layer_idx = self.current_layer_idx
            self.current_layer_idx += 1
        
        if self.generator_layers is None:
            # 未指定层，对所有层使用生成器
            return True
        else:
            # 只对指定的层使用生成器
            return layer_idx in self.generator_layers
    
    def _process_student_feature(self, student_feat, teacher_feat, layer_idx):
        """
        处理学生特征：对齐维度 + 上采样 + Generator
        
        用于Response Map计算，复用content_distill的处理流程但不计算loss
        
        Args:
            student_feat: (B, N_s, C_s) - 学生原始特征
            teacher_feat: (B, N_t, C_t) - 教师特征
            layer_idx: 层索引
        
        Returns:
            processed_feat: (B, N_t, C_t) - 处理后的学生特征
        """
        x = teacher_feat.transpose(-1, -2)  # (B, C_t, N_t)
        y = student_feat.transpose(-1, -2)   # (B, C_s, N_s)
        
        # 1. 通道对齐
        if self.align_linear is not None:
            y = y.transpose(-1, -2)
            y = self.align_linear(y)
            y = y.transpose(-1, -2)
        
        # 2. 空间对齐
        if x.shape[-1] != y.shape[-1]:
            target_N = x.shape[-1]
            if self.upsample_follows_generator:
                use_learnable_upsample = self._should_use_generator(layer_idx)
            else:
                use_learnable_upsample = (self.upsample_template is not None)
            y = self._spatial_upsample_linear(y, target_N, learnable=use_learnable_upsample)
        
        # 3. Generator增强（不使用mask）
        use_gen = self._should_use_generator(layer_idx)
        if self.generator is not None and use_gen:
            B, C, N = y.shape
            if N == 320:
                template_tokens = y[:, :, :64]
                search_tokens = y[:, :, 64:]
                template_2d = template_tokens.view(B, C, 8, 8)
                template_gen = self.generator(template_2d).view(B, C, 64)
                search_2d = search_tokens.view(B, C, 16, 16)
                search_gen = self.generator(search_2d).view(B, C, 256)
                y = torch.cat([template_gen, search_gen], dim=2)
            else:
                hw = int(N**0.5)
                if hw * hw == N:
                    y_2d = y.view(B, C, hw, hw)
                    y = self.generator(y_2d).view(B, C, N)
        
        return y.transpose(-1, -2)  # (B, N, C)
    
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
            # ViTKD 风格：学生上采样到教师分辨率
            target_N = max(x.shape[-1], y.shape[-1])
            if y.shape[-1] != target_N:
                y = F.interpolate(y.unsqueeze(1), size=target_N, mode='linear', align_corners=False).squeeze(1)
            if x.shape[-1] != target_N:
                x = F.interpolate(x.unsqueeze(1), size=target_N, mode='linear', align_corners=False).squeeze(1)
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
            # ViTKD 风格：学生上采样到教师分辨率
            target_N = max(x.shape[-1], y.shape[-1])
            if y.shape[-1] != target_N:
                y = F.interpolate(y.unsqueeze(1), size=target_N, mode='linear', align_corners=False).squeeze(1)
            if x.shape[-1] != target_N:
                x = F.interpolate(x.unsqueeze(1), size=target_N, mode='linear', align_corners=False).squeeze(1)
        x = self.instanceNorm(x)
        y = self.instanceNorm(y)

        B = x.shape[0]
        # 注意：上采样后 mask 需要适配新的 token 数量
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
            # ViTKD 风格：学生上采样到教师分辨率
            target_N = max(x.shape[-1], y.shape[-1])
            if y.shape[-1] != target_N:
                y = F.interpolate(y.unsqueeze(1), size=target_N, mode='linear', align_corners=False).squeeze(1)
            if x.shape[-1] != target_N:
                x = F.interpolate(x.unsqueeze(1), size=target_N, mode='linear', align_corners=False).squeeze(1)
        x = self.instanceNorm(x)
        y = self.instanceNorm(y)

        B = x.shape[0]
        # 注意：上采样后 mask 需要适配新的 token 数量
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

    if name==CKD_ViTKD_loss.NAME:
        print(f"[CKD Loss] Using {name} - ViTKD style (Linear alignment + CNN generator)")
        # 从配置中获取参数
        student_channels = getattr(cfg.MODEL, 'STUDENT_CHANNELS', 768)
        teacher_channels = getattr(cfg.MODEL, 'TEACHER_CHANNELS', 768)
        use_generator = getattr(cfg.TRAIN, 'VITKD_USE_GENERATOR', True)
        mask_ratio = getattr(cfg.TRAIN, 'VITKD_MASK_RATIO', 0.0)
        generator_layers = getattr(cfg.TRAIN, 'VITKD_GENERATOR_LAYERS', None)
        always_use_linear_align = getattr(cfg.TRAIN, 'VITKD_ALWAYS_USE_LINEAR_ALIGN', True)
        use_learnable_upsample = getattr(cfg.TRAIN, 'VITKD_USE_LEARNABLE_UPSAMPLE', True)
        upsample_follows_generator = getattr(cfg.TRAIN, 'VITKD_UPSAMPLE_FOLLOWS_GENERATOR', True)
        
        return CKD_ViTKD_loss(
            content_level=content_level, 
            style_level=style_level,
            student_channels=student_channels,
            teacher_channels=teacher_channels,
            use_generator=use_generator,
            mask_ratio=mask_ratio,
            generator_layers=generator_layers,
            always_use_linear_align=always_use_linear_align,
            use_learnable_upsample=use_learnable_upsample,
            upsample_follows_generator=upsample_follows_generator
        )
    elif name==CKD_loss.NAME:
        print(f"[CKD Loss] Using {name} - Standard CKD (no ViTKD modules)")
        return CKD_loss(content_level=content_level, style_level=style_level)
    elif name==CKD_loss_Cov.NAME:
        return CKD_loss_Cov()  # CKD_loss_Cov 不接受 content_level/style_level 参数
    elif name==CKD_GlobalLocal_soft_loss.NAME:
        return CKD_GlobalLocal_soft_loss(content_level=content_level, style_level=style_level)
    elif name==CKD_GlobalLocal_hard_loss.NAME:
        return CKD_GlobalLocal_hard_loss(content_level=content_level, style_level=style_level)
    
    raise ValueError(f"Unknown CKD loss type: {name}")
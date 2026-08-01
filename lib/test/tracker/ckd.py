import math

from lib.models.ckd import build_ostrack_ckd
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
import torch.nn.functional as F
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack_twobranch(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack_twobranch, self).__init__(params)
        network = build_ostrack_ckd(params.cfg, training=False)
        try:
            m,n = network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=False)
            if m!=[]:
                raise m
        except:
            network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu'), strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        
        # 特征保存相关（默认不启用）
        self.save_features = False
        self.feature_dir = None
        self.feature_save_path = None

    def _save_gradcam_v1(self, feats, grads, search_rgb_img, search_tir_img,
                          template_rgb_img, template_tir_img, seq_name, frame_id):
        # """
        # [原始版本备份 v1] 计算并保存 Grad-CAM 可视化图。
        # scalar = score_map.max()，梯度包含 Template+Search 混合 token。
        # feats: dict {'rgb': tensor[B,L,C], 'tir': tensor[B,L,C]}  (保留计算图)
        # grads: dict {'rgb': tensor[B,L,C], 'tir': tensor[B,L,C]}  (已 detach)
        # search/template_*_img: numpy HxWx3 RGB 图像
        # 四张子图：左上=RGB-Search Grad-CAM, 右上=TIR-Search Grad-CAM,
        #           左下=RGB-Template Grad-CAM, 右下=TIR-Template Grad-CAM
        # """
        try:
            stride = self.cfg.MODEL.BACKBONE.STRIDE
            t_size = self.cfg.TEST.TEMPLATE_SIZE
            s_size = self.cfg.TEST.SEARCH_SIZE
            num_t = (t_size // stride) ** 2   # template token 数，如 16
            num_s = (s_size // stride) ** 2   # search  token 数，如 64
            t_h = t_w = t_size // stride      # 如 4
            s_h = s_w = s_size // stride      # 如 8
            print(f"  Grad-CAM: template={t_h}x{t_w}, search={s_h}x{s_w}, num_t={num_t}, num_s={num_s}")

            def compute_gradcam(feat_tensor, grad_tensor, n_tokens, h, w):
                """
                feat_tensor: [B, L, C]  (L = num_t + num_s)
                grad_tensor: [B, L, C]
                返回归一化到 [0,1] 的热力图 numpy [h, w]
                """
                # 取 search 部分的 token（后 num_s 个）
                f = feat_tensor[0, num_t:num_t + n_tokens, :].detach().cpu().float()  # [S, C]
                g = grad_tensor[0, num_t:num_t + n_tokens, :].float()                 # [S, C]
                # 对每个通道在 token 维度求均值作为权重
                weights = g.mean(dim=0)          # [C]
                cam = (f * weights).sum(dim=-1)  # [S]
                cam = torch.clamp(cam, min=0)    # ReLU
                cam = cam.reshape(h, w).numpy()  # [h, w]
                # 归一化
                cam_min, cam_max = cam.min(), cam.max()
                if cam_max - cam_min > 1e-8:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                return cam

            def compute_gradcam_template(feat_tensor, grad_tensor, n_tokens, h, w):
                """同上，但取 template 部分（前 num_t 个 token）"""
                f = feat_tensor[0, :n_tokens, :].detach().cpu().float()  # [T, C]
                g = grad_tensor[0, :n_tokens, :].float()
                weights = g.mean(dim=0)
                cam = (f * weights).sum(dim=-1)
                cam = torch.clamp(cam, min=0)
                cam = cam.reshape(h, w).numpy()
                cam_min, cam_max = cam.min(), cam.max()
                if cam_max - cam_min > 1e-8:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                return cam

            def overlay(img_np, cam, alpha=0.5):
                """将 cam [h,w] 叠加到 img_np [H,W,3] 上，返回 [H,W,3] uint8"""
                img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cam_uint8 = (cam * 255).astype(np.uint8)
                cam_resized = cv2.resize(cam_uint8, (img_bgr.shape[1], img_bgr.shape[0]),
                                         interpolation=cv2.INTER_CUBIC)
                heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
                blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
                return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            # 计算四张图
            results = {}
            for branch in ['rgb', 'tir']:
                if branch in feats and branch in grads:
                    ft = feats[branch]
                    gr = grads[branch]
                    results[f'{branch}_search']   = compute_gradcam(ft, gr, num_s, s_h, s_w)
                    results[f'{branch}_template'] = compute_gradcam_template(ft, gr, num_t, t_h, t_w)

            # 绘图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Grad-CAM  {seq_name}  Frame {frame_id:04d}', fontsize=14, fontweight='bold')

            pairs = [
                (axes[0, 0], 'rgb_search',   search_rgb_img,   'RGB Search'),
                (axes[0, 1], 'tir_search',   search_tir_img,   'TIR Search'),
                (axes[1, 0], 'rgb_template', template_rgb_img, 'RGB Template'),
                (axes[1, 1], 'tir_template', template_tir_img, 'TIR Template'),
            ]
            for ax, key, img, title in pairs:
                if key in results and img is not None:
                    vis = overlay(img, results[key])
                    ax.imshow(vis)
                elif key in results:
                    ax.imshow(results[key], cmap='jet', vmin=0, vmax=1)
                else:
                    ax.axis('off')
                ax.set_title(title)
                ax.axis('off')

            os.makedirs(self.feature_save_path, exist_ok=True)
            save_path = os.path.join(self.feature_save_path, f'{frame_id:06d}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Grad-CAM 保存: {save_path}")

        except Exception as e:
            print(f"  ✗ Grad-CAM 内部错误: {e}")
            import traceback
            traceback.print_exc()

    def _save_gradcam(self, feats, grads, search_rgb_img, search_tir_img,
                       template_rgb_img, template_tir_img, seq_name, frame_id,
                       gt_bbox_in_search=None):
        """
        [改进版] Target-specific Grad-CAM。
        方案A: 用 GT 框中心位置的 score_map 响应值作为 scalar（而非 max），
               梯度精准指向真实目标位置。
               gt_bbox_in_search: [x1,y1,w,h]（在 search patch 坐标系内，像素单位），
               若为 None 则回退到 score_map.max()。
        方案C: 计算 Search Grad-CAM 时只使用 Search 对应的 token 梯度和特征，
               切断来自 Template token 的干扰，热力图更纯净。
        """
        try:
            stride = self.cfg.MODEL.BACKBONE.STRIDE
            t_size = self.cfg.TEST.TEMPLATE_SIZE
            s_size = self.cfg.TEST.SEARCH_SIZE
            num_t = (t_size // stride) ** 2
            num_s = (s_size // stride) ** 2
            t_h = t_w = t_size // stride
            s_h = s_w = s_size // stride
            print(f"  Grad-CAM v2: template={t_h}x{t_w}, search={s_h}x{s_w}, "
                  f"num_t={num_t}, num_s={num_s}, "
                  f"gt={'yes' if gt_bbox_in_search is not None else 'no(fallback)'}")

            # ------------------------------------------------------------------
            # 方案C: 只取 Search token 的梯度/特征（切断 Template 干扰）
            # ------------------------------------------------------------------
            def compute_search_gradcam_pure(feat_tensor, grad_tensor, h, w):
                """
                只使用 Search 部分（后 num_s 个）的特征和梯度，
                Template token 的梯度完全不参与权重计算。
                """
                f = feat_tensor[0, num_t:num_t + num_s, :].detach().cpu().float()  # [S, C]
                g = grad_tensor[0, num_t:num_t + num_s, :].float()                 # [S, C]
                # 方案C核心：权重只由 Search token 的梯度决定
                weights = g.mean(dim=0)           # [C]  仅 Search 梯度均值
                cam = (f * weights).sum(dim=-1)   # [S]
                cam = torch.clamp(cam, min=0)
                cam = cam.reshape(h, w).numpy()
                v_min, v_max = cam.min(), cam.max()
                if v_max - v_min > 1e-8:
                    cam = (cam - v_min) / (v_max - v_min)
                return cam

            def compute_template_gradcam_pure(feat_tensor, grad_tensor, h, w):
                """Template token（前 num_t 个），梯度同样只用 Template 部分。"""
                f = feat_tensor[0, :num_t, :].detach().cpu().float()  # [T, C]
                g = grad_tensor[0, :num_t, :].float()
                weights = g.mean(dim=0)
                cam = (f * weights).sum(dim=-1)
                cam = torch.clamp(cam, min=0)
                cam = cam.reshape(h, w).numpy()
                v_min, v_max = cam.min(), cam.max()
                if v_max - v_min > 1e-8:
                    cam = (cam - v_min) / (v_max - v_min)
                return cam

            def overlay(img_np, cam, alpha=0.5):
                img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
                cam_uint8 = (cam * 255).astype(np.uint8)
                cam_resized = cv2.resize(cam_uint8, (img_bgr.shape[1], img_bgr.shape[0]),
                                         interpolation=cv2.INTER_CUBIC)
                heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
                blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
                return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            # ------------------------------------------------------------------
            # 方案A: 用 GT 中心的 score_map 响应值反向传播
            # gt_bbox_in_search 已经是在 search patch（search_size x search_size）
            # 内的坐标，需要映射到 score_map（s_h x s_w）上
            # ------------------------------------------------------------------
            # score_map 存在 feats 里的额外 key
            score_map = feats.get('score_map', None)   # [B,1,s_h,s_w]
            if score_map is not None:
                if gt_bbox_in_search is not None:
                    x1, y1, w_gt, h_gt = gt_bbox_in_search
                    cx_px = x1 + w_gt / 2.0
                    cy_px = y1 + h_gt / 2.0
                    # 映射到 score_map 坐标
                    gt_col = int(np.clip(cx_px / s_size * s_w, 0, s_w - 1))
                    gt_row = int(np.clip(cy_px / s_size * s_h, 0, s_h - 1))
                    scalar = score_map[0, 0, gt_row, gt_col]
                    print(f"  方案A: GT中心 ({cx_px:.1f},{cy_px:.1f})px -> "
                          f"score_map[{gt_row},{gt_col}]={scalar.item():.4f}")
                else:
                    scalar = score_map.max()
                    print(f"  方案A fallback: score_map.max()={scalar.item():.4f}")

                # 重新反向传播（用 GT 中心响应值）
                self.network.zero_grad()
                scalar.backward(retain_graph=True)
                # 更新 grads（此时 hook 已捕获新梯度）
                # 注意：grads 在外部 hook 里已经更新，这里 grads 是最新的

            # 计算四张图（方案C：纯 token 分离）
            results = {}
            for branch in ['rgb', 'tir']:
                if branch in feats and branch in grads:
                    ft = feats[branch]
                    gr = grads[branch]
                    results[f'{branch}_search']   = compute_search_gradcam_pure(ft, gr, s_h, s_w)
                    results[f'{branch}_template'] = compute_template_gradcam_pure(ft, gr, t_h, t_w)

            # 绘图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            gt_str = '(GT-center)' if gt_bbox_in_search is not None else '(score_max)'
            fig.suptitle(f'Target-specific Grad-CAM {gt_str}  {seq_name}  Frame {frame_id:04d}',
                         fontsize=13, fontweight='bold')

            pairs = [
                (axes[0, 0], 'rgb_search',   search_rgb_img,   'RGB Search [pure-S]'),
                (axes[0, 1], 'tir_search',   search_tir_img,   'TIR Search [pure-S]'),
                (axes[1, 0], 'rgb_template', template_rgb_img, 'RGB Template [pure-T]'),
                (axes[1, 1], 'tir_template', template_tir_img, 'TIR Template [pure-T]'),
            ]
            for ax, key, img, title in pairs:
                if key in results and img is not None:
                    ax.imshow(overlay(img, results[key]))
                elif key in results:
                    ax.imshow(results[key], cmap='jet', vmin=0, vmax=1)
                else:
                    ax.axis('off')
                ax.set_title(title)
                ax.axis('off')

            os.makedirs(self.feature_save_path, exist_ok=True)
            save_path = os.path.join(self.feature_save_path, f'{frame_id:06d}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Target Grad-CAM 保存: {save_path}")

        except Exception as e:
            print(f"  ✗ Grad-CAM v2 内部错误: {e}")
            import traceback
            traceback.print_exc()

    def _save_features(self, feat, seq_name, frame_id, template_img=None, search_img=None):
        """保存和可视化Backbone特征（template/search分开显示）"""
        try:
            if feat is None or not isinstance(feat, torch.Tensor):
                return

            # 转为numpy
            feat_np = feat.detach().cpu().numpy() if isinstance(feat, torch.Tensor) else feat
            print(f"  原始特征shape: {feat_np.shape}")

            feat_temp_spatial = None
            feat_search_spatial = None
            feat_spatial = None

            if feat_np.ndim == 3:  # [B, L, C] ViT token格式
                B, L, C = feat_np.shape
                print(f"  B={B}, L={L}, C={C}")
                feat_single = feat_np[0]  # [L, C]

                # 从cfg中读取真实的template/search size和stride，计算token数
                stride = self.cfg.MODEL.BACKBONE.STRIDE
                template_size = self.cfg.TEST.TEMPLATE_SIZE if hasattr(self.cfg.TEST, 'TEMPLATE_SIZE') else self.params.template_size
                search_size = self.cfg.TEST.SEARCH_SIZE if hasattr(self.cfg.TEST, 'SEARCH_SIZE') else self.params.search_size
                num_temp_tokens = (template_size // stride) ** 2
                num_search_tokens = (search_size // stride) ** 2
                print(f"  template_size={template_size}, search_size={search_size}, stride={stride}")
                print(f"  num_temp_tokens={num_temp_tokens}, num_search_tokens={num_search_tokens}, L={L}")

                # 验证token数是否匹配
                if num_temp_tokens + num_search_tokens != L:
                    print(f"  警告: token数不匹配({num_temp_tokens}+{num_search_tokens}={num_temp_tokens+num_search_tokens} != {L})，回退为等分")
                    num_temp_tokens = L // 2
                    num_search_tokens = L - num_temp_tokens

                temp_tokens = feat_single[:num_temp_tokens, :]           # [T, C]
                search_tokens = feat_single[num_temp_tokens:num_temp_tokens + num_search_tokens, :]  # [S, C]

                # 计算template的patch grid（一般是正方形）
                t_edge = int(np.sqrt(num_temp_tokens))
                print(f"  template grid: {t_edge}x{t_edge}")
                feat_temp_spatial = temp_tokens.reshape(t_edge, t_edge, C).transpose(2, 0, 1)  # [C, H, W]

                # 计算search的patch grid
                s_edge = int(np.sqrt(num_search_tokens))
                if s_edge * s_edge != num_search_tokens:
                    factors = [(h, num_search_tokens // h) for h in range(1, num_search_tokens + 1) if num_search_tokens % h == 0]
                    s_h, s_w = min(factors, key=lambda x: abs(x[0] - x[1]))
                    print(f"  search grid (非正方形): {s_h}x{s_w}")
                else:
                    s_h, s_w = s_edge, s_edge
                    print(f"  search grid: {s_h}x{s_w}")
                feat_search_spatial = search_tokens.reshape(s_h, s_w, C).transpose(2, 0, 1)  # [C, H, W]

            elif feat_np.ndim == 4:  # [B, C, H, W]
                B, C, H, W = feat_np.shape
                print(f"  4D特征: B={B}, C={C}, H={H}, W={W}")
                feat_spatial = feat_np[0]  # [C, H, W]

            else:
                print(f"  不支持的特征维度: {feat_np.ndim}")
                return

            # ---- 绘图 ----
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f'{seq_name}  Frame {frame_id:04d}', fontsize=14, fontweight='bold')

            if feat_temp_spatial is not None:
                # Template/Search 热力图 + 叠加原图
                temp_map = feat_temp_spatial.mean(axis=0)    # [H, W]
                search_map = feat_search_spatial.mean(axis=0)

                ax = axes[0, 0]
                im0 = ax.imshow(temp_map, cmap='hot', aspect='auto')
                ax.set_title('Template Feature Map')
                plt.colorbar(im0, ax=ax)

                ax = axes[0, 1]
                im1 = ax.imshow(search_map, cmap='hot', aspect='auto')
                ax.set_title('Search Feature Map')
                plt.colorbar(im1, ax=ax)

                ax = axes[1, 0]
                if template_img is not None:
                    ax.imshow(template_img)
                    h_resized = cv2.resize(temp_map, (template_img.shape[1], template_img.shape[0]),
                                           interpolation=cv2.INTER_CUBIC)
                    ax.imshow(h_resized, cmap='jet', alpha=0.5)
                    ax.set_title('Template + Heatmap')
                else:
                    ax.imshow(temp_map, cmap='jet', aspect='auto')
                    ax.set_title('Template Feature Map (jet)')

                ax = axes[1, 1]
                if search_img is not None:
                    ax.imshow(search_img)
                    h_resized = cv2.resize(search_map, (search_img.shape[1], search_img.shape[0]),
                                           interpolation=cv2.INTER_CUBIC)
                    ax.imshow(h_resized, cmap='jet', alpha=0.5)
                    ax.set_title('Search + Heatmap')
                else:
                    ax.imshow(search_map, cmap='jet', aspect='auto')
                    ax.set_title('Search Feature Map (jet)')

            elif feat_spatial is not None:
                # 4D特征：通道均值 + 最大方差通道
                C, H, W = feat_spatial.shape
                spatial_mean = feat_spatial.mean(axis=0)

                ax = axes[0, 0]
                im0 = ax.imshow(spatial_mean, cmap='hot', aspect='auto')
                ax.set_title('Spatial Mean (all channels)')
                plt.colorbar(im0, ax=ax)

                channel_var = feat_spatial.var(axis=(1, 2))
                max_var_ch = int(np.argmax(channel_var))
                ax = axes[0, 1]
                im1 = ax.imshow(feat_spatial[max_var_ch], cmap='hot', aspect='auto')
                ax.set_title(f'Channel {max_var_ch} (Max Variance)')
                plt.colorbar(im1, ax=ax)

                channel_mean = feat_spatial.mean(axis=(1, 2))
                ax = axes[1, 0]
                ax.plot(channel_mean)
                ax.set_xlabel('Channel')
                ax.set_ylabel('Mean Activation')
                ax.set_title('Channel Mean Activation')
                ax.grid(True, alpha=0.3)

                ax = axes[1, 1]
                ax.axis('off')
                stats_text = (f"Shape: {feat_spatial.shape}\n"
                              f"Mean:  {feat_spatial.mean():.4f}\n"
                              f"Std:   {feat_spatial.std():.4f}\n"
                              f"Max:   {feat_spatial.max():.4f}\n"
                              f"Min:   {feat_spatial.min():.4f}")
                ax.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
                        verticalalignment='center', transform=ax.transAxes)

            # 保存图片
            os.makedirs(self.feature_save_path, exist_ok=True)
            save_path = os.path.join(self.feature_save_path, f'{frame_id:06d}.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  ✓ 保存特征图: {save_path}")

        except Exception as e:
            print(f"  ✗ 特征保存失败: {e}")
            import traceback
            traceback.print_exc()

    def initialize(self, image_v, image_i, info: dict):
        self.temps = []
        self.temps_score = []
        # forward the template once
        z_patch_arr_rgb, resize_factor_rgb, z_amask_arr_rgb = sample_target(image_v, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        z_patch_arr_tir, resize_factor_tir, z_amask_arr_tir = sample_target(image_i, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)

        # z_patch_arr_tir[:16,:16,:]*=0
        # z_patch_arr_tir[-16:,:16,:]=255.
        # z_patch_arr_rgb[:,-32:,:]=255
        self.z_patch_arr_rgb = z_patch_arr_rgb
        self.z_patch_arr_tir = z_patch_arr_tir

        template_rgb = self.preprocessor.process(z_patch_arr_rgb, z_amask_arr_rgb)
        template_tir = self.preprocessor.process(z_patch_arr_tir, z_amask_arr_tir)
        with torch.no_grad():
            self.z_dict1_rgb = template_rgb
            self.z_dict1_tir = template_tir
            self.z_dict = [self.z_dict1_rgb,self.z_dict1_tir]
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox_rgb = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_rgb,
                                                        template_rgb.tensors.device).squeeze(1)
            self.box_mask_z_rgb = generate_mask_cond(self.cfg, 1, template_rgb.tensors.device, template_bbox_rgb)

            template_bbox_tir = self.transform_bbox_to_crop(info['init_bbox'], resize_factor_tir,
                                                        template_tir.tensors.device).squeeze(1)
            self.box_mask_z_tir = generate_mask_cond(self.cfg, 1, template_tir.tensors.device, template_bbox_tir)

            self.box_mask_z = [self.box_mask_z_rgb,self.box_mask_z_tir]
            self.box_mask_z = self.box_mask_z[0]
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image_v,image_i, info: dict = None):
        H, W, _ = image_v.shape
        self.frame_id += 1
        x_patch_arr_rgb, resize_factor_rgb, x_amask_arr_rgb = sample_target(image_v, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr_tir, resize_factor_tir, x_amask_arr_tir = sample_target(image_i, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        # x_patch_arr_rgb*=0
        # x_patch_arr_tir*=0
        search_rgb = self.preprocessor.process(x_patch_arr_rgb, x_amask_arr_rgb)
        search_tir = self.preprocessor.process(x_patch_arr_tir, x_amask_arr_tir)

        x_dict_rgb = search_rgb
        x_dict_tir = search_tir
        x_dict = [x_dict_rgb, x_dict_tir]

        # ---- Grad-CAM 或普通推理 ----
        if hasattr(self, 'save_features') and self.save_features and hasattr(self, 'feature_save_path') and self.feature_save_path:
            # Grad-CAM 需要梯度，不能用 no_grad
            grad_cam_feats = {}   # {'rgb': tensor, 'tir': tensor}
            grad_cam_grads = {}   # {'rgb': tensor, 'tir': tensor}
            handles = []

            def _make_fwd_hook(name):
                def hook(module, input, output):
                    output.retain_grad()          # 保留中间张量梯度
                    grad_cam_feats[name] = output  # 保留计算图（不 detach）
                return hook

            def _make_bwd_hook(name):
                def hook(module, grad_input, grad_output):
                    grad_cam_grads[name] = grad_output[0].detach().cpu()
                return hook

            if hasattr(self.network.rgb_branch, 'norm'):
                handles.append(self.network.rgb_branch.norm.register_forward_hook(_make_fwd_hook('rgb')))
                handles.append(self.network.rgb_branch.norm.register_full_backward_hook(_make_bwd_hook('rgb')))
            if hasattr(self.network.tir_branch, 'norm'):
                handles.append(self.network.tir_branch.norm.register_forward_hook(_make_fwd_hook('tir')))
                handles.append(self.network.tir_branch.norm.register_full_backward_hook(_make_bwd_hook('tir')))

            with torch.enable_grad():
                out_dict = self.network.forward(
                    template=[self.z_dict[0].tensors, self.z_dict[1].tensors],
                    search=[x_dict[0].tensors, x_dict[1].tensors],
                    ce_template_mask=self.box_mask_z)

                # 先用 score_map.max() 做一次反向传播（hook 捕获梯度）
                score_map = out_dict['score_map']          # [B, 1, s_h, s_w]
                scalar = score_map.max()
                self.network.zero_grad()
                scalar.backward(retain_graph=True)
                # 将 score_map 存入 feats，供 _save_gradcam 方案A 使用
                grad_cam_feats['score_map'] = score_map

            for h in handles:
                h.remove()

            try:
                seq_name = info.get('seq_name', 'unknown') if info else 'unknown'
                frame_id = info.get('frame_id', self.frame_id) if info else self.frame_id

                # 方案A：将 GT bbox 映射到 search patch 坐标系
                gt_bbox_in_search = None
                if info and 'gt_bbox' in info:
                    gt = info['gt_bbox']  # [x1,y1,w,h] 原图坐标
                    if gt is not None:
                        try:
                            import torch as _torch
                            if isinstance(gt, _torch.Tensor):
                                gt = gt.tolist()
                            # 将原图坐标映射到 search patch 坐标系
                            cx_orig = gt[0] + gt[2] / 2.0
                            cy_orig = gt[1] + gt[3] / 2.0
                            half_side = 0.5 * self.params.search_size / resize_factor_rgb
                            cx_state = self.state[0] + 0.5 * self.state[2]
                            cy_state = self.state[1] + 0.5 * self.state[3]
                            cx_in_search = (cx_orig - cx_state + half_side)
                            cy_in_search = (cy_orig - cy_state + half_side)
                            w_in_search  = gt[2] * resize_factor_rgb
                            h_in_search  = gt[3] * resize_factor_rgb
                            gt_bbox_in_search = [
                                cx_in_search - w_in_search / 2,
                                cy_in_search - h_in_search / 2,
                                w_in_search, h_in_search
                            ]
                        except Exception as _e:
                            print(f"  GT bbox 映射失败: {_e}")

                self._save_gradcam(
                    grad_cam_feats, grad_cam_grads,
                    x_patch_arr_rgb, x_patch_arr_tir,
                    self.z_patch_arr_rgb, self.z_patch_arr_tir,
                    seq_name, frame_id,
                    gt_bbox_in_search=gt_bbox_in_search)
            except Exception as e:
                print(f"  ✗ Grad-CAM 保存失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            with torch.no_grad():
                out_dict = self.network.forward(
                    template=[self.z_dict[0].tensors, self.z_dict[1].tensors],
                    search=[x_dict[0].tensors, x_dict[1].tensors],
                    ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor_rgb).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor_rgb), H, W, margin=10)


        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:

                self.visdom.register((image_v, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(x_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'search_region_t')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_rgb).permute(2, 0, 1), 'image', 1, 'template_v')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_tir).permute(2, 0, 1), 'image', 1, 'template_t')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                
                # enc_opt = out_dict['backbone_feat'][0, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
                # opt = enc_opt.reshape()

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr_rgb, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                    
                if 'removed_indexes_s' in out_dict['aux_dict_rgb'] and out_dict['aux_dict_rgb']['removed_indexes_s']:
                    removed_indexes_s = out_dict['aux_dict_rgb']['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr_rgb, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search_v')
                    
                if 'removed_indexes_s' in out_dict['aux_dict_tir'] and out_dict['aux_dict_tir']['removed_indexes_s']:
                    removed_indexes_s = out_dict['aux_dict_tir']['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr_tir, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search_i')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor_rgb, resize_factor_rgb)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack_twobranch

"""
OSTrack CKD Actor with ViTKD Generator + Response Map Distillation

结合ViTKD生成器和Response Map蒸馏的训练Actor。
- ViTKD生成器：学生特征上采样到教师分辨率（80→320 tokens）
- Response Map蒸馏：在生成器输出上计算depthwise cross-correlation，学习定位分布
"""
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import math
from .ckd_loss_vitkd import get_ckd_loss  # ViTKD loss [ORIGINAL - 已注释]
from .ckd_loss_vitkd_reshape import get_ckd_loss  # ViTKD loss [RESHAPE版本 - 修复view()问题]
# from .ckd_loss_responsemap_new import get_response_map_loss  # Response Map loss (正确实现)



class OSTrack_CKD_ViTKD_Resp_Actor(BaseActor):
    """ 
    Actor for training OSTrack with ViTKD Generator + Response Map Distillation
    
    训练流程:
    1. 前向传播得到teacher和student特征
    2. ViTKD生成器将student特征上采样到320 tokens
    3. Content/Style蒸馏损失（在生成后的特征上）
    4. Response Map蒸馏损失（在生成后的特征上）
    5. Task losses（GIoU, L1, Focal）
    """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        
        # ViTKD loss（包含生成器）
        self.ckd_loss = get_ckd_loss(cfg)
        
        # Response Map loss
        self.response_loss = get_response_map_loss(cfg)
        
        # 将response_loss移到GPU（如果存在）
        if self.response_loss is not None:
            self.response_loss = self.response_loss.cuda()
        
        # 冻结配置
        self.freeze_backbone_epochs = getattr(cfg.TRAIN, 'FREEZE_BACKBONE_EPOCHS', 0)
        self.is_frozen = self.freeze_backbone_epochs > 0
        
        print(f"[ViTKD+Resp Actor] Initialized with:")
        print(f"  - ViTKD Generator: {cfg.TRAIN.get('VITKD_USE_GENERATOR', False)}")
        print(f"  - Response Map KD: {cfg.TRAIN.get('ENABLE_RESPONSE_KD', False)}")
        if self.response_loss:
            print(f"  - Response Temp: {cfg.TRAIN.get('RESPONSE_TEMP', 2.0)}")
            print(f"  - Response Weight: {cfg.TRAIN.get('RESPONSE_KD_WEIGHT', 0.5)}")
        if self.freeze_backbone_epochs > 0:
            print(f"  - Freeze Backbone: First {self.freeze_backbone_epochs} epochs")
    
    def unfreeze_backbone_if_needed(self, epoch):
        """在指定epoch解冻网络参数"""
        if self.is_frozen and epoch > self.freeze_backbone_epochs:
            print(f"\n{'='*80}")
            print(f"[Unfreeze] Epoch {epoch}: Unfreezing backbone parameters")
            print(f"[Unfreeze] Generator/Response Map warm-up completed")
            print(f"{'='*80}\n")
            
            unfrozen_count = 0
            for name, param in self.net.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += 1
            
            print(f"[Unfreeze] Unfrozen {unfrozen_count} parameters")
            self.is_frozen = False

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        template_list = []
        search_list = []
        
        # Check if data is in multi-frame format (tensor) or dual-resolution format (list)
        if isinstance(data['template_images'], list):
            # Dual resolution case
            for i in range(len(data['template_images'])):
                template_img_i = data['template_images'][i].view(-1, *data['template_images'][i].shape[2:])
                template_list.append(template_img_i)
            for i in range(len(data['search_images'])):
                search_img_i = data['search_images'][i].view(-1, *data['search_images'][i].shape[2:])
                search_list.append(search_img_i)
        else:
            # Single resolution multi-frame case
            for i in range(data['template_images'].shape[0]):
                template_img_i = data['template_images'][i]
                template_list.append(template_img_i)
            for i in range(data['search_images'].shape[0]):
                search_img_i = data['search_images'][i]
                search_list.append(search_img_i)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, self.cfg.TRAIN.BATCH_SIZE, template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=torch.stack(template_list),
                            search=torch.stack(search_list),
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        """
        计算所有损失，包括:
        1. Task losses (GIoU, L1, Focal)
        2. Teacher losses (教师分支的监督)
        3. Content/Style distillation (ViTKD生成器)
        4. Response Map distillation (新增)
        """
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]
        
        # 学生网络的 GT heatmap（低分辨率）
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        
        # 教师网络的 GT heatmap（高分辨率）
        teacher_search_size = getattr(self.cfg.TEACHER_SIZE.SEARCH, 'SIZE', 256)
        gt_gaussian_maps_teacher = generate_heatmap(gt_dict['search_anno'], teacher_search_size, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps_teacher = gt_gaussian_maps_teacher[-1].unsqueeze(1)

        # ========== Task Losses ==========
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # ========== Teacher Losses ==========
        loss_t_tir = torch.tensor(0.0, device=l1_loss.device)
        if "out_t_tir" in pred_dict:
            pred_boxes = pred_dict['out_t_tir']['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            
            try:
                giou_loss_t_tir, iou_t_tir = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss_t_tir, iou_t_tir = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            
            l1_loss_t_tir = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            
            if 'score_map' in pred_dict['out_t_tir']:
                location_loss_t_tir = self.objective['focal'](pred_dict['out_t_tir']['score_map'], gt_gaussian_maps_teacher)
            else:
                location_loss_t_tir = torch.tensor(0.0, device=l1_loss.device)

            loss_t_tir = self.loss_weight['giou'] * giou_loss_t_tir + \
                    self.loss_weight['l1'] * l1_loss_t_tir + \
                    self.loss_weight['focal'] * location_loss_t_tir
        
        loss_t_rgb = torch.tensor(0.0, device=l1_loss.device)
        if "out_t_rgb" in pred_dict:
            pred_boxes = pred_dict['out_t_rgb']['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            
            try:
                giou_loss_t_rgb, iou_t_rgb = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss_t_rgb, iou_t_rgb = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            
            l1_loss_t_rgb = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
            
            if 'score_map' in pred_dict['out_t_rgb']:
                location_loss_t_rgb = self.objective['focal'](pred_dict['out_t_rgb']['score_map'], gt_gaussian_maps_teacher)
            else:
                location_loss_t_rgb = torch.tensor(0.0, device=l1_loss.device)

            loss_t_rgb = self.loss_weight['giou'] * giou_loss_t_rgb + \
                    self.loss_weight['l1'] * l1_loss_t_rgb + \
                    self.loss_weight['focal'] * location_loss_t_rgb

        # ========== Content and Style Distillation (ViTKD) ==========
        content_loss = torch.tensor(0.0, device=l1_loss.device)
        style_loss = torch.tensor(0.0, device=l1_loss.device)
        
        if 't_x_tir' in pred_dict:
            if self.cfg.TRAIN.ENABLE_CONTENT:
                if self.cfg.TRAIN.STOP_CONTENT_GRADIENT:
                    for layer_idx, (p,y) in enumerate(zip(pred_dict['aux_dict_t_tir']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values())):
                        content_loss += self.ckd_loss.content_distill(p.detach(), y, layer_idx=layer_idx, score_map_gt=gt_gaussian_maps)
                    for layer_idx, (p,y) in enumerate(zip(pred_dict['aux_dict_t_rgb']['x_list'].values(), pred_dict['aux_dict_rgb']['x_list'].values())):
                        content_loss += self.ckd_loss.content_distill(p.detach(), y, layer_idx=layer_idx, score_map_gt=gt_gaussian_maps)
                else:
                    for layer_idx, (p,y) in enumerate(zip(pred_dict['aux_dict_t_tir']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values())):
                        content_loss += self.ckd_loss.content_distill(p, y, layer_idx=layer_idx, score_map_gt=gt_gaussian_maps)
                    for layer_idx, (p,y) in enumerate(zip(pred_dict['aux_dict_t_rgb']['x_list'].values(), pred_dict['aux_dict_rgb']['x_list'].values())):
                        content_loss += self.ckd_loss.content_distill(p, y, layer_idx=layer_idx, score_map_gt=gt_gaussian_maps)
                    
            if self.cfg.TRAIN.ENABLE_STYLE:
                if self.cfg.TRAIN.STOP_STYLE_GRADIENT:
                    for p,y in zip(pred_dict['aux_dict_rgb']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values()):
                        style_loss += self.ckd_loss.style_distill(p.detach(), y, score_map_gt=gt_gaussian_maps)
                else:
                    for p,y in zip(pred_dict['aux_dict_rgb']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values()):
                        style_loss += self.ckd_loss.style_distill(p, y, score_map_gt=gt_gaussian_maps)

        # ========== Response Map Distillation (新增) ==========
        response_loss = torch.tensor(0.0, device=l1_loss.device)
        R_peak_ratio = 0.0
        scale_factor = 0.0  # 初始化scale_factor，避免UnboundLocalError
        
        # 检查条件
        has_response_module = self.response_loss is not None
        has_teacher_feat = 't_x_tir' in pred_dict
        
        # 诊断输出（无条件打印，但只打印一次）
        if torch.rand(1).item() < 0.01:  # 1%概率
            print(f"[RESP-COND] has_response_module={has_response_module}, has_teacher_feat={has_teacher_feat}")
            print(f"[RESP-COND] pred_dict keys: {list(pred_dict.keys())}")
        
        if has_response_module and has_teacher_feat:
            # ========== 准备特征：320 tokens → 2D特征图 ==========
            # Teacher特征（原始320 tokens）
            teacher_rgb_last = list(pred_dict['aux_dict_t_rgb']['x_list'].values())[-1]  # (B, 320, C)
            teacher_tir_last = list(pred_dict['aux_dict_t_tir']['x_list'].values())[-1]  # (B, 320, C)
            
            # Student特征（原始80 tokens → 通过generator上采样到320）
            student_rgb_raw = list(pred_dict['aux_dict_rgb']['x_list'].values())[-1]  # (B, 80, C)
            student_tir_raw = list(pred_dict['aux_dict_tir']['x_list'].values())[-1]  # (B, 80, C)
            
            # 使用ckd_loss的_process_student_feature获取生成后的特征
            student_rgb_generated = self.ckd_loss._process_student_feature(
                student_rgb_raw, teacher_rgb_last, layer_idx=11)  # (B, 320, C)
            student_tir_generated = self.ckd_loss._process_student_feature(
                student_tir_raw, teacher_tir_last, layer_idx=11)  # (B, 320, C)
            
            # ========== 分割 template 和 search tokens ==========
            # Teacher: 前64个template (8×8) + 后256个search (16×16)
            teacher_rgb_tmpl = teacher_rgb_last[:, :64, :]    # (B, 64, C)
            teacher_rgb_srch = teacher_rgb_last[:, 64:, :]    # (B, 256, C)
            teacher_tir_tmpl = teacher_tir_last[:, :64, :]    # (B, 64, C)
            teacher_tir_srch = teacher_tir_last[:, 64:, :]    # (B, 256, C)
            
            # Student: 前64个template (8×8) + 后256个search (16×16)
            student_rgb_tmpl = student_rgb_generated[:, :64, :]    # (B, 64, C)
            student_rgb_srch = student_rgb_generated[:, 64:, :]    # (B, 256, C)
            student_tir_tmpl = student_tir_generated[:, :64, :]    # (B, 64, C)
            student_tir_srch = student_tir_generated[:, 64:, :]    # (B, 256, C)
            
            # ========== Reshape to 2D特征图 ==========
            B, _, C = teacher_rgb_last.shape
            
            # Template: 64 tokens → 8×8
            teacher_rgb_tmpl_2d = teacher_rgb_tmpl.permute(0, 2, 1).reshape(B, C, 8, 8)  # (B, C, 8, 8)
            teacher_tir_tmpl_2d = teacher_tir_tmpl.permute(0, 2, 1).reshape(B, C, 8, 8)
            student_rgb_tmpl_2d = student_rgb_tmpl.permute(0, 2, 1).reshape(B, C, 8, 8)
            student_tir_tmpl_2d = student_tir_tmpl.permute(0, 2, 1).reshape(B, C, 8, 8)
            
            # Search: 256 tokens → 16×16
            teacher_rgb_srch_2d = teacher_rgb_srch.permute(0, 2, 1).reshape(B, C, 16, 16)  # (B, C, 16, 16)
            teacher_tir_srch_2d = teacher_tir_srch.permute(0, 2, 1).reshape(B, C, 16, 16)
            student_rgb_srch_2d = student_rgb_srch.permute(0, 2, 1).reshape(B, C, 16, 16)
            student_tir_srch_2d = student_tir_srch.permute(0, 2, 1).reshape(B, C, 16, 16)
            
            # ========== 计算Response Map Loss ==========
            # RGB分支：传入4个2D特征图
            loss_resp_rgb, metrics_rgb = self.response_loss(
                student_rgb_srch_2d,   # student search (B, C, 16, 16)
                student_rgb_tmpl_2d,   # student template (B, C, 8, 8)
                teacher_rgb_srch_2d,   # teacher search (B, C, 16, 16)
                teacher_rgb_tmpl_2d    # teacher template (B, C, 8, 8)
            )
            
            # TIR分支：传入4个2D特征图
            loss_resp_tir, metrics_tir = self.response_loss(
                student_tir_srch_2d,   # student search (B, C, 16, 16)
                student_tir_tmpl_2d,   # student template (B, C, 8, 8)
                teacher_tir_srch_2d,   # teacher search (B, C, 16, 16)
                teacher_tir_tmpl_2d    # teacher template (B, C, 8, 8)
            )
            
            # 平均两个分支
            response_loss = (loss_resp_rgb + loss_resp_tir) / 2.0
            R_peak_ratio = (metrics_rgb['R_peak_ratio'] + metrics_tir['R_peak_ratio']) / 2.0
            scale_factor = metrics_rgb['scale_factor']  # 获取可学习的缩放因子
            
            # 诊断：打印loss值
            if torch.rand(1).item() < 0.01:
                print(f"[RESP-LOSS] rgb={loss_resp_rgb.item():.6f}, tir={loss_resp_tir.item():.6f}, avg={response_loss.item():.6f}, R_peak={R_peak_ratio:.4f}, scale={scale_factor:.2f}")

        # ========== Total Loss ==========
        loss_total = self.loss_weight['giou'] * giou_loss + \
                self.loss_weight['l1'] * l1_loss + \
                self.loss_weight['focal'] * location_loss
        
        # 添加蒸馏损失
        loss = loss_total + \
               self.loss_weight['style'] * style_loss + \
               self.loss_weight['content'] * content_loss + \
               loss_t_rgb + loss_t_tir
        
        # 添加response map蒸馏损失
        if self.response_loss is not None:
            response_weight = self.cfg.TRAIN.get('RESPONSE_KD_WEIGHT', 0.5)
            loss = loss + response_weight * response_loss
        
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {
                "Loss/total": loss_total.item(),
                "Loss/style": style_loss.item(),
                "Loss/content": content_loss.item(),
                "Loss/response": response_loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "IoU": mean_iou.item(),
            }
            
            # 添加response map监控指标
            if R_peak_ratio > 0:
                status["R_peak_ratio"] = R_peak_ratio
            if self.response_loss is not None:
                status["scale_factor"] = scale_factor  # 监控可学习的缩放因子
            
            return loss, status
        else:
            return loss

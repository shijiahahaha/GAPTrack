"""
OSTrack CKD Actor for RealHigh Version

RealHigh版本的训练Actor：
- 使用 ckd_loss_realhigh 来处理教师-学生特征尺寸不匹配
- 教师使用真实高分辨率，学生使用低分辨率
"""
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
# from lib.models.ostrack_twobranch_qkv.ostrack import OSTrack_twobranch
import math
from .ckd_loss import get_ckd_loss  # 使用 realhigh 版本的损失函数



class OSTrack_CKD_Actor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.ckd_loss = get_ckd_loss(cfg)

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
        # currently only support 1 template and 1 search region
        # data['template_images'] shape: (num_frames, batch, 3, H, W)
        # For single-resolution: just stack frames
        # For dual-resolution (teacher+student): would be list of 2 tensors
        
        template_list = []
        search_list = []
        
        # Check if data is in multi-frame format (tensor) or dual-resolution format (list)
        if isinstance(data['template_images'], list):
            # Dual resolution case: each element is a tensor for one resolution version
            for i in range(len(data['template_images'])):
                template_img_i = data['template_images'][i].view(-1, *data['template_images'][i].shape[2:])
                template_list.append(template_img_i)
            for i in range(len(data['search_images'])):
                search_img_i = data['search_images'][i].view(-1, *data['search_images'][i].shape[2:])
                search_list.append(search_img_i)
        else:
            # Single resolution multi-frame case: stack all frames
            # data['template_images'].shape: (num_frames, batch, 3, H, W)
            for i in range(data['template_images'].shape[0]):
                template_img_i = data['template_images'][i]  # (batch, 3, H, W)
                template_list.append(template_img_i)
            for i in range(data['search_images'].shape[0]):
                search_img_i = data['search_images'][i]  # (batch, 3, H, W)
                search_list.append(search_img_i)

        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

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
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        
        # 学生的 gt_gaussian_maps (低分辨率)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)    # B, 1, 8, 8 (for search_size=128)
        
        # 教师的 gt_gaussian_maps (高分辨率) - 如果配置中有 TEACHER_SIZE
        if hasattr(self.cfg, 'TEACHER_SIZE') and hasattr(self.cfg.TEACHER_SIZE, 'SEARCH'):
            gt_gaussian_maps_teacher = generate_heatmap(gt_dict['search_anno'], self.cfg.TEACHER_SIZE.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps_teacher = gt_gaussian_maps_teacher[-1].unsqueeze(1)  # B, 1, 16, 16 (for search_size=256)
        else:
            # 如果没有 TEACHER_SIZE 配置，教师和学生使用相同尺寸
            gt_gaussian_maps_teacher = gt_gaussian_maps

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        # for tir teacher
        loss_t_tir = torch.tensor(0.0, device=l1_loss.device)
        if "out_t_tir" in pred_dict:
            # Get boxes
            pred_boxes = pred_dict['out_t_tir']['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            # compute giou and iou
            try:
                giou_loss_t_tir, iou_t_tir = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss_t_tir, iou_t_tir = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss_t_tir = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict['out_t_tir']:
                location_loss_t_tir = self.objective['focal'](pred_dict['out_t_tir']['score_map'], gt_gaussian_maps_teacher)
            else:
                location_loss_t_tir = torch.tensor(0.0, device=l1_loss.device)

            loss_t_tir = self.loss_weight['giou'] * giou_loss_t_tir + \
                    self.loss_weight['l1'] * l1_loss_t_tir + \
                        self.loss_weight['focal'] * location_loss_t_tir
        ###########################################
            
        
        loss_t_rgb = torch.tensor(0.0, device=l1_loss.device)
        if "out_t_rgb" in pred_dict:
            # Get boxes
            pred_boxes = pred_dict['out_t_rgb']['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            # compute giou and iou
            try:
                giou_loss_t_rgb, iou_t_rgb = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss_t_rgb, iou_t_rgb = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss_t_rgb = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict['out_t_rgb']:
                location_loss_t_rgb = self.objective['focal'](pred_dict['out_t_rgb']['score_map'], gt_gaussian_maps_teacher)
            else:
                location_loss_t_rgb = torch.tensor(0.0, device=l1_loss.device)

            loss_t_rgb = self.loss_weight['giou'] * giou_loss_t_rgb + \
                    self.loss_weight['l1'] * l1_loss_t_rgb + \
                        self.loss_weight['focal'] * location_loss_t_rgb
        ###########################################

        # content and style
        content_loss = torch.tensor(0.0, device=l1_loss.device)
        style_loss = torch.tensor(0.0, device=l1_loss.device)
        if 't_x_tir' in pred_dict:
            if self.cfg.TRAIN.ENABLE_CONTENT:
                if self.cfg.TRAIN.STOP_CONTENT_GRADIENT:
                    for p,y in zip(pred_dict['aux_dict_t_tir']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values()):
                        content_loss += self.ckd_loss.content_distill(p.detach(), y, score_map_gt=gt_gaussian_maps)
                    for p,y in zip(pred_dict['aux_dict_t_rgb']['x_list'].values(), pred_dict['aux_dict_rgb']['x_list'].values()):
                        content_loss += self.ckd_loss.content_distill(p.detach(), y, score_map_gt=gt_gaussian_maps)
                else:
                    for p,y in zip(pred_dict['aux_dict_t_tir']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values()):
                        content_loss += self.ckd_loss.content_distill(p, y, score_map_gt=gt_gaussian_maps)
                    for p,y in zip(pred_dict['aux_dict_t_rgb']['x_list'].values(), pred_dict['aux_dict_rgb']['x_list'].values()):
                        content_loss += self.ckd_loss.content_distill(p, y, score_map_gt=gt_gaussian_maps)
                    
            if self.cfg.TRAIN.ENABLE_STYLE:
                if self.cfg.TRAIN.STOP_STYLE_GRADIENT:
                    for p,y in zip(pred_dict['aux_dict_rgb']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values()):
                        style_loss += self.ckd_loss.style_distill(p.detach(), y, score_map_gt=gt_gaussian_maps)
                else:
                    for p,y in zip(pred_dict['aux_dict_rgb']['x_list'].values(), pred_dict['aux_dict_tir']['x_list'].values()):
                        style_loss += self.ckd_loss.style_distill(p, y, score_map_gt=gt_gaussian_maps)

        # weighted sum
        loss_total = self.loss_weight['giou'] * giou_loss + \
                self.loss_weight['l1'] * l1_loss + \
                    self.loss_weight['focal'] * location_loss
        loss = loss_total + self.loss_weight['style'] * style_loss + \
                self.loss_weight['content'] * content_loss + loss_t_rgb + loss_t_tir
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss_total.item(),
                      "Loss/style": style_loss.item(),
                      "Loss/content": content_loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
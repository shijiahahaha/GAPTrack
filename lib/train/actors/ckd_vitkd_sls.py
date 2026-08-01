"""
OSTrack CKD Actor with SLS Loss (ViTKD + SLS)

This actor extends the ViTKD actor by adding SLS (Scale-Location Sensitive) loss
for student model's bounding box predictions.

SLS Loss = LS (Scale Loss) + LL (Location Loss)
- Applied to student predictions only
- Complements existing GIoU, L1, and focal losses

Supports both standard and dynamic SLS loss:
- Standard SLS: Fixed weight for scale and location components
- Dynamic SLS: Adaptive scheduling of scale/location weights during training
"""

from .ckd_vitkd import OSTrack_CKD_Actor
from .sls_loss import SLS_Loss
import torch
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy

# Try to import dynamic SLS loss if available
try:
    from .sls_loss_dynamic import Dynamic_SLS_Loss
    DYNAMIC_SLS_AVAILABLE = True
except:
    DYNAMIC_SLS_AVAILABLE = False


class OSTrack_CKD_ViTKD_SLS_Actor(OSTrack_CKD_Actor):
    """
    Actor with ViTKD distillation + SLS loss for student predictions
    Supports both standard and dynamic SLS loss based on configuration
    """
    
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective, loss_weight, settings, cfg)
        
        # Check if dynamic SLS is enabled in config
        self.enable_dynamic_sls = getattr(cfg.TRAIN, 'ENABLE_DYNAMIC_SLS', False) and DYNAMIC_SLS_AVAILABLE
        
        # Get SLS loss weight from config
        self.sls_weight = getattr(cfg.TRAIN, 'SLS_WEIGHT', 1.0)
        
        # Get search image size for polar coordinate calculation
        self.search_size = cfg.DATA.SEARCH.SIZE
        
        # Initialize appropriate SLS loss module
        if self.enable_dynamic_sls:
            # Initialize Dynamic SLS Loss
            self.sls_loss = Dynamic_SLS_Loss(
                eps=1e-7,
                reduction='mean',
                initial_scale_weight=getattr(cfg.TRAIN, 'DYNAMIC_SLS_INITIAL_SCALE_WEIGHT', 0.3),
                final_scale_weight=getattr(cfg.TRAIN, 'DYNAMIC_SLS_FINAL_SCALE_WEIGHT', 0.7),
                schedule_power=getattr(cfg.TRAIN, 'DYNAMIC_SLS_SCHEDULE_POWER', 1.0),
                use_difficulty_weighting=getattr(cfg.TRAIN, 'DYNAMIC_SLS_USE_DIFFICULTY_WEIGHTING', False),
                normalize_losses=getattr(cfg.TRAIN, 'DYNAMIC_SLS_NORMALIZE_LOSSES', True),
                total_epochs=cfg.TRAIN.EPOCH
            )
            print(f"[SLS Loss] Using DYNAMIC SLS Loss")
            print(f"[SLS Loss] Initial scale weight: {self.sls_loss.initial_scale_weight}")
            print(f"[SLS Loss] Final scale weight: {self.sls_loss.final_scale_weight}")
            print(f"[SLS Loss] Schedule power: {self.sls_loss.schedule_power}")
            print(f"[SLS Loss] Total epochs: {self.sls_loss.total_epochs}")
        else:
            # Initialize Standard SLS Loss
            self.sls_loss = SLS_Loss(eps=1e-7, reduction='mean')
            print(f"[SLS Loss] Using STANDARD SLS Loss")
        
        print(f"[SLS Loss] Loss weight: {self.sls_weight}")
        print(f"[SLS Loss] Using search size={self.search_size} for polar coordinates")
    
    def set_epoch(self, epoch):
        """
        Set current epoch for dynamic SLS scheduling
        
        Args:
            epoch: Current training epoch (0-based)
        """
        if self.enable_dynamic_sls and hasattr(self.sls_loss, 'set_epoch_info'):
            self.sls_loss.set_epoch_info(epoch)
    
    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        """
        Compute losses including SLS loss for student predictions
        
        Override parent method to add SLS loss
        """
        # Get original losses from parent class
        loss, status = super().compute_losses(pred_dict, gt_dict, return_status)
        
        # Add SLS loss for student predictions
        sls_loss = self._compute_sls_loss(pred_dict, gt_dict)
        
        # Add to total loss
        loss = loss + self.sls_weight * sls_loss
        
        # Update status
        if return_status:
            status['Loss/sls'] = sls_loss.item()
            status['Loss/sls_weighted'] = (self.sls_weight * sls_loss).item()
            
            # Log dynamic SLS scheduling weights if enabled
            if self.enable_dynamic_sls and hasattr(self.sls_loss, 'current_scale_weight'):
                status['SLS/scale_weight'] = self.sls_loss.current_scale_weight
                status['SLS/location_weight'] = self.sls_loss.current_location_weight
        
        return loss, status
    
    def _compute_sls_loss(self, pred_dict, gt_dict):
        """
        Compute SLS loss for student model predictions
        
        Args:
            pred_dict: Dictionary containing model predictions
                - 'pred_boxes': Student predictions (B, N, 4) in [cx, cy, w, h] format
            gt_dict: Dictionary containing ground truth
                - 'search_anno': GT boxes (Ns, B, 4) in [x, y, w, h] format
                
        Returns:
            sls_loss: Scalar SLS loss value
        """
        # Get student predictions
        pred_boxes = pred_dict['pred_boxes']  # (B, N, 4)
        
        # Get ground truth boxes
        gt_bbox = gt_dict['search_anno'][-1]  # (B, 4) in [x1, y1, w, h] format
        
        if torch.isnan(pred_boxes).any():
            print("[Warning] NaN detected in pred_boxes, returning zero SLS loss")
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # Convert GT from [x, y, w, h] to [cx, cy, w, h]
        gt_boxes_cxcywh = gt_bbox.clone()
        gt_boxes_cxcywh[:, 0] = gt_bbox[:, 0] + gt_bbox[:, 2] / 2  # cx = x + w/2
        gt_boxes_cxcywh[:, 1] = gt_bbox[:, 1] + gt_bbox[:, 3] / 2  # cy = y + h/2
        # w, h remain the same
        
        # Clamp to valid range [0, 1]
        gt_boxes_cxcywh = gt_boxes_cxcywh.clamp(min=0.0, max=1.0)
        
        # Handle multiple queries (N > 1)
        num_queries = pred_boxes.size(1)
        if num_queries > 1:
            # Repeat GT boxes for each query
            gt_boxes_expanded = gt_boxes_cxcywh[:, None, :].repeat(1, num_queries, 1)
            gt_boxes_expanded = gt_boxes_expanded.view(-1, 4)
            pred_boxes_flat = pred_boxes.view(-1, 4)
        else:
            gt_boxes_expanded = gt_boxes_cxcywh
            pred_boxes_flat = pred_boxes.squeeze(1)
        
        # Compute SLS loss
        # Pass image size for better polar coordinate calculation
        sls_loss = self.sls_loss(
            pred_boxes_flat, 
            gt_boxes_expanded,
            image_size=(self.search_size, self.search_size)
        )
        
        return sls_loss


# For backward compatibility
OSTrack_CKD_SLS_Actor = OSTrack_CKD_ViTKD_SLS_Actor

"""
SLS Loss: Scale-Location Sensitive Loss for Bounding Box Regression

This loss combines:
1. LS (Scale Loss): Weighted IoU loss considering area variance
2. LL (Location Loss): Position loss using polar coordinate system

Formula:
    SLS = LS + LL
    
    LS = 1 - w * IoU
    where:
        IoU = intersection / union
        w = (min(area_pred, area_gt) + var) / (max(area_pred, area_gt) + var)
        var = ((area_pred - area_gt)^2) / 2
    
    LL = radial_loss + angular_loss
    where:
        radial_loss = 1 - min(d_p, d_gt) / max(d_p, d_gt)
        angular_loss = 4/pi^2 * (theta_p - theta_gt)^2
        
        For centroid (x, y):
            d = sqrt(x^2 + y^2)       # polar radius
            theta = arctan2(y, x)      # polar angle
"""

import torch
import torch.nn as nn
import math


class SLS_Loss(nn.Module):
    """
    Scale-Location Sensitive Loss
    
    Args:
        eps: Small value to avoid division by zero
        reduction: 'mean' or 'sum' or 'none'
    """
    
    def __init__(self, eps=1e-7, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, pred_boxes, gt_boxes, image_size=None):
        """
        Args:
            pred_boxes: (N, 4) in format [cx, cy, w, h] (normalized 0-1)
            gt_boxes: (N, 4) in format [cx, cy, w, h] (normalized 0-1)
            image_size: (H, W) for converting to pixel coordinates, optional
            
        Returns:
            loss: SLS loss value
        """
        # If image_size provided, use it for better polar coordinate calculation
        if image_size is not None:
            H, W = image_size
            # Convert to pixel coordinates for more accurate polar calculation
            pred_boxes_pixel = pred_boxes.clone()
            gt_boxes_pixel = gt_boxes.clone()
            pred_boxes_pixel[:, [0, 2]] *= W
            pred_boxes_pixel[:, [1, 3]] *= H
            gt_boxes_pixel[:, [0, 2]] *= W
            gt_boxes_pixel[:, [1, 3]] *= H
        else:
            pred_boxes_pixel = pred_boxes
            gt_boxes_pixel = gt_boxes
        
        # Compute LS (Scale Loss)
        ls_loss = self._compute_scale_loss(pred_boxes, gt_boxes)
        
        # Compute LL (Location Loss) using polar coordinates
        ll_loss = self._compute_location_loss(pred_boxes_pixel, gt_boxes_pixel)
        
        # Total SLS loss
        sls_loss = ls_loss + ll_loss
        
        if self.reduction == 'mean':
            return sls_loss.mean()
        elif self.reduction == 'sum':
            return sls_loss.sum()
        else:
            return sls_loss
    
    def _compute_scale_loss(self, pred_boxes, gt_boxes):
        """
        Compute LS = 1 - w * IoU
        
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h]
            gt_boxes: (N, 4) [cx, cy, w, h]
        """
        # Convert to [x1, y1, x2, y2] for IoU calculation
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
        gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
        gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
        gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2
        
        # Compute intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Compute areas
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
        
        # Compute union
        union_area = pred_area + gt_area - inter_area
        
        # Compute IoU
        iou = inter_area / (union_area + self.eps)
        
        # Compute area variance
        var = ((pred_area - gt_area) ** 2) / 2
        
        # Compute weight w
        area_min = torch.min(pred_area, gt_area)
        area_max = torch.max(pred_area, gt_area)
        w = (area_min + var) / (area_max + var + self.eps)
        
        # Compute LS
        ls = 1 - w * iou
        
        return ls
    
    def _compute_location_loss(self, pred_boxes, gt_boxes):
        """
        Compute LL = radial_loss + angular_loss
        Using polar coordinate system
        
        Args:
            pred_boxes: (N, 4) [cx, cy, w, h] (in pixel coordinates if available)
            gt_boxes: (N, 4) [cx, cy, w, h] (in pixel coordinates if available)
        """
        # Extract centroids
        pred_cx = pred_boxes[:, 0]
        pred_cy = pred_boxes[:, 1]
        gt_cx = gt_boxes[:, 0]
        gt_cy = gt_boxes[:, 1]
        
        # Convert to polar coordinates
        # d = sqrt(x^2 + y^2)
        # theta = arctan2(y, x)
        
        pred_d = torch.sqrt(pred_cx ** 2 + pred_cy ** 2 + self.eps)
        pred_theta = torch.atan2(pred_cy, pred_cx)
        
        gt_d = torch.sqrt(gt_cx ** 2 + gt_cy ** 2 + self.eps)
        gt_theta = torch.atan2(gt_cy, gt_cx)
        
        # Radial loss: 1 - min(d_p, d_gt) / max(d_p, d_gt)
        d_min = torch.min(pred_d, gt_d)
        d_max = torch.max(pred_d, gt_d)
        radial_loss = 1 - d_min / (d_max + self.eps)
        
        # Angular loss: 4/pi^2 * (theta_p - theta_gt)^2
        # Note: atan2 returns values in [-pi, pi]
        theta_diff = pred_theta - gt_theta
        
        # Handle angle wrapping (e.g., -179° to 179° should be 2° not 358°)
        # Normalize to [-pi, pi]
        theta_diff = torch.atan2(torch.sin(theta_diff), torch.cos(theta_diff))
        
        angular_loss = (4 / (math.pi ** 2)) * (theta_diff ** 2)
        
        # Total location loss
        ll = radial_loss + angular_loss
        
        return ll


class SLS_Loss_with_Mask(nn.Module):
    """
    SLS Loss variant that uses predicted and ground truth masks
    This is more accurate for segmentation tasks
    
    Args:
        eps: Small value to avoid division by zero
        reduction: 'mean' or 'sum' or 'none'
    """
    
    def __init__(self, eps=1e-7, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, pred_mask, gt_mask):
        """
        Args:
            pred_mask: (N, H, W) binary mask (0 or 1)
            gt_mask: (N, H, W) binary mask (0 or 1)
            
        Returns:
            loss: SLS loss value
        """
        # Compute LS (Scale Loss)
        ls_loss = self._compute_scale_loss_mask(pred_mask, gt_mask)
        
        # Compute LL (Location Loss)
        ll_loss = self._compute_location_loss_mask(pred_mask, gt_mask)
        
        # Total SLS loss
        sls_loss = ls_loss + ll_loss
        
        if self.reduction == 'mean':
            return sls_loss.mean()
        elif self.reduction == 'sum':
            return sls_loss.sum()
        else:
            return sls_loss
    
    def _compute_scale_loss_mask(self, pred_mask, gt_mask):
        """
        Compute LS = 1 - w * IoU using masks
        """
        N = pred_mask.shape[0]
        ls_list = []
        
        for i in range(N):
            pred = pred_mask[i]
            gt = gt_mask[i]
            
            # Compute IoU
            intersection = (pred & gt).sum().float()
            union = (pred | gt).sum().float()
            iou = intersection / (union + self.eps)
            
            # Compute areas
            pred_area = pred.sum().float()
            gt_area = gt.sum().float()
            
            # Compute variance
            var = ((pred_area - gt_area) ** 2) / 2
            
            # Compute weight
            area_min = torch.min(pred_area, gt_area)
            area_max = torch.max(pred_area, gt_area)
            w = (area_min + var) / (area_max + var + self.eps)
            
            # Compute LS
            ls = 1 - w * iou
            ls_list.append(ls)
        
        return torch.stack(ls_list)
    
    def _compute_location_loss_mask(self, pred_mask, gt_mask):
        """
        Compute LL using centroids of masks
        """
        N, H, W = pred_mask.shape
        ll_list = []
        
        # Create coordinate grids
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=pred_mask.device, dtype=torch.float32),
            torch.arange(W, device=pred_mask.device, dtype=torch.float32),
            indexing='ij'
        )
        
        for i in range(N):
            pred = pred_mask[i].float()
            gt = gt_mask[i].float()
            
            # Compute centroids
            pred_sum = pred.sum() + self.eps
            gt_sum = gt.sum() + self.eps
            
            pred_cx = (pred * x_coords).sum() / pred_sum
            pred_cy = (pred * y_coords).sum() / pred_sum
            
            gt_cx = (gt * x_coords).sum() / gt_sum
            gt_cy = (gt * y_coords).sum() / gt_sum
            
            # Convert to polar coordinates
            pred_d = torch.sqrt(pred_cx ** 2 + pred_cy ** 2 + self.eps)
            pred_theta = torch.atan2(pred_cy, pred_cx)
            
            gt_d = torch.sqrt(gt_cx ** 2 + gt_cy ** 2 + self.eps)
            gt_theta = torch.atan2(gt_cy, gt_cx)
            
            # Radial loss
            d_min = torch.min(pred_d, gt_d)
            d_max = torch.max(pred_d, gt_d)
            radial_loss = 1 - d_min / (d_max + self.eps)
            
            # Angular loss
            theta_diff = pred_theta - gt_theta
            theta_diff = torch.atan2(torch.sin(theta_diff), torch.cos(theta_diff))
            angular_loss = (4 / (math.pi ** 2)) * (theta_diff ** 2)
            
            ll = radial_loss + angular_loss
            ll_list.append(ll)
        
        return torch.stack(ll_list)


# Test code
if __name__ == "__main__":
    # Test SLS_Loss with bounding boxes
    print("Testing SLS_Loss with bounding boxes...")
    sls_criterion = SLS_Loss(reduction='mean')
    
    # Example: pred and gt boxes in [cx, cy, w, h] format (normalized)
    pred_boxes = torch.tensor([
        [0.5, 0.5, 0.3, 0.3],
        [0.6, 0.4, 0.25, 0.25]
    ])
    gt_boxes = torch.tensor([
        [0.5, 0.5, 0.3, 0.3],  # Perfect match
        [0.5, 0.5, 0.3, 0.3]   # Offset prediction
    ])
    
    loss = sls_criterion(pred_boxes, gt_boxes, image_size=(256, 256))
    print(f"SLS Loss: {loss.item():.4f}")
    
    # Test SLS_Loss_with_Mask
    print("\nTesting SLS_Loss_with_Mask...")
    sls_mask_criterion = SLS_Loss_with_Mask(reduction='mean')
    
    # Create sample masks
    pred_mask = torch.zeros(2, 32, 32, dtype=torch.bool)
    gt_mask = torch.zeros(2, 32, 32, dtype=torch.bool)
    
    pred_mask[0, 10:20, 10:20] = True
    gt_mask[0, 10:20, 10:20] = True
    
    pred_mask[1, 12:22, 12:22] = True
    gt_mask[1, 10:20, 10:20] = True
    
    loss_mask = sls_mask_criterion(pred_mask, gt_mask)
    print(f"SLS Loss (with mask): {loss_mask.item():.4f}")

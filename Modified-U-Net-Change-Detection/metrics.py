import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity as ssim

class EnhancedMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_advanced_metrics(self, pred_mask, gt_mask):
        """Calculate comprehensive set of metrics"""
        # Convert to numpy if tensors
        if torch.is_tensor(pred_mask):
            pred_mask = pred_mask.cpu().numpy()
        if torch.is_tensor(gt_mask):
            gt_mask = gt_mask.cpu().numpy()
            
        metrics = {
            'hausdorff_distance': self._hausdorff_distance(pred_mask, gt_mask),
            'ssim': ssim(pred_mask, gt_mask),
            'boundary_f1': self._boundary_f1_score(pred_mask, gt_mask),
            'fragmentation': self._fragmentation_index(pred_mask),
            'change_density': np.sum(pred_mask) / pred_mask.size
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _hausdorff_distance(self, pred, gt):
        """Calculate Hausdorff distance between prediction and ground truth"""
        return max(directed_hausdorff(pred, gt)[0], 
                  directed_hausdorff(gt, pred)[0])
    
    def _boundary_f1_score(self, pred, gt, tolerance=2):
        """Calculate F1 score for boundary pixels"""
        pred_boundary = self._get_boundary(pred)
        gt_boundary = self._get_boundary(gt)
        
        # Calculate distances
        pred_distances = ndimage.distance_transform_edt(~pred_boundary)
        gt_distances = ndimage.distance_transform_edt(~gt_boundary)
        
        # Calculate precision and recall
        pred_match = pred_boundary & (gt_distances <= tolerance)
        gt_match = gt_boundary & (pred_distances <= tolerance)
        
        precision = np.sum(pred_match) / (np.sum(pred_boundary) + 1e-7)
        recall = np.sum(gt_match) / (np.sum(gt_boundary) + 1e-7)
        
        return 2 * precision * recall / (precision + recall + 1e-7)
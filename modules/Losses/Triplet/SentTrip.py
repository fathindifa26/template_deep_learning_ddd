import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class SentenceTriplet(nn.Module):
    def __init__(self, margin=0.5,reducers="mean"):
        super().__init__()
        self.margin = margin
        self.reducers=reducers
    def _cosine_distance(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
        y_norm = F.normalize(y, p=2, dim=1, eps=1e-6)
        sim_matrix = torch.mm(x_norm, y_norm.T)
        return torch.clamp(1- sim_matrix, min=0.0, max=2.0)
 
        
    def forward(self, og_feat, ag_feat, labels):
        device = og_feat.device
        batch_size = og_feat.size(0)
        d_ap = self._cosine_distance(og_feat, ag_feat).diag()
        d_an = self._cosine_distance(og_feat, og_feat)
        # Create masks
        labels = labels.view(-1)
        label_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).bool()
        eye_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        valid_neg_mask = label_mask & eye_mask

        # Semi-hard mining
        d_ap_expanded = d_ap.unsqueeze(1)
        semi_hard_mask = (d_an > d_ap_expanded) & \
                        (d_an < d_ap_expanded + self.margin) & \
                        valid_neg_mask
        d_an_semi = torch.where(semi_hard_mask, d_an, torch.full_like(d_an, float('inf')))
        min_d_an_semi, _ = torch.min(d_an_semi, dim=1)
        valid_semi = min_d_an_semi < float('inf')

        # Fallback to hard mining if no semi-hard negatives
        if not valid_semi.any():
            # Hard negative mining
            d_an_hard = torch.where(valid_neg_mask, d_an, torch.full_like(d_an, float('inf')))
            min_d_an_hard, _ = torch.min(d_an_hard, dim=1)
            valid_hard = min_d_an_hard < float('inf')
            
            if not valid_hard.any():
                return (og_feat * 0.0).sum() + (ag_feat * 0.0).sum()
            
            loss_terms = F.relu(d_ap[valid_hard] - min_d_an_hard[valid_hard] + self.margin)
            if self.reducers == "mean":
                return loss_terms.sum() / (valid_hard.sum().float() + 1e-7)
            if self.reducers == "sum":
                return loss_terms.sum()
        # Original semi-hard loss calculation
        loss_terms = F.relu(d_ap[valid_semi] - min_d_an_semi[valid_semi] + self.margin)
        if self.reducers == "mean":
            return loss_terms.sum() / (valid_semi.sum().float() + 1e-7)
        if self.reducers == "sum":
            return loss_terms.sum()
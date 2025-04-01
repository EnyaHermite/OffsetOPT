import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ReconLoss(nn.Module):

    def __init__(self): 
        super().__init__()
                  
    def forward(self, pred_logits):   
        N, T = pred_logits.shape
        K_ = int(np.sqrt(T))  # K_ = knn-1 (excluding the center point itself)
        pred_logits = pred_logits.reshape(N, K_, K_)

        # create pseudo-labels, size of pred_logits is (N,K_,K_)
        top2_logits, I = torch.topk(pred_logits, 2, dim=-1)
        top2_confidence = nn.Sigmoid()(top2_logits)
        pseudo_labels = torch.zeros_like(pred_logits)
        pseudo_labels.scatter_(2, I, 1)
        mask = (top2_confidence[...,:1]>0.5).float()
        pseudo_labels = pseudo_labels*mask

        loss = F.binary_cross_entropy_with_logits(pred_logits, pseudo_labels, reduction='sum')
        return loss
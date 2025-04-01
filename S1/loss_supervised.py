import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryLoss(nn.Module):

    def __init__(self): 
        super().__init__()
    
    def _torch_meshgrid_2D(self, input):
        assert(input.dim()==2)
        N, K = input.size()
        expand_3rd = input[...,None].tile(1,1,K)
        expand_2nd = input[:,None,:].tile(1,K,1)
        C = torch.stack([expand_3rd,expand_2nd], dim=3)
        output = torch.reshape(C, [N,-1,2])
        return output   
    
    def _build_ground_truth_dense(self, points, knn_indices, gt_triangles):    
        N, K = knn_indices.shape
        num_pts = points.shape[0]
               
        # symmetric matrix
        full_mat = torch.zeros(num_pts, num_pts, dtype=torch.bool).to(knn_indices.device)
        v = [gt_triangles[:,0], gt_triangles[:,1], gt_triangles[:,2]]
        w = [gt_triangles[:,1], gt_triangles[:,2], gt_triangles[:,0]]
        rows = torch.cat(v+w)
        cols = torch.cat(w+v)
        full_mat[rows,cols] = 1
        
        gt_mask = torch.zeros(N, (K-1)*(K-1), dtype=torch.bool).to(knn_indices.device)
        
        gt_cand_edges = self._torch_meshgrid_2D(knn_indices[:,1:])
        gt_cand_triangles = torch.cat([knn_indices[:,:1].repeat(1,gt_cand_edges.shape[1])[...,None], gt_cand_edges], dim=2)
        gt_cand_triangles = torch.sort(gt_cand_triangles, dim=-1).values
        
        all_N_gt = []
        for n in range(N):
            v0 = knn_indices[n,0]
            mask = torch.any(gt_triangles==v0,dim=-1)
            nn_triangles = gt_triangles[mask]
            
            cand_tris = gt_cand_triangles[n]         
            mat_v1 = cand_tris[:,0][:,None]==nn_triangles[:,0][None,:]
            mat_v2 = cand_tris[:,1][:,None]==nn_triangles[:,1][None,:]
            mat_v3 = cand_tris[:,2][:,None]==nn_triangles[:,2][None,:]
            
            cand_mask = mat_v1 & mat_v2 & mat_v3
            gt_mask[n] = torch.any(cand_mask, dim=-1)
            all_N_gt.append(nn_triangles.shape[0])          
        
        all_edges = gt_cand_edges.reshape(-1,2)
        rows, cols = all_edges[:,0], all_edges[:,1]
        gt_labels = full_mat[rows, cols].reshape(N, (K-1)*(K-1)).float()     
        gt_labels_masked = gt_labels*gt_mask.float()
               
        all_N_gt = torch.tensor(all_N_gt).to(knn_indices.device).to(torch.float)
        all_N_pred = torch.sum(gt_labels_masked, dim=-1)
            
        rows = knn_indices[:,:1].tile(1, K-1).reshape(-1)
        cols = knn_indices[:,1:].reshape(-1)
        gt_labels_center = full_mat[rows, cols].reshape(N, K-1).float()    
        gt_labels_masked = torch.cat([gt_labels_center, gt_labels_masked], dim=1)
        manifold_indices = (all_N_gt*2==all_N_pred)
        return gt_labels_masked, manifold_indices      
    
    def _advanced_hard_negative_mining(self, gt_labels, pred_logits, knn_indices):
        N, K = knn_indices.shape
        
        gt_labels = gt_labels[:,:K-1]
        pred_logits = pred_logits.reshape(N, K-1, K-1)
        pred_logits = torch.sort(pred_logits, dim=-1, descending=True).values
        pred_logits_pos = pred_logits[:,:,1]
        pred_logits_neg = pred_logits[:,:,2]
        
        gt_mask = (gt_labels==1)
        pos_pred = pred_logits_pos[gt_mask]
        neg_pred = pred_logits_neg[gt_mask]

        loss_pos = F.binary_cross_entropy_with_logits(pos_pred, torch.ones_like(pos_pred))
        loss_neg = F.binary_cross_entropy_with_logits(neg_pred, torch.zeros_like(neg_pred))
        return loss_pos, loss_neg   

    def forward(self, pred_logits, points, knn_indices, gt_triangles):   
        gt_triangles = torch.sort(gt_triangles, dim=-1).values  
        gt_triangles = torch.unique(gt_triangles, dim=0)          
        
        N, K = knn_indices.shape
        gt_labels, manifold_indices = self._build_ground_truth_dense(points, knn_indices, gt_triangles)
        loss = F.binary_cross_entropy_with_logits(pred_logits[manifold_indices], gt_labels[manifold_indices,K-1:])

        # record the loss for the positive and negative predictions, not used for network optimization
        loss_pos_mine, loss_neg_mine =  self._advanced_hard_negative_mining(gt_labels, pred_logits, knn_indices) # 
        return loss, loss_pos_mine, loss_neg_mine

        


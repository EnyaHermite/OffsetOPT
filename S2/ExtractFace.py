import torch
import numpy as np
import torch.nn as nn



class SurfExtract(nn.Module):

    def __init__(self, confidence_top1=0.8, confidence_top2=0.5, angle=120):
        super().__init__()
        self.conf_1 = confidence_top1  
        self.conf_2 = confidence_top2
        self.angle = angle
    
    def _torch_meshgrid_2D(self, input):
        assert(input.dim()==2)
        N, K = input.size()
        expand_3rd = input[...,None].tile(1,1,K)
        expand_2nd = input[:,None,:].tile(1,K,1)
        C = torch.stack([expand_3rd,expand_2nd], dim=3)
        output = torch.reshape(C, [N,-1,2])
        return output   
    
    def _build_candidate_triangles(self, knn_indices):    
        cand_edges = self._torch_meshgrid_2D(knn_indices[:,1:])
        cand_triangles = torch.cat([knn_indices[:,:1].repeat(1,cand_edges.shape[1])[...,None], cand_edges], dim=2)
        return cand_triangles
    
    def forward(self, points, pred_logits, knn_indices):
        candidate_triangles = self._build_candidate_triangles(knn_indices)

        N, K = knn_indices.shape
        pred_logits = pred_logits.reshape(N, K-1, K-1)
        candidate_triangles = candidate_triangles.reshape(N, K-1, K-1, 3)
        pred_logits, indices = torch.sort(pred_logits, dim=-1, descending=True) # sort from confidence high to low

        sorted_candidate_triangles = torch.stack([torch.gather(candidate_triangles[...,0], 2, indices), 
                                                 torch.gather(candidate_triangles[...,1], 2, indices),
                                                 torch.gather(candidate_triangles[...,2], 2, indices)], dim=3)
        
        # get flip sign of the top 2 triangles
        triangle_A = sorted_candidate_triangles[:,:,0,:]     # top 1st triangle
        triangle_B = sorted_candidate_triangles[:,:,1,:]     # top 2nd triangle

        center_pt = knn_indices[:,:1]
        edge_pt = triangle_A[:,:,1]
        other_edge_A = triangle_A[:,:,2]
        other_edge_B = triangle_B[:,:,2]
        assert(torch.all(triangle_A[:,:,0]==triangle_B[:,:,0]))
        assert(torch.all(triangle_A[:,:,1]==triangle_B[:,:,1]))

        edge = points[edge_pt] - points[center_pt]
        edge_A = points[other_edge_A] - points[edge_pt]
        edge_B = points[other_edge_B] - points[edge_pt]

        normal_A = torch.cross(edge, edge_A, dim=-1)
        normal_B = torch.cross(edge, edge_B, dim=-1)
        normal_A = normal_A/torch.linalg.norm(normal_A, axis=-1, keepdim=True)
        normal_B = normal_B/torch.linalg.norm(normal_B, axis=-1, keepdim=True)
        face_angle_thresh = np.cos(self.angle/180*np.pi)
        flip_sign = torch.sum(normal_A*normal_B, dim=-1)>face_angle_thresh  # true if the two triangles are flipped

        confidence_1 = nn.Sigmoid()(pred_logits[:,:,0])
        confidence_2 = nn.Sigmoid()(pred_logits[:,:,1])
        valid_rows, valid_cols = torch.where((confidence_1>0.8) & (confidence_2>0.5) & (~flip_sign))
        pred_triangles = sorted_candidate_triangles[valid_rows, valid_cols,:2,:]
        pred_triangles = pred_triangles.reshape(-1,3)
        pred_triangles = torch.sort(pred_triangles, dim=-1).values
        pred_triangles, counts = torch.unique(pred_triangles, dim=0, return_counts=True)
        return pred_triangles
    
        

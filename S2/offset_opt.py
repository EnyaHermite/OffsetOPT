import torch
import numpy as np
import open3d as o3d
import torch.nn as nn
from S2.ExtractFace import SurfExtract


class OffsetOPT(nn.Module):
    def __init__(self, model, loss, device, maxIter=100, zero_init=False, slice_per_chunk=5000):
        super().__init__()
        self.model = model
        self.loss = loss
        self.device = device
        self.cycle = 10       # decay the learning rate every 10 iterations (as stated in the paper)
        self.maxIter = maxIter  # maximum iterations for offset optimization (T in the paper)     
        self.zero_init = zero_init  # initialize offsets as zeros if True; False is recommended
        self.SurfaceExtractor = SurfExtract()
        self.slice_per_chunk = slice_per_chunk

        self._freeze_model() # freeze the model, only offsets are updated

    def _freeze_model(self):
        for name, param in self.model.named_parameters():
            if "transformer_posenc" in name: 
                param.requires_grad = False 

    def _config_optim_lr(self, offsets):
        self.optimizer = torch.optim.SGD([{'params': offsets}], lr=0.1, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)    

    def _init_offsets(self, nn_offsets):
        if self.zero_init:
            offsets = nn.Parameter(torch.zeros_like(nn_offsets))
        else:
            offsets = nn.Parameter(nn_offsets/4)
        return offsets
    
    @staticmethod
    def _get_manifold_pct(triangles):
        v1 = triangles[:, 0]
        v2 = triangles[:, 1]
        v3 = triangles[:, 2]
        edges = np.concatenate([np.stack([v1, v2], axis=1),
                                np.stack([v2, v3], axis=1),
                                np.stack([v1, v3], axis=1)], axis=0)
        edges = np.sort(edges, axis=-1)
        _, uni_cnts = np.unique(edges, return_counts=True, axis=0)
        manifold_edge_pct = np.mean(uni_cnts<=2)
        return manifold_edge_pct

    def _process_chunks(self, knn_indices, features, offsets, require_logits=False):
        total_loss = 0
        pred_logits = []
        N, K = knn_indices.shape
        num_chunks = N//self.slice_per_chunk+1

        # ========================= Chunk processing for gradient accumulation =========================
        self.optimizer.zero_grad()
        for chunk_id in range(num_chunks):               
            start_idx = chunk_id*self.slice_per_chunk
            end_idx = (chunk_id+1)*self.slice_per_chunk 
            slice_knn_indices = knn_indices[start_idx:end_idx]
            if slice_knn_indices.shape[0]==0: continue

            slice_features = features[start_idx:end_idx,:,:]
            slice_offsets_features = offsets[slice_knn_indices] - offsets[slice_knn_indices[:,0],None,:]
            slice_logits = self.model(slice_features, slice_offsets_features)

            loss = self.loss(slice_logits)
            loss = loss/(N*(K-1)*(K-1))
            loss.backward()     # assign the accumulated gradients

            if require_logits:
                pred_logits.append(slice_logits)
            total_loss += loss.item()
        # ==============================================================================================
        return total_loss, pred_logits

    def __call__(self, points, knn_indices, center, scale, req_inter_logits=False):
        points = points.to(self.device)
        knn_indices = knn_indices.to(self.device)
      
        nn_offsets = points - points[knn_indices[:,1]]
        features = points[knn_indices] - points[:,None,:]        
        d_0 = torch.linalg.norm(nn_offsets, dim=-1)
        offsets = self._init_offsets(nn_offsets)
        self._config_optim_lr(offsets)

        lr = self.scheduler.get_last_lr()[0]        
        for iter in range(self.maxIter):
            self.model.train()

            if iter==(self.maxIter-1):
                require_logits = True
            else:
                require_logits = req_inter_logits
            total_loss, pred_logits = self._process_chunks(knn_indices, features, offsets, require_logits)

            # uncontrolled updates of the offsets, following Equation (4)
            Ln = torch.linalg.norm(offsets.grad, dim=-1)    # length of each offset-vector
            offsets.grad *= d_0.view(-1, 1)/(Ln.view(-1, 1))
            with torch.no_grad():
                temp_offsets_next = offsets - lr*offsets.grad

            # controlled updates of the offsets, following Equation (7)     
            temp = points+temp_offsets_next
            knn_dists = torch.linalg.norm(temp[:,None,:] - temp[knn_indices[:,1:]], dim=-1)
            d_t = knn_dists.min(dim=-1)[0]
            mask = (d_t>(d_0/2)).float()
            offsets.grad *= mask.view(-1, 1)      
            self.optimizer.step()    # updates offsets with controlled gradients: offsets ← offsets - lr * offsets.grad  

            if iter%self.cycle==0:        
                if iter>0: self.scheduler.step()    

                COLOR = "\033[96m"
                RESET = "\033[0m"
                if req_inter_logits:
                    assert len(pred_logits)>0, "pred_logits list is empty"
                    curr_pred_logits = torch.cat(pred_logits, dim=0)
                    mesh_reconstructed = self._extract_mesh(points, knn_indices, center, scale, curr_pred_logits)
                    triangles = np.asarray(mesh_reconstructed.triangles)
                    manifold_pct = self._get_manifold_pct(triangles)
                    # print('Iter: {}, Loss: {:.6f},  lr: {}, Manifold: {:3.2f}%%'.format(iter, total_loss, self.scheduler.get_last_lr()[0], manifold_pct*100))
                    print(COLOR+'Iter:{}, Loss:{:.6f},  lr:{:.6f}, Manifold:{:3.2f}%%'.format(
                          iter, total_loss, self.scheduler.get_last_lr()[0], manifold_pct*100)+RESET)
                    del curr_pred_logits, pred_logits
                else:
                    # print('Iter: {}, Loss: {:.6f},  lr: {}'.format(iter, total_loss, self.scheduler.get_last_lr()[0]))
                    print(COLOR+'Iter:{}, Loss:{:.6f},  lr:{:.6f}'.format(
                          iter, total_loss, self.scheduler.get_last_lr()[0])+RESET)
          
            lr = self.scheduler.get_last_lr()[0]
                
        pred_logits = torch.cat(pred_logits, dim=0)
        mesh_reconstructed = self._extract_mesh(points, knn_indices, center, scale, pred_logits)
        return mesh_reconstructed

    def _extract_mesh(self, points, knn_indices, center, scale, pred_logits):
        pred_triangles = []
        step = 20000    # Recover face connections for up to 20,000 points at a time to avoid memory overflow
        for i in range(0, points.shape[0], step):
            pred_triangles.append(self.SurfaceExtractor(points, pred_logits[i:i+step], knn_indices[i:i+step]))
        pred_triangles = torch.cat(pred_triangles, dim=0)
        pred_triangles = torch.sort(pred_triangles, dim=-1).values
        pred_triangles = torch.unique(pred_triangles, dim=0)

        points = points.detach().cpu()*scale + center 
        points = points.numpy()
        pred_triangles = pred_triangles.detach().cpu().numpy()
        pred_mesh = o3d.geometry.TriangleMesh()
        pred_mesh.vertices = o3d.utility.Vector3dVector(points)
        pred_mesh.triangles = o3d.utility.Vector3iVector(pred_triangles)
        return pred_mesh
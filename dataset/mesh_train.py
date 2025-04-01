import torch
import numpy as np
from torch.utils.data import Dataset
from pykdtree.kdtree import KDTree
import open3d as o3d


class MeshTrainSet(Dataset):
    
    def __init__(self, meshList, knn=50, Lembed=16, resolution=0.01, normalize=True, 
                       per_mesh_patches=200, sample_patch=True, transform=None):
        super().__init__()
        self.knn = knn
        self.Lembed = Lembed
        self.mesh_files = meshList
        self.normalize = normalize
        self.transform = transform
        self.resolution = resolution
        self.sample_patch = sample_patch
        self.num_patches = per_mesh_patches
        
    def _posEnc(self, x):
        pe_feats = [x]
        for i in range(self.Lembed):
            for fn in [torch.sin, torch.cos]:
                pe_feats.append(fn(2.**i * x))
        return torch.cat(pe_feats, axis=-1)
    
    def _knn_search(self, points):
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        kdt = KDTree(points)
        knn_dists, knn_indices = kdt.query(points, k=self.knn)
        
        points = torch.from_numpy(points)        
        knn_dists = torch.from_numpy(knn_dists)
        knn_indices = torch.from_numpy(knn_indices.astype('int64'))
        
        features = points[knn_indices] - points[:,None,:]
        features = self.resolution*features/knn_dists[:,1:2,None]
        return features, knn_indices
    
    def _sample_patches(self, points, triangles):
        sample_indices = torch.randperm(points.shape[0])[:self.num_patches]
        
        # select gt triangles that contain the sampled center points
        point_mask = torch.zeros_like(points[:,0])
        point_mask[sample_indices] = 1
        triangle_mask_1d = point_mask[triangles.flatten()] # mask triangle vertices
        triangle_mask = triangle_mask_1d.view(-1, 3).any(dim=-1)
        valid_triangles = triangles[triangle_mask]
        return sample_indices, valid_triangles

    def __len__(self):
        return len(self.mesh_files)

    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        
        if self.normalize:
            vertices = np.asarray(mesh.vertices)
            min_bnd = np.min(vertices, axis=0)
            max_bnd = np.max(vertices, axis=0)
            center = (min_bnd + max_bnd)/2
            vertices -= center
            scale = np.linalg.norm(max_bnd-min_bnd)/2
            vertices = vertices/scale
            mesh.vertices = o3d.utility.Vector3dVector(vertices)        
        
        vertices = np.asarray(mesh.vertices).astype('float32')
        triangles = np.asarray(mesh.triangles).astype('int32')  
              
        if self.transform:
            vertices = torch.from_numpy(vertices)
            vertices = self.transform(vertices)
            vertices = vertices.numpy()
        
        points = torch.from_numpy(vertices)
        triangles = torch.from_numpy(triangles)
        assert (triangles.min()==0)

        features, knn_indices = self._knn_search(points)

        if self.sample_patch:
            sample_indices, triangles = self._sample_patches(points, triangles) 
            features = features[sample_indices]
            knn_indices = knn_indices[sample_indices]
            assert torch.equal(sample_indices, knn_indices[:,0])     
        
        nv = points.shape[0]
        mf = triangles.shape[0]
        features = self._posEnc(features)
        
        # new_mesh = o3d.geometry.TriangleMesh()
        # new_mesh.vertices = o3d.utility.Vector3dVector(points.numpy())
        # new_mesh.triangles = o3d.utility.Vector3iVector(triangles.numpy())
        # o3d.visualization.draw_geometries([new_mesh], mesh_show_back_face=True)        
        return points, features, knn_indices, triangles, nv, mf

    def collate_fn(self, batch_data):
        batch_points = [item[0] for item in batch_data]
        batch_features = [item[1] for item in batch_data]
        batch_nv = [item[4] for item in batch_data]
        batch_mf = [item[5] for item in batch_data]
        
        batch_points = torch.cat(batch_points, axis=0)
        batch_features = torch.cat(batch_features, axis=0)
        batch_mf = torch.tensor(batch_mf)
                
        vid_offsets = torch.cumsum(torch.tensor([0] + batch_nv), dim=0)
        batch_knn_indices = [item[2] + vid_offsets[i] for i, item in enumerate(batch_data)]
        batch_triangles = [item[3] + vid_offsets[i] for i, item in enumerate(batch_data)]
        batch_knn_indices = torch.cat(batch_knn_indices, dim=0)
        batch_triangles = torch.cat(batch_triangles, dim=0)
        batch_nv = torch.tensor(batch_nv)
     
        return batch_points, batch_features, batch_knn_indices, \
               batch_triangles, batch_nv, batch_mf


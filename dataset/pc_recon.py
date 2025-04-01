import torch
import numpy as np
from torch.utils.data import Dataset
from pykdtree.kdtree import KDTree
import open3d as o3d


class PCReconSet(Dataset):
    
    def __init__(self, pcloudList, delta=0.01, knn=50, rescale_delta=False):
        super().__init__()
        self.knn = knn
        self.delta = delta
        self.pcloud_files = pcloudList
        self.rescale_delta = rescale_delta
    
    def _knn_search(self, points):
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        kdt = KDTree(points)
        knn_dists, knn_indices = kdt.query(points, k=self.knn)
        knn_dists = torch.from_numpy(knn_dists)
        knn_indices = torch.from_numpy(knn_indices.astype('int64'))
        return knn_indices, knn_dists
    
    def get_center_scale(self, points): 
        min_bnd = np.min(points, axis=0)
        max_bnd = np.max(points, axis=0)
        center = (min_bnd + max_bnd)/2
        scale = np.linalg.norm(max_bnd-min_bnd)/2
        return center, scale

    def __len__(self):
        return len(self.pcloud_files)

    def __getitem__(self, idx):
        pcd_path = self.pcloud_files[idx]
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = pcd.remove_duplicated_points() # to avoid problematic duplicate points
     
        if self.delta==0: # applies to FAUST&MGN mesh vertices, without voxelized downsampling
            points = np.asarray(pcd.points) 
        else:
            if self.rescale_delta:
                X = np.asarray(pcd.points)
                scale = self.get_center_scale(X)[-1]
                voxel_pcd = pcd.voxel_down_sample(voxel_size=(self.delta*scale))
            else:
                voxel_pcd = pcd.voxel_down_sample(voxel_size=self.delta)
            points = np.asarray(voxel_pcd.points)
     
        center, scale = self.get_center_scale(points)
        points = (points - center)/scale
        points = points.astype('float32')
        knn_indices, _ = self._knn_search(points)     
        points = torch.from_numpy(points)
        return points, knn_indices, center, scale  
   

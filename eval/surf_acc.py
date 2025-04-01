import trimesh, os
import numpy as np
from sklearn.neighbors import KDTree
from glob import glob
from tqdm import tqdm
import open3d as o3d


class MeshEvaluator:
    def __init__(self, sample_num=100000, gt_surf_type='mesh'):
        self.sample_num = sample_num
        self.f1_thresh = 0.003
        self.ef1_radius = 0.004
        self.ef1_dot = 0.2
        self.ef1_thresh = 0.005
        self.gt_surf_type = gt_surf_type

    def set_gt_surface_type(self, surf_type):
        self.gt_surf_type = surf_type

    def normalize_diagonal(self, input):
        if isinstance(input, trimesh.Trimesh):
            vertices = np.asarray(input.vertices)
            triangles = np.asarray(input.faces)
            return_mesh = True
        elif isinstance(input, np.ndarray):
            vertices = input
            return_mesh = False
        else:
            raise TypeError("Input must be a trimesh.Trimesh or a numpy ndarray of vertices.")

        center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
        scale = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
        vertices = (vertices - center) / scale

        if return_mesh:
            return trimesh.Trimesh(vertices=vertices, faces=triangles)
        else:
            return vertices

    def compute_angles(self, points, triangles):
        A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
        l_AB = np.linalg.norm(A - B, axis=-1) + 1e-10
        l_AC = np.linalg.norm(A - C, axis=-1) + 1e-10
        l_BC = np.linalg.norm(B - C, axis=-1) + 1e-10

        dot_A = np.sum((B - A) * (C - A), axis=-1) / (l_AB * l_AC)
        dot_B = np.sum((A - B) * (C - B), axis=-1) / (l_AB * l_BC)
        dot_C = np.sum((A - C) * (B - C), axis=-1) / (l_AC * l_BC)

        angles = np.arccos(np.clip(np.stack([dot_A, dot_B, dot_C], axis=1), -1.0, 1.0))
        return angles.reshape(-1) * 180 / np.pi

    def load_gt(self, gt_mesh):
        if self.gt_surf_type=='mesh':
            gt = self.normalize_diagonal(trimesh.load_mesh(gt_mesh))
            gt_points, gt_idx = gt.sample(self.sample_num, return_index=True)
            gt_normals = gt.face_normals[gt_idx]
        else:
            print(gt_mesh)
            gt_pcloud = o3d.io.read_point_cloud(gt_mesh)    
            gt_points = np.asarray(gt_pcloud.points)
            gt_points = self.normalize_diagonal(gt_points)
            gt_normals = np.asarray(gt_pcloud.normals)

            # sample
            N = gt_points.shape[0]
            idx = np.random.choice(N, self.sample_num, replace=False)
            gt_points = gt_points[idx]
            gt_normals = gt_normals[idx]    
        return gt_points, gt_normals

    def get_metrics(self, gt_mesh, pred_mesh):
        # load gt surface and get points and normals
        gt_points, gt_normals = self.load_gt(gt_mesh)

        # load predicted surface and get points and normals
        pred = self.normalize_diagonal(trimesh.load_mesh(pred_mesh))
        pred_points, pred_idx = pred.sample(self.sample_num, return_index=True)
        pred_normals = pred.face_normals[pred_idx]

        # cd and nc and f1
        # from gt to pred
        pred_tree = KDTree(pred_points)
        dist, inds = pred_tree.query(gt_points, k=1)
        recall = np.sum(dist < self.f1_thresh) / float(len(dist))
        gt2pred_mean_cd1 = np.mean(dist)
        dist = np.square(dist)
        gt2pred_mean_cd2 = np.mean(dist)
        neighbor_normals = pred_normals[np.squeeze(inds, axis=1)]
        dotproduct = np.abs(np.sum(gt_normals*neighbor_normals, axis=1))
        gt2pred_nc = np.mean(dotproduct)
        gt2pred_nr = np.mean(np.degrees(np.arccos(np.minimum(dotproduct,1.0))))

        gt2pred_na = []
        for i in range(90):
            gt2pred_na.append( np.mean( (dotproduct<np.cos(i/180.0*np.pi)).astype(np.float32) ) )

        # from pred to gt
        gt_tree = KDTree(gt_points)
        dist, inds = gt_tree.query(pred_points, k=1)
        precision = np.sum(dist<self.f1_thresh)/float(len(dist))
        pred2gt_mean_cd1 = np.mean(dist)
        dist = np.square(dist)
        pred2gt_mean_cd2 = np.mean(dist)
        neighbor_normals = gt_normals[np.squeeze(inds, axis=1)]
        dotproduct = np.abs(np.sum(pred_normals*neighbor_normals, axis=1))
        pred2gt_nc = np.mean(dotproduct)
        pred2gt_nr = np.mean(np.degrees(np.arccos(np.minimum(dotproduct,1.0))))

        pred2gt_na = []
        for i in range(90):
            pred2gt_na.append( np.mean( (dotproduct<np.cos(i/180.0*np.pi)).astype(np.float32) ) )

        cd1 = gt2pred_mean_cd1+pred2gt_mean_cd1
        cd2 = gt2pred_mean_cd2+pred2gt_mean_cd2
        nc = (gt2pred_nc+pred2gt_nc)/2
        nr = (gt2pred_nr+pred2gt_nr)/2
        if recall+precision > 0: f1 = 2 * recall * precision / (recall + precision)
        else: f1 = 0

        #sample gt edge points
        indslist = gt_tree.query_radius(gt_points, self.ef1_radius)
        flags = np.zeros([len(gt_points)],bool)
        for p in range(len(gt_points)):
            inds = indslist[p]
            if len(inds)>0:
                this_normals = gt_normals[p:p+1]
                neighbor_normals = gt_normals[inds]
                dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
                if np.any(dotproduct < self.ef1_dot):
                    flags[p] = True
        gt_edge_points = np.ascontiguousarray(gt_points[flags])

        #sample pred edge points
        indslist = pred_tree.query_radius(pred_points, self.ef1_radius)
        flags = np.zeros([len(pred_points)],bool)
        for p in range(len(pred_points)):
            inds = indslist[p]
            if len(inds)>0:
                this_normals = pred_normals[p:p+1]
                neighbor_normals = pred_normals[inds]
                dotproduct = np.abs(np.sum(this_normals*neighbor_normals, axis=1))
                if np.any(dotproduct < self.ef1_dot):
                    flags[p] = True
        pred_edge_points = np.ascontiguousarray(pred_points[flags])

        #ecd ef1
        if len(pred_edge_points)==0: pred_edge_points=np.zeros([486,3],np.float32)
        if len(gt_edge_points)==0:
            ecd1 = 0
            ef1 = 1
        else:
            # from gt to pred
            tree = KDTree(pred_edge_points)
            dist, inds = tree.query(gt_edge_points, k=1)
            erecall = np.sum(dist < self.ef1_thresh) / float(len(dist))
            gt2pred_mean_ecd1 = np.mean(dist)
            dist = np.square(dist)

            # from pred to gt
            tree = KDTree(gt_edge_points)
            dist, inds = tree.query(pred_edge_points, k=1)
            eprecision = np.sum(dist < self.ef1_thresh) / float(len(dist))
            pred2gt_mean_ecd1 = np.mean(dist)
            dist = np.square(dist)

            ecd1 = gt2pred_mean_ecd1+pred2gt_mean_ecd1
            if erecall+eprecision > 0: ef1 = 2 * erecall * eprecision / (erecall + eprecision)
            else: ef1 = 0

        return cd1*100, cd2*1e5, f1, nc, nr, ecd1*100, ef1

    def run_batch_eval(self, gt_dir, pred_dir):
        pred_files = sorted(glob(os.path.join(pred_dir, '*.ply')))

        results = []
        for fname in tqdm(pred_files):
            name = os.path.basename(fname)
            gt_path = os.path.join(gt_dir, name)
            pred_path = os.path.join(pred_dir, name)
            results.append(self.get_metrics(gt_path, pred_path))

        results = np.array(results)
        avg = np.mean(results, axis=0)
        print("\nAverage metrics (CD1, CD2, F1, NC, NR, ECD1, EF1):")
        print("(%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f)" % tuple(avg))
        return results
    

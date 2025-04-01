import argparse, os
import numpy as np
import torch, glob
from torch.utils.data import DataLoader
from time import time
import open3d as o3d
from dataset.pc_recon import PCReconSet
from S2.loss_unsupervised import ReconLoss
from S2.ReconNet import S2ReconNet
from S2.offset_opt import OffsetOPT


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--delta', type=float, default=0.01, help='Voxel size to use [default: 0.01]')
parser.add_argument('--rescale_delta', action='store_true', help='Rescale delta to match point cloud size [default: False]')
parser.add_argument('--dataset', type=str, default='ABC', help='Name of the dataset to use [default: ABC]')
opt = parser.parse_args()



if __name__=='__main__':
    device = torch.device('cuda:{}'.format(opt.gpu) if torch.cuda.is_available() else 'cpu')

    # hyper-parameter configurations
    dim, Lembed = 3, 8
    Cin = dim + dim*Lembed*2
    knn = 50
  
    # load point clouds to be reconstructed
    test_files = glob.glob(f'./Data/PointClouds/{opt.dataset}/*.ply')
    testSet  = PCReconSet(test_files, knn=knn, delta=opt.delta, rescale_delta=opt.rescale_delta)  
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=0)

    # store folder to reconstructed meshes
    results_folder = f'results/{opt.dataset}/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)    
                              
    # config model, loss function, and load the trained model
    model = S2ReconNet(Cin=Cin, knn=knn, Lembed=Lembed).to(device) 
    loss_fn = ReconLoss()
    model.load_state_dict(torch.load('trained_models/model_knn50.pth', map_location=device))

    # set the OffsetOPT optimizer
    if 'ABC' in opt.dataset:
        # switch off offset optimization, and the proposed offset initialization
        OffsetOPTer = OffsetOPT(model, loss_fn, device, maxIter=1, zero_init=True)
    else:
        OffsetOPTer = OffsetOPT(model, loss_fn, device, maxIter=100, zero_init=False)
    runtime = np.zeros(len(test_files))
    idx = 0
    for data in testLoader:
        data = [item[0] for item in data]
        points, knn_indices, center, scale = data

        start_time = time()        
        recon_mesh = OffsetOPTer(points, knn_indices, center, scale, req_inter_logits=False)
        runtime[idx] = time()-start_time
        print("Time taken:", runtime[idx]) 

        o3d.io.write_triangle_mesh(os.path.join(results_folder, '%s'%test_files[idx].split('/')[-1]), recon_mesh)
        idx += 1

        
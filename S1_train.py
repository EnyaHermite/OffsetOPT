import argparse, os
import glob, torch
from torch.utils.data import DataLoader
from dataset.mesh_train import MeshTrainSet
from utils.augmentor import *
from S1.BaseNet import S1BaseNet
from S1.loss_supervised import BinaryLoss
from S1.fitModel import ModelFit


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--max_epoch', type=int, default=301, help='Epoch to run [default: 301]')
parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint to resume training [default: None]')
opt = parser.parse_args()


if __name__=='__main__':
    device = torch.device('cuda:{}'.format(opt.gpu) if torch.cuda.is_available() else 'cpu')
    MAX_EPOCHS = opt.max_epoch
    
    # hyper-parameter configurations
    dim, Lembed = 3, 8
    Cin = dim + dim*Lembed*2
    knn = 50
        
    # # load file_lists of train/test split
    train_files = glob.glob('Data/ABC/train/*.ply')
    train_files.sort()
    test_files = glob.glob('Data/ABC/test/*.ply')
    test_files.sort()
    Lists = {'train':train_files, 'test':test_files[:100]}   
    print(len(Lists['train']), len(Lists['test']))   
        
    # define transforms and load in the datasets
    transforms = {}
    transforms['train'] = torch.nn.Sequential(rotate_point_cloud(upaxis='x', has_normal=False),
                                              rotate_point_cloud(upaxis='y', has_normal=False),
                                              rotate_point_cloud(upaxis='z', has_normal=False),
                                              jitter_point_cloud(sigma=0.001, prob=1.0),
                                              random_scale_point_cloud(prob=0.95),)
    trainSet = MeshTrainSet(Lists['train'], knn=knn, Lembed=Lembed, sample_patch=True, per_mesh_patches=200, transform=transforms['train'])
    testSet  = MeshTrainSet(Lists['test'], knn=knn, Lembed=Lembed, sample_patch=True, per_mesh_patches=500, transform=None)  

    # build training set dataloader
    trainLoader = DataLoader(trainSet, batch_size=6, shuffle=True, num_workers=3, collate_fn=trainSet.collate_fn) 
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False, num_workers=3, collate_fn=testSet.collate_fn) 
                              
    # create model, loss function, optimizer, learning rate scheduler, and writer for tensorboard 
    ckpt_epoch = 0
    loss_fn = BinaryLoss()
    model = S1BaseNet(Cin=Cin, knn=knn).to(device)
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)   
    write_folder = f'S1_training/model_k{knn}'

    print("write_folder:", write_folder)
    if not os.path.exists(write_folder): os.makedirs(write_folder)
    fout = open(os.path.join(write_folder, 'log_train.txt'), 'a')
    fout.write(str(opt) + '\n')

    if opt.ckpt_path is not None:
        ckpt = torch.load(opt.ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        ckpt_epoch = ckpt['epoch'] + 1  # resume from next epoch
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)   
        print(ckpt_epoch)

    trainer = ModelFit(model=model, optimizer=optimizer, scheduler=scheduler, 
                       loss=loss_fn, device=device, fout=fout)
    trainer(ckpt_epoch, MAX_EPOCHS, trainLoader, testLoader, write_folder)
    fout.close()

    # save the trained model
    torch.save(trainer.model.state_dict(), f"trained_models/retrained_model_knn{knn}.pth")
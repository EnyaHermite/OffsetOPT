import torch
import numpy as np
import torch.nn as nn
from .rot_helpers import rot_x, rot_y, rot_z


class rotate_point_cloud(nn.Module):
    
    def __init__(self, max_angle=2*np.pi, upaxis='x', prob=0.95, has_normal=False):
        super().__init__()
        self.max_angle = max_angle
        self.upaxis = upaxis
        self.prob = prob       
        self.has_normal = has_normal 
    
    def forward(self, ptcloud):
        """ Randomly rotate the point clouds to augment the dataset
            rotation is vertical
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, rotated point clouds
        """
        if torch.rand([])<self.prob:
            angle = torch.rand([1])*self.max_angle
            if self.upaxis=='x':
                R = rot_x(angle)
            elif self.upaxis=='y':
                R = rot_y(angle)
            elif self.upaxis=='z':
                R = rot_z(angle)
            else:
                raise ValueError('unkown axis!')
            
            xyz = torch.matmul(ptcloud[:,:3], R)
            if self.has_normal:
                normals = torch.matmul(ptcloud[:,3:], R)
                ptcloud = torch.cat([xyz, normals], dim=1)
            else:
                ptcloud = xyz
        return ptcloud
    

class jitter_point_cloud(nn.Module):
    
    def __init__(self, sigma=0.001, prob=0.95):
        super().__init__()
        self.sigma = sigma
        self.prob = prob
        
    def forward(self, xyz):
        """ Randomly jitter point heights.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        if torch.rand([])<self.prob:
            noise = torch.randn(xyz.shape[0],3)*self.sigma
            noise = torch.clip(noise, min=-3*self.sigma, max=3*self.sigma)
            xyz[:,:3] += noise
        return xyz

 
class random_scale_point_cloud(nn.Module):
    
    def __init__(self, scale_low=0.8, scale_high=1.2, prob=0.95):
        super().__init__()
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.prob = prob
        
    def forward(self, xyz):
        """ Randomly scale the point cloud.
            Input:
                Nx3 array, original point clouds
            Return:
                Nx3 array, scaled point clouds
        """
        if torch.rand([])<self.prob:
            scales = torch.rand([3])*(self.scale_high-self.scale_low) + self.scale_low
            xyz[:,:3] *= scales
        return xyz
    

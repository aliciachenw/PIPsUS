"""
Random time difference, load pregenerated data directly
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultrasound.dataaug import adjust_intensity


IMAGE_SHAPE = (256, 256)
SEQ_LENGTH = 20

if os.name == 'nt':
    PSEUDO_LABEL_PATH = 'D:/Wanwen/EchoNet/echonetdynamic-2/EchoNet-Dynamic/pseudo20/'
else:
    PSEUDO_LABEL_PATH = '/workspace/us_seq_dataset/EchoNet/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic/pseudo20/'


def generate_pseudo_gt(seq_filename, rgbs, is_train=False):
    
    folder = PSEUDO_LABEL_PATH
    # print('folder', folder)
    filenames = os.listdir(folder)
    # find the seqname
    filenames = list(filter(lambda x:x.find(seq_filename)!=-1, filenames))
    filenames.sort()

    dataset = PseudoDataset(rgbs, folder, filenames, is_train)

    return dataset

class PseudoDataset(Dataset):
    def __init__(self, rgbs, root_dir, pred_trajs_files, is_train):
        # rgbs: (length, H, W, C)
        # pred_trajs: (S+1, N, 2)
        self.rgbs = rgbs
        self.root_dir = root_dir
        pred_trajs_files.sort()
        self.pred_trajs = []
        self.start_idx = []
        self.is_train = is_train
        for f in pred_trajs_files:
            traj = np.loadtxt(os.path.join(root_dir, f))
            traj = np.reshape(traj, (SEQ_LENGTH, -1, 2))

            # remove some points
            first_kp = traj[0]
            mask = (first_kp[:,0] >= 20) & (first_kp[:,0] < 236)  & (first_kp[:,1] >= 20) & (first_kp[:,1] < 236)
            traj = traj[:,mask,:]

            # check if has point left
            if np.any(mask):
            
                self.pred_trajs.append(traj)
                idx = int(f[-8:-4])
                self.start_idx.append(idx)


                
        self.seq_length = SEQ_LENGTH 
        self.step = 1
        self.len = len(self.pred_trajs)

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        start_frame = self.start_idx[idx]
        image = self.rgbs[start_frame:start_frame+self.seq_length]
        image = image.permute(0,3,1,2).float() # S+1, C, H, W
        if self.is_train:
            image[1:] = adjust_intensity(image[1:], max_gain=20, smooth_sigma=3, noise_sigma=5)
        pred_traj = self.pred_trajs[idx] # S+1, N, 2

        return {'images': image.float(), 'trajs_gt': torch.from_numpy(pred_traj).float()}
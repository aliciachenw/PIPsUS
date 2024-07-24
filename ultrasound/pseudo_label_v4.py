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


IMAGE_SHAPE = (256, 256)
SEQ_LENGTH = 10

if os.name == 'nt':
    PSEUDO_LABEL_PATH = 'D:/Wanwen/TORS/us_us_registration_dataset/2D_cleaned_v2_pseudo10'
else:
    PSEUDO_LABEL_PATH = '/workspace/us_seq_dataset/2D_cleaned_v2_pseudo10'

# MAX_PT_NUM = 128

def generate_pseudo_gt(seq_filename, rgbs, use_batch=False):
    # read traj
    filename_split = seq_filename.split('/')
    patient = filename_split[0]
    sub = filename_split[1]
    scan = filename_split[2]
    seqname = filename_split[3]
    
    folder = os.path.join(PSEUDO_LABEL_PATH, patient, sub, scan)
    # print('folder', folder)
    filenames = os.listdir(folder)
    # find the seqname
    filenames = list(filter(lambda x:x.find(seqname)!=-1, filenames))
    filenames.sort()
    # print('seq_name', seqname)
    # print(filenames)
    dataset = PseudoDataset(rgbs, folder, filenames, use_batch)

    return dataset

class PseudoDataset(Dataset):
    def __init__(self, rgbs, root_dir, pred_trajs_files, use_batch):
        # rgbs: (length, H, W, C)
        # pred_trajs: (S+1, N, 2)
        self.rgbs = rgbs
        self.root_dir = root_dir
        pred_trajs_files.sort()
        self.pred_trajs = []
        self.start_idx = []
        for f in pred_trajs_files:
            traj = np.loadtxt(os.path.join(root_dir, f))
            traj = np.reshape(traj, (SEQ_LENGTH+1, -1, 2))

            # remove some points
            first_kp = traj[0]
            mask = (first_kp[:,0] >= 20) & (first_kp[:,0] < 236)  & (first_kp[:,1] >= 20) & (first_kp[:,1] < 236)
            traj = traj[:,mask,:]

            # check if has point left
            if np.any(mask):
            
                self.pred_trajs.append(traj)
                idx = int(f[-8:-4])
                self.start_idx.append(idx)


                
        self.seq_length = SEQ_LENGTH + 1
        self.step = 1
        self.len = len(self.pred_trajs)

    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        start_frame = self.start_idx[idx]
        image = self.rgbs[start_frame:start_frame+self.seq_length]
        image = image.permute(0,3,1,2).float() # S+1, C, H, W
        pred_traj = self.pred_trajs[idx] # S+1, N, 2

        return {'images': image.float(), 'trajs_gt': torch.from_numpy(pred_traj).float()}
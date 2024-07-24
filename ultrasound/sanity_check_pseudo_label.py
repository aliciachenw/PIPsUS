# generate simple translation dataset for sanity check

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

from ultrasound.pseudo_label_v2 import extract_keypoints, cvt_opencv_kps_to_numpy
from ultrasound.sanity_check_data import SANITY_CHECK_FAKE_DATA_LENGTH
from ultrasound.dataaug import adjust_intensity

DEBUG_MODE = False

MAX_PT_NUM = 256
def generate_pseudo_gt(rgbs, motions, is_train=False):
    # extract keypoints
    all_trajs = []
    # for each starting small seq (length as SEQ_LENGTH), we extract keypoint at the first frame & add motion vector!
    video_length = rgbs.shape[0]
    # print('video length: ', video_length)
    motions = motions.numpy()
    # print(motions.shape)


    # kps
    img = rgbs[0]
    kps = extract_keypoints(img, keypoint_type='sift')#Nx2
    if len(kps) == 0:
        print("len kps == 0")
        # try other keypoint
        # save the image in temp output
        if DEBUG_MODE:
            # plot img
            temp_img = img.cpu().numpy().astype(np.uint8)
            # buffer
            filenames = os.listdir("debug_temp_output")
            filenames = list(filter(lambda x:x.find('kp_debug_img_')!=-1, filenames))
            counter = len(filenames)
            if counter < 100:
                cv2.imwrite(os.path.join("debug_temp_output", 'kp_debug_img_'+str(counter).zfill(4) + '.png'), temp_img)
    else:
        kps = cvt_opencv_kps_to_numpy(kps)

        kps = np.reshape(kps, (1, -1, 2))
        trajs = np.tile(kps, (video_length,1,1)) #s+1,n,2
        trajs[1:] += np.reshape(motions, (video_length-1, 1, 2))
        all_trajs.append(trajs)

    
    dataset = PseudoDataset(rgbs, all_trajs, is_train=is_train)

    return dataset

class PseudoDataset(Dataset):
    def __init__(self, rgbs, traj, is_train=True):
        # rgbs: (length, H, W, C)
        # pred_trajs: (S, N, 2)
        self.rgbs = rgbs
        self.pred_trajs = traj

        self.len = len(self.pred_trajs)
        assert(self.len == 1 or self.len == 0)
        assert(rgbs.shape[0] == SANITY_CHECK_FAKE_DATA_LENGTH+1)
        self.is_train = is_train
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        image = self.rgbs
        image = image.permute(0,3,1,2).float() # S, C, H, W
        if self.is_train:
            image[1:] = adjust_intensity(image[1:], max_gain=20, smooth_sigma=3, noise_sigma=5)

        pred_traj = self.pred_trajs[idx] # S, N, 2

        return {'images': image.float(), 'trajs_gt': torch.from_numpy(pred_traj).float()}
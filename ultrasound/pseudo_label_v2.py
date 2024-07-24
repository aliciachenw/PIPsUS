"""
Random time difference
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultrasound.data import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def extract_keypoints(img, keypoint_type='harris', torch_tensor=True, grayscale=False):
    # input image: opencv grayscale, 0-255, (H, W)
    # output: opencv keypoints list
    # img = np.float32(img)
    if torch_tensor:
        img = img.cpu().numpy()[:,:,0]
        img = img.astype(np.uint8)
    else:
        if not grayscale:
            img = img[:,:,0]
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    # print("gray shape", img.shape)
    if keypoint_type == 'harris':
        response = cv2.cornerHarris(img, blockSize=8, ksize=15, k=0.04)
        ret, dst = cv2.threshold(response, 0.01*response.max(), 255, 0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        kps = []
        centroids = centroids.astype(np.float32)
        for i in range(centroids.shape[0]):
            kps.append(cv2.KeyPoint(centroids[i,0], centroids[i,1], size=1, response=response[int(centroids[i,1]), int(centroids[i,0])]))

    elif keypoint_type == 'shi-tomasi':
        corners = cv2.goodFeaturesToTrack(img, maxCorners=25, qualityLevel=0.01, minDistance=10)
        # corners = np.int0(corners)
        kps = []
        for i in corners:
            x, y = i.ravel()
            kps.append(cv2.KeyPoint(x, y, size=1))
    
    elif keypoint_type == 'sift':
        sift = cv2.SIFT_create(contrastThreshold=0.08, edgeThreshold=4) ## for neck
        kps = sift.detect(img, None)

    elif keypoint_type == 'fast':
        fast = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        kps = fast.detect(img, None)
    
    elif keypoint_type == 'orb':
        orb = cv2.ORB_create()
        kps = orb.detect(img, None)    
    else:
        raise NotImplementedError('keypoint type not implemented')
    # remove edgepoints
    kps = [kp for kp in kps if kp.pt[0] > 10 and kp.pt[0] < img.shape[1]-10 and kp.pt[1] > 10 and kp.pt[1] < img.shape[0]-10]
    return kps

def plot_keypoints(img, kps, vis=False):
    # input image: opencv grayscale, 0-255, (H, W)
    # input keypoints: opencv keypoints list
    # output: opencv image with keypoints
    output_img = cv2.drawKeypoints(img, kps, None, color=(0,255,0))
    if vis:
        plt.figure()
        plt.imshow(output_img)
        plt.show()
    return output_img


def cvt_opencv_kps_to_numpy(kps):
    # input: opencv keypoints list
    # output: numpy array, (N, 2)
    kps_np = []
    for kp in kps:
        kps_np.append([kp.pt[0], kp.pt[1]])
    kps_np = np.stack(kps_np, axis=0)
    return kps_np


# def generate_pseudo_gt(model, rgbs, sequence_length=5, keypoint_type='harris', iters=16, step=1, device='cpu', max_data=100000, use_augs=False):
#     # rgbs: (S, H, W, C)
#     # sequence_length: number of history of frames to use
#     all_trajs_e = []
#     all_frame_idx = []
#     video_length = rgbs.shape[0]
#     model.to(device)
#     with torch.no_grad():
#         for i in range(0, rgbs.shape[0] - sequence_length-1, step):
#             if use_augs:
#                 if np.random.rand() < 0.5:
#                     # randomly pick sequence_length+1 frames
#                     st = i
#                     inds = np.random.randint(st, min(rgbs.shape[0], st+12),size=sequence_length+1)
#                     inds.sort()
#                     inds[0] = i
#                     sub_rgbs = rgbs[inds].permute(0,3,1,2).unsqueeze(0).float()
#                     all_frame_idx.append(inds)
#                 else:
#                     sub_rgbs = rgbs[i:i+sequence_length+1].permute(0,3,1,2).unsqueeze(0).float() # B, S+1, C, H, W
#                     all_frame_idx.append(list(range(i,i+sequence_length+1)))

#             else:
#                 sub_rgbs = rgbs[i:i+sequence_length+1].permute(0,3,1,2).unsqueeze(0).float() # B, S+1, C, H, W
#                 all_frame_idx.append(list(range(i,i+sequence_length+1)))
#             # don't need to normalize rgb because it is done in pips2 directly

#             # extract keypoints and init first estimation
#             kps = extract_keypoints(sub_rgbs[0,0].permute(1,2,0), keypoint_type=keypoint_type)
#             if len(kps) == 0:
#                 print("len kps == 0")
#                 # try other keypoint
#                 for kp_tt in ['harris', 'sift', 'orb', 'shi-tomasi']:
#                     kps = extract_keypoints(sub_rgbs[0,0].permute(1,2,0), keypoint_type=kp_tt)
#                     if len(kps) > 0:
#                         kps = cvt_opencv_kps_to_numpy(kps)
#                         print("use other keypoints:", kp_tt)
#                         break
#             else:
#                 kps = cvt_opencv_kps_to_numpy(kps)
#             traj0 = np.expand_dims(kps, axis=0)
#             traj0 = np.repeat(traj0, sequence_length+1, axis=0) # S+1, N, 2
#             traj0 = np.expand_dims(traj0, axis=0)
#             traj0 = torch.from_numpy(traj0).float().to(device)
#             assert(traj0.shape[1] == sub_rgbs.shape[1] == sequence_length+1)
#             # run model
#             preds, _, _, _ = model(traj0, sub_rgbs.to(device), iters=iters, feat_init=None, beautify=True)
#             trajs_e = preds[-1].squeeze(0)  # last prediction is the pseudo ground truth

#             # save trajs_e
#             all_trajs_e.append(trajs_e.cpu().detach())
#             if device == 'cuda:0':
#                 torch.cuda.empty_cache()
            
#             if len(all_trajs_e) >= max_data:
#                 break
#             # break
#     # init dataset
#     dataset = PseudoDataset(rgbs, all_trajs_e, all_frame_idx, S=sequence_length, step=step)
#     # print("generate pseudo gt dataset length", len(dataset))
#     model.to('cpu')

#     return dataset



def generate_pseudo_gt(rgbs, teacher_model, sequence_length=5, keypoint_type='harris', iters=16, step=1, device='cpu', max_data=100000, use_augs=False):#### FOR ECHO
    # sequence_length: number of history of frames to use
    all_trajs_e = []
    all_frame_idx = []
    video_length = rgbs.shape[0]
    teacher_model.to(device)
    # print("rgbs shape", rgbs.shape)
    with torch.no_grad():
        # extract keypoints and init first estimation
        kps = extract_keypoints(rgbs[0], keypoint_type=keypoint_type)
        if len(kps) == 0:
            print("len kps == 0")
        else:
            kps = cvt_opencv_kps_to_numpy(kps)
            traj0 = np.expand_dims(kps, axis=0)
            traj0 = np.repeat(traj0, video_length, axis=0) # S+1, N, 2
            traj0 = np.expand_dims(traj0, axis=0)
            traj0 = torch.from_numpy(traj0).float().to(device)
            # print(traj0.shape)
            # run model
            preds, _, _, _ = teacher_model(traj0, rgbs.to(device).unsqueeze(0).permute(0,1,4,2,3), iters=iters, feat_init=None, beautify=True)
            trajs_e = preds[-1].squeeze(0)  # last prediction is the pseudo ground truth

            # save trajs_e
            all_trajs_e.append(trajs_e.cpu().detach())
        if device == 'cuda:0':
            torch.cuda.empty_cache()

            # break
    # init dataset
    dataset = PseudoDataset(rgbs, all_trajs_e, all_frame_idx, S=sequence_length, step=step)
    # print("generate pseudo gt dataset length", len(dataset))
    teacher_model.to('cpu')

    return dataset


class PseudoDataset(Dataset):
    def __init__(self, rgbs, pred_trajs, all_idx, S=5, step=1):
        # rgbs: (length, H, W, C)
        # pred_trajs: (S+1, N, 2)
        self.rgbs = rgbs
        self.pred_trajs = pred_trajs
        self.S = S
        self.step = step
        self.len = len(self.pred_trajs)
        self.idx_list = all_idx
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # indx = self.idx_list[idx]
        image = self.rgbs
        # print(image.shape)
        image = image.permute(0,3,1,2).float() # S+1, C, H, W
        pred_traj = self.pred_trajs[idx] # S+1, N, 2
        return {'images': image, 'trajs_gt': pred_traj}
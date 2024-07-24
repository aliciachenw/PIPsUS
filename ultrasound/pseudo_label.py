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
        sift = cv2.SIFT_create(contrastThreshold=0.08, edgeThreshold=4)
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

def generate_pseudo_traj(model, grays, sequence_length=5, keypoint_type='harris', iters=16, device='cpu'):
    all_trajs_e = []
    all_images = []
    for i in range(grays.shape[0] - sequence_length):
        sub_grays = grays[i:i+sequence_length]
        rgbs = cvt_grays_to_rgbs(sub_grays).astype(np.float32)
        rgbs = torch.from_numpy(rgbs).permute(0,3,1,2).unsqueeze(0).float() # B, S, C, H, W
        rgbs = rgbs.to(device)
        print("rgbs.shape", rgbs.shape)
        # don't need to normalize rgb because it is done in pips2 directly

        # extract keypoints and init first estimation
        kps = extract_keypoints(sub_grays[0], keypoint_type=keypoint_type, torch_tensor=False)
        if len(kps) == 0:
            continue
        kps = cvt_opencv_kps_to_numpy(kps)
        traj0 = np.expand_dims(kps, axis=0)
        traj0 = np.repeat(traj0, sequence_length, axis=0)
        traj0 = np.expand_dims(traj0, axis=0)
        traj0 = torch.from_numpy(traj0).float().to(device)
        print("traj0.shape", traj0.shape)
        # run model
        preds, preds_anim, _, _ = model(traj0, rgbs, iters=iters, feat_init=None, beautify=True)
        trajs_e = preds[-1]

        # save trajs_e
        all_trajs_e.append(trajs_e.detach())
        all_images.append(rgbs.detach())
        # print(trajs_e[0,:,0])
        if i >= 5:
            break
    
    return all_trajs_e, all_images
    

def generate_pseudo_gt(model, rgbs, sequence_length=5, keypoint_type='harris', iters=16, step=1, device='cpu', max_data=100000, use_augs=False):
    # rgbs: (S, H, W, C)
    # sequence_length: number of frames to use
    all_trajs_e = []
    all_trajs_idx = []
    video_length = rgbs.shape[0]
    model.to(device)
    with torch.no_grad():
        for i in range(0, rgbs.shape[0] - sequence_length-1, step):
            sub_rgbs = rgbs[i:i+sequence_length+1].permute(0,3,1,2).unsqueeze(0).float() # B, S+1, C, H, W
            # don't need to normalize rgb because it is done in pips2 directly

            # extract keypoints and init first estimation
            kps = extract_keypoints(sub_rgbs[0,0].permute(1,2,0), keypoint_type=keypoint_type)
            
            if len(kps) == 0:
                print("len kps == 0") # can not find kps
                continue

            kps = cvt_opencv_kps_to_numpy(kps)
            traj0 = np.expand_dims(kps, axis=0)
            traj0 = np.repeat(traj0, sequence_length+1, axis=0) # S+1, N, 2
            traj0 = np.expand_dims(traj0, axis=0)
            traj0 = torch.from_numpy(traj0).float().to(device)
            assert(traj0.shape[1] == sub_rgbs.shape[1] == sequence_length+1)
            # run model
            preds, _, _, _ = model(traj0, sub_rgbs.to(device), iters=iters, feat_init=None, beautify=True)
            trajs_e = preds[-1].squeeze(0)  # last prediction is the pseudo ground truth

            # save trajs_e
            all_trajs_e.append(trajs_e.cpu().detach())
            all_trajs_idx.append(i)
            if device == 'cuda:0':
                torch.cuda.empty_cache()
            
            if len(all_trajs_e) >= max_data:
                break

    # init dataset
    dataset = PseudoDataset(rgbs, all_trajs_e, all_trajs_idx, S=sequence_length)
    # print("generate pseudo gt dataset length", len(dataset))
    model.to('cpu')

    return dataset



class PseudoDataset(Dataset):
    def __init__(self, rgbs, pred_trajs, start_idx, S=5):
        # rgbs: (length, H, W, C)
        # pred_trajs: (S+1, N, 2)
        self.rgbs = rgbs
        self.pred_trajs = pred_trajs
        self.S = S
        self.start_idx = start_idx
        self.len = len(self.pred_trajs)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        start_frame = self.start_idx[idx]
        image = self.rgbs[start_frame:start_frame+self.S+1] # include S previous images and 1 current images, (S+1, H, W, C)
        image = image.permute(0,3,1,2).float() # S+1, C, H, W
        pred_traj = self.pred_trajs[idx] # S+1, N, 2
        return {'images': image, 'trajs_gt': pred_traj, 'start_frame': start_frame}
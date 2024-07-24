import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from ultrasound.data import USDataset
from torch.utils.data import DataLoader
from nets.pips2 import Pips
from nets.pipsUS_v5 import PipsUS
import torch
from ultrasound.pseudo_label import extract_keypoints, cvt_opencv_kps_to_numpy
from matcher.gms_matcher import *


def extract_features(model, img, kps):
    # img: H x W x C
    # kps: N x 2
    # return: N x 256
    img = img.permute(2, 0, 1).unsqueeze(0).float()
    kps = kps.unsqueeze(0).float()
    features = model.extract_features(img, kps)
    return features

def main():
    # load model
    model = Pips(stride=8)
    model.load_state_dict(torch.load('./reference_model/model-000200000.pth')['model_state_dict'])
    # model.load_state_dict(torch.load('./checkpoints/1_36_64_i6_1e-4_A_smurf_val_w_artificial_173014/model-000000099.pth')['model_state_dict'])

    model = PipsUS(stride=8)
    model.init_realtime_delta()
    model.load_state_dict(torch.load('./checkpoints/pipsUSv5_i6_S8_size256_256_kpsift_lr1e-4_A_Feb21_finetune_w_pipsv2+rand/model-000000003.pth')['model_state_dict'])
    # load data
    dataset = USDataset('valid', shape=(256, 256))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, data in enumerate(dataloader):
        rgbs = data['rgbs']
        img1 = rgbs[0, 0] # H x W x C
        img2 = rgbs[0, 1] # H x W x C
        cv_kp1 = extract_keypoints(img1, keypoint_type='sift')
        cv_kp2 = extract_keypoints(img2, keypoint_type='sift')
        if len(cv_kp1) == 0 or len(cv_kp2) == 0:
            continue
        kps1 = cvt_opencv_kps_to_numpy(cv_kp1)
        kps1 = torch.from_numpy(kps1).float()

        kps2 = cvt_opencv_kps_to_numpy(cv_kp2)
        kps2 = torch.from_numpy(kps2).float()
        # extract features
        features1 = extract_features(model, img1, kps1)
        features2 = extract_features(model, img2, kps2)
        print(features1.shape, features2.shape)
        # exit()
        descriptor1 = features1.squeeze(0).detach().cpu().numpy()
        descriptor2 = features2.squeeze(0).detach().cpu().numpy()


        img1 = img1.detach().cpu().numpy() 
        img1 = img1.astype(np.uint8)
        img2 = img2.detach().cpu().numpy()
        img2 = img2.astype(np.uint8)

        # bfmatcher = cv2.BFMatcher(cv2.NORM_L2)
        gms = GmsMatcher('bfmatcher')

        matches = gms.compute_matches(img1, img2, cv_kp1, descriptor1, cv_kp2, descriptor2)

        gms.draw_matches(img1, img2, DrawingType.COLOR_CODED_POINTS_XpY, visualize=True, filename='bugfix_pipsUS_matches_' + str(i).zfill(3) + '.png')
        gms.empty_matches()

        if i == 10:
            break

if __name__ == '__main__':
    main()
        
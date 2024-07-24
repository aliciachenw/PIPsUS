"""
inference using NCC
"""

from baseline.NCC.ncc_tracker import ncc_matching
import numpy as np

def ncc_tracking(video, start_points, patch_size, search_size):
    # Fast Normalized Cross-Correlation to track keypoints
    # video: (T, C, H, W) tensor
    # start_points: (N, 2) numpy array

    video_length, C, H, W = video.shape
    video = video.permute(0, 2, 3, 1).numpy()
    video = video.astype(np.uint8)
    video = video[:, :, :, 0] # gray scale
    video_length = video.shape[0]


    trajs = np.zeros((video_length, start_points.shape[0], 2))
    trajs[0] = start_points

    for i in range(1, video_length):
        next_kps = ncc_matching(video[i-1], video[i], trajs[i-1], patch_size, search_size)
        trajs[i] = next_kps

    valids = np.ones((video_length, start_points.shape[0]))
    return trajs, valids

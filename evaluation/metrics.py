"""
read the file and implement the metrics
"""


import numpy as np
import csv
import os

def end_point_error(gt, pred):
    """
    Calculate the end point error
    gt: (S, N, 2)
    pred: (S, N, 2)
    """
    err = np.linalg.norm(gt[-1] - pred[-1], axis=1)
    median_err = np.median(err)
    avg_err = np.mean(err)
    std_err = np.std(err)
    return {'err': err, 'median_err': median_err, 'avg_err': avg_err, 'std_err': std_err, 'unit': 'pixel'}


def trajectory_error(gt, pred):
    """
    Calculate the trajectory error
    gt: (S, N, 2)
    pred: (S, N, 2)
    see accumulate error in sequence
    """
    err = np.linalg.norm(gt - pred, axis=2) # (S, N)
    err = np.mean(err, axis=1) # (S,)
    return {'err': err}

def survival_rate(gt, pred, threshold=50):
    """
    Calculate the survival rate
    Survival: L2 distance < 50 pixels (as defined in PointOdyssey)
    gt: (S, N, 2)
    pred: (S, N, 2)
    """
    err = np.linalg.norm(gt - pred, axis=2) # (S, N)
    survival = err < threshold # (S,N) bool
    pt_num = gt.shape[1]
    survival = np.sum(survival, axis=1) / pt_num # (S,)
    return {'survival': survival}
    
    
def cycle_loss(pred_fw, pred_bw):
    """
    Calculate the cycle loss
    pred_fw: (S, N, 2)
    pred_bw: (S, N, 2)
    """
    err = np.linalg.norm(pred_fw[0] - pred_bw[-1], axis=1) # (N)
    mean_err = np.mean(err)
    median_err = np.median(err)
    std_err = np.std(err)
    return {'err': err, 'mean_err': mean_err, 'median_err': median_err, 'std_err': std_err, 'unit': 'pixel'}
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

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

def cvt_opencv_kps_to_numpy(kps):
    # input: opencv keypoints list
    # output: numpy array, (N, 2)
    kps_np = []
    for kp in kps:
        kps_np.append([kp.pt[0], kp.pt[1]])
    kps_np = np.stack(kps_np, axis=0)
    return kps_np


def smurf_generate_training(teacher_model, videos, device, kp_type, iters=16, aug=True):
    # print(videos.shape)
    first_points = extract_keypoints(videos[0][0], keypoint_type=kp_type, torch_tensor=True)
    if len(first_points) == 0:
        return None
    
    first_points = cvt_opencv_kps_to_numpy(first_points)
    
    B, T, H, W, C = videos.shape
    teacher_model.eval()
    teacher_model.to(device)

    with torch.no_grad():
        traj0 = np.expand_dims(first_points, axis=0)
        traj0 = np.repeat(traj0, T, axis=0) # T, N, 2
        traj0 = np.expand_dims(traj0, axis=0)
        traj0 = torch.from_numpy(traj0).float().to(device)
        videos = videos.permute(0, 1, 4, 2, 3).float().to(device) # B, T, C, H, W

        preds, _, _, _ = teacher_model(traj0, videos, iters=iters)

    if aug:
        # generate student training data
        aug_data = augment_training(videos, preds[-1])
        return aug_data
    else:
        return {'rgbs': videos, 'gt_trajs': preds[-1]}

def augment_training(videos, gt_trajs, max_rotate_angle=10, max_translate=10, gaussian_sigma=5, max_gain=10):
    # random augmentation
    B, T, C, H, W = videos.shape
    B, T, N, _ = gt_trajs.shape

    # generate random homogeneous transform
    for i in range(B):
        angle = np.random.randint(-max_rotate_angle, max_rotate_angle)
        center = (W / 2, H / 2)
        aug_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aug_mat[0, 2] += (np.random.rand() - 0.5) * max_translate * 2
        aug_mat[1, 2] += (np.random.rand() - 0.5) * max_translate * 2

        aug_videos = torch.zeros_like(videos)
        aug_trajs = torch.zeros_like(gt_trajs)

        for j in range(T):
            frame = videos[i, j].permute(1,2,0).cpu().numpy().astype(np.uint8)
            frame = cv2.warpAffine(frame, aug_mat, (W, H))
            frame = adjust_intensity(frame, max_gain, gaussian_sigma)
            aug_videos[i, j] = torch.from_numpy(frame).permute(2,0,1)

            aug_trajs[i, j, :, 0] = gt_trajs[i, j, :, 0] * aug_mat[0, 0] + gt_trajs[i, j, :, 1] * aug_mat[0, 1] + aug_mat[0, 2]
            aug_trajs[i, j, :, 1] = gt_trajs[i, j, :, 0] * aug_mat[1, 0] + gt_trajs[i, j, :, 1] * aug_mat[1, 1] + aug_mat[1, 2]
            
    return {'rgbs': aug_videos, 'gt_trajs': aug_trajs}

def adjust_intensity(img, max_gain, sigma):

    gain = np.random.randint(-max_gain, max_gain)
    img = img + gain
    
    gauss = np.random.normal(0, sigma, img.shape)
    img = img + gauss
    img = np.where(img > 0, img, 0)
    img = np.where(img < 255, img, 255)
    img = img.astype(np.uint8)      
    return img


def artificial_generate_training(videos, device, kp_type, aug=True,  max_rotate_angle=10, max_translate=10, gaussian_sigma=5, max_gain=10):
    first_points = extract_keypoints(videos[0][0], keypoint_type=kp_type, torch_tensor=True)
    if len(first_points) == 0:
        return None
    
    first_points = cvt_opencv_kps_to_numpy(first_points)
    
    B, T, H, W, C = videos.shape
    
    videos = videos[:,0:1,:,:,:].repeat(1, T, 1, 1, 1)

    traj0 = np.expand_dims(first_points, axis=0)
    traj0 = np.repeat(traj0, T, axis=0) # T, N, 2
    traj0 = np.expand_dims(traj0, axis=0)
    gt_trajs = torch.from_numpy(traj0).float().to(device)
    videos = videos.permute(0, 1, 4, 2, 3).float().to(device) # B, T, C, H, W

    if aug:
        aug_videos = torch.zeros_like(videos)
        aug_trajs = torch.zeros_like(gt_trajs)
        # generate student training data
        for i in range(B):
            for j in range(T):
                angle = np.random.randint(-max_rotate_angle, max_rotate_angle)
                center = (W / 2, H / 2)
                aug_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                aug_mat[0, 2] += (np.random.rand() - 0.5) * max_translate * 2
                aug_mat[1, 2] += (np.random.rand() - 0.5) * max_translate * 2

                frame = videos[i, j].permute(1,2,0).cpu().numpy().astype(np.uint8)
                frame = cv2.warpAffine(frame, aug_mat, (W, H))
                frame = adjust_intensity(frame, max_gain, gaussian_sigma)
                
                aug_videos[i, j] = torch.from_numpy(frame).permute(2,0,1)

                aug_trajs[i, j, :, 0] = gt_trajs[i, j, :, 0] * aug_mat[0, 0] + gt_trajs[i, j, :, 1] * aug_mat[0, 1] + aug_mat[0, 2]
                aug_trajs[i, j, :, 1] = gt_trajs[i, j, :, 0] * aug_mat[1, 0] + gt_trajs[i, j, :, 1] * aug_mat[1, 1] + aug_mat[1, 2]
                
        return {'rgbs': aug_videos, 'gt_trajs': aug_trajs}
    else:
        return {'rgbs': videos, 'gt_trajs': traj0}
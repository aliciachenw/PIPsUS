from ultrasound.data import USDataset, ToyUSDataset
from ultrasound.echodata import EchoUSDataset
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from inference_ncc import ncc_tracking
from inference_raft import raft_tracking, load_raft_model
from inference_pipsUS import pipsUS_tracking, load_pipsUS_model, pipsUS_tracking_pipstarts
from inference_pips2 import pips2_tracking, load_pips2_model
from inference_pips2ncc import pips2ncc_tracking, load_pips2ncc_model
from inference_pipsUSv8 import pipsUSv8_tracking, load_pipsUSv8_model
from inference_sift import sift_tracking
from inference_classicflow import flow_tracking
import torch
import cv2
import torch.nn as nn
from ultrasound.pseudo_label_v2 import extract_keypoints, cvt_opencv_kps_to_numpy
from ultrasound.pseudo_label_v3 import generate_pseudo_gt
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from ultrasound.sanity_check_data_v2 import USDataset as RandomUSDataset
from ultrasound.sanity_check_pseudo_label_v2 import generate_pseudo_gt as generate_pseudo_gt_random
from ultrasound.sanity_check_echodata import EchoUSDataset as RandomEchoUSDataset
from ultrasound.sanity_check_echo_pseudo_label import generate_pseudo_gt as generate_pseudo_gt_echo
from ultrasound.pseudo_label_v3_echo import generate_pseudo_gt as generate_pseudo_gt_echo_v3
import utils.improc
import time

def quick_video_write(video, trajs, filename):
    S, C, H, W = video.shape
    _, N, D = trajs.shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 3, (W, H))

    video = video.permute(0, 2, 3, 1).cpu().numpy()
    for i in range(video.shape[0]):
        img = video[i].astype(np.uint8)
        for j in range(N):
            cv2.circle(img, (int(trajs[i, j, 0]), int(trajs[i, j, 1])), 2, (0, 0, 255), -1)
        out.write(img)
    out.release()

def detect_startpoints(img, kps, kp_num, margin=10):
    # detect start points
    
    if kps == 'grid':
        _, H, W = img.shape
        x = np.linspace(margin, W-margin, int(np.sqrt(kp_num)))
        y = np.linspace(margin, H-margin, int(np.sqrt(kp_num)))
        xx, yy = np.meshgrid(x, y)
        start_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)

        # remove the points in black patch
        start_points = start_points.astype(np.int32)
        valid = np.zeros(start_points.shape[0])
        for i in range(start_points.shape[0]):
            patch = img[0, start_points[i, 1]-margin:start_points[i, 1]+margin, start_points[i, 0]-margin:start_points[i, 0]+margin]
            patch = patch.float()
            if torch.mean(patch) > 10:
                valid[i] = 1
        valid = valid > 0
        start_points = start_points[valid]
    else:
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        start_points = extract_keypoints(img, keypoint_type=kps)#Nx2
        if len(start_points) > kp_num:
            start_points = start_points[:kp_num]
        elif len(start_points) == 0:
            return np.array([])
        
        start_points = cvt_opencv_kps_to_numpy(start_points)

    return start_points


def patch_similarity(video, trajs_gt, trajs_pred, patch_size=10):
    # video: S, C, H, W, tensor
    # trajs_gt: S, N, 2, numpy
    # trajs_pred: S, N, 2, numpy
    S, _, H, W = video.shape
    # print('video shape:', video.shape)
    N = trajs_gt.shape[1]
    video = video.permute(0, 2, 3, 1).numpy()[:,:,:,0] # S, H, W - gray scale only
    ssim = np.zeros((S, N))
    ncc = np.zeros((S, N))
    rmse = np.zeros((S, N))
    half_size = patch_size // 2
    for i in range(S):
        # sample the patch
        for j in range(N):
            x_left_gt = max(0, int(trajs_gt[i, j, 0]) - half_size)
            x_right_gt = min(W, int(trajs_gt[i, j, 0]) + half_size+1)
            y_top_gt = max(0, int(trajs_gt[i, j, 1]) - half_size)
            y_bottom_gt = min(H, int(trajs_gt[i, j, 1]) + half_size+1)

            x_left_pred = max(0, int(trajs_pred[i, j, 0]) - half_size)
            x_right_pred = min(W, int(trajs_pred[i, j, 0]) + half_size+1)
            y_top_pred = max(0, int(trajs_pred[i, j, 1]) - half_size)
            y_bottom_pred = min(H, int(trajs_pred[i, j, 1]) + half_size+1)

            if x_right_pred-x_left_pred != x_right_gt-x_left_gt:
                # the patch is not the same size, crop the larger one
                pred_left_margin = int(trajs_pred[i,j,0]) - x_left_pred
                pred_right_margin = x_right_pred - int(trajs_pred[i,j,0])
                gt_left_margin = int(trajs_gt[i,j,0]) - x_left_gt
                gt_right_margin = x_right_gt - int(trajs_gt[i,j,0])
                if pred_left_margin > gt_left_margin:
                    pred_left_margin = gt_left_margin
                    x_left_pred = int(trajs_pred[i,j,0]) - pred_left_margin
                elif pred_left_margin < gt_left_margin:
                    gt_left_margin = pred_left_margin
                    x_left_gt = int(trajs_gt[i,j,0]) - gt_left_margin
                if pred_right_margin > gt_right_margin:
                    pred_right_margin = gt_right_margin
                    x_right_pred = int(trajs_pred[i,j,0]) + pred_right_margin
                elif pred_right_margin < gt_right_margin:
                    gt_right_margin = pred_right_margin
                    x_right_gt = int(trajs_gt[i,j,0]) + gt_right_margin
            if y_bottom_pred-y_top_pred != y_bottom_gt-y_top_gt:
                # the patch is not the same size, crop the larger one
                pred_top_margin = int(trajs_pred[i,j,1]) - y_top_pred
                pred_bottom_margin = y_bottom_pred - int(trajs_pred[i,j,1])
                gt_top_margin = int(trajs_gt[i,j,1]) - y_top_gt
                gt_bottom_margin = y_bottom_gt - int(trajs_gt[i,j,1])
                if pred_top_margin > gt_top_margin:
                    pred_top_margin = gt_top_margin
                    y_top_pred = int(trajs_pred[i,j,1]) - pred_top_margin
                elif pred_top_margin < gt_top_margin:
                    gt_top_margin = pred_top_margin
                    y_top_gt = int(trajs_gt[i,j,1]) - gt_top_margin
                if pred_bottom_margin > gt_bottom_margin:
                    pred_bottom_margin = gt_bottom_margin
                    y_bottom_pred = int(trajs_pred[i,j,1]) + pred_bottom_margin
                elif pred_bottom_margin < gt_bottom_margin:
                    gt_bottom_margin = pred_bottom_margin
                    y_bottom_gt = int(trajs_gt[i,j,1]) + gt_bottom_margin


            patch_gt = video[i, y_top_gt:y_bottom_gt, x_left_gt:x_right_gt]
            patch_pred = video[i, y_top_pred:y_bottom_pred, x_left_pred:x_right_pred]

            if patch_gt.shape[0] == 0 or patch_gt.shape[1] == 0 or patch_pred.shape[0] == 0 or patch_pred.shape[1] == 0 or patch_gt.shape[0] != patch_pred.shape[0] or patch_gt.shape[1] != patch_pred.shape[1]:
                ssim[i, j] = np.nan
                ncc[i, j] = np.nan
                rmse[i, j] = np.nan
            elif patch_gt.shape[0] < 7 or patch_gt.shape[1] < 7 or patch_pred.shape[0] < 7 or patch_pred.shape[1] < 7: # patch too small
                ssim[i, j] = np.nan
                ncc[i, j] = np.nan
                rmse[i, j] = np.nan
            else:
                patch_gt = patch_gt.astype(np.float32)
                patch_pred = patch_pred.astype(np.float32)
                ssim[i, j] = structural_similarity(patch_gt, patch_pred, data_range=255-0)
                rmse[i, j] = np.sqrt(mean_squared_error(patch_gt, patch_pred))
                patch_gt_flat = patch_gt.flatten()
                patch_gt_flat /= np.linalg.norm(patch_gt_flat)
                patch_pred_flat = patch_pred.flatten()
                patch_pred_flat /= np.linalg.norm(patch_pred_flat)
                ncc[i, j] = np.correlate(patch_gt_flat, patch_pred_flat)
            
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(patch_gt, cmap='gray')
        # axes[1].imshow(patch_pred, cmap='gray')
        
        # print('ssim:', ssim[i, j], 'mse:', rmse[i, j], 'ncc:', ncc[i, j])
        # plt.show()
    return ssim, rmse, ncc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='test')
    parser.add_argument('--method', type=str, default='ncc')
    parser.add_argument('--patch_size', type=int, default=20)
    parser.add_argument('--val_patch_size', type=int, default=10)
    parser.add_argument('--search_size', type=int, default=40)
    parser.add_argument('--kps', type=str, default='grid')
    parser.add_argument('--kp_num', type=int, default=200)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--savetxt', action='store_true', help='save the tracking results to txt')
    parser.add_argument('--saveimg', action='store_true', help='save the tracking results to images')
    parser.add_argument('--savemp4', action='store_true', help='save the tracking results to videos')
    parser.add_argument('--readtxt', action='store_true', help='read the tracking results from txt')
    parser.add_argument('--compare_pips2', action='store_true', help='read the tracking results from args and pips2 and calculate the difference')

    # read parse
    args = parser.parse_args()

    if args.data not in ['test', 'train', 'valid', 'inplane', 'artificial', 'echo', 'echo_artificial']:
        raise ValueError('Invalid data name')

    if args.method not in ['ncc', 'raft', 'pipsUS', 'pipsUScorr', 'pips2', 'sift', 'raft_tune', 'classical_flow']:
        raise ValueError('Invalid method name')
    
    if args.kps not in ['grid', 'harris', 'sift', 'orb']:
        raise ValueError('Invalid kps name')

    # prep model
    if args.method == 'raft' or args.method == 'raft_tune':
        raft_model = load_raft_model(args)
        raft_model.eval()
    elif args.method == 'pipsUS':
        pipsUS_model = load_pipsUS_model(args)
        pipsUS_model.eval()
    elif args.method == 'pips2':
        pips2_model = load_pips2_model()
        pips2_model.eval()
    # elif args.method == 'pips2ncc':
    #     pips2_model = load_pips2ncc_model()
    #     pips2_model.eval()
    elif args.method == 'pipsUScorr':
        pipsUScorr_model = load_pipsUSv8_model(args)
        pipsUScorr_model.eval()

    # prep data
    if args.data == 'inplane':
        dataset = ToyUSDataset((256, 256))
    elif args.data == 'artificial':
        dataset = RandomUSDataset('test', (256, 256), randomseed=10, smooth=True)
    elif args.data == 'echo':
        dataset = EchoUSDataset('test', (256, 256), use_mini=True)
        print('echo dataset length:', len(dataset))
    elif args.data == 'echo_artificial':
        dataset = RandomEchoUSDataset('test', (256, 256), randomseed=10, smooth=True, use_mini=True)

    else:
        # get dataset
        dataset = USDataset(args.data, (256, 256))
    dataloder = DataLoader(dataset, batch_size=1, shuffle=False)

    if args.savetxt:  ## only support sift for echo and test, can support any keypoints for artificial
        all_l1 = []
        all_l2 = []
        all_time = []
        all_survival = []
        all_ssim = []
        all_rmse = []
        all_ncc = []
        all_mask = []
    
        for i, data in enumerate(dataloder):
            video = data['rgbs'][0]
            filename = data['filename'][0]

            #### generate pseudo ground truth
            if args.data == 'artificial':
                # can give ground truth to any start points
                start_points = detect_startpoints(video[0], args.kps, args.kp_num)
                sub_dataset = generate_pseudo_gt_random(video, data['motion'][0], is_train=True, kps=start_points)  
            elif args.data == 'echo' and args.kps == 'sift':
                sub_dataset = generate_pseudo_gt_echo_v3(filename, video)
            elif args.data in ['train', 'valid', 'test'] and args.kps == 'sift':
                sub_dataset = generate_pseudo_gt(filename, video)
            elif args.data == 'echo_artificial' and args.kps == 'sift':
                # start_points = detect_startpoints(video[0], args.kps, args.kp_num)
                sub_dataset = generate_pseudo_gt_echo(video, data['motion'][0], is_train=True)
            else:
                raise ValueError('Invalid data and kps combo!')
            
            sub_dataloder = DataLoader(sub_dataset, batch_size=1, shuffle=False)

            for j, sub_data in enumerate(sub_dataloder):
                sub_video = sub_data['images'][0] # S, C, H, W
                trajs_gt = sub_data['trajs_gt'][0].numpy()

                start_points = trajs_gt[0]

                if start_points.shape[0] == 0:
                    continue

                start_time = time.time()
                if args.method == 'ncc':
                    trajs, _ = ncc_tracking(sub_video, start_points, args.patch_size, args.search_size)
                elif args.method == 'raft' or args.method == 'raft_tune':
                    trajs, _ = raft_tracking(raft_model, sub_video, start_points, iters=args.iter)
                elif args.method == 'pipsUS':
                    trajs, _ = pipsUS_tracking(pipsUS_model, sub_video, start_points, iters=args.iter)
                elif args.method == 'pips2':
                    trajs, _ = pips2_tracking(pips2_model, sub_video, start_points, iters=args.iter)
                elif args.method == 'pipsUScorr':
                    trajs, _ = pipsUSv8_tracking(pipsUScorr_model, sub_video, start_points, iters=args.iter)
                elif args.method == 'classical_flow':
                    trajs, _ = flow_tracking(sub_video, start_points)
                else:
                    raise ValueError('Invalid method name')
                time_use = time.time() - start_time

                save_path = os.path.join('results', args.method, args.data, args.kps)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                paths = filename.split('/')
                temp_path = save_path
                for p in paths[:-1]:
                    temp_path = os.path.join(temp_path, p)
                    if not os.path.exists(temp_path):
                        os.makedirs(temp_path)

                #### JUST FOR DEBUGGING
                # sw_t = utils.improc.Summ_writer()
                # sw_t.save_this=True
                # trajs = torch.from_numpy(trajs).float().unsqueeze(0) # B, S, N, 2   
                # rgb_save = sw_t.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs',trajs=trajs, rgbs=utils.improc.preprocess_color(sub_video.unsqueeze(0)), cmap='hot', linewidth=1, show_dots=False, only_return=True)

                # out = cv2.VideoWriter('debugging_' + args.method + '_' + args.data + '_' + args.kps + '_' + str(i).zfill(3) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4, (256, 256))
                # for j in range(len(rgb_save)):
                #     out.write(cv2.cvtColor(rgb_save[j].astype(np.uint8), cv2.COLOR_RGB2BGR))
                # out.release()


                np.save(os.path.join(save_path, filename + '_' + str(j).zfill(4) + '.npy'), trajs)

                # evaluation
                if(args.kps == 'sift' and args.data in ['train', 'valid', 'test', 'echo', 'echo_artificial']) or args.data == 'artificial': # only these data has ground truth

                    trajs = torch.from_numpy(trajs).float().unsqueeze(0) # B, S, N, 2
                    trajs_gt = torch.from_numpy(trajs_gt).float().unsqueeze(0) # B, S, N, 2
                    l1 = torch.abs(trajs - trajs_gt).sum(dim=-1) # B, S, N
                    l2 = torch.sqrt(torch.sum((trajs - trajs_gt)**2, dim=-1)) # B, S, N
                    survival = (l2 < 50).float() # B, S, N
                    mask = (trajs_gt[:,:,:,0]>0) & (trajs_gt[:,:,:,1]>0) & (trajs_gt[:,:,:,0]<256) & (trajs_gt[:,:,:,1]<256)
                    all_l1.append(l1.numpy()[0]) # S, N
                    all_l2.append(l2.numpy()[0]) # S, N
                    all_time.append(time_use / trajs.shape[1]) # second per frame
                    all_survival.append(survival.numpy()[0]) # S,N
                    ssim, rmse, ncc = patch_similarity(sub_video, trajs_gt[0].numpy(), trajs[0].numpy(), args.val_patch_size) # S, N
                    all_ssim.append(ssim)
                    all_rmse.append(rmse)
                    all_ncc.append(ncc)
                    all_mask.append(mask.numpy()[0]) # S, N

                    print('time:', time_use, 'l1:', np.mean(l1.numpy()), 'l2:', np.mean(l2.numpy()), 'survival:', np.mean(survival.numpy()), 'ssim:', np.nanmean(ssim), 'rmse:', np.nanmean(rmse), 'ncc:', np.nanmean(ncc))
                else:
                    all_time.append(time_use / trajs.shape[1]) # second per frame
                    print('time:', time_use)
                        

        # save the results
        if(args.kps == 'sift' and args.data in ['train', 'valid', 'test', 'echo', 'echo_artificial']) or args.data == 'artificial':  # only these data has ground truth

            all_l1 = np.concatenate(all_l1, axis=1) # S, N_all
            all_l2 = np.concatenate(all_l2, axis=1) # S, N_all
            all_time = np.array(all_time) # B
            all_survival = np.concatenate(all_survival, axis=1) # B, N_all
            all_ssim = np.concatenate(all_ssim, axis=1) # S, N_all
            all_rmse = np.concatenate(all_rmse, axis=1) # S, N_all
            all_ncc = np.concatenate(all_ncc, axis=1) # S, N_all
            all_mask = np.concatenate(all_mask, axis=1) # S, N_all
            # print(all_mse.shape, all_ssim.shape, all_ncc.shape)
            print(np.mean(all_l1), np.mean(all_l2), np.nanmean(all_ssim), np.nanmean(all_rmse), np.nanmean(all_ncc), np.mean(all_time), np.mean(all_survival))

            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'ssim.txt'), np.array(all_ssim))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'rmse.txt'), np.array(all_rmse))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps,  'ncc.txt'), np.array(all_ncc))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps,  'survival.txt'), np.array(all_survival))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'l1.txt'), np.array(all_l1))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'l2.txt'), np.array(all_l2))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'time.txt'), np.array(all_time))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'mask.txt'), np.array(all_mask))
            all_results = np.array([np.mean(all_l1), np.mean(all_l2), np.nanmean(all_ssim), np.nanmean(all_rmse), np.nanmean(all_ncc), np.mean(all_time), np.mean(all_survival)])
            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'all_results.txt'), all_results)
        else:
            all_time = np.array(all_time)
            print(np.mean(all_time))
            np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'time.txt'), all_time)
        print('Finish inference %d' % i)
        
    elif args.saveimg or args.savemp4:
        # inference on the whole sequence       
        sw_t = utils.improc.Summ_writer()
        sw_t.save_this=True

        for i, data in enumerate(dataloder):
            video = data['rgbs'][0].permute(0,3,1,2)
            filename = data['filename'][0]

            if args.data in ['train', 'valid', 'test']:
                video = video[:20]
            if args.data == 'echo':
                video = video[:20]

            S, C, H, W = video.shape
            print('video shape:', video.shape)

            # get start points
            start_points = detect_startpoints(video[0], args.kps, args.kp_num)
            if start_points.shape[0] == 0:
                print('No keypoints detected in %s' % filename)
                continue

            if args.method == 'ncc':
                trajs, valids = ncc_tracking(video, start_points, args.patch_size, args.search_size)
            elif args.method == 'raft' or args.method == 'raft_tune':
                trajs, valids = raft_tracking(raft_model, video, start_points, iters=args.iter)
            elif args.method == 'pipsUS':
                trajs, valids = pipsUS_tracking(pipsUS_model, video, start_points, iters=args.iter)
            elif args.method == 'pipsUScorr':
                trajs, valids = pipsUSv8_tracking(pipsUScorr_model, video, start_points, iters=args.iter)
            elif args.method == 'pips2':
                trajs, valids = pips2_tracking(pips2_model, video, start_points, iters=args.iter)
            elif args.method == 'pips2ncc':
                trajs, valids = pips2ncc_tracking(pips2_model, video, start_points, iters=args.iter)
            elif args.method == 'sift':
                trajs, valids = sift_tracking(video)
            elif args.method == 'classical_flow':
                trajs, _ = flow_tracking(video, start_points)
            else:
                raise ValueError('Invalid method name')

            trajs = torch.from_numpy(trajs).float().unsqueeze(0) # B, S, C, H, W
            # valids = torch.from_numpy(valids).float().unsqueeze(0) # B, S, C, H, W
            video = video.unsqueeze(0) # B, S, C, H, W
    
            # rgb_save = sw_t.summ_traj2ds_on_rgbs2('outputs/trajs_on_rgbs',trajs=trajs[0:1], rgbs=utils.improc.preprocess_color(video[0:1]), visibles=valids[0:1], cmap='hot', linewidth=1, show_dots=False, only_return=True)
            rgb_save = sw_t.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs',trajs=trajs[0:1], rgbs=utils.improc.preprocess_color(video[0:1]), cmap='hot', linewidth=1, show_dots=False, only_return=True)

            # save the video
            if args.savemp4:
                out = cv2.VideoWriter('results/' + args.method + '_' + args.data + '_' + args.kps + '_' + str(i).zfill(3) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4, (W, H))
                for j in range(len(rgb_save)):
                    out.write(cv2.cvtColor(rgb_save[j].astype(np.uint8), cv2.COLOR_RGB2BGR))
                out.release()
            if args.saveimg:
                # save to image instead of video
                img_path = os.path.join('results', args.method, args.data, args.kps)
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                for j in range(len(rgb_save)):
                    frame = cv2.cvtColor(rgb_save[j].astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(img_path, 'video_' + str(i).zfill(4) + '_frame_' + str(j).zfill(4) + '.png'), frame)
            
            # break
            if i > 29:
                break
        print('Finish inference %d' % i)

    elif args.readtxt: ############ this is not finished yet
        all_l1 = []
        all_l2 = []
        all_time = []
        all_survival = []
        all_ssim = []
        all_rmse = []
        all_ncc = []
        save_path = os.path.join('results', args.method, args.data)
        for i, data in enumerate(dataloder):
            video = data['rgbs'][0]
            filename = data['filename'][0]

            sub_dataset = generate_pseudo_gt(filename, video)

            sub_dataloder = DataLoader(sub_dataset, batch_size=1, shuffle=False)

            for j, sub_data in enumerate(sub_dataloder):
                sub_video = sub_data['images'][0]
                trajs_gt = sub_data['trajs_gt'][0].numpy()
                # get start points
                start_points = trajs_gt[0]
                if start_points.shape[0] == 0:
                    continue
                # read npy
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                paths = filename.split('/')
                temp_path = save_path
                trajs = np.load(os.path.join(save_path, filename + '_' + str(j).zfill(4) + '.npy'))

                # evaluation
                trajs = torch.from_numpy(trajs).float().unsqueeze(0) # B, S, N, 2
                trajs_gt = torch.from_numpy(trajs_gt).float().unsqueeze(0) # B, S, N, 2
                l1 = torch.abs(trajs - trajs_gt).sum(dim=-1) # B, S, N
                l2 = torch.sqrt(torch.sum((trajs - trajs_gt)**2, dim=-1)) # B, S, N
                survival = (l2 < 50).float() # B, S, N
                all_l1.append(l1.numpy()[0])
                all_l2.append(l2.numpy()[0])
                all_survival.append(survival.numpy()[0])
                ssim, rmse, ncc = patch_similarity(sub_video, trajs_gt[0].numpy(), trajs[0].numpy(), args.val_patch_size) # S, N
                all_ssim.append(ssim)
                all_rmse.append(rmse)
                all_ncc.append(ncc)

                print('l1:', np.mean(l1.numpy()), 'l2:', np.mean(l2.numpy()), 'survival:', np.mean(survival.numpy()), 'ssim:', np.nanmean(ssim), 'rmse:', np.nanmean(rmse), 'ncc:', np.nanmean(ncc))
                
        # save the results
        all_l1 = np.concatenate(all_l1, axis=1) # S, N_all
        all_l2 = np.concatenate(all_l2, axis=1) # S, N_all
        all_time = np.array(all_time) # B
        all_survival = np.concatenate(all_survival, axis=1) # B, N_all
        all_ssim = np.concatenate(all_ssim, axis=1) # S, N_all
        all_rmse = np.concatenate(all_rmse, axis=1) # S, N_all
        all_ncc = np.concatenate(all_ncc, axis=1) # S, N_all
        # print(all_mse.shape, all_ssim.shape, all_ncc.shape)
        print(np.mean(all_l1), np.mean(all_l2), np.nanmean(all_ssim), np.nanmean(all_rmse), np.nanmean(all_ncc), np.mean(all_survival))
        all_results = np.loadtxt(os.path.join('results', args.method, args.data, 'all_results.txt'))
        print(all_results)    



    elif args.compare_pips2:  # need an on-the-fly pips2 inference
        if args.method == 'pips2':
            raise ValueError('No need to compare with pips2')
        if args.data == 'artificial':
            raise ValueError('Automatic ground truth for artificial data')

        all_l1 = []
        all_l2 = []
        all_time = []
        all_survival = []
        all_ssim = []
        all_rmse = []
        all_ncc = []
        
        pips2_model = load_pips2_model()
        pips2_model.eval()

        for i, data in enumerate(dataloder):
            video = data['rgbs'][0]
            filename = data['filename'][0]

            if args.data == 'echo':
                seq_length = video_length
            else:
                seq_length = 20

            video_length = video.shape[0]
            if video_length < seq_length:
                continue
            for j in range(0, video_length - seq_length + 1, seq_length):
                sub_video = video[j:j+seq_length]
                start_points = detect_startpoints(sub_video[0], args.kps, args.kp_num)
                sub_video = sub_video.permute(0,3,1,2)
                if start_points.shape[0] == 0:
                    continue
                trajs_gt, _ = pips2_tracking(pips2_model, sub_video, start_points, iters=6)

                start_time = time.time()
                if args.method == 'ncc':
                    trajs, _ = ncc_tracking(sub_video, start_points, args.patch_size, args.search_size)
                elif args.method == 'raft':
                    trajs, _ = raft_tracking(raft_model, sub_video, start_points, iters=args.iter)
                elif args.method == 'pipsUS':
                    trajs, _ = pipsUS_tracking(pipsUS_model, sub_video, start_points, iters=args.iter)
                elif args.method == 'pipsUScorr':
                    trajs, _ = pipsUSv8_tracking(pipsUScorr_model, sub_video, start_points, iters=args.iter)
                else:
                    raise ValueError('Invalid method name')
                time_use = time.time() - start_time

                save_path = os.path.join('results', args.method, args.data, args.kps)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                paths = filename.split('/')
                temp_path = save_path
                for p in paths[:-1]:
                    temp_path = os.path.join(temp_path, p)
                    if not os.path.exists(temp_path):
                        os.makedirs(temp_path)
                np.save(os.path.join(save_path, filename + '_' + str(j).zfill(4) + '.npy'), trajs)

                # evaluation
                trajs = torch.from_numpy(trajs).float().unsqueeze(0) # B, S, N, 2
                trajs_gt = torch.from_numpy(trajs_gt).float().unsqueeze(0) # B, S, N, 2
                l1 = torch.abs(trajs - trajs_gt).sum(dim=-1) # B, S, N
                l2 = torch.sqrt(torch.sum((trajs - trajs_gt)**2, dim=-1)) # B, S, N
                survival = (l2 < 50).float() # B, S, N
                all_l1.append(l1.numpy()[0])
                all_l2.append(l2.numpy()[0])
                all_time.append(time_use / trajs.shape[1]) # second per frame
                all_survival.append(survival.numpy()[0])
                ssim, rmse, ncc = patch_similarity(sub_video, trajs_gt[0].numpy(), trajs[0].numpy(), args.val_patch_size) # S, N
                all_ssim.append(ssim)
                all_rmse.append(rmse)
                all_ncc.append(ncc)

                print('time:', time_use, 'l1:', np.mean(l1.numpy()), 'l2:', np.mean(l2.numpy()), 'survival:', np.mean(survival.numpy()), 'ssim:', np.nanmean(ssim), 'rmse:', np.nanmean(rmse), 'ncc:', np.nanmean(ncc))

        # save the results

        all_l1 = np.concatenate(all_l1, axis=1) # S, N_all
        all_l2 = np.concatenate(all_l2, axis=1) # S, N_all
        all_time = np.array(all_time) # B
        all_survival = np.concatenate(all_survival, axis=1) # B, N_all
        all_ssim = np.concatenate(all_ssim, axis=1) # S, N_all
        all_rmse = np.concatenate(all_rmse, axis=1) # S, N_all
        all_ncc = np.concatenate(all_ncc, axis=1) # S, N_all
        # print(all_mse.shape, all_ssim.shape, all_ncc.shape)
        print(np.mean(all_l1), np.mean(all_l2), np.nanmean(all_ssim), np.nanmean(all_rmse), np.nanmean(all_ncc), np.mean(all_time), np.mean(all_survival))

        np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'ssim.txt'), np.array(all_ssim))
        np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'rmse.txt'), np.array(all_rmse))
        np.savetxt(os.path.join('results', args.method, args.data, args.kps,  'ncc.txt'), np.array(all_ncc))
        np.savetxt(os.path.join('results', args.method, args.data, args.kps,  'survival.txt'), np.array(all_survival))
        np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'l1.txt'), np.array(all_l1))
        np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'l2.txt'), np.array(all_l2))
        np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'time.txt'), np.array(all_time))

        all_results = np.array([np.mean(all_l1), np.mean(all_l2), np.nanmean(all_ssim), np.nanmean(all_rmse), np.nanmean(all_ncc), np.mean(all_time), np.mean(all_survival)])
        np.savetxt(os.path.join('results', args.method, args.data, args.kps, 'all_results.txt'), all_results)

        print('Finish inference %d' % i)
        
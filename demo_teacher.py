import time
import numpy as np
import saverloader
import utils.improc
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
import cv2
from pathlib import Path
import os
from ultrasound.data import USDataset
from torch.utils.data import DataLoader
from ultrasound.pseudo_label import extract_keypoints, cvt_opencv_kps_to_numpy
import re
from nets.pips2 import Pips


def generate_translation_sequence(img, quat_seq_len=25):
    img = img.cpu().numpy() # H,W,3
    print(img.shape)
    seq = [img]
    dx = np.linspace(0, 51, quat_seq_len)
    print(dx.shape)

    dx = np.concatenate((dx, dx[::-1][1:], -dx[1:], -dx[::-1][1:]))
    for i in range(dx.shape[0]):
        new_image = translate_image(img, dx[i], 0)
        seq.append(new_image)
    seq = np.array(seq)
    seq = torch.from_numpy(seq)
    return seq


def translate_image(img, dx, dy):
    aug_mat = np.zeros((2, 3))
    aug_mat[0, 0] = 1
    aug_mat[1, 1] = 1
    aug_mat[0, 2] = dx
    aug_mat[1, 2] = dy
    h, w, c = img.shape
    aug_img = cv2.warpAffine(img, aug_mat, (w, h), flags=cv2.INTER_LINEAR)
    return aug_img


def get_trajs(model, rgbs, iters=16, sw=None, counter=0, keypoint_type='harris', device='cpu'):

    # rgbs: video_length x H x W x C

    # cut shorter
    video_length = min(50, rgbs.shape[0])
    rgbs = rgbs[0:video_length]

    
    video_length, H, W, _ = rgbs.shape
    rgbs = rgbs.float() # video_length, H, W, C
    rgbs = rgbs.to(device)

    kps = extract_keypoints(rgbs[0], keypoint_type=keypoint_type)
    kps = cvt_opencv_kps_to_numpy(kps) # N x 2

    traj0 = np.expand_dims(kps, axis=0)
    traj0 = np.repeat(traj0, video_length, axis=0) # S x N x 2
    traj0 = np.expand_dims(traj0, axis=0) # 1 x S x N x 2
    traj0 = torch.from_numpy(traj0).float().to(device)
    rgbs = rgbs.permute(0,3,1,2).unsqueeze(0)
    preds, _, _, _ = model(traj0, rgbs, iters=iters, feat_init=None, beautify=True)
    trajs_e = preds[-1]  # last prediction is the pseudo ground truth
 
    if sw is not None and sw.save_this:

        rgbs_prep = utils.improc.preprocess_color(rgbs)
        rgb_save = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False, only_return=True)

        rgb_save = rgb_save[0]
        rgb_save = rgb_save.permute(0,2,3,1)
        # save the video
        out = cv2.VideoWriter('realtime_valid_teacher_vis_' + str(counter).zfill(3) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4, (W, H))
        for i in range(rgb_save.shape[0]):
            out.write(cv2.cvtColor(rgb_save[i].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.release()

    return trajs_e




def main(keypoint_type, device='cpu', log_freq=1, log_dir='./logs_demo', model_name='model'):
    # read params from model path
    print("find keypoint type:", keypoint_type)

    # find iter
    iters = 16
    
    # find image shape
    H, W = 256, 256
   
    # load default path
    model_path = './reference_model'

    # build model
    model = Pips(stride=8)

    # load checkpoint
    saverloader.load(model_path, model, model_name=model_name)

    if device == 'cuda:0':
        model = model.cuda()
    
    # load data
    print("loading data...")
    dataset = USDataset('valid', (256, 256))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    print("finish loading data! Dataset size: ", len(dataset))

    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/pips2' + '/t', max_queue=10, flush_secs=60)

    # run model
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i > 4:
                break
            sw_t = utils.improc.Summ_writer(
                writer=writer_t,
                global_step=i+1,
                log_freq=log_freq,
                fps=16,
                scalar_freq=int(log_freq/2),
                just_gif=True)

            us_clips = data['rgbs'][0].to(device)
            print(us_clips.shape)
            # stable motion
            # us_clips = generate_translation_sequence(us_clips[0]).to(device)

            print('us_clips', us_clips.shape)
            # if us_clips.shape[0] < history_seq_len+5:
            #     continue
            # run model
            trajs_e = get_trajs(model, us_clips, iters=iters, sw=sw_t, counter=i, keypoint_type=keypoint_type, device=device)
            print('trajs_e', trajs_e.shape)




if __name__ == '__main__':  
    main(keypoint_type='sift', device='cuda:0', model_name='model')
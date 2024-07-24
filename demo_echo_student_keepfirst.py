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
from nets.old_files.pipsUS_v3 import PipsUS
from ultrasound.echodata import EchoUSDataset
from torch.utils.data import DataLoader
from ultrasound.pseudo_label import extract_keypoints, cvt_opencv_kps_to_numpy
import re
from nets.pips2 import Pips


# def padding_sequence(video,desire_shape):
#     # video: S x H x W x 3
#     S, H, W, C = video.shape
    
#     if (H >= desire_shape[0] or W >= desire_shape[1]):
#         print("wrong padding size!")
#         exit()


#     new_pad_video = torch.zeros((S, desire_shape[0], desire_shape[1], C))
#     # calculate 
#     H_pad = (desire_shape[0] - H) // 2
#     W_pad = (desire_shape[1] - W) // 2
#     new_pad_video[:,H_pad:H_pad+H,W_pad:W_pad+W,:] = video
#     return new_pad_video


def padding_sequence(video,desire_shape):
    # video: S x H x W x 3
    S, H, W, C = video.shape

    video = video.cpu().numpy().astype(np.uint8)

    new_pad_video = np.zeros((S, desire_shape[0], desire_shape[1], C)).astype(np.uint8)
    for i in range(S):
        new_pad_video[i] = cv2.resize(video[i], desire_shape)

    new_pad_video = torch.from_numpy(new_pad_video)
    return new_pad_video



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


def get_trajs(model, rgbs, S, iters=16, sw=None, counter=0, keypoint_type='harris', device='cpu', pips2start=False):

    # rgbs: video_length x H x W x C

    # cut shorter
    # video_length = min(50, rgbs.shape[0])
    # rgbs = rgbs[0:video_length]


    video_length, H, W, _ = rgbs.shape
    rgbs = rgbs.float() # video_length, H, W, C
    rgbs = rgbs.to(device)


    # get init keypoints
    init_rgbs = rgbs[0:1].permute(0,3,1,2).repeat(S,1,1,1).unsqueeze(0).float() # 1 x S x C x H x W
    kps = extract_keypoints(init_rgbs[0,0].permute(1,2,0), keypoint_type=keypoint_type)
    kps = cvt_opencv_kps_to_numpy(kps) # N x 2

    traj0 = np.expand_dims(kps, axis=0)
    traj0 = np.repeat(traj0, S, axis=0) # S x N x 2
    traj0 = np.expand_dims(traj0, axis=0) # 1 x S x N x 2
    traj0 = torch.from_numpy(traj0).float().to(device)
    print("traj0.shape", traj0.shape)
    assert(traj0.shape[1] == init_rgbs.shape[1] == S)

    # get sub seq
    # number of points
    N = traj0.shape[2]
    traj_pre = traj0.clone()
    trajs_e = torch.zeros((1, video_length, N, 2)).to(device)
    trajs_e[0, 0:1] = traj0[0,-1:]
    model.eval()
    prev = init_rgbs.clone()
    with torch.no_grad():
        for i in range(1, video_length):            
            curr = rgbs[i].unsqueeze(0).permute(0,3,1,2).float() # 1 x C x H x W
            preds_coords, _, _ = model(traj_pre, prev, curr, iters=iters, beautify=True)
            pred_point = preds_coords[-1] # 1 x N x 2
            trajs_e[0,i] = pred_point[0]

            # update traj_pre
            traj_pre = torch.cat([trajs_e[0,0:1], traj_pre[0,2:], pred_point], dim=0).unsqueeze(0)
            prev = torch.cat([init_rgbs[:,0:1], prev[:,2:], curr.unsqueeze(0)], dim=1)
            if device == 'cuda:0':
                torch.cuda.empty_cache()


    if sw is not None and sw.save_this:
        rgbs = rgbs.permute(0,3,1,2).unsqueeze(0)

        rgbs_prep = utils.improc.preprocess_color(rgbs)
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
        rgb_save = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False, only_return=True)

        rgb_save = rgb_save[0]
        rgb_save = rgb_save.permute(0,2,3,1)
        # save the video
        out = cv2.VideoWriter('realtime_echo_vis_' + str(counter).zfill(3) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 4, (W, H))
        for i in range(rgb_save.shape[0]):
            out.write(cv2.cvtColor(rgb_save[i].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.release()

    return trajs_e




def main(model_path, device='cpu', log_freq=1, log_dir='./logs_demo', model_name='model', stride=8):
    # read params from model path
    model_params = model_path.split('/')[-1]
    if model_params.find('harris') != -1:
        keypoint_type = 'harris'
    elif model_params.find('sift') != -1:
        keypoint_type = 'sift'
    elif model_params.find('shi-tomasi') != -1:
        keypoint_type = 'shi-tomasi'
    elif model_params.find('orb') != -1:
        keypoint_type = 'orb'
    print("find keypoint type:", keypoint_type)

    # find iter
    iters = -1
    for i in range(10000):
        if model_params.find("_i%d" % i)!= -1:
            iters = i
            print("model iteration time:", iters)
            break
    if iters == -1:
        print("can not read iteration number from model, set as default (16)")
        iters = 16
    
    # find history len
    history_seq_len = -1
    for i in range(100):
        if model_params.find("_S%d" % i)!= -1:
            history_seq_len = i
            print("model history seq len S:", history_seq_len)
            break
    if history_seq_len == -1:
        print("can not read history seq len S from model!!")
        return
    
    # find image shape
    H, W = -1, -1
    for i in range(128, 1025):
        if model_params.find("_size%d" % i) != -1:
            st = model_params.find("_size%d" % i)
            sub_string = model_params[st:st + 15] # should be long enough
            num = re.findall(r'\d+', sub_string)
            H = num[0]
            W = num[1]
            print("reshape size:", H, W)
            reshape_size = (int(H), int(W))
            break
    if H == -1 or W == -1:
        print("can not read reshape size from model!!")
        return
    
    

    # build model
    model = PipsUS(stride=stride, history_seq_len=history_seq_len)
    model.init_realtime_delta()
    if device == 'cuda:0':
        model = model.cuda()
    
    # load checkpoint
    saverloader.load(model_path, model, model_name=model_name)

    # load data
    print("loading data...")
    dataset = EchoUSDataset('test', reshape_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    print("finish loading data! Dataset size: ", len(dataset))

    log_dir = 'logs_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_params + '/t', max_queue=10, flush_secs=60)

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
            # # stable motion
            # us_clips = generate_translation_sequence(us_clips[0]).to(device)
            # us_clips = padding_sequence(us_clips, (512, 512))

            print('us_clips', us_clips.shape)
            if us_clips.shape[0] < history_seq_len+5:
                continue
            # run model
            trajs_e = get_trajs(model, us_clips, S=history_seq_len, iters=iters, sw=sw_t, counter=i, keypoint_type=keypoint_type, device=device)
            print('trajs_e', trajs_e.shape)




if __name__ == '__main__':
    experiment = 'checkpoints/finetune_i6_S8_size256_256_kpsift_lr5e-5_s_A_Feb65_finetune_w_pipsv2_and_random'  
    main(experiment, device='cuda:0', model_name='best_val', stride=8)
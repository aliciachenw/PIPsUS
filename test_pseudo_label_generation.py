
import matplotlib.pyplot as plt
import ultrasound.data as data
import ultrasound.pseudo_label as pseudo_label

import time
import numpy as np
import saverloader
from nets.pips2 import Pips
import utils.improc
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from fire import Fire
import sys
import cv2
from pathlib import Path



def run_model(model, grays, sequence_length=128, iters=16, sw=None, device='cpu'):
    S, H, W = grays.shape
    print("grays.shape", grays.shape)
    iter_start_time = time.time()

    all_trajs_e, all_images = pseudo_label.generate_pseudo_traj(model, grays, sequence_length=sequence_length, keypoint_type='sift', iters=iters, device=device)

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds' % (iter_time))


    # if sw is not None and sw.save_this:

    for counter in range(0, len(all_trajs_e)):
        trajs_e = all_trajs_e[counter]
        rgbs = data.cvt_grays_to_rgbs(grays[counter:counter+sequence_length])
        # rgbs = all_images[counter]

        rgbs = rgbs.astype(np.float32)
        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).unsqueeze(0).float()  # 1, S, C, H, W
        print("rgbs.shape", rgbs.shape)
        # sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
        rgb_save = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False, only_return=True)
        # print('rgb_save', rgb_save.shape)
        # print(rgb_save.dtype)
        rgb_save = rgb_save[0]
        rgb_save = rgb_save.permute(0,2,3,1)
        # save the video
        out = cv2.VideoWriter('pseudo_label_vis_' + str(counter).zfill(3) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 3, (W, H))
        for i in range(rgb_save.shape[0]):
            out.write(cv2.cvtColor(rgb_save[i].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.release()
    return all_trajs_e



def run_model_v2(model, grays, sequence_length=128, iters=16, sw=None, device='cpu'):
    S, H, W = grays.shape
    print("grays.shape", grays.shape)
    iter_start_time = time.time()

    rgbs = data.cvt_grays_to_rgbs(grays).astype(np.float32)
    rgbs = torch.from_numpy(rgbs)
    dataset = pseudo_label.generate_pseudo_gt(model, rgbs, sequence_length=sequence_length, keypoint_type='sift', iters=iters, step=5, device=device)

    iter_time = time.time()-iter_start_time
    print('inference time: %.2f seconds' % (iter_time))


    # if sw is not None and sw.save_this:

    for counter in range(0, 6):
        item = dataset.__getitem__(counter)
        trajs_e = item['trajs_gt'].unsqueeze(0)
        rgbs = item['images'].unsqueeze(0)

        print("rgbs.shape", rgbs.shape)
        rgb_save = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False, only_return=True)

        rgb_save = rgb_save[0]
        rgb_save = rgb_save.permute(0,2,3,1)
        # save the video
        out = cv2.VideoWriter('pseudo_label_vis_' + str(counter).zfill(3) + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 3, (W, H))
        for i in range(rgb_save.shape[0]):
            out.write(cv2.cvtColor(rgb_save[i].numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
        out.release()
    return dataset


if __name__ == '__main__':
    # # # test image
    # mha_filename = 'D:/Wanwen/TORS/us_us_registration_dataset/inplane_motion_test_data/1/preop/carotid/RecordingTest.igs_20230929_124611.mha'
    # grays = data.read_mha(mha_filename, reshape_size=(512, 512))

    # # for i in range(0, grays.shape[0], 30):
    # #     kps = pseudo_label.extract_keypoints(grays[i], keypoint_type='orb', torch_tensor=False)
    # #     output_img = pseudo_label.plot_keypoints(grays[i], kps, vis=True)

    
    # exp_name = 'de00' # copy from dev repo

    # S = sequence_length = 20
    # log_freq = 1

    # S_here,H,W = grays.shape
    # print('grays', grays.shape)
    # init_dir='./reference_model'
    # # autogen a name
    # model_name = "test_pseudo_labeling"
    # import datetime
    # model_date = datetime.datetime.now().strftime('%H:%M:%S')
    # model_name = model_name # + '_' + model_date ## this will cause OS error in windows
    # print('model_name', model_name)
    
    # log_dir = 'logs_demo'
    # writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    # global_step = 0

    # model = Pips(stride=8) #.cuda()
    # parameters = list(model.parameters())
    # if init_dir:
    #     _ = saverloader.load(init_dir, model)
    # global_step = 0
    # model.eval()


    # with torch.no_grad():
        
    #     sw_t = utils.improc.Summ_writer(
    #         writer=writer_t,
    #         global_step=global_step,
    #         log_freq=log_freq,
    #         fps=16,
    #         scalar_freq=int(log_freq/2),
    #         just_gif=True)

    #     run_model_v2(model, grays, sequence_length=sequence_length, sw=sw_t)

    

    
    ## test generated pseudo label
    from ultrasound.pseudo_label_v3 import generate_pseudo_gt
    from ultrasound.data import USDataset

    dataset = USDataset('train', (256, 256))
    data = dataset.__getitem__(6)
    videos = data['rgbs']
    filename = data['filename']
    print(filename)
    tracking_dataset = generate_pseudo_gt(filename, videos)
"""
Always keep the first frame as reference (regularization)
"""
import numpy as np
import saverloader
from nets.pipsUS import PipsUS
import utils.improc
import utils.geom
import utils.misc
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from fire import Fire
from torch.utils.data import Dataset, DataLoader
from ultrasound.pseudo_label_v3_echo import generate_pseudo_gt as generate_pseudo_gt_pips2
from ultrasound.echodata import EchoUSDataset  as RealUSDataset
from ultrasound.sanity_check_echodata import EchoUSDataset as RandomUSDataset
from ultrasound.sanity_check_echo_pseudo_label import generate_pseudo_gt as generate_pseudo_gt_random
import random

IMAGE_SIZE = 256
USE_BATCH = False # set to True somehow makes the model predict ~0
USE_GT_RATIO = 0.7
SEQ_DECAY_GAMMA = 0.95
USE_MINI = True

def pt_sequence_loss(pt_preds, pt_gt, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    B, N, D = pt_gt.shape
    assert(D==2)
    n_predictions = len(pt_preds)    
    flow_loss = 0.0
    # generate mask for invalid point
    mask = (pt_gt[:, :, 0] >= 0) & (pt_gt[:,:,1] >= 0) & (pt_gt[:,:,0] <IMAGE_SIZE) & (pt_gt[:,:,1] < IMAGE_SIZE) 
    mask = mask.unsqueeze(2)
    loss_func = nn.HuberLoss(reduction='sum')

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        # i_loss = F.l1_loss(pt_preds[i] * mask, pt_gt * mask, reduction='sum')
        i_loss = loss_func(pt_preds[i] * mask, pt_gt * mask)
        flow_loss += i_weight * i_loss / torch.sum(mask)
    flow_loss = flow_loss/n_predictions


    if torch.isnan(flow_loss):
        if torch.sum(mask) > 0:
            for i in range(n_predictions):
                print("Iteration:", i, "pred has NaN:", torch.isnan(pt_preds[i]).any()) # HAS NaN HERE!!!!! OUTPUT EXPLODED :( - mixing with gt previous traj resolved this XD https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/4
    return flow_loss


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, num_steps+100, pct_start=0.1, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


def train_model_random(student_model, data, device, sequence_length, optimizer, scheduler=None, iters=8, sw=None, use_augs=True, batch_size=4):

    videos = data['rgbs'][0]
    motion = data['motion'][0]
    _, _, H, W = videos.shape

    total_loss = 0
    metrics = {}
    
    # use teacher model to get the ground truth
    # use two times of the sequence, and use schedule sampling to insert model pred into trajs_g
    tracking_dataset = generate_pseudo_gt_random(videos, motion, is_train=True)
    if tracking_dataset.__len__() == 0:
        print("dataset length is 0! exit")
        return 0, metrics
    
    if USE_BATCH:
        dataloader = DataLoader(tracking_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        dataloader = DataLoader(tracking_dataset, batch_size=1, shuffle=True, num_workers=0)

    # start iteration
    student_model.to(device)
    student_model.train()

    seq_counter = 0
    for i, data in enumerate(dataloader):

        if np.random.rand() > 0.5: # skip half of the data
            seq_counter += 1
            seq_loss = 0
            rgbs = data['images'] # B,S,C,H,W
            trajs_g = data['trajs_gt'] # B,S,N,2

            if np.random.rand() < 0.5:  ## zero flow constraint
                rgbs_pad = rgbs[:,0:1].repeat(1,sequence_length*2,1,1,1)
                trajs_g_pad = trajs_g[:,0:1].repeat(1,sequence_length*2,1,1)
                rgbs = torch.cat((rgbs_pad,rgbs[:,0:sequence_length]),dim=1)
                trajs_g = torch.cat((trajs_g_pad, trajs_g[:,0:sequence_length]),dim=1)
            else:
                rgbs_pad = rgbs[:,0:1].repeat(1,sequence_length-1,1,1,1)
                trajs_g_pad = trajs_g[:,0:1].repeat(1,sequence_length-1,1,1)
                rgbs = torch.cat((rgbs_pad,rgbs),dim=1)
                trajs_g = torch.cat((trajs_g_pad, trajs_g),dim=1)

            if use_augs:
                if np.random.rand() < 0.5: # rot90 aug
                    rgbs = rgbs.permute(0,1,2,4,3) # swap xy
                    trajs_g = trajs_g.flip([3]) # swap xy

                if np.random.rand() < 0.5: # time inverse
                    rgbs = rgbs.flip([1])
                    trajs_g = trajs_g.flip([1])


            B, S, C, H, W = rgbs.shape
            assert(C==3)
            B, S, N, D = trajs_g.shape
            assert(D==2)

            loss = torch.tensor(0.0).to(device)
            rgbs = rgbs.to(device)
            trajs_g = trajs_g.to(device)

            valid_loss_counter = 0
            for jj in range(S-sequence_length):

                if jj == 0:
                    image_previous = rgbs[:,jj:jj+sequence_length]
                    trajs_previous = trajs_g[:,:sequence_length]
                    preds_coords = student_model(trajs_previous, image_previous=image_previous, image_curr=rgbs[:,jj+sequence_length], iters=iters, beautify=False)
                else:
                    image_previous = torch.cat((rgbs[:,0:1],rgbs[:,jj+1:jj+sequence_length]), dim=1)

                    if np.random.rand() < USE_GT_RATIO:
                        trajs_previous_ = torch.cat((trajs_g[:,0:1], trajs_g[:,jj+1:jj+sequence_length]), dim=1)
                        # add noise
                        trajs_previous_[:,1:] = trajs_previous_[:,1:] + torch.from_numpy(np.random.normal(0, 1, trajs_previous_[:,1:].shape)).float().to(trajs_previous_.device)
                    else:
                        trajs_previous_ = torch.cat((trajs_g[:,0:1], trajs_previous[:,1:]), dim=1)
                    preds_coords = student_model(trajs_previous_, image_previous=image_previous, image_curr=rgbs[:,jj+sequence_length], iters=iters, beautify=False)

                preds_e = preds_coords[-1] # prediction at the last iteration, B,N,2
                # update trajs previous
                trajs_previous = torch.cat((trajs_previous[:,1:], preds_e.detach().unsqueeze(1)), dim=1)
                loss = pt_sequence_loss(preds_coords, trajs_g[:,jj+sequence_length])

                if torch.isnan(loss):
                    continue

                loss.backward()
                seq_loss = seq_loss + loss.item()
                valid_loss_counter += 1
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                if device == 'cuda:0':
                    torch.cuda.empty_cache()

            if valid_loss_counter > 0:
                seq_loss = seq_loss / valid_loss_counter
            total_loss += seq_loss
    if seq_counter > 0:
        total_loss = total_loss / seq_counter
    student_model.to('cpu')
    metrics['total_loss'] = total_loss

    # # visualize current training for the last batch
    # if sw is not None and sw.save_this:
    #     prep_rgbs = utils.improc.preprocess_color(rgbs)

    #     trajs_pred = trajs_g.clone()
    #     trajs_pred[:,-1] = preds_e[:]
    #     sw.summ_traj2ds_on_rgbs('training/trajs_pred_on_rgbs', trajs_pred[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
    #     sw.summ_traj2ds_on_rgbs('training/trajs_gt_on_rgbs',  trajs_g[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)

    #     sw.summ_traj2ds_on_rgb('training/trajs_pred_on_rgb_curr', trajs_pred[0:1], prep_rgbs[0:1,-1], cmap='spring', linewidth=2)
    #     sw.summ_traj2ds_on_rgb('training/trajs_gt_on_rgb_curr', trajs_g[0:1], prep_rgbs[0:1,-1], cmap='spring', linewidth=2)

    return total_loss, metrics
    

def val_model_pips2(student_model, data, device, sequence_length, iters=8, sw=None, batch_size=4):
    metrics = {}
    videos = data['rgbs'][0]
    filename = data['filename'][0]

    tracking_dataset = generate_pseudo_gt_pips2(filename, videos, is_train=False)

    if tracking_dataset.__len__() == 0:
        print("dataset length is 0! exit")
        return {'total_loss': 0}

    if USE_BATCH:
        dataloader = DataLoader(tracking_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        dataloader = DataLoader(tracking_dataset, batch_size=1, shuffle=False, num_workers=0)
    _, _, H, W = videos.shape

    # metric to calculate
    total_loss = 0    
    student_model.to(device)
    student_model.eval()

    with torch.no_grad():
        
        # start iteration
        for i, data in enumerate(dataloader):
            
            rgbs = data['images'] # B,video_length,C,H,W
            trajs_g = data['trajs_gt'] # B,video_length,N,2
            
            # pad with static start
            rgbs_pad = rgbs[:,0:1].repeat(1,sequence_length-1,1,1,1)
            trajs_g_pad = trajs_g[:,0:1].repeat(1,sequence_length-1,1,1)
            rgbs = torch.cat((rgbs_pad,rgbs),dim=1)
            trajs_g = torch.cat((trajs_g_pad, trajs_g),dim=1)

            B, S, C, H, W = rgbs.shape
            assert(C==3)
            B, S, N, D = trajs_g.shape
            assert(D==2)

            loss = torch.tensor(0.0).to(device)
            rgbs = rgbs.to(device)
            trajs_g = trajs_g.to(device)
            valid_loss_counter = 0
            
            for jj in range(S-sequence_length-1):

                if jj == 0:
                    trajs_previous_from_model = trajs_g[:,:sequence_length] # buffer to save the prediction from model
                    trajs_previous = trajs_g[:,:sequence_length]
                    image_previous = rgbs[:,jj:jj+sequence_length]
                else:
                    trajs_previous = torch.cat((trajs_g[:,0:1], trajs_previous_from_model[:,1:]), dim=1)
                    # trajs_previous = torch.cat((trajs_g[:,0:1], trajs_g[:,jj+1:jj+sequence_length]), dim=1)
                    image_previous = torch.cat((rgbs[:,0:1],rgbs[:,jj+1:jj+sequence_length]), dim=1)

                preds_coords = student_model(trajs_previous, image_previous=image_previous, image_curr=rgbs[:,jj+sequence_length], iters=iters, beautify=True)
                preds_e = preds_coords[-1] # prediction at the last iteration, B,N,2
                trajs_previous_from_model = torch.cat((trajs_previous_from_model[:,1:], preds_e.detach().unsqueeze(1)), dim=1)

                # for now just MSE, in the future add regularization
                curr_loss = pt_sequence_loss(preds_coords, trajs_g[:,jj+sequence_length])

                if torch.isnan(curr_loss):
                    print('nan in loss; skipping')
                    continue
                
                loss += curr_loss
                valid_loss_counter += 1

            if valid_loss_counter > 0:
                loss = loss / valid_loss_counter
            total_loss = loss.item() + total_loss
    
        total_loss = total_loss / len(dataloader)

    # # visualize current training for the last batch
    # if sw is not None and sw.save_this:
    #     prep_rgbs = utils.improc.preprocess_color(rgbs)

    #     trajs_pred = trajs_g.clone()
    #     trajs_pred[:,-1] = preds_e[:]
    #     sw.summ_traj2ds_on_rgbs('valid/trajs_pred_on_rgbs', trajs_pred[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
    #     sw.summ_traj2ds_on_rgbs('valid/trajs_gt_on_rgbs',  trajs_g[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)

    #     sw.summ_traj2ds_on_rgb('valid/trajs_pred_on_rgb_curr', trajs_pred[0:1], prep_rgbs[0:1,-1], cmap='spring', linewidth=2)
    #     sw.summ_traj2ds_on_rgb('valid/trajs_gt_on_rgb_curr', trajs_g[0:1], prep_rgbs[0:1,-1], cmap='spring', linewidth=2)


    # analyze stats for this run
    metrics['total_loss'] = total_loss
    student_model.to('cpu')

    return metrics

def train_model_pips2(student_model, data, device, sequence_length, optimizer, scheduler=None, iters=8, sw=None, use_augs=True, batch_size=4):

    videos = data['rgbs'][0]
    filename = data['filename'][0]

    total_loss = 0
    metrics = {}
    
    # use teacher model to get the ground truth
    # use two times of the sequence, and use schedule sampling to insert model pred into trajs_g
    tracking_dataset = generate_pseudo_gt_pips2(filename, videos, is_train=True)
    if tracking_dataset.__len__() == 0:
        print("dataset length is 0! exit")
        return 0, metrics
    
    if USE_BATCH:
        dataloader = DataLoader(tracking_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    else:
        dataloader = DataLoader(tracking_dataset, batch_size=1, shuffle=True, num_workers=0)

    # start iteration
    student_model.to(device)
    student_model.train()

    for i, data in enumerate(dataloader):
        seq_loss = 0
        rgbs = data['images'] # B,21,C,H,W
        trajs_g = data['trajs_gt'] # B,21,N,2

        if np.random.rand() < 0.5:
            # zero flow
            rgbs_pad = rgbs[:,0:1].repeat(1,sequence_length*2,1,1,1)
            trajs_g_pad = trajs_g[:,0:1].repeat(1,sequence_length*2,1,1)
            rgbs = torch.cat((rgbs_pad,rgbs[:,0:sequence_length]),dim=1)
            trajs_g = torch.cat((trajs_g_pad, trajs_g[:,0:sequence_length]),dim=1)
        else:
            rgbs_pad = rgbs[:,0:1].repeat(1,sequence_length-1,1,1,1)
            trajs_g_pad = trajs_g[:,0:1].repeat(1,sequence_length-1,1,1)
            rgbs = torch.cat((rgbs_pad,rgbs),dim=1)
            trajs_g = torch.cat((trajs_g_pad, trajs_g),dim=1)
        if use_augs:
            if np.random.rand() < 0.5: # rot90 aug
                rgbs = rgbs.permute(0,1,2,4,3) # swap xy
                trajs_g = trajs_g.flip([3]) # swap xy

            if np.random.rand() < 0.5: # time inverse
                rgbs = rgbs.flip([1])
                trajs_g = trajs_g.flip([1])


        B, S, C, H, W = rgbs.shape
        # print("video length", S)
        assert(C==3)
        B, S, N, D = trajs_g.shape
        assert(D==2)
        # print("video length", S)

        loss = torch.tensor(0.0).to(device)
        rgbs = rgbs.to(device)
        trajs_g = trajs_g.to(device)

        valid_loss_counter = 0
        for jj in range(S-sequence_length):

            if jj == 0:
                image_previous = rgbs[:,jj:jj+sequence_length]
                trajs_previous = trajs_g[:,:sequence_length]
                preds_coords = student_model(trajs_previous, image_previous=image_previous, image_curr=rgbs[:,jj+sequence_length], iters=iters, beautify=False)
            else:
                image_previous = torch.cat((rgbs[:,0:1],rgbs[:,jj+1:jj+sequence_length]), dim=1)

                if np.random.rand() < USE_GT_RATIO:
                    trajs_previous_ = torch.cat((trajs_g[:,0:1], trajs_g[:,jj+1:jj+sequence_length]), dim=1)
                    # add noise
                    trajs_previous_[:,1:] = trajs_previous_[:,1:] +  torch.from_numpy(np.random.normal(0, 1, trajs_previous_[:,1:].shape)).float().to(trajs_previous_.device)
                else:
                    trajs_previous_ = torch.cat((trajs_g[:,0:1], trajs_previous[:,1:]), dim=1)
                preds_coords = student_model(trajs_previous_, image_previous=image_previous, image_curr=rgbs[:,jj+sequence_length], iters=iters, beautify=False)

            preds_e = preds_coords[-1] # prediction at the last iteration, B,N,2
            # update trajs previous
            trajs_previous = torch.cat((trajs_previous[:,1:], preds_e.detach().unsqueeze(1)), dim=1)
            i_weight = SEQ_DECAY_GAMMA**(S - sequence_length - jj)
            loss = pt_sequence_loss(preds_coords, trajs_g[:,jj+sequence_length]) * i_weight

            if torch.isnan(loss):
                continue

            loss.backward()
            seq_loss = seq_loss + loss.item()
            valid_loss_counter += 1
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            if device == 'cuda:0':
                torch.cuda.empty_cache()

        if valid_loss_counter > 0:

            seq_loss = seq_loss / valid_loss_counter
        total_loss += seq_loss

    total_loss = total_loss / len(dataloader)


    student_model.to('cpu')
    metrics['total_loss'] = total_loss

    # # visualize current training for the last batch
    # if sw is not None and sw.save_this:
    #     prep_rgbs = utils.improc.preprocess_color(rgbs)

    #     trajs_pred = trajs_g.clone()
    #     trajs_pred[:,-1] = preds_e[:]
    #     sw.summ_traj2ds_on_rgbs('training/trajs_pred_on_rgbs', trajs_pred[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)
    #     sw.summ_traj2ds_on_rgbs('training/trajs_gt_on_rgbs',  trajs_g[0:1], utils.improc.preprocess_color(rgbs[0:1]), cmap='hot', linewidth=1, show_dots=False)

    #     sw.summ_traj2ds_on_rgb('training/trajs_pred_on_rgb_curr', trajs_pred[0:1], prep_rgbs[0:1,-1], cmap='spring', linewidth=2)
    #     sw.summ_traj2ds_on_rgb('training/trajs_gt_on_rgb_curr', trajs_g[0:1], prep_rgbs[0:1,-1], cmap='spring', linewidth=2)

    return total_loss, metrics
   

def train(
        S=5, # seqlen
        stride=8, # spatial stride of the model 
        iters=6, # inference steps of the model
        use_augs=True,
        reshape_size=(IMAGE_SIZE,IMAGE_SIZE), # size of the input to the model
        keypoint = 'sift',
        # optimization
        lr=1e-4,
        use_scheduler=False,
        max_epoch=50,
        # summaries
        log_dir='./logs_train',
        log_freq=5,
        backup_freq=5,
        # saving/loading
        ckpt_dir='./checkpoints',
        keep_latest=2,
        init_dir='', # previous checkpoint to initialize with
        load_optimizer=True,
        load_step=True,
        ignore_load=None,
):
    device = 'cuda:0'

    exp_name = 'Feb27_finetune' 
        
    if init_dir:
        init_dir = '%s/%s' % (ckpt_dir, init_dir)
        
    # autogen a descriptive name
    model_name = "pipsUSMICCAI_echo"
    model_name += "_i%d" % (iters)
    model_name += "_S%d" % (S)
    model_name += "_size%d_%d" % (reshape_size[0], reshape_size[1])
    model_name += "_kp%s" % (keypoint)

    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_lr%s" % lrn
    if use_scheduler:
        model_name += "_s"
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name

    print('model_name', model_name)
    
    ckpt_path = '%s/%s' % (ckpt_dir, model_name)
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    # load dataset
    print("loading data...")
    dataset_t = RealUSDataset('train', reshape_size, use_mini=USE_MINI)
    dataset_v = RealUSDataset('val', reshape_size, use_mini=USE_MINI)
    dataset_t2 = RandomUSDataset('train', reshape_size, use_mini=USE_MINI)
    # dataset_v2 = RandomUSDataset('valid', reshape_size)
    dataloader_t = DataLoader(dataset_t, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    dataloader_v = DataLoader(dataset_v, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    dataloader_t2 = DataLoader(dataset_t2, batch_size=1, shuffle=True, num_workers=0, drop_last=False)
    # dataloader_v2 = DataLoader(dataset_v2, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    print("finish loading data! Dataset size: ", len(dataset_t), "and", len(dataset_v))
    
    max_iters = max_epoch * len(dataset_t) * 100

    # setup model and optimizer
    print("setting up model and optimizer...")
    student_model = PipsUS(stride=stride) #.to(device)
    student_model.init_realtime_delta()
    _ = saverloader.load('./checkpoints/pipsUSMICCAI_echo_i6_S5_size256_256_kpsift_lr5e-4_A_Feb27_warmup', student_model, model_name='model')

    student_model.to(device)

    parameters = list(student_model.parameters())

    weight_decay = 1e-6
    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, max_iters, student_model.parameters())
    else:
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay) 
        scheduler = None

    utils.misc.count_parameters(student_model)

    global_step = 0
    if init_dir:
        if load_step and load_optimizer:
            global_step = saverloader.load(init_dir, student_model, optimizer=optimizer, scheduler=scheduler, ignore_load=ignore_load)
        elif load_step:
            global_step = saverloader.load(init_dir, student_model, ignore_load=ignore_load)
        else:
            _ = saverloader.load(init_dir, student_model.module, ignore_load=ignore_load)
            global_step = 0
    
    requires_grad(parameters, True)
    student_model.train()

    best_val_l1 = 999999.999
    last_epoch = global_step
    sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=min(S,8),
            scalar_freq=log_freq//4,
            just_gif=True)


    for epoch in range(last_epoch, max_epoch):

        # training loop
        for i, data in enumerate(dataloader_t):

            if use_scheduler:
                total_loss, metrics = train_model_pips2(student_model, data, device, S, iters=iters, optimizer=optimizer, scheduler=scheduler, use_augs=use_augs, sw=None)
            else:
                total_loss, metrics = train_model_pips2(student_model, data, device, S, iters=iters, optimizer=optimizer, use_augs=use_augs, sw=None)

            if i % log_freq == 0 or i == len(dataloader_t) - 1:
                print("Training epoch ", global_step, " video ", i, "/", len(dataloader_t), ", total loss", total_loss)

            sw_t.summ_scalar('total_loss', total_loss)

            current_lr = optimizer.param_groups[0]['lr']
            sw_t.summ_scalar('_/current_lr', current_lr)

        # training loop
        for i, data in enumerate(dataloader_t2):

            if use_scheduler:
                total_loss, metrics = train_model_random(student_model, data, device, S, iters=iters, optimizer=optimizer, scheduler=scheduler, use_augs=use_augs, sw=None)
            else:
                total_loss, metrics = train_model_random(student_model, data, device, S, iters=iters, optimizer=optimizer, use_augs=use_augs, sw=None)

            if i % log_freq == 0 or i == len(dataloader_t) - 1:
                print("Training epoch ", global_step, " video ", i, "/", len(dataloader_t), ", total loss", total_loss)

            sw_t.summ_scalar('total_loss', total_loss)

            current_lr = optimizer.param_groups[0]['lr']
            sw_t.summ_scalar('_/current_lr', current_lr)

        global_step += 1

        saverloader.save(ckpt_path, optimizer, student_model, global_step, scheduler=scheduler, keep_latest=keep_latest)
        if global_step % backup_freq == 0:
            saverloader.save(ckpt_path, optimizer, student_model, global_step, scheduler=scheduler, model_name='backup_model')
        # validation loop
        val_loss = 0
        for i, data in enumerate(dataloader_v):
            student_model.eval()

            with torch.no_grad():

                metrics = val_model_pips2(student_model, data, device, S, iters=iters, sw=None)
                val_loss += metrics['total_loss']
                if i % log_freq == 0 or i == len(dataloader_v) - 1:
                    print("Valid video ", i, "/", len(dataloader_v), ", total loss", metrics['total_loss'])

        if val_loss < best_val_l1:
            saverloader.save(ckpt_path, optimizer, student_model, global_step, scheduler=scheduler, keep_latest=keep_latest, model_name='best_val')
            best_val_l1 = val_loss
            print("update best checkpoint! Current epoch: ", global_step)
            if global_step % backup_freq == 0:
                saverloader.save(ckpt_path, optimizer, student_model, global_step, scheduler=scheduler, model_name='best_val_backup')

        student_model.train()
                
                    
    writer_t.close()



if __name__ == '__main__':
    Fire(train)

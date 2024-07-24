from nets.pipsUS_v5 import PipsUS
import torch

import numpy as np
import cv2
import matplotlib.pyplot as plt

def translate_image(img, dx, dy):
    aug_mat = np.zeros((2, 3))
    aug_mat[0, 0] = 1
    aug_mat[1, 1] = 1
    aug_mat[0, 2] = dx
    aug_mat[1, 2] = dy
    h, w, c = img.shape
    aug_img = cv2.warpAffine(img, aug_mat, (w, h), flags=cv2.INTER_LINEAR)
    return aug_img

if __name__ == '__main__':
    with torch.no_grad():
        # load model
        model = PipsUS(stride=8, history_seq_len=8)
        model.init_realtime_delta()

        # todo: load checkpoint
        checkpoint_path = 'checkpoints/pipsUSv5_i6_S8_size256_256_kpsift_lr5e-5_A_Feb11_finetune_zero_flow/best_val-000000010.pth'
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # generate fake data
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # draw a circle
        cv2.circle(img, (128, 128), 5, (255, 255, 255), -1)

        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        
        #
        center_img = img.astype(np.float32)
        center_img = torch.from_numpy(center_img)
        center_img = 2 * (center_img / 255.0) - 1.0
        center_img = np.transpose(center_img, (2, 0, 1)).unsqueeze(0)
        feature = model.fnet(center_img)[0]

        print(feature.shape)

        # translate img
        aug_feats = []
        for trans_x in [20,  40, 60, 80, 100, 120]:

            aug_img = translate_image(img, trans_x, 0)
            aug_img = aug_img.astype(np.float32)
            aug_img = torch.from_numpy(aug_img)
            aug_img = 2 * (aug_img / 255.0) - 1.0
            aug_img = np.transpose(aug_img, (2, 0, 1)).unsqueeze(0)
            feature_aug = model.fnet(aug_img)[0]
            aug_feats.append(feature_aug)

        num_plots = len(aug_feats) + 1
        fig, axes = plt.subplots(4, num_plots)

        # axes[0, 0].imshow(feature[0].cpu().numpy())
        # axes[1, 0].imshow(feature[64].cpu().numpy())
        # axes[2, 0].imshow(feature[64+96].cpu().numpy())
        # axes[3, 0].imshow(feature[64+96+128].cpu().numpy())

        # for i in range(len(aug_feats)):
        #     axes[0, i+1].imshow(aug_feats[i][0].cpu().numpy())
        #     axes[1, i+1].imshow(aug_feats[i][64].cpu().numpy())
        #     axes[2, i+1].imshow(aug_feats[i][64+96].cpu().numpy())
        #     axes[3, i+1].imshow(aug_feats[i][64+96+128].cpu().numpy())



        axes[0, 0].imshow(feature[0].cpu().numpy())
        axes[1, 0].imshow(feature[63].cpu().numpy())
        axes[2, 0].imshow(feature[95].cpu().numpy())
        axes[3, 0].imshow(feature[127].cpu().numpy())

        for i in range(len(aug_feats)):
            axes[0, i+1].imshow(aug_feats[i][0].cpu().numpy())
            axes[1, i+1].imshow(aug_feats[i][63].cpu().numpy())
            axes[2, i+1].imshow(aug_feats[i][95].cpu().numpy())
            axes[3, i+1].imshow(aug_feats[i][127].cpu().numpy())

        plt.show()

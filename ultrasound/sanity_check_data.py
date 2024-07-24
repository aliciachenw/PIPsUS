import SimpleITK as sitk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
# generate simple translation dataset for sanity check

SANITY_CHECK_FAKE_DATA_LENGTH = 40



# generate simple translation dataset for sanity check
def conv_smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def rand_trans_smooth(max_x, max_y, seq_length):
    # random dx
    dx = np.linspace(0, max_x, seq_length)

    m_type = np.random.randint(4)
    if m_type == 0:
        dx = dx
    elif m_type == 1:
        dx = np.concatenate((dx, dx[::-1][1:]))
    elif m_type == 2:
        dx = np.concatenate((dx, dx[::-1][1:], -dx[1:]))
    elif m_type == 3:
        dx = np.concatenate((dx, dx[::-1][1:], -dx[1:], -dx[::-1][1:]))
    
    # resample to the fixed seq length
    x = np.linspace(0, len(dx), seq_length)
    dx = np.interp(x, range(0, len(dx)), dx)


    dy = np.linspace(0, max_y, seq_length)
    m_type = np.random.randint(4)
    if m_type == 0:
        dy = dy
    elif m_type == 1:
        ddyx = np.concatenate((dy, dy[::-1][1:]))
    elif m_type == 2:
        dy = np.concatenate((dy, dy[::-1][1:], -dy[1:]))
    elif m_type == 3:
        dy = np.concatenate((dx, dx[::-1][1:], -dy[1:], -dy[::-1][1:]))
    
    # resample to the fixed seq length
    y = np.linspace(0, len(dy), seq_length)
    dy = np.interp(y, np.arange(0, len(dy)), dy)

    assert len(dx) == len(dy)
    return dx, dy


def rand_trans(max_trans, seq_length, smooth=False, window=10):
    dx = np.random.rand(seq_length)
    dy = np.random.rand(seq_length)
    # scale to -max_trans, +max_trans
    dx = (dx - 0.5) / 0.5 * max_trans
    dy = (dy - 0.5) / 0.5 * max_trans

    # conv filtering
    if smooth:
        dx = conv_smooth(dx, window)
        dy = conv_smooth(dy, window)
    return dx, dy

def generate_translation_sequence(img, seq_length=SANITY_CHECK_FAKE_DATA_LENGTH, smooth=False):
    # img: H,W,3
    seq = [img]
    
    # random dx
    # max_x = np.random.randint(-50, 51)
    # max_y = np.random.randint(-50, 51)
    # dx, dy = rand_trans_smooth(max_x, max_y, seq_length)

    max_x = 20 # 20 pixel motion??
    # dx, dy = rand_trans(max_x, seq_length, smooth=smooth)
    if smooth:
        dx, dy = rand_trans_smooth(max_x, max_x, seq_length)
    else:
        dx, dy = rand_trans(max_x, seq_length)

    for i in range(dx.shape[0]):
        new_image = translate_image(img, dx[i], dy[i])
        seq.append(new_image)
    seq = np.array(seq)
    seq = torch.from_numpy(seq)
    motion = np.vstack((dx, dy)).T # (seq_length, 2)
    motion = torch.from_numpy(motion)
    # print(seq.shape, motion.shape)
    return seq, motion


def translate_image(img, dx, dy):
    aug_mat = np.zeros((2, 3))
    aug_mat[0, 0] = 1
    aug_mat[1, 1] = 1
    aug_mat[0, 2] = dx
    aug_mat[1, 2] = dy
    h, w, c = img.shape
    aug_img = cv2.warpAffine(img, aug_mat, (w, h), flags=cv2.INTER_LINEAR)
    return aug_img



def read_mha(mha_filename, reshape_size=None):
    # Read the image using SimpleITK
    # reshape_size: (W, H) if need to resize, None if no need to resize
    image = sitk.ReadImage(mha_filename)
    image = sitk.GetArrayFromImage(image)
    image = image.astype(np.uint8)
    # print(image.shape)
    if reshape_size is not None: 
        resize_image = []
        for i in range(image.shape[0]):
            resize_image.append(cv2.resize(image[i], reshape_size, interpolation=cv2.INTER_LINEAR))
        image = np.stack(resize_image, axis=0)
        image = image.astype(np.uint8)
    return image


def cvt_grays_to_rgbs(grays):
    rgbs = []
    for i in range(grays.shape[0]):
        rgbs.append(cv2.cvtColor(grays[i], cv2.COLOR_GRAY2RGB))
    rgbs = np.stack(rgbs, axis=0)
    return rgbs



class USDataset(Dataset):
    def __init__(self, split, shape, smooth=False, randomseed=None):
        # shape = (w, h)
        if os.name == 'nt':
            root_dir = 'D:/Wanwen/TORS/us_us_registration_dataset/2D_cleaned_v2'
        else:
            root_dir = '/workspace/us_seq_dataset/2D_cleaned_v2'
        if split == 'train':
            patient_list =  ["OR_01192023_case_1",  "OR_04202023_case6", "OR_05122023_case8",\
                        "OR_04202023_case7", "OR_05122023_case9", "OR_07212023_Surgery1", "OR_07212023_Surgery2",\
                            "OR_07212023_Surgery3", "OR_08032023_Surgery1", "OR_09072023_Surgery1",\
                            "OR_09072023_Surgery2", "OR_09072023_Surgery3"]
        elif split == 'valid':
            patient_list =  ["OR_01262023_case_3", "OR_02232023_case4", "OR_06152023_Surgery2", "OR_03152023_case5"]
        elif split == 'test':
            patient_list = ["OR_01192023_case_2", "OR_06152023_Surgery1", "OR_06152023_Surgery3"]

        if randomseed is not None:
            np.random.seed(randomseed)
        # if split == 'train':
        #     patient_list =  ["OR_01192023_case_1"]
        # elif split == 'valid':
        #     patient_list =  ["OR_01262023_case_3"]
        # elif split == 'test':
        #     patient_list = ["OR_01192023_case_2"]
            
        sub_floders = ['1_BeforeRetraction', '2_AfterRetraction']
        scanning_types = ['1_Neck', '2_SMG', '2_Below_Chin', '3_BOT']
        
        self.video_list = []
        self.shape = shape

        self.filepaths = []

        for patient in patient_list:
            for sub_floder in sub_floders:
                for scan in scanning_types:
                    video_path = os.path.join(root_dir, patient, sub_floder, scan)
                    if os.path.exists(video_path):
                        filenames = os.listdir(video_path)
                        for fn in filenames:
                            if fn.endswith('.igs.mha'):
                                # print('reading', os.path.join(video_path, fn))
                                video = read_mha(os.path.join(video_path, fn), reshape_size=shape)
                                self.video_list.append(video)
                                self.filepaths.append(patient + '/' + sub_floder + '/' + scan + '/' + fn[:-8])
            #                     break
            #             break
            #         break
            #     break
            # break
        self.len = len(self.video_list)

        self.smooth = smooth


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rgbs = cvt_grays_to_rgbs(self.video_list[idx])

        rgbs, motion = generate_translation_sequence(rgbs[0], smooth=self.smooth)  # NOTE: motion is the displacement relative to the first frame!
        filename = self.filepaths[idx]
        return {'rgbs': rgbs, 'motion': motion, 'filename': filename} # (S, H, W, C)
import SimpleITK as sitk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
# generate affine transformed dataset for sanity check

SANITY_CHECK_FAKE_DATA_LENGTH = 40


def apply_affine(img, aug_mat):
    # img: H,W,3
    # mat: 2,3
    h, w, c = img.shape
    img = cv2.warpAffine(img, aug_mat, (w, h), flags=cv2.INTER_LINEAR)
    return img


def generate_random_affine(max_trans=20.0, max_rot=10.0, affine_scale=(0.8, 1.2)):
    """
    max_trans: pixel, max_rot: degree, affine_scale: (min, max)
    """
    T = np.eye(3)
    T[0, 2] = np.random.uniform(-max_trans, max_trans+1)
    T[1, 2] = np.random.uniform(-max_trans, max_trans+1)
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D((128, 128), np.random.uniform(-max_rot, max_rot+1), 1)

    A = np.eye(3)
    aff = np.random.uniform(affine_scale[0], affine_scale[1])
    A[0, 0] = aff
    A[1, 1] = aff

    mat = A @ T @ R
    return mat[:2]


def generate_smooth_affine(seq_length, max_trans=40.0, max_rot=20.0, affine_scale=(0.8, 1.2)):
    """
    max_trans: pixel, max_rot: degree, affine_scale: (min, max)
    """
    # use interpolation to smooth
    trans = []

    max_x = np.random.uniform(max_trans // 2, max_trans+1)
    max_y = np.random.uniform(max_trans // 2, max_trans+1)
    max_angle = np.random.uniform(max_rot // 2, max_rot+1)
    max_scale = np.random.uniform(affine_scale[0], affine_scale[1])

    # interpolate
    x = np.linspace(0, max_x, seq_length)
    y = np.linspace(0, max_y, seq_length)
    angle = np.linspace(0, max_angle, seq_length)
    scale = np.linspace(1, max_scale, seq_length)

    for i in range(seq_length):
        T = np.eye(3)
        T[0, 2] = x[i]
        T[1, 2] = y[i]
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D((128, 128), angle[i], 1)

        A = np.eye(3)
        A[0, 0] = scale[i]
        A[1, 1] = scale[i]

        mat = A @ T @ R
        trans.append(mat[:2])
    return trans

def generate_random_sequence(img, seq_length=SANITY_CHECK_FAKE_DATA_LENGTH, smooth=False):
    # img: H,W,3
    seq = [img]
    trans = []
    # identity = np.eye(2, 3)
    # trans.append(identity)
    if smooth:
        trans = generate_smooth_affine(seq_length)
    else: # just random
        for i in range(seq_length):
            aff = generate_random_affine()
            trans.append(aff)

    for i in range(seq_length):
        aff = trans[i]
        new_image = apply_affine(img, aff)
        seq.append(new_image)

    seq = np.array(seq)
    seq = torch.from_numpy(seq)
    motion = np.array(trans)
    motion = torch.from_numpy(motion)

    return seq, motion


# generate simple translation dataset for sanity check
def conv_smooth(trans, box_pts):
    x = np.zeros(len(trans))
    y = np.zeros(len(trans))
    for i in range(len(trans)):
        x[i] = trans[i][0, 2]
        y[i] = trans[i][1, 2]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    x_smooth = np.convolve(x, box, mode='same')

    for i in range(len(trans)):
        trans[i][0, 2] = x_smooth[i]
        trans[i][1, 2] = y_smooth[i]
    return trans


def affine_transform_points(points, mat):
    # points: N x 2
    # mat: 3 x 3
    N = points.shape[0]
    points_new = np.zeros_like(points)
    points_new[:, 0] = points[:, 0] * mat[0, 0] + points[:, 1] * mat[0, 1] + mat[0, 2]
    points_new[:, 1] = points[:, 0] * mat[1, 0] + points[:, 1] * mat[1, 1] + mat[1, 2]
    return points_new


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
    def __init__(self, split, shape, randomseed=None, smooth=True):
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

        # if split == 'train':
        #     patient_list =  ["OR_01192023_case_1"]
        # elif split == 'valid':
        #     patient_list =  ["OR_01262023_case_3"]
        # elif split == 'test':
        #     patient_list = ["OR_01192023_case_2"]
        if randomseed is not None:
            np.random.seed(randomseed)
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

        rgbs, motion = generate_random_sequence(rgbs[0], smooth=self.smooth)  # NOTE: motion is the displacement relative to the first frame!
        filename = self.filepaths[idx]
        return {'rgbs': rgbs, 'motion': motion, 'filename': filename} # (S, H, W, C)
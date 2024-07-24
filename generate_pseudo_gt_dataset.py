from ultrasound.pseudo_label import *
from ultrasound.data import *
from nets.pips2 import *
import saverloader
import torch

new_path = '/workspace/us_seq_dataset/2D_cleaned_v2_pseudo10'

patient_lists =  ["OR_01192023_case_1",  "OR_04202023_case6", "OR_05122023_case8",\
                    "OR_04202023_case7", "OR_05122023_case9", "OR_07212023_Surgery1", "OR_07212023_Surgery2",\
                        "OR_07212023_Surgery3", "OR_08032023_Surgery1", "OR_09072023_Surgery1",\
                        "OR_09072023_Surgery2", "OR_09072023_Surgery3", "OR_01262023_case_3",\
                         "OR_02232023_case4", "OR_06152023_Surgery2", "OR_03152023_case5",\
                         "OR_01192023_case_2", "OR_06152023_Surgery1", "OR_06152023_Surgery3"]

sub_folders = ['1_BeforeRetraction', '2_AfterRetraction']
scanning_types = ['1_Neck', '2_SMG', '2_Below_Chin', '3_BOT']


step=seq_length=10

def main(reshape_size=(256, 256)):

    # generate path
    for p in patient_lists:
        if not os.path.exists(os.path.join(new_path, p)):
            os.mkdir(os.path.join(new_path, p))
        for s in sub_folders:
            if not os.path.exists(os.path.join(new_path, p, s)):
                os.mkdir(os.path.join(new_path, p, s))
            for scan in scanning_types:
                if not os.path.exists(os.path.join(new_path, p, s, scan)):
                    os.mkdir(os.path.join(new_path, p, s, scan))

    device = 'cuda:0'

    # load dataset
    print("loading data...")
    dataset_t = USDataset('train', reshape_size)
    dataset_v = USDataset('valid', reshape_size)
    dataset_test = USDataset('test', reshape_size)
    
    # setup model and optimizer
    print("setting up model...")
    teacher_model = Pips(stride=8) #.to(device)
    _ = saverloader.load('./reference_model', teacher_model)
    teacher_model.eval()
    
    inference(dataset_t, teacher_model, seq_length, step, device)
    inference(dataset_v, teacher_model, seq_length, step, device)
    inference(dataset_test, teacher_model, seq_length, step, device)


def inference(dataset, model, S=16, step=10, device='cpu'):
    for idx in range(dataset.__len__()):
        data = dataset.__getitem__(idx)
        rgbs = torch.from_numpy(data['rgbs']) # S,H,W,C
        filename = data['filename']
        # run inference
        with torch.no_grad():
            sub_dataset = generate_pseudo_gt(model, rgbs, S, step=step, device=device, keypoint_type='sift')
            # write the sub dataset
            for i in range(sub_dataset.__len__()):
                mini_traj = sub_dataset.__getitem__(i)
                start_frame_i = mini_traj['start_frame']
                filename_i = filename + '_' + str(start_frame_i).zfill(4) + '.csv'
                print(filename_i)
                traj = mini_traj['trajs_gt'].to('cpu').numpy()
                _, N, _ = traj.shape
                traj = np.reshape(traj, ((S+1)*N, 2))
                np.savetxt(os.path.join(new_path, filename_i), traj)




if __name__ == '__main__':
    main()
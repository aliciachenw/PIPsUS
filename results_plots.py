import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# methods = ['ncc', 'raft_tune', 'pips2', 'pipsUScorr', 'pipsUS']
methods = ['ncc', 'raft_tune', 'pipsUScorr', 'pipsUS']
kps = 'sift'
# kps = 'grid'

dataset = 'artificial'
dataset = 'test'
# dataset = 'echo'
# dataset = 'echo_artificial'

# read data
save_path = 'results/'
metrics = ['l1', 'l2', 'ssim', 'ncc', 'survival', 'rmse', 'mask']
image_title = ['L1 difference', 'L2 difference', 'SSIM', 'NCC', 'Survival rate', 'RMSE']
# color_map = ['#1f78b4', '#33a02c', '#ff7f00', '#fb9a99', '#e31a1c', '#6a3d9a']
color_map = ['#1f78b4', '#33a02c','#ff7f00', '#e31a1c', '#6a3d9a']

y_labels = ['pixel', 'pixel', '', '', 'Percentage', '']


data_dict = {}
for method in methods:
    for metric in metrics:
        filename = os.path.join(save_path, method, dataset, kps, metric + '.txt')
        if os.path.exists(filename):
            print('reading', filename)
            data = np.loadtxt(filename)

            # save the results
            data_dict[method + '_' + metric] = data



# for k, metric in enumerate(metrics):
#     if metric == 'mask':
#         continue
#     plt.figure()
#     for i, method in enumerate(methods):
#         if method + '_' + metric not in data_dict:
#             continue
#         data = data_dict[method + '_' + metric]
#         mask = data_dict[method + '_mask']
#         if metric == 'survival':
#             data = data * 100
        
#         # data = np.ma.masked_where(mask==0, data)
#         # mean_data = np.mean(data, axis=1)
#         # std_data = np.std(data, axis=1)
#         # x_axis = np.arange(len(mean_data))
#         # plt.plot(x_axis, mean_data, label=method, color=color_map[i])
#         # # fill between
#         # plt.fill_between(x_axis, mean_data - std_data, mean_data + std_data, alpha=0.2, color=color_map[i])

#         ## percentile plot
#         data[mask==0] = np.nan
#         mean_data = np.nanmean(data, axis=1)
#         std_data = np.nanstd(data, axis=1)
#         x_axis = np.arange(len(mean_data))
#         plt.plot(x_axis, mean_data, label=method, color=color_map[i])
#         # fill between
#         p95 = np.nanpercentile(data, 90, axis=1)
#         print(p95)
#         p5 = np.nanpercentile(data, 10, axis=1)
#         plt.fill_between(x_axis, p5, p95, alpha=0.2, color=color_map[i])

#         # ## percentile plot
#         # data[mask==0] = np.nan
#         # mean_data = np.nanmean(data, axis=1)
#         # std_data = np.nanstd(data, axis=1)
#         # x_axis = np.arange(len(mean_data))
#         # plt.plot(x_axis, mean_data, label=method, color=color_map[i])
#         # # fill between
#         # if metric == 'survival':
#         #     lower = mean_data - std_data
#         #     lower[lower < 0] = 0
#         #     upper = mean_data + std_data
#         #     upper[upper > 100] = 100
#         # if metric == 'l1' or metric == 'l2' or metric == 'rmse':
#         #     lower = mean_data - std_data
#         #     lower[lower < 0] = 0
#         #     upper = mean_data + std_data
#         # plt.fill_between(x_axis, lower, upper, alpha=0.2, color=color_map[i])

#     plt.title(image_title[k])
#     plt.xlabel('Frame')
#     plt.ylabel(y_labels[k])
#     plt.legend()
#     plt.savefig(os.path.join(save_path, dataset + '_' + kps + '_' + metric + '_percentile.png'))
#     # plt.show()
#     plt.close()
#     # break



# # write table
# tab_to_write = np.zeros((len(methods), len(metrics)*2-2))
# for i, method in enumerate(methods):
#     for j, metric in enumerate(metrics):
#         if metric == 'mask':
#             continue
#         if method + '_' + metric not in data_dict:
#             continue
#         data = data_dict[method + '_' + metric]
#         mask = data_dict[method + '_mask']
#         if metric == 'survival':
#             data = data * 100
#         data = np.ma.masked_where(mask==0, data)
#         mean_data = np.nanmean(data)
#         std_data = np.nanstd(data)
#         tab_to_write[i, j*2] = mean_data
#         tab_to_write[i, j*2+1] = std_data

# print(tab_to_write)
# tab_to_write = pd.DataFrame(tab_to_write, columns=['L1', 'L1_std', 'L2', 'L2_std', 'SSIM', 'SSIM_std', 'NCC', 'NCC_std', 'Survival', 'Survival_std', 'RMSE', 'RMSE_std'], index=methods)
# tab_to_write.to_csv(os.path.join(save_path, dataset + '_' + kps +'_table_all.csv'))


# write table
for i, method in enumerate(methods):
    for j, metric in enumerate(metrics):
        if metric == 'mask':
            continue
        if method + '_' + metric not in data_dict:
            continue
        data = data_dict[method + '_' + metric]
        mask = data_dict[method + '_mask']
        if metric == 'survival':
            data = data * 100
            data = np.ma.masked_where(mask==0, data)
            data = data[-1]
            mean_data = np.nanmean(data)
            std_data = np.nanstd(data)
            print("survival", method, mean_data, std_data)


# print(tab_to_write)
# tab_to_write = pd.DataFrame(tab_to_write, columns=['L1', 'L1_std', 'L2', 'L2_std', 'SSIM', 'SSIM_std', 'NCC', 'NCC_std', 'Survival', 'Survival_std', 'RMSE', 'RMSE_std'], index=methods)
# tab_to_write.to_csv(os.path.join(save_path, dataset + '_' + kps +'_table_all.csv'))


# get frame rate

for method in methods:
    all_time = []
    filename = os.path.join(save_path, method, dataset, kps, 'time' + '.txt')
    if os.path.exists(filename):
        print('reading', filename)
        data = np.loadtxt(filename)
        all_time.append(data)
    if len(all_time) == 0:
        continue
    all_time = np.stack(all_time, axis=0)
    fps = 1 / all_time

    print(method, 'mean fps:', np.mean(fps), 'std:', np.std(fps))
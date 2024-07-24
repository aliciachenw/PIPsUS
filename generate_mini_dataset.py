import os
import csv
ECHONET_PATH = 'D:/Wanwen/EchoNet/echonetdynamic-2/EchoNet-Dynamic/EchoNet-Dynamic/'

# randomly sample videos to generate a small dataset
train_list = []
valid_list = []
test_list = []
video_list = []
with open(os.path.join(ECHONET_PATH,'FileList.csv'), 'r') as file:
    reader = csv.reader(file)
    for i, row in enumerate(reader):
        if row[-1].lower() == 'train':
            train_list.append(row)
        elif row[-1].lower() == 'val':
            valid_list.append(row)
        elif row[-1].lower() == 'test':
            test_list.append(row)

# random sampling
import random
train_list = random.sample(train_list, 200)
valid_list = random.sample(valid_list, 50)
test_list = random.sample(test_list, 50)
# write to csv
with open(os.path.join(ECHONET_PATH,'FileList_mini.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_list)   
    writer.writerows(valid_list)
    writer.writerows(test_list)        
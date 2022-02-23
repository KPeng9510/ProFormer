import numpy as np
import os
import csv
import pickle
import shutil
train_path = "/cvhci/data/activity/NTU_RGBD/train_set.csv"
test_path = "/cvhci/data/activity/NTU_RGBD/test_set.csv"

with open(train_path, 'r') as f:
    data_train = list(csv.reader(f, delimiter=";"))
with open(test_path, 'r') as f:
    data_test = list(csv.reader(f, delimiter=";"))
# print(data_train[0])
train_key_list = []
test_key_list = []
for index in data_train:
    train_key_list.append(index[0].split('/')[-1].split('_rgb')[0])
for index in data_test:
    test_key_list.append(index[0].split('/')[-1].split('_rgb')[0])


def find_files(directory, suffix='.png'):
    if not os.path.exists(directory):
        raise ValueError("Directory not found {}".format(directory))

    matches = []
    for root, dirnames, filenames in os.walk(directory):
        # print(filenames)
        for dirname in dirnames:
            for root, dirnames, filenames in os.walk(directory + dirname):
                # print(dirnames)
                for filename in filenames:
                    full_path = os.path.join(directory + dirname, filename)
                    # if filename.endswith(suffix):
                    matches.append(full_path)
    return matches


all_list = []

one_shot_train = find_files("/cvhci/data/activity/kpeng/ntu_oneshot/one_shot/train/", '.skeleton.png')
#print(one_shot_train)
one_shot_sample = find_files("/cvhci/data/activity/kpeng/ntu_oneshot/one_shot/samples/", '.skeleton.png')
one_shot_test = find_files("/cvhci/data/activity/kpeng/ntu_oneshot/one_shot/test/", '.skeleton.png')
#all_list = one_shot_train + one_shot_sample + one_shot_test

key_list_train = [index.split('.skeleton')[0].split('/')[-1] for index in one_shot_train]
label_list_train = [index.split('/')[-2] for index in one_shot_train]
key_list_test = [index.split('.skeleton')[0].split('/')[-1] for index in one_shot_test]
label_list_test = [index.split('/')[-2] for index in one_shot_test]
key_list_sample = [index.split('.skeleton')[0].split('/')[-1] for index in one_shot_sample]
label_list_sample = [index.split('/')[-2] for index in one_shot_sample]
print(key_list_train)
print(label_list_train)
for sample in train_key_list:
    if sample in key_list_train:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/train/"
        position = key_list_train.index(sample)
        os.makedirs(save_path + label_list_train[position], exist_ok=True)
        save_path = save_path + label_list_train[position] + '/' + sample + '.skeleton.png'
        data_path = one_shot_train[position]
        shutil.copyfile(data_path, save_path)
    if sample in key_list_test:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/train/"
        position = key_list_test.index(sample)
        os.makedirs(save_path + label_list_test[position], exist_ok=True)
        save_path = save_path + label_list_test[position] + '/' + sample + '.skeleton.png'
        data_path = one_shot_test[position]
        shutil.copyfile(data_path, save_path)
    if sample in key_list_sample:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/train/"
        position = key_list_sample.index(sample)
        os.makedirs(save_path + label_list_sample[position], exist_ok=True)
        save_path = save_path + label_list_sample[position] + '/' + sample + '.skeleton.png'
        data_path = one_shot_sample[position]
        shutil.copyfile(data_path, save_path)
for sample in test_key_list:
    if sample in key_list_train:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/test/"
        position = key_list_train.index(sample)
        os.makedirs(save_path + label_list_train[position], exist_ok=True)
        save_path = save_path + label_list_train[position] + '/' + sample + '.skeleton.png'
        data_path = one_shot_train[position]
        shutil.copyfile(data_path, save_path)
    if sample in key_list_test:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/test/"
        position = key_list_test.index(sample)
        os.makedirs(save_path + label_list_test[position], exist_ok=True)
        save_path = save_path + label_list_test[position] + '/' + sample + '.skeleton.png'
        data_path = one_shot_test[position]
        shutil.copyfile(data_path, save_path)
    if sample in key_list_sample:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/test/"
        position = key_list_sample.index(sample)
        os.makedirs(save_path + label_list_sample[position], exist_ok=True)
        save_path = save_path + label_list_sample[position] + '/' + sample + '.skeleton.png'
        data_path = one_shot_sample[position]
        shutil.copyfile(data_path, save_path)



# with open('ntu_train.pkl', 'wb') as f:
#    pickle.dump(train_key_list, f)
# with open('ntu_test.pkl', 'wb') as f:
#    pickle.dump(test_key_list, f)


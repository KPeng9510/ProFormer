
import numpy as np
import os
import csv
import pickle
import shutil




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
all_list = one_shot_train + one_shot_sample + one_shot_test

key_list_train = [index.split('.skeleton')[0].split('/')[-1] for index in one_shot_train]
label_list_train = [index.split('/')[-2] for index in one_shot_train]
key_list_test = [index.split('.skeleton')[0].split('/')[-1] for index in one_shot_test]
label_list_test = [index.split('/')[-2] for index in one_shot_test]
key_list_sample = [index.split('.skeleton')[0].split('/')[-1] for index in one_shot_sample]
label_list_sample = [index.split('/')[-2] for index in one_shot_sample]
#print(key_list_train)
#print(label_list_train)
all_key_list = key_list_train + key_list_sample + key_list_test
all_label_list = label_list_train + label_list_sample + label_list_test
training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,
38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
for sample in all_key_list:
    person_number = int(sample.split('P')[-1].split('R')[0])
    if person_number in training_subjects:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/train/"
        position = all_key_list.index(sample)
        os.makedirs(save_path + all_label_list[position], exist_ok=True)
        save_path = save_path + all_label_list[position] + '/' + sample + '.skeleton.png'
        data_path = all_list[position]
        shutil.copyfile(data_path, save_path)
    else:
        save_path = "/cvhci/data/activity/kpeng/ntu_train_test/test/"
        position = all_key_list.index(sample)
        os.makedirs(save_path + all_label_list[position], exist_ok=True)
        save_path = save_path + all_label_list[position] + '/' + sample + '.skeleton.png'
        data_path = all_list[position]
        shutil.copyfile(data_path, save_path)



# with open('ntu_train.pkl', 'wb') as f:
#    pickle.dump(train_key_list, f)
# with open('ntu_test.pkl', 'wb') as f:
#    pickle.dump(test_key_list, f)


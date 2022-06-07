import numpy as np
import os
from os import path as osp
np.random.seed(1)
import random 

# change the data_root to the root directory of dataset with all objects
abspath = osp.abspath(__file__)
dname = osp.dirname(abspath)
os.chdir(dname)

data_root_path = "/mnt/sda/suddhu/shape-closures/data/real/"
objects = sorted(os.listdir(data_root_path))

# write training/validation/testing data loader files
save_path = osp.join('..', 'data_files')
test_data_file = open(osp.join(save_path,'test_data_real.txt'),'w')

global_test_idx = 0
for object in objects:
    obj_path = osp.join(data_root_path, object)
    if not osp.isdir(obj_path):
        continue
    datasets = sorted(os.listdir(obj_path))
    print("dataset: ", object)

    for dataset in datasets:
        dataset_path = osp.join(obj_path, dataset)
        if dataset == 'bg' or not osp.isdir(dataset_path):
            continue
        # load in tactile images from real sensor
        tactile_path = osp.join(dataset_path, 'frames')
        imgs = sorted(os.listdir(tactile_path))
        imgs = [ x for x in imgs if ".jpg" in x ]

        if len(imgs) > 10:
            imgs = random.sample(imgs, 10)
        for i, img in enumerate(imgs):
            test_data_file.write(str(i) + "," + tactile_path + "/" + img + "\n")
            global_test_idx += 1

print("Real test data size: {}".format(global_test_idx))
test_data_file.close()

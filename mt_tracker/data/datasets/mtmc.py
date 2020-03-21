import os, random
import numpy as np


class mtmc(object):

    def __init__(self):
        self.root = "/home/apie/projects/AIC20-MTMC/dataset/AIC20_T3"
        self.train_dir = os.path.join(self.root, 'train')
        self.val_dir   = os.path.join(self.root, "validation")

        self.train_data, self.train_target = self.process_data(self.train_dir)

    def loadfeature(self, file_list):
        feature_dict = dict()

        for f in file_list:
            with open(f, "r") as f:
                for line in f.readlines():
                    split_line = line.strip("\n").split(",")
                    _id = int(split_line[1])
                    feature = [float(f) for f in split_line[11:]]
                    if _id not in feature_dict:
                        feature_dict[_id] = [feature]
                    else:
                        feature_dict[_id].append(feature)
        return feature_dict

    def process_data(self, data_dir):
        scene_dirs, data, gt = list(), list(), list()

        for dirname in os.listdir(data_dir):
            if dirname.startswith("S0"):
                scene_dirs.append(dirname)
        for scene_dir in scene_dirs:
            camera_dirs, file_list = list(), list()
            scene_path  = os.path.join(data_dir, scene_dir)
            for dirname in os.listdir(scene_path):
                if dirname.startswith("c0"):
                    camera_dirs.append(dirname)
            for camera_dir in camera_dirs:
                filename = os.path.join(scene_path, camera_dir, "det_reid_features.txt")
                file_list.append(filename)

            feature_dict = self.loadfeature(file_list)
            ids = list(feature_dict.keys())

            for i in ids:
                features = feature_dict[i]
                num = 0

                # Positive Data
                for j in range(len(features)):
                    for k in range(len(features)):
                        if j != k:
                            _input = torch.FloatTensor(features[j] + features[k])
                            data.append(_input)
                            gt.append(1)
                            num += 1

                # Negtive Data
                for _ in range(num):
                    while True:
                        _id = random.choice(ids)
                        if _id != i:
                            break
                    
                    negtive_feature = random.choice(features[_id])
                    feature = random.choice(features)
                    _input = feature + negtive_feature
                    data.append(_input)
                    gt.append(0)

            return torch.FloatTensor(data), torch.LongTensor(gt)
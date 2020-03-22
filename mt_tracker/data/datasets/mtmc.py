import os, random, torch
import numpy as np


class mtmc(object):

    def __init__(self):
        self.root = "/home/apie/projects/AIC20-MTMC/dataset/AIC20_T3"
        self.train_dir = os.path.join(self.root, 'train')
        self.val_dir   = os.path.join(self.root, "validation")

        self.trainset = self.process_data(self.train_dir)
        self.valset   = self.process_data(self.val_dir)

    def loadfeature(self, file_list):
        feature_dict = dict()

        for fn in file_list:
            print (f"Load {fn}...")
            with open(fn, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i]
                    split_line = line.strip("\n").split(",")
                    camid = fn.split("/")[-2]
                    _id = int(split_line[2])
                    if _id not in feature_dict:
                        feature_dict[_id] = dict()
                        feature_dict[_id][camid] = [i]
                    else:
                        if camid not in feature_dict[_id]:
                            feature_dict[_id][camid] = [i]
                        else:
                            feature_dict[_id][camid].append(i)
        return feature_dict

    def process_data(self, data_dir):
        scene_dirs, dataset = list(), list()
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
                filename = os.path.join(scene_path, camera_dir, "tracker_train_file.txt")
                file_list.append(filename)
            feature_dict = self.loadfeature(file_list)
            ids = list(feature_dict.keys())
            for i in ids:
                print (f"Producing id={i} Data")
                camids = list(feature_dict[i].keys())
                
                # Positive Data
                for j in range(len(camids)):
                    camj = camids[j]
                    features1 = feature_dict[i][camj]
                    num = 0
                    for k in range(j + 1, len(camids)):
                        camk = camids[k]
                        if j != k:
                            features2 = feature_dict[i][camk]
                            for f1 in features1:
                                for f2 in features2:
                                    _input = [scene_dir, camj, f1, camk, f2, 1]
                                    dataset.append(_input)
                                    num += 1
                
                    # Negtive Data
                    for _ in range(num):
                        
                        while True:
                            _id = random.choice(ids)
                            if _id != i:
                                break
                        while True:
                            neg_camid = random.choice(list(feature_dict[_id].keys()))
                            if neg_camid != j:
                                break
                        neg_feature  = random.choice(feature_dict[_id][neg_camid])
                        feature = random.choice(features1)
                        _input = [scene_dir, camj, feature, neg_camid, neg_feature, 0]
                        dataset.append(_input)

        return dataset
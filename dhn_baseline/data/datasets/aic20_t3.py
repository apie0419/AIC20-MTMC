import os
import numpy as np

class aic20_t3(object):
    
    def __init__(self, **kwargs):
        self.root = "/home/apie/projects/AIC20-MTMC/dataset/AIC20_T3"
        self.train_dir = os.path.join(self.root, 'train')
        self.val_dir   = os.path.join(self.root, "validation")

        self.trainset = self.process_data(self.train_dir)
        self._min, self._max = self.trainset[:, 0].min(), self.trainset[:, 0].max()
        self.valset = self.process_data(self.val_dir)
        self.trainset = self.normalize(self.trainset, self._min, self._max)
        self.valset = self.normalize(self.valset, self._min, self._max)


    def normalize(self, data, _min, _max):
        data[:, 0] = (data[:, 0] - _min)/(_max - _min)
        return data

    def produce_distance_matrix(self, pre_boxes, now_boxes):
        pre_num = len(pre_boxes)
        now_num = len(now_boxes)
        mat_size = max(pre_num, now_num)
        cost_matrix = np.zeros((mat_size, mat_size))
        gt_matrix = np.zeros((mat_size, mat_size))
        for i in range(pre_num):
            pre_features = pre_boxes[i]["features"]
            pre_gps = pre_boxes[i]["gps"]
            pre_id = pre_boxes[i]["id"]
            for j in range(now_num):        
                now_features = now_boxes[i]["features"]
                now_gps = now_boxes[i]["gps"]
                now_id  = now_boxes[i]["id"]
                gps_dis_vec = ((pre_gps[0]-now_gps[0]), (pre_gps[1]-now_gps[1]))
                gps_dis = (gps_dis_vec[0]*100000)**2 + (gps_dis_vec[1]*100000)**2
                feature_dis_vec = now_features - pre_features
                feature_dis = np.dot(feature_dis_vec.T, feature_dis_vec)
                total_dis = gps_dis*0.05 + feature_dis*0.95
                cost_matrix[i][j] = total_dis
                if now_id == pre_id:
                    gt_matrix[i][j] = 1
        
        return cost_matrix, gt_matrix

    def read_gt_file(self, gt_file_path):
        now_frame_idx = 0
        dataset, pre_boxes= list(), list()
        with open(gt_file_path, "r") as f:
            lines = f.readlines()
            now_boxes = list()
            for line in f.readlines():
                split_line = line.strip("\n").split(",")
                frame_idx = int(split_line[0])
                vehicle_id = int(split_line[1])
                GPS_coor = np.array([split_line[2], split_line[3]], dtype=np.float32)
                features = np.array(split_line[4:], dtype=np.float32)
                if frame_idx != now_frame_idx:
                    if len(now_boxes) > 0 and len(pre_boxes) > 0:
                        cost, gt = self.produce_distance_matrix(pre_boxes, now_boxes)
                        dataset.append([data, target])
                    pre_boxes = now_boxes
                    now_boxes = list()
                    now_frame_idx = frame_idx
                
                now_boxes.append({
                    "id": vehicle_id,
                    "gps": GPS_coor,
                    "features": features
                })

        return dataset

    def process_data(self, data_dir):
        scene_dirs, all_data = list(), list()
         
        for dirname in os.listdir(data_dir):
            if dirname.startswith("S0"):
                scene_dirs.append(dirname)

        for scene_dir in scene_dirs:
            camera_dirs = list()
            scene_path  = os.path.join(data_dir, scene_dir)
            for dirname in os.listdir(scene_path):
                if dirname.startswith("c0"):
                    camera_dirs.append(dirname)
            for camera_dir in camera_dirs:
                gt_file = os.path.join(scene_path, camera_dir, "dhn_gt_file.txt")
                dataset = self.read_gt_file(gt_file)
                all_data.extend(dataset)
        
        return np.array(all_data)
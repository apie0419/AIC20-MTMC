import os, random
import numpy as np


class mtmc(object):

    def __init__(self):
        self.root = "/home/apie/projects/AIC20-MTMC/dataset/AIC20_T3"
        self.train_dir = os.path.join(self.root, 'train')
        self.val_dir   = os.path.join(self.root, "validation")
        
        if os.path.exists(os.path.join(self.train_dir, "data.txt")):
            self.trainset = self.load_data_file(os.path.join(self.train_dir, "data.txt"))
        else:
            self.trainset = self.process_data(self.train_dir)
        if os.path.exists(os.path.join(self.val_dir, "data.txt")):
            self.valset = self.load_data_file(os.path.join(self.val_dir, "data.txt"))
        else:
            self.valset   = self.process_data(self.val_dir)

    def loadfeature(self, file_list, fps_dict):
        feature_dict = dict()
          
        ts_dict = self.get_timestamp_dict()
        for fn in file_list:
            print (f"Load {fn}...")
            with open(fn, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    line = lines[i]
                    split_line = line.strip("\n").split(",")
                    camid = fn.split("/")[-2]
                    ts_per_frame = 1/fps_dict[camid]
                    ts = float(split_line[1]) * ts_per_frame + ts_dict[camid]
                    gps = (float(split_line[3]), float(split_line[4]))
                    feature = [float(ft) for ft in split_line[5:]]
                    _id = int(split_line[2])
                    if _id not in feature_dict:
                        feature_dict[_id] = dict()
                        feature_dict[_id][camid] = [[np.array(feature), gps, ts]]
                    else:
                        if camid not in feature_dict[_id]:
                            feature_dict[_id][camid] = [[np.array(feature), gps, ts]]
                        else:
                            feature_dict[_id][camid].append([np.array(feature), gps, ts])
                            
        return feature_dict

    def get_timestamp_dict(self):
        ts_dict = dict()
        for filename in os.listdir(os.path.join(self.root, "cam_timestamp")):
            with open(os.path.join(self.root, "cam_timestamp", filename), "r") as f:
                lines = f.readlines()
                temp = dict()
                for line in lines:
                    split_line = line.strip("\n").split(" ")
                    temp[split_line[0]] = float(split_line[1])
                _max = np.array(list(temp.values())).max()
                for camid, ts in temp.items():
                    ts_dict[camid] = ts * -1 + _max

        return ts_dict

    def get_fps_dict(self, dirname):
        fps_dict = dict()
        with open(os.path.join(self.root, dirname, "fps_file.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                split_line = line.strip("\n").split(" ")
                fps_dict[split_line[0]] = float(split_line[1])
        return fps_dict

    def load_data_file(self, data_file):
        dataset = list()
        with open(data_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n").split(",")
                avg_ft1 = [float(ele) for ele in line[:2048]]
                avg_ft2 = [float(ele) for ele in line[2048:4096]]
                dis_gps1 = float(line[-6])
                dis_gps2 = float(line[-5])
                dis_gps3 = float(line[-4])
                dis_ts1 = float(line[-3])
                dis_ts2 = float(line[-2])
                label = int(line[-1])
                dataset.append([avg_ft1, avg_ft2, dis_gps1, dis_gps2, dis_gps3, dis_ts1, dis_ts2, label])
        
        return dataset

    def process_data(self, data_dir):

        scene_dirs, dataset = list(), list()
        fps_dict = self.get_fps_dict(data_dir)

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
            
            feature_dict = self.loadfeature(file_list, fps_dict)
            ids = list(feature_dict.keys())

            for i in ids:
                print (f"Producing id={i} Data")
                camids = list(feature_dict[i].keys())
                
                # Positive Data
                for j in range(len(camids)):
                    camj = camids[j]
                    track1 = np.array(feature_dict[i][camj])
                    avg_feature1 = np.average(track1[:, 0], axis=0)
                    gps1 = track1[:, 1]
                    ts1 = track1[:, 2]
                    num = 0
                    for k in range(j + 1, len(camids)):
                        camk = camids[k]
                        if j != k:
                            track2 = np.array(feature_dict[i][camk])
                            avg_feature2 = np.average(track2[:, 0], axis=0)
                            gps2 = track2[:, 1]
                            ts2 = track2[:, 2]
                            dis_gps_1 = (gps1[0][0] - gps2[0][0]) ** 2 + (gps1[0][1] - gps2[0][1]) ** 2
                            dis_gps_2 = (gps1[int(len(gps1)/2)][0] - gps2[int(len(gps2)/2)][0]) ** 2 + (gps1[int(len(gps1)/2)][1] - gps2[int(len(gps2)/2)][1]) ** 2
                            dis_gps_3 = (gps1[-1][0] - gps2[-1][0]) ** 2 + (gps1[-1][0] - gps2[-1][0]) ** 2
                            dis_ts_1 = abs(ts1[0] - ts2[0])
                            dis_ts_2 = abs(ts1[-1] - ts2[-1])
                            _input = [avg_feature1, avg_feature2, dis_gps_1, dis_gps_2, dis_gps_3, dis_ts_1, dis_ts_2, 1]
                            dataset.append(_input)
                            with open(os.path.join(data_dir, "data.txt"), "a+") as f:
                                avg_ft1 = [str(ft) for ft in avg_feature1]
                                avg_ft2 = [str(ft) for ft in avg_feature2]
                                f.write(",".join(avg_ft1) + "," + ",".join(avg_ft2) + "," + str(dis_gps_1) + "," + str(dis_gps_2) + "," + str(dis_gps_3) + "," + str(dis_ts_1) + "," + str(dis_ts_2) + ",1\n")
                            num += 1

                    num = num + int(num*0.5)
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
                        track2  = np.array(feature_dict[_id][neg_camid])
                        avg_feature2 = np.average(track2[:, 0], axis=0)
                        gps2 = track2[:, 1]
                        ts2 = track2[:, 2]
                        dis_gps_1 = (gps1[0][0] - gps2[0][0]) ** 2 + (gps1[0][1] - gps2[0][1]) ** 2
                        dis_gps_2 = (gps1[int(len(gps1)/2)][0] - gps2[int(len(gps2)/2)][0]) ** 2 + (gps1[int(len(gps1)/2)][1] - gps2[int(len(gps2)/2)][1]) ** 2
                        dis_gps_3 = (gps1[-1][0] - gps2[-1][0]) ** 2 + (gps1[-1][0] - gps2[-1][0]) ** 2
                        dis_ts_1 = abs(ts1[0] - ts2[0])
                        dis_ts_2 = abs(ts1[-1] - ts2[-1])
                        _input = [avg_feature1, avg_feature2, dis_gps_1, dis_gps_2, dis_gps_3, dis_ts_1, dis_ts_2, 0]
                        dataset.append(_input)
                        with open(os.path.join(data_dir, "data.txt"), "a+") as f:
                            avg_ft1 = [str(ft) for ft in avg_feature1]
                            avg_ft2 = [str(ft) for ft in avg_feature2]
                            f.write(",".join(avg_ft1) + "," + ",".join(avg_ft2) + "," + str(dis_gps_1) + "," + str(dis_gps_2) + "," + str(dis_gps_3) + "," + str(dis_ts_1) + "," + str(dis_ts_2) + ",0\n")
        return dataset



if __name__ == "__main__":
    dataset = mtmc()
    trainset = np.array(dataset.trainset)
    print (trainset[:, 2].max(), trainset[:, 2].min())
    print (trainset[:, 3].max(), trainset[:, 3].min())
    print (trainset[:, 4].max(), trainset[:, 4].min())
    print (trainset[:, 5].max(), trainset[:, 5].min())
    print (trainset[:, 6].max(), trainset[:, 6].min())
    """
    5.636946243413381e-05 0.00013657720170511732
    5.611853823862016e-05 0.0001372030191373347
    2.797372882596438e-05 0.0005720367961958972
    27.260769012485873 38.03783122595934
    25.960272417707124 38.19534731865891
    """
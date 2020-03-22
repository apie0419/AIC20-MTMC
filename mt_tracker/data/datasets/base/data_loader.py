from torch.utils.data import Dataset
import os

class FeatureDataset(Dataset):

    def __init__(self, dataset, cfg, _dir):
        self.dataset = dataset
        self.root    = os.path.join(cfg.PATH.ROOT_PATH, _dir)
        self.all_features = dict()
        self.load_features()

    def load_features(self):
        scene_dirs = list()
        for dirname in os.listdir(self.root):
            if dirname.startswith("S0"):
                scene_dirs.append(dirname)

        for scene_dir in scene_dirs:
            camera_dirs, file_list = list(), list()
            scene_path  = os.path.join(self.root, scene_dir)
            self.all_features[scene_dir] = dict()
            for dirname in os.listdir(scene_path):
                if dirname.startswith("c0"):
                    camera_dirs.append(dirname)
            for camera_dir in camera_dirs:
                print (f"Load Feature... Scene: {scene_dir}, Camera: {camera_dir}")
                self.all_features[scene_dir][camera_dir] = list()
                with open(os.path.join(scene_path, camera_dir, "tracker_train_file.txt"), "r") as f:
                    for line in f.readlines():
                        feature = [float(f) for f in line.strip("\n").split(",")[5:]]
                        self.all_features[scene_dir][camera_dir].append(feature)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        scene, cam1, f1, cam2, f2, target = self.dataset[index]
        feature1 = self.all_features[scene][cam1][f1]
        feature2 = self.all_features[scene][cam2][f2]

        return feature1 + feature2, target
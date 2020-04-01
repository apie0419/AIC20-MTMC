from torch.utils.data import Dataset
import os

class FeatureDataset(Dataset):

    def __init__(self, dataset, cfg):
        self.dataset = dataset
        self.gps_min = cfg.INPUT.GPS_MIN
        self.gps_max = cfg.INPUT.GPS_MAX
        self.ts_min = cfg.INPUT.TS_MIN
        self.ts_max = cfg.INPUT.TS_MAX

    def __len__(self):
        return len(self.dataset)

    def standarize(self, x, mean, std):
        return (x - mean) / std

    def normalize(self, x, _min, _max):
        return (x - _min) / (_max - _min)

    def __getitem__(self, index):
        avg_feature1, avg_feature2, dis_gps_1, dis_gps_2, dis_gps_3, dis_ts_1, dis_ts_2, target = self.dataset[index]
        dis_gps_1 = self.normalize(dis_gps_1, self.gps_min[0], self.gps_max[0])
        dis_gps_2 = self.normalize(dis_gps_2, self.gps_min[1], self.gps_max[1])
        dis_gps_3 = self.normalize(dis_gps_3, self.gps_min[2], self.gps_max[2])
        dis_ts_1 = self.normalize(dis_ts_1, self.ts_min[0], self.ts_max[0])
        dis_ts_2 = self.normalize(dis_ts_2, self.ts_min[1], self.ts_max[1])
        physic_feature = [dis_gps_1, dis_gps_2, dis_gps_3, dis_ts_1, dis_ts_2]
        
        return avg_feature1, avg_feature2, physic_feature, target
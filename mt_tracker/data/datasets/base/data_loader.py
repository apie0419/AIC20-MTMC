from torch.utils.data import Dataset
import os

class FeatureDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        avg_feature1, avg_feature2, dis_gps_1, dis_gps_2, dis_gps_3, dis_ts_1, dis_ts_2, target = self.dataset[index]
        

        return feature1 + feature2, target
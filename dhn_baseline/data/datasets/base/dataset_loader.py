import torch
from torch.utils.data import Dataset


class MatDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        dis_matrix, gt, size = self.dataset[index]
        dis_matrix = torch.FloatTensor(dis_matrix)
        
        gt = torch.LongTensor(gt)
        return dis_matrix, gt, size
import torch


def collate_fn(batch):
    features, targets = zip(*batch)
    targets = torch.LongTensor(targets)
    features = torch.FloatTensor(features)
    return features, targets
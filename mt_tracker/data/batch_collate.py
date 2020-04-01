import torch


def collate_fn(batch):
    f1, f2, physic_feature, targets = zip(*batch)
    targets = torch.LongTensor(targets)
    f1 = torch.FloatTensor(f1)
    f2 = torch.FloatTensor(f2)
    physic_feature = torch.FloatTensor(physic_feature)
    _input = torch.cat((f1, f2, physic_feature), 1)
    
    return _input, targets
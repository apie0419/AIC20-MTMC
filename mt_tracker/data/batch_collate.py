import torch


def collate_fn(batch):
    _input, targets = zip(*batch)
    targets = torch.LongTensor(targets)
    _input = torch.FloatTensor(_input)
    
    return _input, targets
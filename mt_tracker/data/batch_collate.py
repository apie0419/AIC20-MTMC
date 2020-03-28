import torch


def collate_fn(batch):
    f1, f2, dis_gps, dis_ts, targets = zip(*batch)
    targets = torch.LongTensor(targets)
    f1 = torch.FloatTensor(f1)
    f2 = torch.FloatTensor(f2)
    dis_gps = torch.FloatTensor(dis_gps)
    dis_ts = torch.FloatTensor(dis_ts)
    _input = torch.cat((f1, f2, dis_gps, dis_ts), 1)

    return _input, targets
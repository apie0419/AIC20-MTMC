import torch


def collate_fn(batch):
<<<<<<< HEAD
    _input, targets = zip(*batch)
    targets = torch.LongTensor(targets)
    _input = torch.FloatTensor(_input)
    
=======
    f1, f2, dis_gps, dis_ts, targets = zip(*batch)
    targets = torch.LongTensor(targets)
    f1 = torch.FloatTensor(f1)
    f2 = torch.FloatTensor(f2)
    dis_gps = torch.FloatTensor(dis_gps)
    dis_ts = torch.FloatTensor(dis_ts)
    _input = torch.cat((f1, f2, dis_gps, dis_ts), 1)

>>>>>>> parent of 2349a86... add expected time limit
    return _input, targets
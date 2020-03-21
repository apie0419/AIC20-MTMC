import torch.utils.data as Data
from torch.utils.data import DataLoader

def make_data_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAME)

    trainset = Data.TensorDataset(data_tensor=dataset.train_data, target_tensor=dataset.train_target)
    
    train_loader = DataLoader(trainset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers)
    
    # val_loader = DataLoader(valset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers)
    
    return train_loader
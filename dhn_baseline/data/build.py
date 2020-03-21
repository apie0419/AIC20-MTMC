import torch
from torch.utils.data import DataLoader
from .datasets        import init_dataset
from .datasets.base   import MatDataset
from .collate_batch   import *

def make_data_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    print(cfg.DATASETS.NAME)
    dataset = init_dataset(cfg.DATASETS.NAME)
    trainset = MatDataset(dataset.trainset)
    valset   = MatDataset(dataset.valset)
    train_loader = DataLoader(trainset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers)
    
    val_loader = DataLoader(valset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
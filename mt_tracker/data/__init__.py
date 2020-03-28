from torch.utils.data import DataLoader
from .datasets      import init_dataset
from .datasets.base import FeatureDataset
from .batch_collate import collate_fn

def make_train_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAME)
    trainset = FeatureDataset(dataset.trainset, cfg, "train")
    train_loader = DataLoader(trainset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader

def make_val_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAME)
    valset = FeatureDataset(dataset.valset, cfg, "validation")
    val_loader = DataLoader(valset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return val_loader
from torch.utils.data import DataLoader
from .datasets      import init_dataset
from .datasets.base import FeatureDataset
from .batch_collate import collate_fn

def make_data_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAME)
    trainset = FeatureDataset(dataset.trainset, cfg)
    valset = FeatureDataset(dataset.valset, cfg)
    train_loader = DataLoader(trainset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(valset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_loader, val_loader

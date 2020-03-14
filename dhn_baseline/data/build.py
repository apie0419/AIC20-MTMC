from torch.utils.data import DataLoader
from .dataset         import init_dataset


def make_data_loader(cfg):
    num_workers = cfg.DATALOADER.NUM_WORKERS
    print(cfg.DATASETS.NAMES)
    dataset = init_dataset(cfg.DATASETS.NAME)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_loader = DataLoader(dataset.trainset, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers)
    
    val_loader = DataLoader(dataset.valset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
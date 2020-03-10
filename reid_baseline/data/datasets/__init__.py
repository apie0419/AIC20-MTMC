from .aic20_t3 import aic20_t3
from .aic20_t2 import aic20_t2
from .veri     import veri
from .base     import ImageDataset

__factory = {
    "aic20_t3": aic20_t3,
    "aic20_t2": aic20_t2,
    "veri": veri
}

def get_names():
    return __factory.keys()

def merge_dataset(d1, d2):
    for i in range(len(d2.train)):
        d2.train[i][1] = d2.train[i][1] + d1.num_vids
        d2.train[i][2] = d2.train[i][2] + d1.num_cams
    for i in range(len(d2.query)):
        d2.query[i][1] = d2.query[i][1] + d1.num_vids
        d2.query[i][2] = d2.query[i][2] + d1.num_cams
    for i in range(len(d2.gallery)):
        d2.gallery[i][1] = d2.gallery[i][1] + d1.num_vids
        d2.gallery[i][2] = d2.gallery[i][2] + d1.num_cams
    d1.train.extend(d2.train)
    d1.query.extend(d2.query)
    d1.gallery.extend(d2.gallery)
    d1.num_imgs += d2.num_imgs
    d1.num_vids += d2.num_vids
    d1.num_cams += d2.num_cams
    return d1

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

from .aic20_t3 import aic20_t3

__factory = {
    "aic20_t3": aic20_t3
}

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
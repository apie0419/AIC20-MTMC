from .mtmc import mtmc


__factory = {
    "mtmc": mtmc
}

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
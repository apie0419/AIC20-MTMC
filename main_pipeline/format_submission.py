import os, sys

sys.path.append("..")

from config import cfg

output_path = os.path.join(cfg.PATH.ROOT_PATH, "AIC20_res")
dirname = cfg.PATH.INPUT_PATH.split("/")[-1]

if not os.path.exists(output_path):
    os.mkdir(output_path)
res = dict()

with open(os.path.join(cfg.PATH.ROOT_PATH, "track3.txt"), "r") as f:
    lines = f.readlines()
    for line in lines:
        data_list = line.split(" ")
        camera = data_list[0]
        data = [data_list[2], data_list[1]]
        data.extend(data_list[3:])
        if camera not in res:
            res[camera] = [data]
        else:
            res[camera].append(data)
        
            
for camera, data in res.items():
    print (camera)
    # print (data)
    with open(os.path.join(output_path, "c" + camera.zfill(3) + f"_{dirname}.txt"), "w") as f:
        data.sort(key=lambda x:int(x[0]))
        for d in data:
            f.write(",".join(d))


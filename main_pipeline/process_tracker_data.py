import os, sys, torch, cv2
import numpy as np
from PIL import Image

sys.path.append("..")

from model  import build_reid_model, build_transforms
from config import cfg

INPUT_DIR     = cfg.PATH.ROOT_PATH
DEVICE        = cfg.DEVICE.TYPE
if DEVICE == "cuda":
    torch.cuda.set_device(cfg.DEVICE.GPU)

train_dir = os.path.join(INPUT_DIR, "train")
val_dir   = os.path.join(INPUT_DIR, "validation")

def process_data(data_dir):
    scene_dirs = list()
    model = build_reid_model(cfg)
    model = model.to(DEVICE)
    model.eval()
    transform = build_transforms(cfg)
    for dirname in os.listdir(data_dir):
        if dirname.startswith("S0"):
            scene_dirs.append(dirname)
    for scene_dir in scene_dirs:
        camera_dirs = list()
        scene_path  = os.path.join(data_dir, scene_dir)
        for dirname in os.listdir(scene_path):
            if dirname.startswith("c0"):
                camera_dirs.append(dirname)
        for camera_dir in camera_dirs:
            print (scene_path + "/" + camera_dir)
            img_path    = os.path.join(scene_path, camera_dir, "gt_images")
            gt_gps_file = os.path.join(scene_path, camera_dir, "gt_gps_file.txt")
            output_file = os.path.join(scene_path, camera_dir, "tracker_train_file.txt")

            if os.path.exists(output_file):
                os.remove(output_file)
            
            gt_gps_lines = list()
            with open(gt_gps_file, "r") as f:
                for line in f.readlines():
                    
                    gt_gps_lines.append(line.strip("\n").split(","))

            for line in gt_gps_lines:
                img_name = line[0]
                _id = line[2]
                file_path = os.path.join(img_path, _id, img_name)
                img = Image.open(file_path).convert('RGB')
                with open(output_file, 'a+') as f:
                    with torch.no_grad():
                        img = transform(img)
                        img = torch.Tensor(img).view(1, 3, 256, 256).cuda()
                        feature = list(model(img)[0].cpu().numpy())
                        feature = [str(f) for f in feature]
                        
                        f.write(",".join(line) + "," + ",".join(feature) + "\n")
                        
if __name__ == '__main__': 
    process_data(train_dir)
    process_data(val_dir)
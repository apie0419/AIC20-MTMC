import os, sys, torch, cv2
import numpy as np
from PIL import Image

sys.path.append("..")

from reid_model import build_model, build_transforms
from config     import cfg


INPUT_DIR = cfg.PATH.ROOT_PATH
DEVICE    = cfg.DEVICE.TYPE
if DEVICE == "cuda":
    torch.cuda.set_device(cfg.DEVICE.GPU)

train_dir = os.path.join(INPUT_DIR, "train")
val_dir   = os.path.join(INPUT_DIR, "validation")

class Box(object):
    def __init__(self, id, box):
        self.id = id
        self.box = box
        self.center = (self.box[0] + self.box[2]/2, self.box[1] + self.box[3]/2)

def analysis_to_frame_dict(file_path):
    frame_dict = dict()
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        cur_box = Box(id, box)
        if index not in frame_dict:
            frame_dict[index] = list()
        frame_dict[index].append(cur_box)
    return frame_dict

def analysis_transfrom_mat(cali_path):
    first_line = open(cali_path).readlines()[0].strip('\r\n')
    cols = first_line.lstrip('Homography matrix: ').split(';')
    transfrom_mat = np.ones((3, 3))
    for i in range(3):
        values_string = cols[i].split()
        for j in range(3):
            value = float(values_string[j])
            transfrom_mat[i][j] = value
    inv_transfrom_mat = np.linalg.inv(transfrom_mat)
    return inv_transfrom_mat

def process_data(data_dir):
    scene_dirs = list()
    model = build_model()
    model = model.to(DEVICE)
    model.eval()
    transform = build_transforms()
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
            gt_file = os.path.join(scene_path, camera_dir, "gt/gt.txt")
            video_path = os.path.join(scene_path, camera_dir, "vdo.avi")
            cali_file = os.path.join(scene_path, camera_dir, "calibration.txt")
            trans_mat = analysis_transfrom_mat(cali_file)
            output_file = os.path.join(scene_path, camera_dir, "dhn_gt_file.txt")

            cap = cv2.VideoCapture(video_path)
            frame_num = int(cap.get(7))
            frame_dict = analysis_to_frame_dict(gt_file)
            if os.path.exists(output_file):
                os.remove(output_file)
            with open(output_file, 'a+') as f:
                for i in range(frame_num):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    tmp_i = i + 1
                    if tmp_i in frame_dict:
                        src_boxes = frame_dict[tmp_i]
                        for det_box in src_boxes:
                            box = det_box.box
                            coor = det_box.center
                            image_coor = [coor[0], coor[1], 1]
                            GPS_coor = np.dot(trans_mat, image_coor)
                            GPS_coor = GPS_coor / GPS_coor[2]
                            GPS_coor = GPS_coor.tolist()
                            cropped_img = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                            with torch.no_grad():
                                img = transform(Image.fromarray(cropped_img))

                                img = torch.Tensor(img).view(1, 3, 256, 256).cuda()
                                feature = list(model(img)[0].cpu().numpy())
                                feature = [str(f) for f in feature]
                            f.write(str(tmp_i) + "," + str(det_box.id) + "," + ",".join(feature) + "," + str(GPS_coor[0]) + "," + str(GPS_coor[1]) + "\n")
                            
if __name__ == '__main__': 
    process_data(train_dir)
    process_data(val_dir)
    
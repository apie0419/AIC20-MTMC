import os

INPUT_DIR = "/home/apie/projects/AIC20-MTMC/dataset/AIC20_T3"
train_dir = os.path.join(INPUT_DIR, "train")
val_dir   = os.path.join(INPUT_DIR, "validation")

class Box(object):
    def __init__(id, box):
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

def process_train():
    scene_dirs = list()
    for dirname in os.listdir(train_dir):
        if dirname.startswith("S0"):
            scene_dirs.append(dirname)
    for scene_dir in scene_dirs:
        camera_dirs = list()
        scene_path  = os.path.join(train_dir, scene_dir)
        for dirname in os.listdir(scene_path):
            if dirname.startswith("c0"):
                camera_dirs.append(dirname)
        for camera_dir in camera_dirs:
            gt_file = os.path.join(scene_path, camera_dir, "gt/gt.txt")
            frame_dict = analysis_to_frame_dict(gt_file)

def process_val():
    pass

if __name__ == '__main__':
    scene_dirs = list()
    
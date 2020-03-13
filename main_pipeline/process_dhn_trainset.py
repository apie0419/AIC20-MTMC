import os
from config as cfg

def analysis_to_frame_dict(file_path):
    frame_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        score = float(words[6])
        center = (self.box[0] + self.box[2]/2, self.box[1] + self.box[3]/2)
        cur_gt_box = GtBox(id, box, score)
        if index not in frame_dict:
            frame_dict[index] = []
        frame_dict[index].append(cur_gt_box)
    return frame_dict

if __name__ == "__main__":
    cfg.PATH.INPUT_PATH
    analysis_to_frame_dict()
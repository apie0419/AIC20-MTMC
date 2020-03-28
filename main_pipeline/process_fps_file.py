import cv2, os, sys

sys.path.append("..")

from config import cfg

BASE_PATH = cfg.PATH.ROOT_PATH
TIMESTAMP_PATH = os.path.join(BASE_PATH, "cam_timestamp")

def main(dirname):
    scene_paths = os.listdir(os.path.join(BASE_PATH, dirname))
    
    fps_dict = dict()
    for s in scene_paths:
        if not s.startswith("S0"):
            continue
        video_paths = os.listdir(os.path.join(BASE_PATH, dirname, s))
        for p in video_paths:
            if not p.startswith("c0"):
                continue
            videodir = os.path.join(BASE_PATH, dirname, s, p)
            print ("Processing " + videodir + "...")
            cap = cv2.VideoCapture(os.path.join(videodir, "vdo.avi"))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps_dict[p] = float(fps)
    
    with open(os.path.join(BASE_PATH, dirname, "fps_file.txt"), "w") as f:
        for p, fps in fps_dict.items():
            f.write(p + " " + str(fps) + "\n")

if __name__ == "__main__":
    for d in ["train", "validation", "test"]:
        main(d)
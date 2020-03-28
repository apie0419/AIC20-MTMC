import cv2, os, random, sys

sys.path.append("..")

from config import cfg

BASE_PATH = cfg.PATH.ROOT_PATH
RESULT_PATH = os.path.join(BASE_PATH, "AIC20_res")
dirname = "train"


frame_dict = dict()
colors = dict()

def read_result_file(filename):
    res = dict()
    with open(os.path.join(RESULT_PATH, filename), "r") as f:
        for line in f.readlines():
            data = line.split(",")
            data = list(map(int, data))
            frameid = data[0]
            if frameid not in res:
                res[frameid] = [data[1:6]]
            else:
                res[frameid].append(data[1:6])
    return res

def main():
    scene_paths = os.listdir(os.path.join(BASE_PATH, dirname))
    
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
            resolution = (int(cap.get(4)), int(cap.get(3)))
            
            fps = cap.get(cv2.CAP_PROP_FPS)

            framelist = list()
            result_dict = read_result_file(p + "_train.txt")
            framenum = int(cap.get(7))
            filename = os.path.join(videodir, "output.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, fps, resolution)

            for i in range(framenum):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                results = result_dict.get(i+1)
                if results is None:
                    continue
                for r in results:
                    id, x, y, w, h = r
                    color = colors.get(id)
                    if color is None:
                        color = (random.randint(1, 200), random.randint(1, 200), random.randint(1, 200))
                        colors[id] = color
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    cv2.putText(frame, str(id), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 4, color, 3)
                width, height, _ = frame.shape
                
                frame = cv2.resize(frame, (int(width), int(height)))
                out.write(frame)
            out.release()
            cap.release()

if __name__ == "__main__":
    main()

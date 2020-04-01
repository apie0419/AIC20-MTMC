import cv2, os, random, sys

sys.path.append("..")

from config import cfg

BASE_PATH = cfg.PATH.ROOT_PATH
INPUT_PATH = cfg.PATH.INPUT_PATH
RESULT_FILE = os.path.join(BASE_PATH, "track3.txt")

frame_dict = dict()
colors = dict()

def read_result_file():
    res = dict()
    with open(RESULT_FILE, "r") as f:
        for line in f.readlines():
            data = line.split(" ")
            data = list(map(int, data))
            camid = data[0]
            frameid = data[2]
            if camid not in res:
                res[camid] = dict()
                res[camid][frameid] = [[data[1]] + data[3:7]]
            else:
                if frameid not in res[camid]:
                    res[camid][frameid] = [[data[1]] + data[3:7]]
                else:
                    res[camid][frameid].append([data[1]] + data[3:7])
    return res

def main():
    scene_paths = os.listdir(INPUT_PATH)
    result_dict = read_result_file()
    for s in scene_paths:
        if not s.startswith("S0"):
            continue
        video_paths = os.listdir(os.path.join(INPUT_PATH, s))
        for p in video_paths:
            if not p.startswith("c0"):
                continue
            videodir = os.path.join(INPUT_PATH, s, p)
            
            print ("Processing " + videodir + "...")
            cap = cv2.VideoCapture(os.path.join(videodir, "vdo.avi"))
            resolution = (int(cap.get(4)), int(cap.get(3)))
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            camid = int(p[1:])
            framelist = list()
            framenum = int(cap.get(7))
            filename = os.path.join(videodir, "output.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, fps, resolution)

            for i in range(framenum):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break
                results = result_dict[camid].get(i+1)
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

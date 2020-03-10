import numpy as np
import cv2, os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAINSET_PATH = os.path.join(BASE_PATH, "../dataset/AIC20_T3/train")

IMAGE_COUNT = 1
TH_SCORE = 0.5
PAD_SIZE = 10
IOU_TH = 0.7
W_PAD = 0
H_PAD = 0


def crop(frame, boxes, ids, img_path):
    global IMAGE_COUNT
    idx = 0
    for box in boxes:
        cropped_img = frame[box[0]:box[2], box[1]:box[3]]
        output_path = os.path.join(img_path, str(ids[idx]))
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_path, str(IMAGE_COUNT).zfill(10) + ".jpg"), cropped_img)
        print ("Cropped Image: " + os.path.join(output_path, str(IMAGE_COUNT).zfill(10) + ".jpg"))
        idx += 1
    
    IMAGE_COUNT += 1

def compute_iou(box1, box2):

    rec1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    rec2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    sum_area = S_rec1 + S_rec2

    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    if left_line >= right_line or top_line >= bottom_line:
        return 0.0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return float(intersect) / (sum_area - intersect)

def preprocess_boxes(src_boxes, src_ids):
    boxes, ids = list(), list()
    
    for idx in range(len(src_boxes)):
        src_box = src_boxes[idx]
        src_id  = src_ids[idx]
        intersection = False
        for box in boxes:
            iou = compute_iou(src_box, box)
            if iou > IOU_TH:
                intersection = True
        if not intersection:
            boxes.append(src_box)
            ids.append(src_id)
    return boxes, ids

def main():
    
    for scene_dir in os.listdir(TRAINSET_PATH):
        if not scene_dir.startswith("S0"):
            continue
        for camera_dir in os.listdir(os.path.join(TRAINSET_PATH, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
            video_path  = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "vdo.avi")
            gt_path     = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "gt/gt.txt")
            image_path  = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "reid_images")
            output_path = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "det_gps_feature.txt")
            
            cap = cv2.VideoCapture(video_path)
            gt = np.loadtxt(gt_path, delimiter=",")

            frame_count = 0
            i = 0
            (fnum, id, left, top, width, height) = gt[i][:6]
            
            while cap.isOpened():
                try :
                    ret, frame = cap.read()
                    frame_count += 1
                    boxes = list()
                    ids = list()
                    if not ret:
                        break
                    if frame_count != fnum:
                        continue
                    
                except:
                    break
                
                while fnum == frame_count:
                    boxes.append((int(top), int(left), int(top+height), int(left+width)))
                    
                    ids.append(int(id))
                    i += 1
                    if i >= len(gt):
                        break
                    (fnum, id, left, top, width, height) = gt[i][:6]
                
                boxes, ids = preprocess_boxes(boxes, ids)
                crop(frame, boxes, ids, image_path)

            cap.release()

    print ("Finish")

if __name__ == "__main__":
    main()
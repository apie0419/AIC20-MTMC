import numpy as np
import cv2, os, sys

sys.path.append("..") 

from config import cfg


IMAGE_COUNT = 1
TH_SCORE = 0.5
PAD_SIZE = 10
IOU_TH = 0.7
W_PAD = 0
H_PAD = 0

def crop(fnum, frame, boxes, ids, img_path, trans_mat):
    global IMAGE_COUNT
    idx = 0
    data = list()
    for box in boxes:
        coor = (box[0] + box[2]/2, box[1] + box[3]/2)
        image_coor = [coor[0], coor[1], 1]
        GPS_coor = np.dot(trans_mat, image_coor)
        GPS_coor = GPS_coor / GPS_coor[2]
        GPS_coor = GPS_coor.tolist()
        cropped_img = frame[box[0]:box[2], box[1]:box[3]]
        output_path = os.path.join(img_path, str(ids[idx]))
        img_name = str(IMAGE_COUNT).zfill(10) + ".jpg"

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_path, str(IMAGE_COUNT).zfill(10) + ".jpg"), cropped_img)
        print ("Cropped Image: " + os.path.join(output_path, img_name))
        data.append(img_name + "," + str(fnum) + "," + str(ids[idx]) + "," + str(GPS_coor[0]) + "," + str(GPS_coor[1]))
        idx += 1

    IMAGE_COUNT += 1

    return data

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

def main(TRAINSET_PATH):
    
    for scene_dir in os.listdir(TRAINSET_PATH):
        if not scene_dir.startswith("S0"):
            continue
        for camera_dir in os.listdir(os.path.join(TRAINSET_PATH, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
            video_path  = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "vdo.avi")
            gt_path     = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "gt/gt.txt")
            image_path  = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "gt_images")
            cali_file   = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "calibration.txt")
            output_file = os.path.join(TRAINSET_PATH, scene_dir, camera_dir, "gt_gps_file.txt")
            
            trans_mat = analysis_transfrom_mat(cali_file)

            cap = cv2.VideoCapture(video_path)
            gt = np.loadtxt(gt_path, delimiter=",")

            frame_count = 0
            i = 0
            (fnum, id, left, top, width, height) = gt[i][:6]
            
            if os.path.exists(output_file):
                os.remove(output_file)

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
                data = crop(fnum, frame, boxes, ids, image_path, trans_mat)
                with open(output_file, "a+") as f:
                    for d in data:
                        f.write(d + "\n")


            cap.release()

    print ("Finish")

if __name__ == "__main__":
    for path in ["train", "validation"]:

        main(os.path.join(cfg.PATH.ROOT_PATH, path))
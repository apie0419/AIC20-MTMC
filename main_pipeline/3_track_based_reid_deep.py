
import numpy as np
import os, math, operator, sys, torch, cv2
from scipy.spatial import distance

sys.path.append("..")

from config import cfg
from model import build_mtmc_model

MATCHED = True
NO_MATCHED = False
input_dir = cfg.PATH.INPUT_PATH
root_dir = cfg.PATH.ROOT_PATH

INNER_SIMILAR_TH = 10
MOVE_TH = 4  # todo 待修订
ALL_TIME = 210
TIME_TH = 4.5
SCALE = 10.0


class Box(object):
    """
    match_state:一个box是否匹配到一个track中，若没有，应该生成新的track
    """
    def __init__(self, camera, frame_index, id, box, score, gps_coor, orientation, time, feature):
        self.camera = camera
        self.frame_index = frame_index
        self.id = id
        self.box = box
        self.score = score
        self.gps_coor = gps_coor
        self.orientation = orientation
        self.time = time
        self.feature = feature
        self.center = (self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2)
        self.match_state = NO_MATCHED

    def get_area(self):
        return self.box[2]*self.box[3]


# 只能用于单视频内的一条track
class Track(object):

    def __init__(self, id, sequence, cams):
        self.id = id
        self.sequence = sequence
        self.match_state = MATCHED
        self.cams = cams
        self.gps_move_vec = np.zeros(2)

    def get_average_feature(self):
        dim = self.sequence[0].feature.size
        ft = np.zeros(dim)
        l = len(self.sequence)
        for bx in self.sequence:
            ft = ft + bx.feature
        ft = ft*(1.0/l)
        return ft

    def get_orientation(self):
        return self.sequence[0].orientation

    def get_camera(self):
        return self.cams

    def append(self, box):
        self.sequence.append(box)

    def get_last(self):
        return self.sequence[-1]

    def get_first(self):
        return self.sequence[0]

    def get_last_feature(self):
        return self.sequence[-1].feature

    def get_first_feature(self):
        return self.sequence[0].feature

    def get_length(self):
        return len(self.sequence)

    # 跟踪轨迹结束的位置，用于判定结束位置的合理性
    def get_last_gps(self):
        return self.sequence[-1].gps_coor

    def get_first_gps(self):
        return self.sequence[0].gps_coor

    # 移动距离，用于判断一个track是否根本没移动
    def get_moving_distance(self):
        start_p = self.sequence[0].center
        end_p = self.sequence[-1].center
        move_vec = ((end_p[0] - start_p[0]), (end_p[1] - start_p[1]))
        move_dis = move_vec[0]**2 + move_vec[1]**2
        # print gps_dis
        return move_dis

    # 整体移动向量，用于判断整体移动方向
    def get_moving_vector(self):
        start_p = self.sequence[0].gps_coor
        end_p = self.sequence[-1].gps_coor
        move_vec = np.zeros(2)
        move_vec[0] = end_p[0] - start_p[0]
        move_vec[1] = end_p[1] - start_p[1]
        # print gps_dis_vec
        return move_vec

    def get_moving_gps_distance(self):
        move_vec = self.get_moving_vector()*100000
        move_gps_dis = move_vec[0] ** 2 + move_vec[1] ** 2
        return math.sqrt(move_gps_dis)

    def get_moving_time(self):
        start_t = self.sequence[0].time
        end_t = self.sequence[-1].time
        return end_t - start_t

    def show(self):
        print("For track-" + str(self.id) + ' : ', "length-" + str(len(self.sequence)))
        print(self.get_moving_distance())

    # 获取track的时间段
    def get_time_slot(self):
        # if self.sequence[-1].time < self.sequence[0].time:
        #     print [self.sequence[0].time, self.sequence[-1].time]

        return [self.sequence[0].time, self.sequence[-1].time]

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

def getdistance(pt1, pt2):
    EARTH_RADIUS = 6378.137
    lat1, lon1 = pt1[0], pt1[1]
    lat2, lon2 = pt2[0], pt2[1]
    radlat1 = lat1 * math.pi / 180
    radlat2 = lat2 * math.pi / 180
    lat_dis = radlat1 - radlat2
    lon_dis = (lon1 * math.pi - lon2 * math.pi) / 180
    distance = 2 * math.asin(math.sqrt((math.sin(lat_dis/2) ** 2) + math.cos(radlat1) * math.cos(radlat2) * (math.sin(lon_dis/2) ** 2)))
    distance *= EARTH_RADIUS
    distance = round(distance * 10000) / 10000
    return distance

def get_timestamp_dict():
    ts_dict = dict()
    for filename in os.listdir(os.path.join(root_dir, "cam_timestamp")):
        with open(os.path.join(root_dir, "cam_timestamp", filename), "r") as f:
            lines = f.readlines()
            temp = dict()
            for line in lines:
                split_line = line.strip("\n").split(" ")
                temp[split_line[0]] = float(split_line[1])
            _max = np.array(list(temp.values())).max()
            for camid, ts in temp.items():
                ts_dict[camid] = ts * -1 + _max

    return ts_dict

def get_fps_dict():
    fps_dict = dict()
    with open(os.path.join(input_dir, "fps_file.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.strip("\n").split(" ")
            fps_dict[split_line[0]] = float(split_line[1])
    return fps_dict

def analysis_to_track_dict(file_path):
    camera = file_path.split('/')[-2]
    track_dict = {}
    lines = open(file_path, 'r').readlines()
    for line in lines:
        words = line.strip('\n').split(',')
        index = int(words[0])
        id = int(words[1])
        box = [int(float(words[2])), int(float(words[3])), int(float(words[4])), int(float(words[5]))]
        score = float(words[6])
        gps = words[7].split('-')
        # print gps
        gps_x = float(gps[0])
        gps_y = float(gps[2])
        orientation = words[8]
        time = float(words[9])
        ft = np.zeros(len(words) - 10)
        for i in range(10, len(words)):
            ft[i - 10] = float(words[i])
        cur_box = Box(camera, index, id, box, score, (gps_x, gps_y), orientation, time, ft)
        if id not in track_dict:
            track_dict[id] = Track(id, [], camera)
        track_dict[id].append(cur_box)
        track_dict[id].gps_move_vec = track_dict[id].get_moving_vector() 
        cmpfun = operator.attrgetter('time') 
        track_dict[id].sequence.sort(key=cmpfun)
    return track_dict

def normalize(x, _min, _max):
    return (x - _min) / (_max - _min)

def match_track(model, q_tracks, g_tracks):
    ranks = list()
    ts_dict = get_timestamp_dict()
    fps_dict = get_fps_dict()
    for qt in q_tracks:
        qt_seq = qt.sequence
        qt_ts = ts_dict[qt.cams]
        qt_ts_per_frame = 1/fps_dict[qt.cams]
        dis = getdistance(qt_seq[0].gps_coor, qt_seq[-1].gps_coor)
        ts = abs(qt_seq[0].frame_index - qt_seq[-1].frame_index) * qt_ts_per_frame
        if dis == 0 or ts == 0:
            continue
        speed = dis / ts
        filter_num = 0
        match = 0
        for gt in g_tracks:
            
            gt_seq = gt.sequence
            gt_ts = ts_dict[gt.cams]
            gt_ts_per_frame = 1/fps_dict[gt.cams]
            
            with torch.no_grad():
                model.eval()
                model.to(cfg.DEVICE.TYPE)
                
                m = torch.nn.Softmax(dim=1)
                gps_min, gps_max, ts_min, ts_max = cfg.MTMC.GPS_MIN, cfg.MTMC.GPS_MAX, cfg.MTMC.TS_MIN, cfg.MTMC.TS_MAX
                
                vec1 = [gt_seq[int(len(gt_seq)/2)].gps_coor[0] - gt_seq[0].gps_coor[0], gt_seq[int(len(gt_seq)/2)].gps_coor[1] - gt_seq[0].gps_coor[1]]
                vec2 = [gt_seq[-1].gps_coor[0] - gt_seq[0].gps_coor[0], gt_seq[-1].gps_coor[1] - gt_seq[0].gps_coor[1]]
                
                dis_gps_1 = (gt_seq[0].gps_coor[0] - qt_seq[0].gps_coor[0]) ** 2 + (gt_seq[0].gps_coor[1] - qt_seq[0].gps_coor[1]) ** 2
                dis_gps_2 = (gt_seq[int(len(gt_seq)/2)].gps_coor[0] - qt_seq[int(len(qt_seq)/2)].gps_coor[0]) ** 2 + (gt_seq[int(len(gt_seq)/2)].gps_coor[1] - qt_seq[int(len(qt_seq)/2)].gps_coor[1]) ** 2
                dis_gps_3 = (gt_seq[-1].gps_coor[0] - qt_seq[-1].gps_coor[0]) ** 2 + (gt_seq[-1].gps_coor[1] - qt_seq[-1].gps_coor[1]) ** 2
                dis_ts_1 = abs((gt_seq[0].frame_index * gt_ts_per_frame + gt_ts) - (qt_seq[0].frame_index * qt_ts_per_frame + qt_ts))
                dis_ts_2 = abs((gt_seq[-1].frame_index * gt_ts_per_frame + gt_ts) - (qt_seq[-1].frame_index * qt_ts_per_frame + qt_ts))
                
                dis_gps_1 = normalize(dis_gps_1, gps_min[0], gps_max[0])
                dis_gps_2 = normalize(dis_gps_2, gps_min[1], gps_max[1])
                dis_gps_3 = normalize(dis_gps_3, gps_min[2], gps_max[2])
                norm_dis_ts_1 = normalize(dis_ts_1, ts_min[0], ts_max[0])
                norm_dis_ts_2 = normalize(dis_ts_2, ts_min[1], ts_max[1])
                _input = qt.average_feature.tolist() + gt.average_feature.tolist()
                _input += [dis_gps_1, dis_gps_2, dis_gps_3, norm_dis_ts_1, norm_dis_ts_2]
                feature = torch.FloatTensor(_input).view(1, cfg.MTMC.APPEARANCE_DIM + cfg.MTMC.PHYSIC_DIM).cuda()
                prob = m(model(feature))[0][1]
                
                expected_time = getdistance(qt_seq[0].gps_coor, gt_seq[0].gps_coor) / speed
                
                if abs(dis_ts_1 - expected_time) > 90 + 10 * abs(int(qt.cams[1:]) - int(gt.cams[1:])):
                    filter_num += 1
                    continue
                if prob > 0.5:
                    match += 1
                    ranks.append([qt, gt.id, prob]) 

        print (f"Track Matched: {match}/{len(g_tracks) - filter_num}")
    ranks = sorted(ranks, key=lambda x: x[2], reverse=True)
    already_matched = list()
    for rank in ranks:
        qt, match_id = rank[0], rank[1]
        if match_id not in already_matched and qt.id not in already_matched:
            already_matched.append(match_id)
            qt.id = match_id

    print (f"Final Matched: {len(already_matched)}/{len(q_tracks)}")
    
def main():

    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    result_file = os.path.join(cfg.PATH.ROOT_PATH, 'submission')
    if os.path.exists(result_file):
        os.remove(result_file)
    for scene_fd in scene_fds:
        if scene_fd.startswith("S0"):
            scene_dirs.append(os.path.join(input_dir, scene_fd))
    for scene_dir in scene_dirs:
        
        camera_dirs, all_track = list(), dict()
        fds = os.listdir(scene_dir)
        for fd in fds:
            if fd.startswith('c0'):
                camera_dirs.append(os.path.join(scene_dir, fd))
        for camera_dir in camera_dirs:
            print(camera_dir)
            camid = int(camera_dir.split("/")[-1][2:])
            track_file_path = os.path.join(camera_dir, 'optimized_track_no_overlapped.txt')
            tk_dict = analysis_to_track_dict(track_file_path)
            track_list = list()
            for k in tk_dict:
                track_list.append(tk_dict[k])
            all_track[camid] = track_list

        print("calu average feature.")
        for c in all_track:
            l = len(all_track[c])
            for i in range(l):
                tk = all_track[c][i]
                tk.average_feature = tk.get_average_feature()

        model = build_mtmc_model(cfg)
        
        cams = sorted(list(all_track.keys()))
        for i in range(1, len(cams)):
            q_camid = cams[i]
            q_tracks = all_track[q_camid]
            g_tracks, exists = list(), list()
            print (f"Query Track: {q_camid}")
            print (f"Gallery Tracks: {cams[:i]}")
            for g_camid in cams[:i]:
                g_tracks += all_track[g_camid]

            match_track(model, q_tracks, g_tracks)
        
        with open(result_file, 'a+') as f:
            for camid in all_track:
                for track in all_track[camid]:
                    cam = track.get_camera()
                    _id = track.id
                    for bx in track.sequence:
                        frame_id = bx.frame_index
                        ww = cam + "," + str(_id) + "," + str(frame_id) + "," + str(bx.box[0]) + \
                        "," + str(bx.box[1]) + "," + str(bx.box[2]) + "," + str(bx.box[3]) + ",-1,-1\n"
                        f.write(ww)


if __name__ == '__main__':
    main()



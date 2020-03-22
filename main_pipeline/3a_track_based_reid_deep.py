
import numpy as np
import os, math, operator, sys, torch

sys.path.append("..")

from config import cfg
from model import build_mtmc_model

MATCHED = True
NO_MATCHED = False
input_dir = cfg.PATH.INPUT_PATH

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

    def get_feature_list(self):
        ft_list = []
        for bx in self.sequence:
            cur_ft = bx.feature
            similar = False
            for ft in ft_list:
                dis = calu_feature_distance(ft, cur_ft)
                if dis < INNER_SIMILAR_TH:
                    similar = True
                    break
            if not similar:
                ft_list.append(cur_ft)
        return ft_list

    # 获取track的时间段
    def get_time_slot(self):
        # if self.sequence[-1].time < self.sequence[0].time:
        #     print [self.sequence[0].time, self.sequence[-1].time]

        return [self.sequence[0].time, self.sequence[-1].time]


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
        track_dict[id].gps_move_vec = track_dict[id].get_moving_vector()  # 第一次初始化的是box加入的方式，需要手动设置移动方向，后面每次update的时候会更新移动方向到最新的
        cmpfun = operator.attrgetter('time')  # 参数为排序依据的属性，可以有多个，这里优先id，使用时按需求改换参数即可
        track_dict[id].sequence.sort(key=cmpfun)
    return track_dict

def match_track(model, q_tracks, g_tracks):
    ranks = list()
    for qt in q_tracks:
        rank = list()
        for gt in g_tracks:
            with torch.no_grad():
                model.eval()
                m = torch.nn.Softmax(dim=1)
                feature = torch.FloatTensor(qt.average_feature.tolist()[:-1] + gt.average_feature.tolist()[:-1]).view(1, cfg.MTMC.HIDDEN_DIM)
                prob = m(model(feature))[0][1]
                rank.append([gt.id, prob])
                
        rank = sorted(rank, key=lambda x: x[1], reverse=True)
        ranks.append([qt, rank])

    already_matched = list()
    num_tracks = len(g_tracks)
    for i in range(num_tracks):
        match = dict()
        for j in range(len(ranks)):
            rank = ranks[j]
            if rank == None:
                continue
            match_id = rank[1][i][0]
            if match_id in already_matched:
                continue
            prob = rank[1][i][1]
            if prob < 0.5:
                continue
            if match_id not in match:
                match[match_id] = [[j, prob]]
            else:
                match[match_id].append([j, prob])

        for match_id in match:
            
            match_rank = sorted(match[match_id], key=lambda x: x[1], reverse=True)
            idx = match_rank[0][0]
            qt = ranks[idx][0]
            qt.id = match_id
            already_matched.append(match_id)
            ranks[idx] = None


def main():

    scene_dirs = []
    scene_fds = os.listdir(input_dir)
    result_file = os.path.join(cfg.PATH.ROOT_PATH, 'submission')
    if os.path.exists(result_file):
        os.remove(result_file
        )
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
            for g_camid in cams[:i]:
                match_track(model, all_track[q_camid], all_track[g_camid])
        
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



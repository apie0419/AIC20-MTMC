# -*- coding: utf-8 -*-
from . import Point
import numpy as np

OUTLIER_DIS = 0.007
SLICE_THRESHOLD = 10
DIRECTION_THRESHOLD = 10
COS_DIS_SAME_DIRECTION = 0.93
COS_DIS_OPP_DIRECTION = -0.5 
MIN_DIS_THRESHOLD = 10 
SPEED_THRESHOLD = 80

def cos_vector(vec_1, vec_2):
    '''
    numerator = vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]
    denominator = np.sqrt(np.square(vec_1[0]) + np.square(vec_1[1])) *  np.sqrt(np.square(vec_2[0]) + np.square(vec_2[1]))
    if denominator == 0:
        print(vec_2)
    return numerator/denominator
    '''
    if np.linalg.norm(np.array(vec_1)) * np.linalg.norm(np.array(vec_2)) == 0:  # 分母判断
        return -1
    return np.dot(np.array(vec_1), np.array(vec_2)) / (
                np.linalg.norm(np.array(vec_1)) * (np.linalg.norm(np.array(vec_2))))


def min_distance(middle_point_list_1, middle_point_list_2):
    len_1 = len(middle_point_list_1)
    len_2 = len(middle_point_list_2)
    if len_1 <= 1 or len_2 <= 1:
        return 0.
    dis_list = []
    for i in range(0, len_1):
        for j in range(0, len_2):
            x1 = middle_point_list_1[i][0] * 1e5
            y1 = middle_point_list_1[i][1] * 1e5
            x2 = middle_point_list_2[j][0] * 1e5
            y2 = middle_point_list_2[j][1] * 1e5
            # dis_list.append(np.sqrt(np.square(x1-x2)+np.square(y1-y2)))
            dis_list.append(((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) ** 0.5)
    return min(dis_list)


def mean_vector(vector_list, begin, end):
    if end <= begin:
        return [0., 0.]
    length = end - begin
    mean_x = 0.
    mean_y = 0.
    for i in range(begin, end):
        mean_x += vector_list[i][0]  # 不scale attention!
        mean_y += vector_list[i][1]  # 不scale attention!
    return [mean_x / length, mean_y / length]


class Track(object):
    '''
    track:
        id,
        sequence(point类的序列),
        video_id 所在的video

    '''

    def __init__(self, id, sequence):
        self.id = id
        self.sequence = sequence
        # self.simple_sequence = simple_sequence
        self.length = 0
        self.valid_speed = 0
        self.move_vector = [0, 0]
        self.speed_list = []

    def add_point(self, point):
        if point.id == self.id:
            self.sequence.append(point)
            self.length += 1

    def print_sequence(self):
        for item in self.sequence:
            item.print_point()

    def sort_every_point_by_time(self):
        # 对sequence进行排序

        self.sequence.sort(key=lambda x: x.t)  # 正序，时间由早到晚
        # self.print_sequence()

    def sort_by_time_and_smooth(self):  # 删除离群点
        # 对sequence按照时间先后进行排序
        self.sequence.sort(key=lambda x: x.t)

        length = len(self.sequence)
        if length <= 2:
            return 0
        vector_list = []
        index_list = []
        point_num_list = []
        speed_list = []
        for i in range(1, length):
            det_x = self.sequence[i].x - self.sequence[i - 1].x
            det_y = self.sequence[i].y - self.sequence[i - 1].y
            vector_list.append([det_x * 1e5, det_y * 1e5])  # attention !!!!!
            speed = self.sequence[i].speed_between_point(self.sequence[i - 1])

            speed_list.append(speed)
            point_num_list.append(i)

            if speed > SPEED_THRESHOLD:  # 速度超出了限制
                if i not in index_list:
                    index_list.append(i)  # 起点的index
                if i == 1:
                    index_list.append(0)

            if i - 1 in index_list:  # todo 如果上一个点跟上上一个点的速度超出限制
                if i >= 2:
                    speed_with_prepre = self.sequence[i].speed_between_point(self.sequence[i - 2])
                    if speed_with_prepre > SPEED_THRESHOLD:
                        if i not in index_list:
                            index_list.append(i)

        # print(speed_list)
        over_speed_num = len(index_list)

        begin = 0
        end = 0
        max_gap = 0
        for i in range(0, len(index_list) - 1):
            if index_list[i + 1] - index_list[i] >= max_gap:
                begin = index_list[i]
                end = index_list[i + 1]
                max_gap = index_list[i + 1] - index_list[i]
        # 0到begin ,begin-end,end到尾,取最长的连续一段
        if len(index_list) != 0:
            gap_to_tail = len(self.sequence) - 1 - index_list[-1]
            gap_from_begin = index_list[0]
            if gap_to_tail > gap_from_begin and gap_to_tail > max_gap:
                begin = index_list[-1]
                end = len(self.sequence) - 1
            elif gap_from_begin > gap_to_tail and gap_from_begin > max_gap:
                end = index_list[0]
                begin = 0
            # test

        # 一定要先算运动向量和速度，再删除离群点
        self.move_vector = mean_vector(vector_list, begin, end)
        if begin == 0 and end == 0:  # 如果没有离群点，index_list里面只有尾巴
            self.valid_speed = self.sequence[0].speed_between_point(self.sequence[-1])
        else:
            self.valid_speed = self.sequence[begin + 1].speed_between_point(
                self.sequence[min(end + 1, len(self.sequence) - 1)])

        # 删除离群点
        # if 1 not in index_list:
        #    index_list.append(1)
        # if 2 not in index_list:
        #    index_list.append(2)  # todo 一开始就非常偏离的点 把前半部分删掉
        index_list.sort()

        if len(index_list) > len(self.sequence) / 3:
            return over_speed_num
        for i in range(len(index_list) - 1, -1, -1):  # todo attention!!!! 倒序删除
            self.sequence.pop(index_list[i])

        return over_speed_num

    def sort_by_time_and_deduplicate(self):  # , anchor)#todo 希望anchor是整个map的也就是整个场景之下的中间点
        # 对sequence进行排序
        # TODO计算排序时间
        self.sequence.sort(key=lambda x: x.t, reverse=True)  # 倒序
        # todo 如果时间不连续 插值
        # 如果相距太近，删除点（O(1) pop()，O(nlogn) sort())
        length = len(self.sequence)
        if length <= 5:
            return
        for i in range(length - 1, 0, -1):
            if self.sequence[i].similar_point_in_pos(self.sequence[i - 1]):
                self.sequence[-1], self.sequence[i - 1] = self.sequence[i - 1], self.sequence[-1]
                self.sequence.pop()

        self.sequence.sort(key=lambda x: x.t)

    def get_middle_point(self, begin, end, step=1):
        # 算一个点列表的中间坐标，代表这个点
        # 返回[x,y,t]
        middle_x = 0
        middle_y = 0
        middle_t = 0
        length = end - begin
        if length == 0:
            return [0., 0., 0.]
        number = 0
        for i in range(begin, end, step):  # todo 步长可以继续调
            if i >= end:
                break  # 因为有step
            middle_x += self.sequence[i].x
            middle_y += self.sequence[i].y
            middle_t += self.sequence[i].t
            number += 1
        if number == 0:
            return [0., 0., 0.]
        middle_point = [middle_x / number, middle_y / number, middle_t / number]
        # print(middle_point)
        return middle_point

    def get_middle_points_list(self, threshold):
        length = len(self.sequence)
        # SLICE_THRESHOLD = 10
        # DIRECTION_THRESHOLD = 50
        middle_points_list = []
        if length < 2 * threshold:  # 长度甚至小于两倍的阈值的，
            if length < threshold:
                return [self.get_middle_point(0, int(length / 2), 1), self.get_middle_point(int(length / 2), length, 1)]
            else:
                return [self.get_middle_point(0, int(length / 2), 1), self.get_middle_point(int(length / 2), length, 1),
                        self.get_middle_point(threshold, length, 1)]
        else:
            for i in range(0, length, threshold):
                middle_points_list.append(self.get_middle_point(i, min(i + threshold, length)))

            return middle_points_list

    def vector_list(self, middle_point_list):
        vector_list = []
        length = len(middle_point_list)
        if length <= 1:
            return vector_list
        for i in range(0, length - 1, 1):  # todo
            det_x = middle_point_list[i + 1][0] - middle_point_list[i][0]
            det_y = middle_point_list[i + 1][1] - middle_point_list[i][1]
            vector_list.append([det_x * 1e5, det_y * 1e5])  # attention !!!!!
        return vector_list

    def angle_with_another_track(self, track_2):  # track已经按时间进行了排序 #TODO
        middle_points_list_1 = self.get_middle_points_list(threshold=int(DIRECTION_THRESHOLD / 2))

        middle_points_list_2 = track_2.get_middle_points_list(threshold=int(DIRECTION_THRESHOLD / 2))

        vector_list_1 = self.vector_list(middle_points_list_1)
        # if len(vector_list_1) == 0:
        # print(middle_points_list_1)
        # print(self.id)
        vector_list_2 = track_2.vector_list(middle_points_list_2)
        # if len(vector_list_2) == 0:
        # print(middle_points_list_2)
        # print(track_2.id)
        angle_list = []
        len_1 = len(vector_list_1)
        len_2 = len(vector_list_2)
        if len_1 <= 0 or len_2 <= 0:
            return [0, 0, 0]
        else:
            for i in range(0, len_1):
                for j in range(0, len_2):
                    angle_list.append(cos_vector(vector_list_1[i], vector_list_2[j]))

        return [min(angle_list), max(angle_list), np.mean(angle_list)]
        # return angle_list

    def angle_and_mindis_with_another_track(self, track_2):
        middle_points_list_1 = self.get_middle_points_list(threshold=int(DIRECTION_THRESHOLD / 2))

        middle_points_list_2 = track_2.get_middle_points_list(threshold=int(DIRECTION_THRESHOLD / 2))

        vector_list_1 = self.vector_list(middle_points_list_1)
        vector_list_2 = track_2.vector_list(middle_points_list_2)

        min_dis = min_distance(middle_points_list_1, middle_points_list_2)  # todo new

        angle_list = []
        len_1 = len(vector_list_1)
        len_2 = len(vector_list_2)
        if len_1 <= 0 or len_2 <= 0:
            return [0, 0, 0]
        else:
            for i in range(0, len_1):
                for j in range(0, len_2):
                    angle_list.append(cos_vector(vector_list_1[i], vector_list_2[j]))

        return [min(angle_list), max(angle_list), np.mean(angle_list), min_dis]

    def can_not_reach(self, track_2):
        vector_between_track = [(track_2.sequence[0].x - self.sequence[0].x),
                                (track_2.sequence[0].y - self.sequence[0].y)]
        cos_dis = cos_vector(self.move_vector, vector_between_track)
        if cos_dis < COS_DIS_OPP_DIRECTION:
            return 0
        else:
            return 1

    def time_to_another_track(self, track_2):
        # return [5.0]
        # 方向不能完全不一致
        time = 0  # 为了跟angle_and_mindis_with_another_track的返回值合并，用list返回
        min_max_mean_mindis = self.angle_and_mindis_with_another_track(track_2)
        if min_max_mean_mindis[-1] < MIN_DIS_THRESHOLD:
            return [0]  # 方向完全不一致，认为不可能到达,  两条track
        if min_max_mean_mindis[1] < COS_DIS_OPP_DIRECTION or self.can_not_reach(track_2):
            return [-99]
        else:
            return [99]

    def angle_with_another_track_coarse(self, track_2):  # 使用首尾的方向向量，用于验证
        if len(self.sequence) == 1 or len(track_2.sequence) == 1:
            return [0, 0, 0]
        vec_1 = [(self.sequence[1].x - self.sequence[0].x) * 1e5, (self.sequence[1].y - self.sequence[0].y) * 1e5]
        vec_2 = [(track_2.sequence[1].x - track_2.sequence[0].x) * 1e5,
                 (track_2.sequence[1].y - track_2.sequence[0].y) * 1e5]
        cos = cos_vector(vec_1, vec_2)
        return [cos, cos, cos]

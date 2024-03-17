import math
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
from cmaes import CMA, CMAwM
import json

X = 0
Y = 1
START = 0
END = 1

plt.rcParams['font.sans-serif'] = 'Times new roman'

def getTransitionAngleForHypar(main_folding_angle):
    return np.arccos(np.sqrt(2 * np.cos(main_folding_angle / 2.0) / (1 + np.cos(main_folding_angle / 2.0))))

def getTransitionAngle(main_folding_angle, beta):
    Z = np.sin(main_folding_angle / 2) ** 2 * np.tan(beta) ** 2
    RHS = (1 - Z) / (1 + Z)
    return np.arccos(RHS)

def getRotationMatrix2D(angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])
    return R
    
class Line:
    def __init__(self, start, end, dim) -> None:
        self.start = start
        self.end = end
        self.dim = dim

    def length(self):
        return math.sqrt(sum([(self.end[i] - self.start[i]) ** 2 for i in range(self.dim)]))
    
    def getData(self):
        return [
            self.start, self.end
        ]

    def getMinX(self):
        return min(self.start[X], self.end[X])
    
    def getMaxX(self):
        return max(self.start[X], self.end[X])
    
    def getMinY(self):
        return min(self.start[Y], self.end[Y])
    
    def getMaxY(self):
        return max(self.start[Y], self.end[Y])

def calculateIntersection(lines):
    intersect = False
    length = len(lines)
    for i in range(length):
        if lines[i].getData()[END][X] >= 140.0 and lines[i].getData()[END][Y] >= 30.0:
            intersect = True
            break
        for j in range(i + 1, length):
            line1 = lines[i]
            line2 = lines[j]
            AC = np.array([line2.getData()[START][X] - line1.getData()[START][X], line2.getData()[START][Y] - line1.getData()[START][Y]])
            AD = np.array([line2.getData()[END][X] - line1.getData()[START][X], line2.getData()[END][Y] - line1.getData()[START][Y]])
            BC = np.array([line2.getData()[START][X] - line1.getData()[END][X], line2.getData()[START][Y] - line1.getData()[END][Y]])
            BD = np.array([line2.getData()[END][X] - line1.getData()[END][X], line2.getData()[END][Y] - line1.getData()[END][Y]])
            if (np.cross(AC, AD) * np.cross(BC, BD) < -1e-5 and np.cross(AC, BC) * np.cross(AD, BD) < -1e-5):
                intersect = True
                break
        if intersect:
            break
    return intersect

class TransitionModel:
    def __init__(self) -> None:
        self.src = []
        self.main_folding_angle = 0
        self.tl = []
        self.plot_tl = []
        self.end_ef = []
        self.end_ef_dir = []
        self.all_end_ef = []
        self.all_end_ef_dir = []
        self.self_intersection_number = 0
        self.enable_hypar_connection = False

    def setSource(self, src):
        self.src = src #[[25, 0, 0], [35, 0.7, 0], [45, 0.7, 1]]

    def setMainFoldingAngle(self, fa):
        self.main_folding_angle = fa

    def calculateTransition(self):
        points = [np.array([0, 0])]
        alphas = [0]
        lines = []
        
        for i in range(len(self.src)):
            start = points[i]
            length, beta = self.src[i][0], self.src[i][1]
            if i > 0:
                angle = getTransitionAngle(self.main_folding_angle, beta)
                if type(self.src[i][2]) == float:
                    if self.src[i][2] < 0.5:
                        angle = -angle
                else:
                    if not self.src[i][2]:
                        angle = -angle
            else:
                if self.enable_hypar_connection:
                    angle = getTransitionAngleForHypar(self.main_folding_angle)
                else:
                    angle = getTransitionAngle(self.main_folding_angle, beta)
                    if type(self.src[i][2]) == float:
                        if self.src[i][2] < 0.5:
                            angle = -angle
                    else:
                        if not self.src[i][2]:
                            angle = -angle
            line = Line(start, np.array([start[X] + length * np.cos(alphas[i] + angle), start[Y] + length * np.sin(alphas[i] + angle)]), 2)
            new_start = deepcopy(line.end)
            points.append(new_start)
            alphas.append(alphas[i] + angle)
            lines.append(line)
        self.tl = lines
        self.end_ef = points[-1]
        self.end_ef_dir = alphas[-1]

    def getAABB(self, exp=0.2):
        old_folding_angle = self.main_folding_angle

        min_x_list = []
        max_x_list = []
        min_y_list = []
        max_y_list = []

        self.all_end_ef.clear()
        self.all_end_ef_dir.clear()

        for i in range(180 + 1):
            self.main_folding_angle = i / 180.0 * math.pi
            new_tl = self.getTransitionLines()

            min_x = min([ele.getMinX() for ele in new_tl])
            max_x = max([ele.getMaxX() for ele in new_tl])
            min_y = min([ele.getMinY() for ele in new_tl])
            max_y = max([ele.getMaxY() for ele in new_tl])

            min_x_list.append(min_x)
            max_x_list.append(max_x)
            min_y_list.append(min_y)
            max_y_list.append(max_y)

            self.all_end_ef.append(deepcopy(self.end_ef))
            self.all_end_ef_dir.append(deepcopy(self.end_ef_dir))
    
        real_min_x = min(min_x_list)
        real_max_x = max(max_x_list)
        real_min_y = min(min_y_list)
        real_max_y = max(max_y_list)

        deltax = real_max_x - real_min_x
        deltay = real_max_y - real_min_y

        self.main_folding_angle = old_folding_angle
        self.calculateTransition()

        return real_min_x - deltax * 0.2, real_max_x + deltax * 0.2, real_min_y - deltay * 0.2, real_max_y + deltay * 0.2

    def getTransitionLines(self):
        self.calculateTransition()
        return self.tl
    
    def plotUi(self, ui=False):
        # Get a AABB box
        min_x, max_x, min_y, max_y = self.getAABB()

        self.plot_tl.clear()
        # 创建一个Figure对象和一个Axes对象
        self.fig, self.ax = plt.subplots()

        # 创建一个Slider对象，用于控制折角
        slider_ax = plt.axes([0.22, 0.03, 0.6, 0.03])
        slider = Slider(slider_ax, "folding_angle", 0, math.pi, valinit=self.main_folding_angle)
        slider.on_changed(self.update_line)

        # 绘制所有线段
        for i in range(len(self.tl)):
            start, end = self.tl[i].getData()
            line, = self.ax.plot([start[0], end[0]], [start[1], end[1]], label=f"线段{i+1}")
            self.plot_tl.append(line)

        self.ax.set_xlim(min_x, max_x)
        self.ax.set_ylim(min_y, max_y)
        
        # 显示图像
        if ui:
            plt.show()

    def getAllEndEffector(self):
        self.all_end_ef.clear()
        self.all_end_ef_dir.clear()
        self.self_intersection_number = 0
        for i in range(180):
            self.main_folding_angle = i / 180.0 * math.pi
            tl = self.getTransitionLines()
            self.all_end_ef.append(deepcopy(self.end_ef))
            self.all_end_ef_dir.append(deepcopy(self.end_ef_dir))
            if calculateIntersection(tl):
                self.self_intersection_number += 1
        self.main_folding_angle = math.pi
        self.getTransitionLines()
        self.all_end_ef.append(deepcopy(self.end_ef))
        self.all_end_ef_dir.append(deepcopy(self.end_ef_dir))
        return self.all_end_ef, self.all_end_ef_dir

    def getPartEndEffector(self, number):
        number = number - 1
        self.all_end_ef.clear()
        self.all_end_ef_dir.clear()
        self.self_intersection_number = 0
        for i in range(0, 180, 180 // number):
            self.main_folding_angle = i / 180.0 * math.pi
            tl = self.getTransitionLines()
            self.all_end_ef.append(deepcopy(self.end_ef))
            self.all_end_ef_dir.append(deepcopy(self.end_ef_dir))
            if calculateIntersection(tl):
                self.self_intersection_number += 1
        self.main_folding_angle = math.pi
        self.getTransitionLines()
        self.all_end_ef.append(deepcopy(self.end_ef))
        self.all_end_ef_dir.append(deepcopy(self.end_ef_dir))

        return self.all_end_ef, self.all_end_ef_dir
    
    def update_line(self, val):
        self.main_folding_angle = val
        new_tl = self.getTransitionLines()
        # self.ax.plot(self.end_ef[X], self.end_ef[Y], 'ro')
        self.ax.quiver(self.end_ef[X], self.end_ef[Y], np.cos(self.end_ef_dir), np.sin(self.end_ef_dir), angles='xy', scale_units='xy', scale=0.4)
        for i in range(len(self.plot_tl)):
            start, end = new_tl[i].getData()
            self.plot_tl[i].set_xdata([start[X], end[X]])
            self.plot_tl[i].set_ydata([start[Y], end[Y]])

        self.fig.canvas.draw_idle()

    def update_line_using_timer(self, val):
        self.main_folding_angle = val
        new_tl = self.getTransitionLines()
        # self.ax.plot(self.end_ef[X], self.end_ef[Y], 'ro')
        self.ax.quiver(self.end_ef[X], self.end_ef[Y], np.cos(self.end_ef_dir), np.sin(self.end_ef_dir), angles='xy', scale_units='xy', scale=0.4)
        for i in range(len(self.plot_tl)):
            start, end = new_tl[i].getData()
            self.plot_tl[i].set_xdata([start[X], end[X]])
            self.plot_tl[i].set_ydata([start[Y], end[Y]])

        self.fig.canvas.draw_idle()

class CurveFittingHelper:
    def __init__(self) -> None:
        self.goal = list()
        self.origin = list()
        self.position_match_flag = False
        self.precision = 6
        self.intersection_time = 0
        self.enable_direction_match = False
        self.direction_goal = list()
        self.direction_origin = list()

    def setPositionMatch(self, flag: bool):
        self.position_match_flag = flag

    def setGoalList(self, goal):
        self.goal = goal

    def setOriginList(self, origin):
        self.origin = origin

    def setDirectionGoalList(self, goal):
        self.direction_goal = goal

    def setDirectionOriginList(self, origin):
        self.direction_origin = origin

    def setPrecision(self, digit):
        self.precision = digit

    def setIntersectionTime(self, time):
        self.intersection_time = time

    def strictMatch(self):
        p1, p2 = self.match('regular')
        return max(p1, p2)
    
    def directionMatch(self, number):
        number = number - 1
        errors = []
        step = len(self.direction_goal) // number
        for i in range(0, number):
            target_direction = self.direction_goal[i * step]
            origin_direction = self.direction_origin[i * step]
            point1 = [math.cos(target_direction), math.sin(target_direction)]
            point2 = [math.cos(origin_direction), math.sin(origin_direction)]
            errors.append(math.sqrt((point1[X] - point2[X]) ** 2 + (point1[Y] - point2[Y]) ** 2))
        target_direction = self.direction_goal[-1]
        origin_direction = self.direction_origin[-1]
        point1 = [math.cos(target_direction), math.sin(target_direction)]
        point2 = [math.cos(origin_direction), math.sin(origin_direction)]
        errors.append(math.sqrt((point1[X] - point2[X]) ** 2 + (point1[Y] - point2[Y]) ** 2))

        total_error = sum(errors) / 2. / len(errors)
        return 1. - total_error

    def partMatch(self, number):
        number = number - 1
        accuracy = [0.0, 0.0, 0.0, 0.0]
        step = len(self.goal) // number
        max_dis = []
        for i in range(0, number):
            goal = self.goal[i * step]
            if i == 0:
                origin = self.origin[0]
                dis = math.sqrt((origin[X] - goal[X]) ** 2 + (origin[Y] - goal[Y]) ** 2)
                max_dis.append(dis)
            else:
                origin = self.origin[0]
                standard = math.sqrt((origin[X] - goal[X]) ** 2 + (origin[Y] - goal[Y]) ** 2)
                for j in range(1, len(self.origin)):
                    origin = self.origin[j]
                    dis = math.sqrt((origin[X] - goal[X]) ** 2 + (origin[Y] - goal[Y]) ** 2)
                    if dis < standard:
                        standard = dis
                max_dis.append(standard)

        origin = self.origin[-1]
        goal = self.goal[-1]
        dis = math.sqrt((origin[X] - goal[X]) ** 2 + (origin[Y] - goal[Y]) ** 2)
        max_dis.append(dis)
        mean_dis = np.mean(max_dis)

        # total_distance, 20 points
        for i in range(0, number + 1):
            accuracy[0] += (50. / (50. + max_dis[i]) / (number + 1)) * 20.

        # var and mean, 10 points
        accuracy[1] = (20. / (40. + np.var(max_dis)) + 10. / (20. + mean_dis)) * 10.
        
        # start and end, 30 points
        accuracy[2] = (2. / (10. + max_dis[0]) + 8. / (10. + max_dis[-1])) * 30.

        # self-intersection, 40 points
        accuracy[3] = 40. - 4. * self.intersection_time

        return sum(accuracy), accuracy

    
    def distanceMatch(self):
        data1 = np.array(self.origin)
        data2 = np.array(self.goal)

        length1 = data1.shape[0]
        length2 = data2.shape[0]

        dist1 = []
        dist2 = []

        for i in range(length1 - 1):
            next_point = data1[i + 1]
            cur_point = data1[i]
            distance = math.sqrt((next_point[X] - cur_point[X]) ** 2 + (next_point[Y] - cur_point[Y]) ** 2)
            dist1.append(distance)

        for i in range(length2 - 1):
            next_point = data2[i + 1]
            cur_point = data2[i]
            distance = math.sqrt((next_point[X] - cur_point[X]) ** 2 + (next_point[Y] - cur_point[Y]) ** 2)
            dist2.append(distance)

        total_dist1 = sum(dist1)
        total_dist2 = sum(dist2)

        dist1 = np.array(dist1) / total_dist1
        dist2 = np.array(dist2) / total_dist2

        loss1_2 = []

        percent1 = 0.0
        percent2 = 0.0
        old_percent2 = 0.0
        i = 0
        j = 0
        while i < length1:
            point1 = data1[i]
            while percent2 < percent1-1e-5:
                old_percent2 = percent2
                percent2 += dist2[j]
                j += 1
            
            if i > 0:
                percent = (percent1 - old_percent2) / (percent2 - old_percent2)

                point2 = [
                    percent * (data2[j][X] - data2[j - 1][X]) + data2[j - 1][X],
                    percent * (data2[j][Y] - data2[j - 1][Y]) + data2[j - 1][Y]
                ]
            else:
                point2 = data2[0]
            
            distance = math.sqrt((point2[X] - point1[X]) ** 2 + (point2[Y] - point1[Y]) ** 2)
            loss1_2.append(distance)

            if percent1 >= 1.0-1e-5:
                break
            percent1 += dist1[i]
            i += 1
        
        accuracy = 0.0
        for i in range(len(loss1_2) - 1):
            accuracy += dist1[i] * ((total_dist1 + total_dist2) / (total_dist1 + total_dist2 + loss1_2[i] + loss1_2[i + 1]))

        return accuracy

    def bestMatch(self):
        p1, p2 = self.match('regular')
        mp1, mp2 = self.match('mirror')

        goal_length = len(self.goal)
        
        max1 = max(p1, p2, mp1, mp2)

        self.setGoalList([[self.goal[goal_length - 1 - i][X], self.goal[goal_length - 1 - i][Y]] for i in range(goal_length)])
        p1, p2 = self.match('regular')
        mp1, mp2 = self.match('mirror')

        max2 = max(p1, p2, mp1, mp2)

        return max(max1, max2)
    
    def match(self, mode='regular') -> float:
        data1 = np.array(self.origin)
        data2 = np.array(self.goal)

        length1 = data1.shape[0]
        length2 = data2.shape[0]

        dist1 = []
        dir1 = []
        dir1_change_flag = [0]

        dist2 = []
        dir2 = []
        dir2_change_flag = [0]

        for i in range(length1 - 1):
            next_point = data1[i + 1]
            cur_point = data1[i]
            distance = math.sqrt((next_point[X] - cur_point[X]) ** 2 + (next_point[Y] - cur_point[Y]) ** 2)
            if mode == "regular":
                try:
                    if next_point[Y] == cur_point[Y] and next_point[X] == cur_point[X]:
                        dir = dir1[-1]
                    else:
                        dir = math.atan((next_point[Y] - cur_point[Y]) / (next_point[X] - cur_point[X]))
                except:
                    dir = math.pi / 2
                if i >= 1:
                    if dir - dir1[-1] > math.pi * 3 / 4:
                        dir1_change_flag.append(-math.pi)
                    elif dir - dir1[-1] < -math.pi * 3 / 4:
                        dir1_change_flag.append(math.pi)
                    else:
                        dir1_change_flag.append(0)

            elif mode == "mirror":
                try:
                    if next_point[Y] == cur_point[Y] and next_point[X] == cur_point[X]:
                        dir = dir1[-1]
                    else:
                        dir = math.atan((-next_point[Y] + cur_point[Y]) / (next_point[X] - cur_point[X]))
                except:
                    dir = math.pi / 2
                if i >= 1:
                    if dir - dir1[-1] > math.pi * 3 / 4:
                        dir1_change_flag.append(-math.pi)
                    elif dir - dir1[-1] < -math.pi * 3 / 4:
                        dir1_change_flag.append(math.pi)
                    else:
                        dir1_change_flag.append(0)

            dist1.append(distance)
            dir1.append(dir)

        for i in range(length2 - 1):
            next_point = data2[i + 1]
            cur_point = data2[i]
            distance = math.sqrt((next_point[X] - cur_point[X]) ** 2 + (next_point[Y] - cur_point[Y]) ** 2)
            try:
                if next_point[Y] == cur_point[Y] and next_point[X] == cur_point[X]:
                    dir = dir2[-1]
                else:
                    dir = math.atan((next_point[Y] - cur_point[Y]) / (next_point[X] - cur_point[X]))
            except:
                dir = math.pi / 2
            if i >= 1:
                if dir - dir2[-1] > math.pi * 3 / 4:
                    dir2_change_flag.append(-math.pi)
                elif dir - dir2[-1] < -math.pi * 3 / 4:
                    dir2_change_flag.append(math.pi)
                else:
                    dir2_change_flag.append(0)
            dist2.append(distance)
            dir2.append(dir)

        total_dist1 = sum(dist1)
        total_dist2 = sum(dist2)

        dist1 = np.array(dist1) / total_dist1
        dist2 = np.array(dist2) / total_dist2

        loss_dist1 = deepcopy(dist1)
        loss_dist2 = deepcopy(dist2)

        for i in range(1, dist1.shape[0]):
            dist1[i] += dist1[i - 1]

        for i in range(1, dist2.shape[0]):
            dist2[i] += dist2[i - 1]

        delta_dir1 = []
        delta_dir2 = []

        for i in range(1, length1 - 1):
            delta_dir = (dir1[i] - dir1[i - 1] + dir1_change_flag[i]) / (loss_dist1[i] + loss_dist1[i - 1])
            delta_dir1.append(delta_dir)

        for i in range(1, length2 - 1):
            delta_dir = (dir2[i] - dir2[i - 1] + dir2_change_flag[i]) / (loss_dist2[i] + loss_dist2[i - 1])
            delta_dir2.append(delta_dir)

        delta_dir1 = np.array(delta_dir1)
        delta_dir2 = np.array(delta_dir2)

        loss1_2 = []
        loss2_1 = []

        for i in range(length1 - 2):
            distance = dist1[i]
            dir = delta_dir1[i]

            lower = 0
            lower_dir = delta_dir2[0]
            upper = dist2[0]
            upper_dir = delta_dir2[0]
            for j in range(1, length2 - 1):
                if distance > lower and distance <= upper:
                    break
                else:
                    lower = upper
                    lower_dir = upper_dir
                    upper = dist2[j]
                    if j == length2 - 2:
                        upper_dir = delta_dir2[j - 1]
                    else:
                        upper_dir = delta_dir2[j]
            
            percent = (distance - lower) / (upper - lower) * (upper_dir - lower_dir) + lower_dir #value
            loss1_2.append((percent - dir) ** 2)

        for i in range(length2 - 2):
            distance = dist2[i]
            dir = delta_dir2[i]

            lower = 0
            lower_dir = 0
            upper = dist1[0]
            upper_dir = delta_dir1[0]
            for j in range(1, length1 - 1):
                if distance > lower and distance <= upper:
                    break
                else:
                    lower = upper
                    lower_dir = upper_dir
                    upper = dist1[j]
                    if j == length1 - 2:
                        upper_dir = delta_dir1[j - 1]
                    else:
                        upper_dir = delta_dir1[j]
            
            percent = (distance - lower) / (upper - lower) * (upper_dir - lower_dir) + lower_dir #value
            loss2_1.append((percent - dir) ** 2)

        match_percent1 = (loss_dist1[0] + loss_dist1[-1]) / 2
        match_percent2 = (loss_dist2[0] + loss_dist2[-1]) / 2

        for i in range(length1 - 2):
            length = (loss_dist1[i + 1] + loss_dist1[i]) / 2
            match_percent1 += length * 1 / (1 + loss1_2[i])

        for i in range(length2 - 2):
            length = (loss_dist2[i + 1] + loss_dist2[i]) / 2
            match_percent2 += length * 1 / (1 + loss2_1[i])

        return np.round(match_percent1, self.precision), np.round(match_percent2, self.precision)

class Pack:
    def __init__(self, data=None, score=None) -> None:
        self.data = data
        self.score = score

    def setScore(self, score):
        self.score = score
    
class GA:
    def __init__(self, storage_number=100) -> None:
        self.iteration = 0
        self.row = 1
        self.col = 1
        self.storage = []
        self.storage_number = storage_number
        self.cur_data = []
        self.cur_data_to_tell = []

        self.prob_select = 0
        self.prob_cross = 0.9
        self.prob_muta = 0.1

        self.mutation_probability = 0.05

    def initialize(self, row, col):
        self.iteration = 0
        self.storage.clear()
        self.setDataType(row, col)
        self.mapper = [
            [0, 1] for i in range(col)
        ]

    def ask(self):
        self.cur_data.clear()
        self.cur_data_to_tell.clear()
        if len(self.storage) < self.storage_number:
            for i in range(self.row):
                data = [np.random.random() for j in range(self.col)]
                self.cur_data_to_tell.append(deepcopy(data))
                for j in range(self.col):
                    if isinstance(self.mapper[j][0], list):
                        map_length = len(self.mapper[j])
                        unit = 1.0 / map_length
                        unit_number = int(data[j] / unit)
                        if unit_number >= map_length:
                            unit_number = map_length - 1
                        mapper = self.mapper[j][unit_number]
                        data[j] = (data[j] - unit * unit_number) / unit * (mapper[1] - mapper[0]) + mapper[0]
                    else:
                        data[j] = data[j] * (self.mapper[j][1] - self.mapper[j][0]) + self.mapper[j][0]
                self.cur_data.append(Pack(data, 0))
        else:
            cross_num = int(self.prob_cross * self.row)
            muta_num = int(self.prob_muta * self.row)

            indice_array = np.zeros(self.storage_number)
            prob_array = np.zeros(self.storage_number)
            fitness_sum = 0

            for i in range(self.storage_number):
                indice_array[i] = i
                prob_array[i] = self.storage[i].score - self.storage[0].score
                fitness_sum += prob_array[i]

            prob_array /= fitness_sum

            for i in range(cross_num):
                sample = np.random.choice(indice_array, 2, replace=False, p=prob_array)
                p1 = self.storage[int(sample[0])]
                p2 = self.storage[int(sample[1])]
                # mode = np.random.random()
                # if mode <= 0.4:
                # data = [(p1.data[j] + p2.data[j]) / (np.random.random() * 3.0 + 1.0) for j in range(self.col)]
                # elif mode > 0.4 and mode <= 0.7:
                data = []
                data_to_tell = []
                for j in range(self.col):
                    if np.random.random() > self.mutation_probability:
                        val = (p1.data[j] + p2.data[j]) / (np.random.random() * 2.0 + 1.0)
                    else:
                        val = (p1.data[j] - p2.data[j]) / (np.random.random() * 2.0 + 1.0)
                    if val > 1.0:
                        val = 1.0
                    elif val < 0.0:
                        val = 0.0
                    data_to_tell.append(val)
                    val = val * (self.mapper[j][1] - self.mapper[j][0]) + self.mapper[j][0]
                    data.append(val)
                # else:
                #     data = []
                #     for j in range(self.col):
                #         val = (p1.data[j] - p2.data[j])
                #         if val < 0.0:
                #             val += 1.0
                #         data.append(val)

                self.cur_data.append(Pack(data, 0))
                self.cur_data_to_tell.append(data_to_tell)

            muta_indice = np.random.choice(self.storage_number, muta_num, replace=True)
            for ele in muta_indice:
                p = deepcopy(self.storage[ele])
                if np.random.random() < self.mutation_probability:
                    for j in range(self.col):
                        p.data[j] += np.random.random() - 0.5
                        if p.data[j] > 1.0:
                            p.data[j] = 1.0
                        elif p.data[j] < 0.0:
                            p.data[j] = 0.0
                self.cur_data_to_tell.append(deepcopy(p))
                for j in range(self.col):
                    p.data[j] = p.data[j] * (self.mapper[j][1] - self.mapper[j][0]) + self.mapper[j][0]
                self.cur_data.append(p)

        return self.cur_data
    
    def getFitnessArray(self):
        fitness_array = []
        for i in range(len(self.storage)):
            fitness_array.append(self.storage[i].score)
        return fitness_array

    def update(self, d: Pack):
        true_length = len(self.storage)
        fitness_array = self.getFitnessArray()

        if true_length < self.storage_number:
            self.storage.append(d)
            self.storage = sorted(self.storage, key=lambda d:d.score)
        elif true_length == self.storage_number:
            if d.score > self.storage[0].score and d.score not in fitness_array:
                self.storage[0] = d
                self.storage = sorted(self.storage, key=lambda d:d.score)
        else:
            self.store = self.store[true_length - self.storage_number: true_length]

    def updateStorage(self):
        for i in range(self.row):
            self.update(Pack(self.cur_data_to_tell[i], self.cur_data[i].score))

    def evaluate(self, fitness_list):
        for i in range(self.row):
            self.cur_data[i].setScore(fitness_list[i])
        self.updateStorage()
        self.iteration += 1

    def setDataType(self, row, col):
        self.row = row
        self.col = col

    def getCurrentBest(self):
        return self.storage[-1].score
    
    def setMapper(self, mapper):
        self.mapper = mapper

class ES():
    def __init__(self, storage_number=100) -> None:
        self.iteration = 0
        self.row = 1
        self.col = 1
        self.storage = []
        self.storage_number = storage_number
        self.cur_data = []
        self.cur_data_to_tell = []

        self.sample_mode = 'es'

    def mode(self, mode):
        self.sample_mode = mode

    def initialize(self, row, col, random_loc_and_scale=True):
        self.iteration = 0
        self.storage.clear()
        self.setDataType(row, col)
        if random_loc_and_scale:
            self.loc = [np.random.random() for i in range(col)]
            self.scale = [(np.random.random() * 0.75 + 0.25) for i in range(col)]
        else:
            self.loc = [0.5 for i in range(col)]
            self.scale = [0.25 for i in range(col)]
        self.mapper = [
            [0, 1] for i in range(col)
        ]
        if self.sample_mode == "cmaes":
            self.optimizer = CMA(mean=np.array(self.loc), sigma=0.5, bounds=np.array([[0.0, 1.0] for i in range(self.col)]), population_size=row)

    def changeModeAndContinue(self, mode):
        if mode == 'cmaes' and self.sample_mode != 'cmaes':
            self.optimizer = CMA(mean=np.array(self.loc), sigma=0.05, bounds=np.array([[0.0, 1.0] for i in range(self.col)]), population_size=self.row, cov=np.diag(self.scale))
            self.sample_mode = mode
        elif mode == 'es' and self.sample_mode != 'es':
            self.loc = self.optimizer._mean.tolist()
            self.sample_mode = mode

    def setDataType(self, row, col):
        self.row = row
        self.col = col

    def ask(self):
        self.cur_data.clear()
        self.cur_data_to_tell.clear()
        if self.sample_mode == 'es':
            data = np.random.normal(self.loc, self.scale, size=(self.row, self.col))
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data[i][j] < 0.0:
                        data[i][j] = 0.0
                    elif data[i][j] > 1.0:
                        data[i][j] = 1.0
            for i in range(data.shape[0]):
                self.cur_data_to_tell.append(deepcopy(data[i]))
                for j in range(data.shape[1]):
                    if isinstance(self.mapper[j][0], list):
                        map_length = len(self.mapper[j])
                        unit = 1.0 / map_length
                        unit_number = int(data[i][j] / unit)
                        if unit_number >= map_length:
                            unit_number = map_length - 1
                        mapper = self.mapper[j][unit_number]
                        data[i][j] = (data[i][j] * - unit * unit_number) / unit * (mapper[1] - mapper[0]) + mapper[0]
                    else:
                        data[i][j] = data[i][j] * (self.mapper[j][1] - self.mapper[j][0]) + self.mapper[j][0]
                self.cur_data.append(Pack(data[i], 0.0))
            return self.cur_data
        elif self.sample_mode == 'cmaes':
            for i in range(self.optimizer.population_size):
                data = self.optimizer.ask()
                self.cur_data_to_tell.append(deepcopy(data))
                for j in range(data.shape[0]):
                    if isinstance(self.mapper[j][0], list):
                        map_length = len(self.mapper[j])
                        unit = 1.0 / map_length
                        unit_number = int(data[j] / unit)
                        if unit_number >= map_length:
                            unit_number = map_length - 1
                        mapper = self.mapper[j][unit_number]
                        data[j] = (data[j] - unit * unit_number) / unit * (mapper[1] - mapper[0]) + mapper[0]
                    else:
                        data[j] = data[j] * (self.mapper[j][1] - self.mapper[j][0]) + self.mapper[j][0]
                self.cur_data.append(Pack(data, 0.0))
            return self.cur_data
    
    def update(self, d: Pack):
        true_length = len(self.storage)
        if true_length < self.storage_number:
            self.storage.append(d)
            self.storage = sorted(self.storage, key=lambda d:d.score)
        elif true_length == self.storage_number:
            if d.score > self.storage[0].score:
                self.storage[0] = d
                self.storage = sorted(self.storage, key=lambda d:d.score)
        else:
            self.store = self.store[true_length - self.storage_number: true_length]

    def calculateMeanAndScale(self):
        data_list = [[] for i in range(self.col)]
        for ele in self.storage:
            data = ele.data
            for i in range(self.col):
                data_list[i].append(data[i])
        for i in range(self.col):
            self.loc[i] = np.mean(data_list[i])
            # self.scale[i] = np.var(data_list[i])

    def updateStorage(self):
        for i in range(self.row):
            self.update(Pack(self.cur_data_to_tell[i], self.cur_data[i].score))
        
    def evaluate(self, fitness_list):
        for i in range(self.row):
            self.cur_data[i].setScore(fitness_list[i])
        self.updateStorage()

        if self.sample_mode == "es":
            self.calculateMeanAndScale()
        else:
            solution = []
            for i in range(self.row):
                solution.append((self.cur_data_to_tell[i], -self.cur_data[i].score))
            self.optimizer.tell(solution)
            if self.optimizer.should_stop():
                self.changeModeAndContinue('es')

        self.iteration += 1

    def getCurrentBest(self):
        return self.storage[-1].score
    
    def updateScale(self, scale):
        for i in range(self.col):
            self.scale[i] = scale

    def setMapper(self, mapper):
        self.mapper = mapper

def workerMultiEvaluator(mlist, data, tm, cfh, row_num, interval_list, id):
    reward_list = []
    for pack in data:
        k_data = pack.data
        # Set source and get all_ef
        tm.setSource([
            k_data[3*k: 3*k+3] for k in range(row_num)
        ])
        all_ef, _ = tm.getAllEndEffector()
        
        cfh.setOriginList(all_ef)

        p = cfh.bestMatch()

        reward_list.append(p)

    #修改数组
    for i in range(len(interval_list)):
        mlist[interval_list[i]] = reward_list[i]

    print('Sub-process ' + str(id) + ' done')

# import multiprocessing
# def train(data, row_num, tm, cfh):
#     fitness_result = np.zeros(process_num * batch_size).tolist()
            
#     p_list = []
#     # communication list
#     mlist = multiprocessing.Manager().list(fitness_result)

#     for k in range(process_num):
#         #创建进程
#         p = multiprocessing.Process(target=workerMultiEvaluator, args=(mlist, data[batch_size * k: batch_size * k + batch_size], tm, cfh, row_num, [l for l in range(batch_size * k, batch_size * k + batch_size)], k))
#         #放入进程列表
#         p_list.append(p)
    
#     #进程id标识号
#     process_id = 0

#     #现有进程数量
#     current_process_number = 0

#     #进程开始
#     while process_id < process_num:
#         while current_process_number < process_num:
#             #开始进程
#             p_list[process_id].start()
#             current_process_number += 1
#             process_id += 1
#             if process_id == process_num:
#                 break

#         while current_process_number > 0:
#             #进程阻塞，主进程将等待子进程结束后才进行
#             p_list[process_id - current_process_number].join()
#             #进程退出
#             current_process_number -= 1

#     #得到适应度数组
#     fitness_array = list(mlist)

#     return fitness_array

class GraphModel:
    def __init__(self, units, creases, kps) -> None:
        self.units = units
        self.creases = creases # have been calculated
        self.kps = kps
        self.connection_matrix = np.array([[None for i in range(len(self.units))] for j in range(len(self.units))])
        self.connection_bonus_matrix = np.array([[0. for i in range(len(self.units))] for j in range(len(self.units))])
        self.value_array = [None for i in range(len(self.units))]
        self.percent_array = [0.0 for i in range(len(self.units))]
        self.hyper_parameters_self = np.random.normal(0., 1., 5)
        self.hyper_parameters_self_b = np.random.normal(0., 1., 5)
        self.hyper_parameters_others = np.random.normal(0., 1., 5)
        self.hyper_parameters_others_b = np.random.normal(0., 1., 5)

    def distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
    
    def calculateConnectionMatrix(self):
        unit_distance_max = 0.0
        crease_length_max = 0.0
        for i in range(len(self.units)):
            for j in range(i + 1, len(self.units)):
                u1 = self.units[i]
                u2 = self.units[j]
                for c in u1.getCrease():
                    c1_start = c[0]
                    c1_end = c[1]
                    for c2 in u2.getCrease():
                        c2_start = c2[0]
                        c2_end = c2[1]
                        if self.distance(c1_start, c2_end) < 1e-5 and self.distance(c1_end, c2_start) < 1e-5:
                            u1_center = u1.getCenter()
                            u2_center = u2.getCenter()
                            for crease in self.creases:
                                crease_start = crease[0]
                                crease_end = crease[1]
                                if (self.distance(crease_start, c1_start) < 1e-5 and self.distance(crease_end, c1_end) < 1e-5) or (self.distance(crease_start, c1_end) < 1e-5 and self.distance(crease_end, c1_start) < 1e-5):
                                    break
                            self.connection_matrix[i][j] = [
                                self.distance(u1_center, u2_center), 
                                self.distance(crease_start, crease_end), 
                                math.atan(crease.level) * 2. / math.pi, 1. / (1. + crease.coeff), .5 if crease.getType() else -.5
                            ]
                            self.connection_matrix[j][i] = self.connection_matrix[i][j]
                            self.connection_bonus_matrix[i][i] += 1.
                            self.connection_bonus_matrix[j][j] += 1.
                            self.connection_bonus_matrix[i][j] = 1.
                            self.connection_bonus_matrix[j][i] = 1.

                            if self.connection_matrix[i][j][0] > unit_distance_max:
                                unit_distance_max = self.connection_matrix[i][j][0]
                            if self.connection_matrix[i][j][1] > crease_length_max:
                                crease_length_max = self.connection_matrix[i][j][1]
            
        for i in range(len(self.units)):
            for j in range(len(self.units)):
                if self.connection_matrix[i][j] != None:
                    self.connection_matrix[i][j][0] /= unit_distance_max
                    self.connection_matrix[i][j][1] /= crease_length_max
                    self.connection_bonus_matrix[i][j] = 1. / math.sqrt(self.connection_bonus_matrix[i][i])
            self.connection_bonus_matrix[i][i] = 1. / self.connection_bonus_matrix[i][i]
                

    def nonLinearMethod(self, value, method="periodic"):
        if method == "periodic":
            if value > 1.0:
                temp = int((value + 1.0) / 2.0)
                value -= temp * 2.0
                return value
            elif value < -1.0:
                temp = int((1.0 - value) / 2.0)
                value += temp * 2.0
                return value

    def calculateValue(self, level):
        if level == 0:
            for i in range(len(self.units)):
                self.value_array[i] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            for i in range(len(self.units)):
                self.value_array[i] = self.connection_bonus_matrix[i][i] * (self.hyper_parameters_self * self.value_array[i] + self.hyper_parameters_self_b)
                for j in range(len(self.units)):
                    if self.connection_matrix[i][j] != None:
                        self.value_array[i] += self.connection_bonus_matrix[i][j] * (self.hyper_parameters_others * self.connection_matrix[i][j] + self.hyper_parameters_others_b)
        for i in range(len(self.units)):
            for j in range(5):
                if abs(self.value_array[i][j]) > 1.0:
                    self.value_array[i][j] = self.nonLinearMethod(self.value_array[i][j])
        
    def aggregation(self):
        for i in range(4):
            self.calculateValue(i)

    def calculateP(self):
        for i in range(len(self.units)):
            self.percent_array[i] = 1 / (1 + math.exp(-sum(self.value_array[i])))

    def generateIndexList(self, id=None):
        id_list = []
        if id == None: #border
            for i in range(len(self.units)):
                unit = self.units[i]
                if unit.isBorder():
                    id_list.append(i)
        else:
            for i in range(len(self.units)):
                if self.connection_matrix[id][i] != None:
                    id_list.append(i)
        return id_list

    def getMaximumLikelihood(self, id_list):
        max_value = 0.0
        max_value_index = 0
        for id in id_list:
            if self.percent_array[id] > max_value:
                max_value = self.percent_array[id]
                max_value_index = id
        return max_value_index

class MCTS:
    def __init__(self, units, creases, kps, panel_size, panel_resolution, origami_size, tsa_number = 1, string_root_number = 0) -> None:
        self.units = units
        self.creases = creases
        self.kps = kps

        self.tsa_number = tsa_number
        self.string_root_number = string_root_number

        self.origami_size = origami_size

        #Point state
        self.NO_VISITED = 0 
        self.VISITED = 1
        self.FIXED = 2

        self.total_string_number = 2 * self.tsa_number + self.string_root_number

        self.values = [[[] for j in range(self.total_string_number)] for i in range(len(self.units))]  # XI + SQRT(2LNn/NI)
        self.n = [0 for j in range(self.total_string_number)]
        # self.ns = np.array([[[] for j in range(self.total_string_number)] for i in range(len(self.units))])
        self.node_state = np.array([self.NO_VISITED for i in range(len(self.units))])

        self.panel_resolution = panel_resolution
        self.panel_size = panel_size
        self.n_panel = [0 for j in range(self.total_string_number)]
        self.values_panel = [[[] for j in range(self.total_string_number)] for i in range(panel_resolution)]
        # self.ns_panel = np.array([[[] for j in range(self.total_string_number)] for i in range(panel_resolution)])
        self.node_state_panel = np.array([self.NO_VISITED for i in range(panel_resolution)])

        self.ucb_A = []
        self.ucb_B = []

        self.connection_matrix = np.array([[0 for i in range(len(self.units))] for j in range(len(self.units))])

        self.method = []

        self.update_list = []

        self.valid = False

        self.tsa_combos = []

        self.calculateConnectionMatrix()
    
    def checkValid(self, aid, bid1, bid2):
        from utils import Crease, calculateIntersectionWithinCrease, BORDER, sameCrease
        a_point = self.panel_size * np.array([math.cos(2. * math.pi * aid / self.panel_resolution), math.sin(2. * math.pi * aid / self.panel_resolution)]) + np.array(self.origami_size) / 2.0
        b_point1 = self.units[bid1].getCenter()
        b_point2 = self.units[bid2].getCenter()
        c1 = Crease(a_point, b_point1, 0)
        c2 = Crease(a_point, b_point2, 0)

        intersection_time = 0
        for lines in self.creases:
            if lines.getType() == BORDER:
                percent = calculateIntersectionWithinCrease(c1, lines, True)
                if percent != None:
                    intersection_time += 1
                    break
        
        if intersection_time == 1:
            have_border = False
            for unit_id in [bid1, bid2]:
                unit = self.units[unit_id]
                for crease in unit.getCrease():
                    if sameCrease(lines, crease) or sameCrease(lines, crease.getReverse()):
                        have_border = True
                        break
            
            if have_border:
                intersection_time = 0
                for lines in self.creases:
                    if lines.getType() == BORDER:
                        percent = calculateIntersectionWithinCrease(c2, lines, True)
                        if percent != None:
                            intersection_time += 1
                            break

                if intersection_time == 1:
                    have_border = False
                    for unit_id in [bid1, bid2]:
                        unit = self.units[unit_id]
                        for crease in unit.getCrease():
                            if sameCrease(lines, crease) or sameCrease(lines, crease.getReverse()):
                                have_border = True
                                break
                    if have_border:
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                return False
        else:
            return False
        

    def distance(self, x1, x2):
        return math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
    
    def vertical(self, dir1, dir2):
        value = (dir1[0] * dir2[0] + dir1[1] * dir2[1]) / math.sqrt(dir1[0] ** 2 + dir1[1] ** 2) / math.sqrt(dir2[0] ** 2 + dir2[1] ** 2)
        return value

    def getBorderIndex(self, consider_visited = False, consider_fixed = False):
        border_list = []
        for i in range(len(self.units)):
            if self.units[i].isBorder():
                if not (consider_visited and self.node_state[i] == self.VISITED):
                    if not (consider_fixed and self.node_state[i] == self.FIXED):
                        border_list.append(i)
        return border_list

    def calculateConnectionMatrix(self):
        for i in range(len(self.units)):
            for j in range(i + 1, len(self.units)):
                u1 = self.units[i]
                u2 = self.units[j]
                for c in u1.getCrease():
                    c1_start = c[0]
                    c1_end = c[1]
                    for c2 in u2.getCrease():
                        c2_start = c2[0]
                        c2_end = c2[1]
                        if self.distance(c1_start, c2_end) < 1e-5 and self.distance(c1_end, c2_start) < 1e-5:
                            for crease in self.creases:
                                crease_start = crease[0]
                                crease_end = crease[1]
                                if (self.distance(crease_start, c1_start) < 1e-5 and self.distance(crease_end, c1_end) < 1e-5) or (self.distance(crease_start, c1_end) < 1e-5 and self.distance(crease_end, c1_start) < 1e-5):
                                    break
                            self.connection_matrix[i][j] = -1 if crease.getType() else 1
                            self.connection_matrix[j][i] = self.connection_matrix[i][j]

    def findPlLocation(self, past_list, input_id):
        past_list_location = -1
        for j in range(len(past_list)): # 0: parent, 1: ns, 2: value 
            ele = past_list[j]
            if len(ele[0]) >= 2:
                new_list1 = sorted(ele[0])
                new_list2 = sorted(input_id)
                equal = True
                if len(new_list1) != len(new_list2):
                    equal = False
                else:
                    for k in range(len(new_list1)):
                        if new_list1[k] != new_list2[k]:
                            equal = False
                            break
                if equal:
                    past_list_location = j
                    break
            else:
                if ele[0][0] == input_id[0]:
                    past_list_location = j
                    break

        return past_list_location

    def calculateUCB(self, id_list, id_type, string_id, input_id=[]):
        if id_type == 'A':
            self.ucb_A.clear()
            pl_list = []
            for i in range(len(id_list)):
                id = id_list[i]
                past_list = deepcopy(self.values_panel[id][string_id[0]])
                past_list_location = self.findPlLocation(past_list, input_id)
                ns = past_list[past_list_location][1] if past_list_location >= 0 else 0
                value = past_list[past_list_location][2] if past_list_location >= 0 else 0.0

                n = self.n_panel[string_id[0]]
                pl_list.append(past_list_location)
                if ns > 0:
                    self.ucb_A.append(value / ns + math.sqrt(2 * math.log(n) / ns))
                else:
                    self.ucb_A.append(math.inf)
            
            max_value = 0.0
            max_index_list = []
            for i in range(len(self.ucb_A)):
                if self.ucb_A[i] > max_value:
                    max_value = self.ucb_A[i]
                    max_index_list = [i]
                elif self.ucb_A[i] == max_value:
                    max_index_list.append(i)
            index = np.random.choice(max_index_list, 1, replace=False)

            return id_list[index[0]], max_value, pl_list[index[0]]

        if id_type == 'B':
            self.ucb_B.clear()
            pl_list = []
            for i in range(len(id_list)):
                id = id_list[i]
                past_list = deepcopy(self.values[id][string_id[0]])
                past_list_location = self.findPlLocation(past_list, input_id)
                ns = past_list[past_list_location][1] if past_list_location >= 0 else 0
                value = past_list[past_list_location][2] if past_list_location >= 0 else 0.0
                # past_list_location = -1
                # for j in range(len(past_list)): # 0: parent, 1: ns, 2: value 
                #     ele = past_list[j]
                #     if len(ele[0]) >= 2:
                #         new_list1 = sorted(ele[0])
                #         new_list2 = sorted(input_id)
                #         equal = True
                #         if len(new_list1) != len(new_list2):
                #             equal = False
                #         else:
                #             for k in range(len(new_list1)):
                #                 if new_list1[k] != new_list2[k]:
                #                     equal = False
                #                     break
                #         if equal:
                #             # print(id)
                #             # print(ele[0])
                #             # print(input_id)
                #             # print("FIND!")
                #             ns = ele[1]
                #             value = ele[2]
                #             past_list_location = j
                #             break
                #     else:
                #         if ele[0][0] == input_id[0]:
                #             ns = ele[1]
                #             value = ele[2]
                #             past_list_location = j
                #             break
                n = self.n[string_id[0]]
                pl_list.append(past_list_location)
                if ns > 0:
                    self.ucb_B.append(value / ns + math.sqrt(2 * math.log(n) / ns))
                else:
                    self.ucb_B.append(math.inf)
            
            max_value = 0.0
            max_index_list = []
            for i in range(len(self.ucb_B)):
                if self.ucb_B[i] > max_value:
                    max_value = self.ucb_B[i]
                    max_index_list = [i]
                elif self.ucb_B[i] == max_value:
                    max_index_list.append(i)
            # print(max_index_list)
            index = np.random.choice(max_index_list, 1, replace=False)
                
            return id_list[index[0]], max_value, pl_list[index[0]]
        
    def getNeighbor(self, id, consider_visited = False, consider_fixed = False, consider_reserve = False, dir = 1):
        id_neighbor = []
        for i in range(len(self.units)):
            if self.connection_matrix[id][i] != 0:
                if not (consider_visited and self.node_state[i] == self.VISITED):
                    if not (consider_fixed and self.node_state[i] == self.FIXED):
                        if not (consider_reserve and self.connection_matrix[id][i] != dir):
                            id_neighbor.append(i)
        return id_neighbor
    
    def getNeighborA(self, id, consider_visited = False, consider_fixed = False, range_a=1):
        id_neighbor = []
        if self.units[id].isBorder():
            point = self.units[id].getCenter()
            if abs(point[X]) < 1e-5:
                if point[X] >= 0.0:
                    point[X] = 1e-5
                else:
                    point[X] = -1e-5
            alpha = math.atan2(point[Y] - self.origami_size[Y] / 2., point[X] - self.origami_size[X] / 2.)
            resolution = math.pi * 2.0 / self.panel_resolution
            aid = (int(alpha / resolution) + self.panel_resolution) % self.panel_resolution
            a_list = [(aid + i) % self.panel_resolution for i in range(-range_a, range_a+1)]
            for ele in a_list:
                if not (consider_visited and self.node_state_panel[ele] == self.VISITED):
                    if not (consider_fixed and self.node_state_panel[ele] == self.FIXED):
                        id_neighbor.append(ele)
        return id_neighbor

    def createEntryB(self, string_index = [], confirm_input_id=[]):
        border_list = self.getBorderIndex(consider_visited=True, consider_fixed=True)

        return_bid = []
        return_string_index = string_index
        return_mv_list = []
        return_pl_location = []
        for index in string_index:
            if len(return_bid) == 0:
                bid, mv, pl_location = self.calculateUCB(border_list, 'B', [index], [-1] + confirm_input_id + return_bid)
                return_bid.append(bid)
                return_mv_list.append(mv)
                return_pl_location.append(pl_location)
            else:
                new_border_list = list(set(border_list) & set(self.getNeighbor(return_bid[-1])))
                bid, mv, pl_location = self.calculateUCB(new_border_list, 'B', [index], [-1] + confirm_input_id + return_bid)
                return_bid.append(bid)
                return_mv_list.append(mv)
                return_pl_location.append(pl_location)

        # bid_1, mv1, pl_location_1 = self.calculateUCB(border_list, 'B', [0], [-1])
        # # print("[0] " + str(self.ucb_B))
        # bid_2, mv2, pl_location_2 = self.calculateUCB(list(set(border_list) & set(self.getNeighbor(bid_1))), 'B', [1], [-1, bid_1])
        # # print("[1] " + str(self.ucb_B))
        new_return_bid = sorted(return_bid)
        sorted_list = []
        for ele in new_return_bid:
            sorted_list.append(return_bid.index(ele))

        return new_return_bid, return_string_index, return_mv_list, return_pl_location, sorted_list
    
    def createTSAPoint(self, bid_list, string_list):
        unit_center = [self.units[bid_list[i]].getCenter() for i in range(len(bid_list))]
        center_point = sum([np.array(unit_center[i]) for i in range(len(bid_list))]) / len(bid_list)
        dir_vector = np.array(unit_center[1]) - np.array(unit_center[0])
        dir_vector /= np.linalg.norm(dir_vector)
        if abs(dir_vector[X]) < 1e-5:
            if dir_vector[X] >= 0.0:
                dir_vector[X] = 1e-5
            else:
                dir_vector[X] = -1e-5

        all_vertical_values = []
        all_points = []
        for id in range(self.panel_resolution):
            a_point = self.panel_size * np.array([math.cos(2. * math.pi * id / self.panel_resolution), math.sin(2. * math.pi * id / self.panel_resolution)]) + np.array(self.origami_size) / 2.0
            dir2 = a_point - center_point
            value = abs(self.vertical(dir_vector, dir2))
            all_points.append(a_point)
            all_vertical_values.append(value)
        
        min_vertical_values = []
        min_vertical_values_index = []
        for i in range(4):
            min_vertical_value = min(all_vertical_values)
            min_value_index = all_vertical_values.index(min_vertical_value)
            all_vertical_values[min_value_index] = math.inf

            min_vertical_values.append(min_vertical_value)
            min_vertical_values_index.append(min_value_index)

        aid = min_vertical_values_index[0]
        distance = self.distance(center_point, all_points[aid])
        for i in range(1, 4):
            new_aid = min_vertical_values_index[i]
            new_distance = self.distance(center_point, all_points[new_aid])
            if new_distance < distance:
                aid = new_aid
                distance = new_distance

        
        # alpha = math.atan(dir_vector[Y] / dir_vector[X])
        # resolution = math.pi * 2.0 / self.panel_resolution

        # number_of_resolution = int(alpha / resolution)
        # aid_1 = int(number_of_resolution + self.panel_resolution / 4.0)
        # aid_2 = int(number_of_resolution + self.panel_resolution / 4.0 * 3.0) % self.panel_resolution
        # a_1 = self.panel_size * np.array([math.cos(2. * math.pi * aid_1 / self.panel_resolution), math.sin(2. * math.pi * aid_1 / self.panel_resolution)]) + np.array(self.origami_size) / 2.0
        # a_2 = self.panel_size * np.array([math.cos(2. * math.pi * aid_2 / self.panel_resolution), math.sin(2. * math.pi * aid_2 / self.panel_resolution)]) + np.array(self.origami_size) / 2.0

        # aid = aid_1
        # if self.distance(a_1, center_point) > self.distance(a_2, center_point):
        #     aid = aid_2

        a_list = [(aid + i) % self.panel_resolution for i in range(0, 1)]

        aid, mv, pl_location = self.calculateUCB(a_list, 'A', string_id=string_list, input_id=bid_list)
        return aid, mv, pl_location

    def endExploration(self, mv_list):
        end = False
        for ele in mv_list:
            if ele == math.inf:
                end = True
                break
        return end
    
    def endExplorationAllInf(self, mv_list):
        end = True
        for ele in mv_list:
            if ele != math.inf:
                end = False
                break
        return end
    
    def ask(self, batch_size, a_number_limit=4):
        a_number = 0
        self.method.clear()
        self.update_list.clear()

        self.node_state.fill(self.NO_VISITED)
        self.node_state_panel.fill(self.NO_VISITED)

        total_bid_list = []
        total_string_list = []
        total_mv_list = []
        total_mvs = []

        for i in range(self.tsa_number):
            bid_list, string_list, mv_list, pl_location_list, sorted_list = self.createEntryB([2 * i, 2 * i + 1])
            # print(bid_list)
            aid, mv_a, pl_location_a = self.createTSAPoint(deepcopy(bid_list), string_list)

            crease_type = self.connection_matrix[bid_list[0], bid_list[1]]

            self.method.append([('A', aid, crease_type), ('B', bid_list[0], crease_type)])
            self.method.append([('A', aid, crease_type), ('B', bid_list[1], crease_type)])

            self.update_list.append([['A', aid, string_list[0], pl_location_a], [bid_list, 0, 0.0] if mv_a == math.inf else deepcopy(self.values_panel[aid][string_list[0]][pl_location_a])])
            self.update_list.append([['B', bid_list[sorted_list[0]], string_list[0], pl_location_list[0]], [[-1], 0, 0.0] if mv_list[0] == math.inf else deepcopy(self.values[bid_list[sorted_list[0]]][string_list[0]][pl_location_list[0]])])
            self.update_list.append([['B', bid_list[sorted_list[1]], string_list[1], pl_location_list[1]], [[-1, bid_list[sorted_list[0]]], 0, 0.0] if mv_list[1] == math.inf else deepcopy(self.values[bid_list[sorted_list[1]]][string_list[1]][pl_location_list[1]])])

            # symmetric update
            past_list_symmetric = deepcopy(self.values[bid_list[sorted_list[1]]][string_list[0]])
            pl_location_symmetric = self.findPlLocation(past_list_symmetric, [-1])
            self.update_list.append([['B', bid_list[sorted_list[1]], string_list[0], pl_location_symmetric], [[-1], 0, 0.0] if pl_location_symmetric < 0 else deepcopy(past_list_symmetric[pl_location_symmetric])])
            past_list_symmetric = deepcopy(self.values[bid_list[sorted_list[0]]][string_list[1]])
            pl_location_symmetric = self.findPlLocation(past_list_symmetric, [-1, bid_list[sorted_list[1]]])
            self.update_list.append([['B', bid_list[sorted_list[0]], string_list[1], pl_location_symmetric], [[-1, bid_list[sorted_list[1]]], 0, 0.0] if pl_location_symmetric < 0 else deepcopy(past_list_symmetric[pl_location_symmetric])])

            self.valid = self.checkValid(aid, bid_list[0], bid_list[1])

            a_number += 2

            self.node_state[bid_list[0]] = self.VISITED
            self.node_state[bid_list[1]] = self.VISITED

            self.node_state_panel[aid] = self.VISITED

            total_bid_list += bid_list
            total_string_list += string_list
            total_mv_list += mv_list
            total_mvs += (mv_list + [mv_a])
        
        for i in range(self.string_root_number):
            bid_list, string_list, mv_list, pl_location_list, sorted_list = self.createEntryB([2 * self.tsa_number + i])
            # print(bid_list)

            neighbor_list_A = self.getNeighborA(bid_list[0], True, True, 1)
            aid, mv_a, pl_location_a = self.calculateUCB(neighbor_list_A, 'A', [2 * self.tsa_number + i], [-1] + bid_list)

            # to get crease_type
            neighbor_list_B = self.getNeighbor(bid_list[0])
            new_bid, new_mvb, new_pl_location_b = self.calculateUCB(neighbor_list_B, 'B', [2 * self.tsa_number + i], deepcopy(bid_list))
            crease_type = -self.connection_matrix[bid_list[0], new_bid]

            self.method.append([('A', aid, crease_type), ('B', bid_list[0], crease_type)])

            self.update_list.append([['A', aid, string_list[0], pl_location_a], [[-1] + bid_list, 0, 0.0] if mv_a == math.inf else deepcopy(self.values_panel[aid][string_list[0]][pl_location_a])])
            self.update_list.append([['B', bid_list[0], string_list[0], pl_location_list[0]], [[-1], 0, 0.0] if mv_list[0] == math.inf else deepcopy(self.values[bid_list[0]][string_list[0]][pl_location_list[0]])])

            a_number += 1

            self.node_state[bid_list[0]] = self.VISITED

            self.node_state_panel[aid] = self.VISITED

            total_bid_list += bid_list
            total_string_list += string_list
            total_mv_list += mv_list
            total_mvs += (mv_list + [mv_a])

        expl_bid_list = deepcopy(total_bid_list)

        expl_all_string_list = [[total_bid_list[i]] for i in range(len(total_bid_list))]

        # expand method
        perfect_answer = False # 是不是完美结果？

        while not self.endExploration(total_mvs):
            temp_mvs = []
            temp_mv_list = [] # for B
            for i in range(len(total_mv_list)):
                if total_mv_list[i] != math.inf:
                    ucb_B_old_value = total_mv_list[i]
                    current_dir = self.method[total_string_list[i]][-1][-1]
                    input_id = deepcopy(expl_all_string_list[i])
                    last_id = expl_bid_list[i]
                    neighbor_list = self.getNeighbor(last_id, consider_fixed=True, consider_visited=True, consider_reserve=True, dir=-current_dir)
                    if len(neighbor_list):
                        new_bid, new_mv, new_pl_location = self.calculateUCB(neighbor_list, 'B', [total_string_list[i]], input_id)
                        if ucb_B_old_value > new_mv:
                            temp_mv_list.append(math.inf)
                            # temp_mvs.append(ucb_B_old_value)
                            self.node_state[last_id] = self.FIXED
                            continue
                        else:
                            self.method[total_string_list[i]].append(('B', new_bid, -current_dir))
                            self.update_list.append([['B', new_bid, total_string_list[i], new_pl_location], [input_id, 0, 0.0] if new_mv == math.inf else deepcopy(self.values[new_bid][total_string_list[i]][new_pl_location])])
                            self.node_state[new_bid] = self.VISITED
                            
                            # temp_mvs.append(new_mv)
                            temp_mv_list.append(new_mv)

                            expl_bid_list[i] = new_bid
                            expl_all_string_list[i].append(new_bid)
                    else:
                        neighbor_list = self.getNeighborA(last_id, consider_fixed=True, consider_visited=True)
                        if len(neighbor_list):
                            new_aid, new_mv, new_pl_location = self.calculateUCB(neighbor_list, 'A', [total_string_list[i]], input_id)
                            if ucb_B_old_value > new_mv:
                                temp_mv_list.append(math.inf)
                                self.node_state[last_id] = self.FIXED
                                # temp_mvs.append(ucb_B_old_value)
                                continue
                            else:
                                self.method[total_string_list[i]].append(('A', new_aid, -current_dir))
                                self.update_list.append([['A', new_aid, total_string_list[i], new_pl_location], [input_id, 0, 0.0] if new_mv == math.inf else deepcopy(self.values_panel[new_aid][total_string_list[i]][new_pl_location])])
                                self.node_state_panel[new_aid] = self.FIXED
                                temp_mv_list.append(math.inf)
                                # temp_mvs.append(ucb_B_old_value)
                                a_number += 1
                        else:
                            temp_mv_list.append(math.inf)
                            self.node_state[last_id] = self.FIXED
                            continue
                else:
                    temp_mv_list.append(math.inf)
            total_mvs = deepcopy(temp_mvs)
            total_mv_list = deepcopy(temp_mv_list)
            
            if (not self.endExploration(total_mvs)) and self.endExplorationAllInf(total_mv_list):
                perfect_answer = True
                break

        initial_method = deepcopy(self.method)
        print(initial_method)
        
        if (not perfect_answer) or (perfect_answer and a_number < a_number_limit):
            # simulated method
            methods = []
            current_string_index = 0
            backup_node_state = deepcopy(self.node_state)
            backup_node_state_panel = deepcopy(self.node_state_panel)
            
            pointer = 0

            while (len(methods) < batch_size) and pointer < batch_size * 8:
                new_a_number = a_number

                temp_method = deepcopy(self.method)
                self.node_state = deepcopy(backup_node_state)
                self.node_state_panel = deepcopy(backup_node_state_panel)

                new_blood = False
                stop_generation = [False for _ in range(self.total_string_number)]
                while not all(stop_generation):
                    if not stop_generation[current_string_index]:
                        past_point_type = temp_method[current_string_index][-1][0]
                        past_point_id = temp_method[current_string_index][-1][1]
                        past_dir = temp_method[current_string_index][-1][2]

                        if past_point_type == 'A':
                            stop_generation[current_string_index] = True
                            current_string_index = (current_string_index + 1) % self.total_string_number
                            continue

                        if (past_point_type == 'B') and (self.node_state[past_point_id] == self.FIXED):
                            stop_generation[current_string_index] = True
                            current_string_index = (current_string_index + 1) % self.total_string_number
                            continue

                        new_dir = -past_dir

                        neighbor_list = self.getNeighbor(past_point_id, consider_visited=True, consider_fixed=True, consider_reserve=True, dir=new_dir)
                        neighbor_list_A = self.getNeighborA(past_point_id, consider_visited=True, consider_fixed=True)

                        random_action = np.random.random()
                        
                        if random_action > ((len(neighbor_list) + 1.) / (len(neighbor_list) + 2.) + 1.) / 2.:
                            stop_generation[current_string_index] = True
                            current_string_index = (current_string_index + 1) % self.total_string_number
                            new_blood = True

                        elif random_action < 1. / (len(neighbor_list) + 2.) and len(neighbor_list_A): #find A
                            new_aid = np.random.choice(np.array(neighbor_list_A), 1, replace=False)
                            temp_method[current_string_index].append(('A', new_aid[0], new_dir))
                            self.node_state_panel[new_aid] = self.FIXED
                            new_blood = True
                            stop_generation[current_string_index] = True

                            new_a_number += 1
                        else:
                            if len(neighbor_list):
                                new_bid = np.random.choice(np.array(neighbor_list), 1, replace=False)
                                temp_method[current_string_index].append(('B', new_bid[0], new_dir))
                                self.node_state[new_bid] = self.VISITED
                                new_blood = True
                            else:
                                stop_generation[current_string_index] = True
                                current_string_index = (current_string_index + 1) % self.total_string_number
                                new_blood = True

                    current_string_index = (current_string_index + 1) % self.total_string_number
                
                if new_a_number >= a_number_limit:
                    methods.append(temp_method)
                if not new_blood:
                    break
        else:
            methods = [deepcopy(self.method)]
        if len(methods) == 0:
            methods = [deepcopy(self.method)]
        return methods, initial_method
    
    def tell(self, reward_list):
        batch_size = len(reward_list)
        for ele in self.update_list: # ['A', aid, string_list[0], pl_location_a], [bid_list, 0, 0.0]
            infos = ele[0]
            values = ele[1]
            if infos[0] == 'A':
                aid = infos[1]
                string_index = infos[2]
                pl_location = infos[3]

                values[1] += 1
                values[2] += sum(reward_list) / batch_size

                self.n_panel[string_index] += 1

                if pl_location != -1:
                    self.values_panel[aid][string_index][pl_location] = deepcopy(values)
                else:
                    self.values_panel[aid][string_index].append(deepcopy(values))
            else:
                bid = infos[1]
                string_index = infos[2]
                pl_location = infos[3]

                values[1] += 1
                values[2] += sum(reward_list) / batch_size

                self.n[string_index] += 1

                if pl_location != -1:
                    self.values[bid][string_index][pl_location] = deepcopy(values)
                else:
                    self.values[bid][string_index].append(deepcopy(values))

        for i in range(len(self.values)):
            print(self.values[i])

if __name__ == "__main__":
    gm = GraphModel([1], None, None)
    # tm = TransitionModel()
    # tm.setSource([
    #     [30, 0.7, 0.3],
    #     [25, 1.1, 0.72],
    #     [35, 0.6, 0.41],
    #     [28, 0.95, 0.13],
    #     [23, 1.05, 0.95],
    # ])
    # all_ef, _ = tm.getAllEndEffector()

    # cfh = CurveFittingHelper()
    # cfh.setGoalList(deepcopy(all_ef))

    # tm.setSource([
    #     [
    #         40.0,
    #         1.1350118428589586,
    #         0.0
    #     ],
    #     [
    #         40.0,
    #         0.4391013886514122,
    #         1.0
    #     ],
    #     [
    #         24.460848516379517,
    #         0.4363323129985824,
    #         0.27241904571311837
    #     ],
    #     [
    #         31.058997782191327,
    #         0.9847232585353759,
    #         0.7831427751802336
    #     ],
    #     [
    #         40.0,
    #         1.2033803321402874,
    #         0.21439431807441559
    #     ]
    # ])
    # all_ef, _ = tm.getAllEndEffector()

    # cfh.setOriginList(deepcopy(all_ef))
    # acc1 = cfh.strictMatch()
    # acc2 = cfh.distanceMatch()
    # print(str(acc1 * acc2))
    # cfh = CurveFittingHelper()
    # x = np.linspace(-1, 1, 180)
    # y1 = -x ** 2
    # y2 = x ** 2

    # cfh.setGoalList([[x[i], y1[i]] for i in range(x.shape[0])])
    # cfh.setOriginList([[x[i], y2[i]] for i in range(x.shape[0])])

    # p = cfh.bestMatch()

    # print(str(100*p) + "%")

    # # ga = GA()
    # # # ga.mode("cmaes")
    # # ga.initialize(10, 10)
    # # ga.setMapper([
    # #     [i - 1, 2 * i] for i in range(1, 11)
    # # ])

    # # fitness_result = []

    # # while(ga.iteration < 100000):
    # #     fitness_result.clear()
    # #     data = ga.ask()
    # #     for pack in data:
    # #         k_data = pack.data
    # #         fitness_result.append(sum([(1 - (k_data[i] - 1.5 * (i + 1)) ** 2) for i in range(10)]))
    # #     ga.evaluate(fitness_result)
    # #     score = ga.getCurrentBest()
    # #     print("ITER: " + str(ga.iteration) + "    Current Best Score: " + str(np.round(score, 6)) + " / " + "10.0000")
    # #     # if ga.iteration % 5000 == 0:
    # #     #     ga.updateScale(0.25 * (1 - ga.iteration / 100000.0) ** 2)
    # #     # if ga.iteration == 1000:
    # #     #     ga.changeModeAndContinue("es")
    # t = np.linspace(0, math.pi, 180)
    # x = np.array([180 * math.cos(t[i]) * (2 * math.pi - t[i]) / (2 * math.pi) for i in range(180)])
    # y = np.array([180 * math.sin(t[i]) * (2 * math.pi - t[i]) / (2 * math.pi) for i in range(180)])
    t = np.linspace(0, 200, 180)
    x = 250 - 200 * np.sin(t / 400.0 * np.pi)
    y = (t) ** 2 / 300.0

    t2 = np.linspace(-1.33, -60, 45)
    x2 = t2
    y2 = (60 + 1 / 3 * x2) * np.sin(np.pi / 2 - x2 * np.pi / 120.0)
    # x = [all_ef[i][0] for i in range(len(all_ef))]
    # y = [all_ef[i][1] for i in range(len(all_ef))]
    s = {
        "x": x.tolist(),
        "y": y.tolist(),
    }

    with open('./curve/curve_qua3_3v.json', 'w', encoding="utf-8") as f:
        json.dump(s, f, indent=4)
    # step_min = 5
    # step_max = 5
    # algorithm = 'es'
    # batch_size = 10

    # process_num = 1

    # tm = TransitionModel()

    # update_scale_step = 100

    # output_best_flag = 100

    # change_mode_flag = 100

    # s = {
    #     "origin": [
    #         [0.0, 0.0]
    #     ],
    #     "kl": [
        
    #     ],
    #     "add_width":[
    #         False
    #     ],
    #     "score": 0.0
    # }

    # for i in range(int(step_min), int(step_max + 1)):
    #     if algorithm == 'es':
    #         algo = ES()
    #         algo.mode("cmaes")
    #         algo.initialize(batch_size * process_num, 3 * i)
    #     elif algorithm == 'ga':
    #         algo = GA()
    #         algo.initialize(batch_size * process_num, 3 * i)
    #     mapper = []
    #     for j in range(i):
    #         mapper.append([20, 40])
    #         mapper.append([25 * math.pi / 180.0, 75 * math.pi / 180.0])
    #         mapper.append([0, 1])
    #     algo.setMapper(mapper)
        
    #     while(algo.iteration < 2000):
    #         reward_list = []
    #         # self._emit.emit(((i - step_min) * self.pref_pack["generation"] + algo.iteration) / total_step)
    #         data = algo.ask()

    #         # fitness_array = train(data, i, tm, cfh)
    #         for pack in data:
    #             k_data = pack.data
    #             # Set source and get all_ef
    #             tm.setSource([
    #                 k_data[3*k: 3*k+3] for k in range(i)
    #             ])
    #             all_ef, _ = tm.getAllEndEffector()
                
    #             cfh.setOriginList(deepcopy(all_ef))

    #             p = cfh.strictMatch() * cfh.distanceMatch()

    #             reward_list.append(p)

    #         algo.evaluate(reward_list)
    #         # algo.evaluate(reward_list)
    #         score = algo.getCurrentBest()
    #         print("ITER: " + str(algo.iteration) + "    Current Best Score: " + str(np.round(score, 6)) + " / " + "1.0000")
    #         if algorithm == "es":
    #             if algo.iteration % update_scale_step == 0:
    #                 algo.updateScale(0.25 * (1 - algo.iteration / 5000) ** 2)
                
    #             if algo.iteration == change_mode_flag:
    #                 algo.changeModeAndContinue("es")

    #             if algo.iteration % output_best_flag == 0:
    #                 design_result = deepcopy(algo.storage[-1])
    #                 kl = [
    #                     [0.1, 0]
    #                 ]
    #                 data_result = []
    #                 alpha = 0.0
    #                 for k in range(i):
    #                     data = design_result.data[3*k: 3*k+3] #length, angle and up/down
    #                     length = data[0] * (20) + 20
    #                     angle = (data[1] * (50) + 25) * math.pi / 180.0
    #                     if data[2] < 0.5:
    #                         alpha -= angle * 2
    #                     else:
    #                         alpha += angle * 2
    #                     kl.append([length, alpha])
    #                     data_result.append([length, angle, data[2]])
    #                 s["kl"] = [kl]
    #                 s["score"] = design_result.score
    #                 with open('./cdfResult/row' + str(i) + '.json', 'w', encoding="utf-8") as f:
    #                     json.dump(s, f, indent=4)
    #                 data_json = {
    #                     "data": data_result
    #                 }
    #                 with open('./cdfResult/data' + str(i) + '.json', 'w', encoding="utf-8") as f:
    #                     json.dump(data_json, f, indent=4)


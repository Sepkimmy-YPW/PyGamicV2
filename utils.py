import math
import numpy as np
from copy import deepcopy

VALLEY = 0
MOUNTAIN = 1
BORDER = 2
CUTTING = 3

UP = 1
DOWN = 0
MIDDLE = 2

FOLDING_MAXIMUM = 0.95

X = 0
Y = 1
Z = 2

START = 0
END = 1
LEFT = 0
RIGHT = 1

V = True
ACTIVE_MIURA = 1
PASSIVE_MIURA = 0

EMPTY = 0
HAVE_BORDER = 1
HAVE_CONNECTION = 2

SHOW_COLOR = 1
BLACK_COLOR = 0

LEFT_HALF = 0
RIGHT_HALF = 1
NO_HALF = 2

BOTTOM = 0
PASS = 1
TOP = 2

FREE = 0
FIXED = 1

class Vertex:
    def __init__(self, kp):
        self.kp = kp
        self.dim = len(kp)
        self.dn = 0
        self.connection_index = []
        self.is_border_node = False
        self.coeff_list = []
        self.level_list = []
    
    def getKp(self):
        return self.kp

class Crease:
    def __repr__(self):
        return f"Start: {self.points[START]}, End: {self.points[END]}"
    def __init__(self, start, end, crease_type, show_color_flag=True, hard=False) -> None:
        self.points = [start, end]
        if(show_color_flag):
            self.crease_type = crease_type
        else:
            self.crease_type = BORDER
        self.hard = hard
        self.hard_angle = math.pi
        self.index = -1
        self.origin_index = -1

        self.start_index = 0
        self.end_index = 0
        self.level = 0
        self.coeff = 1.0

        self.visited = False
        self.undefined = False
        self.recover_level = 0

        self.sign = "equal"
    
    def setHard(self, hard: bool):
        self.hard = hard

    def getType(self):
        return self.crease_type

    def setIndex(self, index):
        self.index = index

    def setOriginIndex(self, index):
        self.origin_index = index

    def getReverse(self):
        new = deepcopy(self)
        new.points.reverse()
        return new
    
    def getMidPoint(self):
        return [(self.points[START][X] + self.points[END][X]) / 2, (self.points[START][Y] + self.points[END][Y]) / 2]

    def getDirection(self):
        return [(self.points[END][X] - self.points[START][X]) / distance(self.points[START], self.points[END]), (self.points[END][Y] - self.points[START][Y]) / distance(self.points[START], self.points[END])]

    def getNormal(self):
        return [(self.points[END][Y] - self.points[START][Y]) / distance(self.points[START], self.points[END]), -(self.points[END][X] - self.points[START][X]) / distance(self.points[START], self.points[END])]
    
    def getPercentPoint(self, percent):
        return [
            percent * (self.points[END][X] - self.points[START][X]) + self.points[START][X], 
            percent * (self.points[END][Y] - self.points[START][Y]) + self.points[START][Y], 
        ]

    def k(self):
        if self.points[END][X] != self.points[START][X]:
            k = (self.points[END][Y] - self.points[START][Y]) / (self.points[END][X] - self.points[START][X])
            return k
        else:
            return math.inf

    def b(self):
        if self.points[START][X] != self.points[END][X]:
            b = (self.points[END][X] * self.points[START][Y] - self.points[START][X] * self.points[END][Y]) / (self.points[END][X] - self.points[START][X])
            return b
        else:
            return self.points[END][X]

    def getPointAxis(self):
        start = self.points[START]
        end = self.points[END]
        # if start[0] <= end[0]:
        return start[X], start[Y], end[X], end[Y]
        # else:
        #     return end[0], end[1], start[0], start[1]
    
    def getLength(self):
        return distance(self.points[START], self.points[END])
    
    def pointIsStartAndEnd(self, p1, p2):
        if (distance(p1, self.points[START]) < 1e-5 and distance(p2, self.points[END]) < 1e-5):
            return True
        else:
            if (distance(p1, self.points[END]) < 1e-5 and distance(p2, self.points[START]) < 1e-5):
                return True
            else:
                return False

    def __getitem__(self, index):
        return self.points[index]

def calculatePercent(c1: Crease, c2: Crease):
    try:
        k1 = c1.k()
        k2 = c2.k()
        b1 = c1.b()
        b2 = c2.b()
        x11, y11, x21, y21 = c1.getPointAxis()
        x12, y12, x22, y22 = c2.getPointAxis()
        if(k1 != math.inf and k2 != math.inf):
            if(k1 == k2):
                return None
            x0 = (b2 - b1) / (k1 - k2)
            y0 = (k1 * b2 - k2 * b1) / (k1 - k2)
            cross_percent = (x0 - x11) / (x21 - x11)
            percent_list = [
                [
                    0.0, (x11 - x12) / (x22 - x12)
                ],
                [
                    1.0, (x21 - x12) / (x22 - x12)
                ],
                [
                    (x12 - x11) / (x21 - x11), 0.0
                ],
                [
                    (x22 - x11) / (x21 - x11), 1.0
                ],
                [
                    cross_percent, (x0 - x12) / (x22 - x12)
                ]
            ]
            if cross_percent >= 0.0 and cross_percent <= 1.0:
                return percent_list
            else:
                return None
        elif (k1 == math.inf and k2 == math.inf):
            return None
        elif (k1 == math.inf):
            x0 = x11
            y0 = y12 + (x0 - x12) / (x22 - x12) * (y22 - y12)
            cross_percent = (x0 - x12) / (x22 - x12)
            percent_list = [
                [
                    0.0, (x11 - x12) / (x22 - x12)
                ],
                [
                    1.0, (x21 - x12) / (x22 - x12)
                ],
                [
                    None, 0.0
                ],
                [
                    None, 1.0
                ],
                [
                    None, cross_percent
                ]
            ]
            if cross_percent >= 0.0 and cross_percent <= 1.0:
                return percent_list
            else:
                return None
        else:
            x0 = x12
            y0 = y11 + (x0 - x11) / (x21 - x11) * (y21 - y11)
            cross_percent = (x0 - x11) / (x21 - x11)
            percent_list = [
                [
                    0.0, None
                ],
                [
                    1.0, None
                ],
                [
                    (x12 - x11) / (x21 - x11), 0.0
                ],
                [
                    (x22 - x11) / (x21 - x11), 1.0
                ],
                [
                    cross_percent, None
                ]
            ]
            if cross_percent >= 0.0 and cross_percent <= 1.0:
                return percent_list
            else:
                return None
    except:
        return None

def percent_limit(x):
    try:
        if x > 1.0:
            return 1.0
        elif x < 0.0:
            return 0.0
        else:
            return x
    except:
        return x

def limit(data, down, upper, matchtype="any", strict=False, tolerance_expand=0.0) -> bool:
    flag = True
    truly_down = down - tolerance_expand
    truly_upper = upper + tolerance_expand
    # If any data is in the margin, funtion will return true
    if matchtype == "any":
        if strict:
            for ele in data:
                if ele > truly_down and ele < truly_upper:
                    return True
            flag = False
        else:
            for ele in data:
                if ele >= truly_down and ele <= truly_upper:
                    return True
            flag = False
    # If all data is in the margin, funtion will return true
    else:
        if strict:
            for ele in data:
                if not (ele > truly_down and ele < truly_upper):
                    return False
        else:
            for ele in data:
                if not (ele >= truly_down and ele <= truly_upper):
                    return False
    return flag

def calculateIntersectionWithinCrease(c1: Crease, c2: Crease, strict_flag = False):
    try:
        k1 = c1.k()
        k2 = c2.k()
        b1 = c1.b()
        b2 = c2.b()
        x11, y11, x21, y21 = c1.getPointAxis()
        x12, y12, x22, y22 = c2.getPointAxis()
        if(k1 != math.inf and k2 != math.inf):
            if(k1 == k2):
                return None
            x0 = (b2 - b1) / (k1 - k2)
            y0 = (k1 * b2 - k2 * b1) / (k1 - k2)
            cross_percent_list = []
            index_1 = [0, -1]
            index_2 = [0, -1]
            if abs(x11 - x21) >= 1e-5:
                cross_percent_list.append((x0 - x11) / (x21 - x11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(y11 - y21) >= 1e-5:
                cross_percent_list.append((y0 - y11) / (y21 - y11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(x12 - x22) >= 1e-5:
                cross_percent_list.append((x0 - x12) / (x22 - x12))
                index_2[END] += 1
            if abs(y12 - y22) >= 1e-5:
                cross_percent_list.append((y0 - y12) / (y22 - y12))
                index_2[END] += 1
            intersection = [x0, y0]
            if strict_flag:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="any", strict=True, tolerance_expand=-1e-5):
                    if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                        return intersection
            else:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                    return intersection
        elif (k1 == math.inf and k2 == math.inf):
            if abs(b1 - b2) < 1e-5:
                if strict_flag:
                    y1 = c1[START][Y]
                    y2 = c1[END][Y]
                    if y1 < y2:
                        y3 = c2[START][Y]
                        y4 = c2[END][Y]
                        if (y3 > y1 and y3 < y2) or (y4 > y1 and y4 < y2):
                            return [b1, 0]
                        else:
                            return None
                    if y2 < y1:
                        y3 = c2[START][Y]
                        y4 = c2[END][Y]
                        if (y3 > y2 and y3 < y1) or (y4 > y2 and y4 < y1):
                            return [b1, 0]
                        else:
                            return None
                else:
                    return [b1, 0]
            else:
                return None
        elif (k1 == math.inf):
            x0 = x11
            y0 = y12 + (x0 - x12) / (x22 - x12) * (y22 - y12)
            cross_percent_list = []
            index_1 = [0, -1]
            index_2 = [0, -1]
            if abs(x11 - x21) >= 1e-5:
                cross_percent_list.append((x0 - x11) / (x21 - x11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(y11 - y21) >= 1e-5:
                cross_percent_list.append((y0 - y11) / (y21 - y11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(x12 - x22) >= 1e-5:
                cross_percent_list.append((x0 - x12) / (x22 - x12))
                index_2[END] += 1
            if abs(y12 - y22) >= 1e-5:
                cross_percent_list.append((y0 - y12) / (y22 - y12))
                index_2[END] += 1
            intersection = [x0, y0]
            if strict_flag:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="any", strict=True, tolerance_expand=-1e-5):
                    if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                        return intersection
            else:
                if limit(cross_percent_list[index_2[START]:index_2[END]], 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                    return intersection
        else:
            x0 = x12
            y0 = y11 + (x0 - x11) / (x21 - x11) * (y21 - y11)
            cross_percent_list = []
            index_1 = [0, -1]
            index_2 = [0, -1]
            if abs(x11 - x21) >= 1e-5:
                cross_percent_list.append((x0 - x11) / (x21 - x11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(y11 - y21) >= 1e-5:
                cross_percent_list.append((y0 - y11) / (y21 - y11))
                index_1[END] += 1
                index_2[START] += 1
                index_2[END] += 1
            if abs(x12 - x22) >= 1e-5:
                cross_percent_list.append((x0 - x12) / (x22 - x12))
                index_2[END] += 1
            if abs(y12 - y22) >= 1e-5:
                cross_percent_list.append((y0 - y12) / (y22 - y12))
                index_2[END] += 1
            intersection = [x0, y0]
            if strict_flag:
                if limit(cross_percent_list, 0.0, 1.0, matchtype="any", strict=True, tolerance_expand=-1e-5):
                    if limit(cross_percent_list, 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                        return intersection
            else:
                if limit(cross_percent_list[index_1[START]:index_1[END]], 0.0, 1.0, matchtype="all", strict=False, tolerance_expand=1e-5):
                    return intersection
        return None
    except:
        return None

def distance(p1, p2):
    return math.sqrt((p1[X] - p2[X]) ** 2 + (p1[Y] - p2[Y]) ** 2)

def pointToCrease(p, crease: Crease):
    k = crease.k()
    b = crease.b()
    if k == math.inf or k == -math.inf:
        return abs(b - p[X])
    else:
        return abs(k * p[X] - p[Y] + b) / math.sqrt(k ** 2 + 1)
    
def calculateIntersection(kb1, kb2):
    if kb1[0] == kb2[0] or (kb1[0] == math.inf and kb2[0] == math.inf):
        return None
    elif kb1[0] == math.inf:
        return [
            kb1[1],
            kb2[0] * kb1[1] + kb2[1]
        ]
    elif kb2[0] == math.inf:
        return [
            kb2[1],
            kb1[0] * kb2[1] + kb1[1]
        ]
    else:
        return [
            (kb1[1] - kb2[1]) / (kb2[0] - kb1[0]),
            (kb1[1] * kb2[0] - kb2[1] * kb1[0]) / (kb2[0] - kb1[0]),
        ]

def norm(vec: list):
    norm = 0
    for ele in vec:
        norm += ele ** 2
    return math.sqrt(norm)

def R(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

def calculateCenterPoint(points):
    x = 0.0
    y = 0.0
    for p in points:
        x += p[X]
        y += p[Y]
    kp_len = len(points)
    x /= kp_len
    y /= kp_len
    return [x, y]

def calculateCenterPoint3D(points):
    x = 0.0
    y = 0.0
    z = 0.0
    for p in points:
        x += p[X]
        y += p[Y]
        z += p[Z]
    kp_len = len(points)
    x /= kp_len
    y /= kp_len
    z /= kp_len
    return [x, y, z]

def pointOnCrease(point, c: Crease, tolerance=1e-5):
    start = c[START]
    end = c[END]
    k = c.k()
    if abs(k) > 1e5:
        percent_x = 0.5
        percent_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
    else:
        percent_x = (point[X] - start[X]) / (end[X] - start[X])
        if abs(k) < 1e-5:
            percent_y = 0.5
        else:
            percent_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
    if pointToCrease(point, c) < tolerance and (percent_limit(percent_x) == percent_x and percent_limit(percent_y) == percent_y):
        return True
    else:
        return False
    
    start = c[START]
    end = c[END]
    if abs(end[X] - start[X]) >= 1e-5 and abs(end[Y] - start[Y]) >= 1e-5:
        p_x = (point[X] - start[X]) / (end[X] - start[X])
        p_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
        if abs(p_x - p_y) < tolerance and p_x >= 0.0 and p_x <= 1.0:
            return True
        else:
            return False
    else:
        tolerance *= 10
        if abs(end[X] - start[X]) < tolerance:
            p_y = (point[Y] - start[Y]) / (end[Y] - start[Y])
            if abs(point[X] - start[X]) < tolerance and p_y >= 0.0 and p_y <= 1.0:
                return True
            else:
                return False
        else:
            p_x = (point[X] - start[X]) / (end[X] - start[X])
            if abs(point[Y] - start[Y]) < tolerance and p_x >= 0.0 and p_x <= 1.0:
                return True
            else:
                return False

def packTrajectory(trajectory, P_number):
    dict = {
        "type": [],
        "id": [],
        "reverse": []
    }
    for i in range(len(trajectory)):
        if trajectory[i][0] < P_number:
            dict["type"].append("A")
            dict["id"].append(trajectory[i][0])
        else:
            dict["type"].append("B")
            dict["id"].append(trajectory[i][0] - P_number)
    exist_non_zero_side = 0
    for i in range(len(trajectory)):
        if trajectory[i][1] != 0:
            exist_non_zero_side = trajectory[i][1]
            break
    if exist_non_zero_side:
        for j in range(len(trajectory)):
            if (j - i) % 2 == 0:
                dict["reverse"].append(trajectory[i][1])
            else:
                dict["reverse"].append(-trajectory[i][1])
        dict["reverse"][0] = dict["reverse"][1]
    else:
        for j in range(len(trajectory)):
            if j % 2 == 0:
                dict["reverse"].append(-1)
            else:
                dict["reverse"].append(1)
        dict["reverse"][0] = 1
    return dict

def pointInPolygon(point, polygon: list, return_min_distance=False, upper_x_bound=math.inf, lower_x_bound=-math.inf):
    angle = 0.0
    polygon_resolution = len(polygon)
    h_list = []
    delta_x_lower = point[X] - lower_x_bound
    delta_x_upper = upper_x_bound - point[X]
    for i in range(0, polygon_resolution):
        next_ele = (i + 1) % polygon_resolution
        vec1 = [polygon[i][x] - point[x] for x in range(len(point))]
        vec2 = [polygon[next_ele][x] - point[x] for x in range(len(point))]
        vec3 = [polygon[next_ele][x] - polygon[i][x] for x in range(len(point))]
        upper = sum([vec1[x] * vec2[x] for x in range(len(vec1))])
        norm1 = norm(vec1)
        norm2 = norm(vec2)
        norm3 = norm(vec3)
        divider = norm1 * norm2
        try:
            val = upper / divider
            if abs(val) > 1.0 and abs(val) < 1.0 + 1e-5:
                if val < 0:
                    val = -1.0
                else:
                    val = 1.0
            theta = math.acos(val)
            h_list.append(norm1 * norm2 * math.sin(theta) / norm3)
            angle += theta
        except:
            if return_min_distance:
                return 0.0
            else:
                return True
    if abs(angle - 2 * math.pi) < 1e-5:
        if return_min_distance:
            return min(min(h_list), delta_x_lower, delta_x_upper)
        else:
            return True
    else:
        return False

def generatePolygonByCenter(center, size, resolution):
    points = []
    step = math.pi * 2 / resolution
    for i in range(0, resolution):
        points.append(
            [
                center[0] + math.cos(i * step) * size, 
                center[1] + math.sin(i * step) * size,
                center[2]
            ]
        )
    return points

def calculateAngle(kp, kp_child1, kp_child2):
    vec1 = [kp_child1[0] - kp[0], kp_child1[1] - kp[1]]
    vec2 = [kp_child2[0] - kp[0], kp_child2[1] - kp[1]]
    len1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
    len2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
    dot_multiple_result = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    cos_result = dot_multiple_result / (len1 * len2)
    return math.acos(cos_result)

def samePoint(p1, p2, dim):
    if distance(p1, p2) > 1e-5:
        return False
    return True

def sameCrease(c1: Crease, c2: Crease):
    if abs(c1[END][Y] - c2[END][Y]) < 1e-5 and abs(c1[START][X] - c2[START][X]) < 1e-5 \
          and abs(c1[END][X] - c2[END][X]) < 1e-5 and abs(c1[START][Y] - c2[START][Y]) < 1e-5:
        return True
    else:
        return False

def crossProduct(c1, c2):
    v1 = [c1[END][X] - c1[START][X], c1[END][Y] - c1[START][Y]] 
    v2 = [c2[END][X] - c2[START][X], c2[END][Y] - c2[START][Y]]
    return v1[X] * v2[Y] - v1[Y] * v2[X]

def angleBetweenCreases(c1, c2, from_same_start=True):
    if from_same_start:
        if distance(c1[END], c2[START]) < 1e-5:
            c1 = c1.getReverse()
        elif distance(c1[START], c2[END]) < 1e-5:
            c2 = c2.getReverse()
    v1 = [c1[END][X] - c1[START][X], c1[END][Y] - c1[START][Y]] 
    v2 = [c2[END][X] - c2[START][X], c2[END][Y] - c2[START][Y]]
    sign = v1[X] * v2[Y] - v1[Y] * v2[X]
    angle = (v1[X] * v2[X] + v1[Y] * v2[Y]) / (norm(v1) * norm(v2))
    if angle > 1.0:
        angle = 1.0
    elif angle < -1.0:
        angle = -1.0
    if sign >= 0:
        return math.acos(angle)
    else:
        return -math.acos(angle)

def methodToTotalInformation(method, P_points, O_points):
    total_information = []
    for i in range(len(method["type"])):
        point_list = []
        for j in range(len(method["type"][i])):
            tsa_point = TSAPoint()
            tsa_point.point_type = method["type"][i][j]
            tsa_point.id = method["id"][i][j]
            tsa_point.dir = method["reverse"][i][j]
            if tsa_point.point_type == 'A':
                tsa_point.point = P_points[tsa_point.id]
            else:
                tsa_point.point = O_points[tsa_point.id]
            point_list.append(tsa_point)
        total_information.append(point_list)
    return total_information

def getMaxDistance(kps):
    max_x = max([kps[i][X] for i in range(len(kps))])
    min_x = min([kps[i][X] for i in range(len(kps))])
    max_y = max([kps[i][Y] for i in range(len(kps))])
    min_y = min([kps[i][Y] for i in range(len(kps))])
    return max(max_x - min_x, max_y - min_y), max_x - min_x, max_y - min_y
    
def getTotalBias(units):
    total_x = 0.0
    total_y = 0.0
    count = 0
    for unit in units:
        seq_points = unit.getSeqPoint()
        for p in seq_points:
            total_x += p[X]
            total_y += p[Y]
            count += 1
    return [total_x / count, total_y / count]

class Unit:
    def __init__(self) -> None:
        self.crease = []
        self.seq_point = []
        self.connection = None
        self.connection_number = None
        self.mass = []

    def reset(self):
        self.crease.clear()

    def addCrease(self, c: Crease):
        self.crease.append(c)

    def repair(self, tolerance = 2.):
        while 1:
            problem = False
            i = 0
            while i < len(self.crease):
                crease = self.crease[i]
                length = crease.getLength()
                if length < tolerance:
                    problem = True
                    mid_point = crease.getMidPoint()
                    self.crease[i - 1].points[END] = mid_point
                    self.crease[(i + 1) % len(self.crease)].points[START] = mid_point
                    del(self.crease[i])
                i += 1
            if not problem:
                break

    def setupMass(self, rho):
        kp_number = len(self.getSeqPoint())
        self.mass = [0. for _ in range(kp_number)]
        new_rho = rho / float(kp_number - 2)
        for i in range(kp_number - 2):
            for j in range(i + 1, i + 1 + kp_number - 2):
                current = j % kp_number
                next = (j + 1) % kp_number
                area = self.calculateTriArea(i, current, next)
                self.mass[i] += new_rho * area / 3.0
                self.mass[current] += new_rho * area / 3.0
                self.mass[next] += new_rho * area / 3.0

    
    def setConnection(self, con):
        self.connection = con

    def setConnectionNumber(self, num):
        self.connection_number = num

    def calculateTriArea(self, i0, i1, i2):
        kps = self.getSeqPoint()
        x0 = kps[i0]
        x1 = kps[i1]
        x2 = kps[i2]
        vec1 = [x1[X] - x0[X], x1[Y] - x0[Y]]
        vec2 = [x2[X] - x0[X], x2[Y] - x0[Y]]
        return abs(0.5 * (vec1[X] * vec2[Y] - vec1[Y] * vec2[X]))
    
    def calculateArea(self):
        kps = self.getSeqPoint()
        start_point = kps[0]
        area = 0.0
        for i in range(1, len(kps) - 1):
            cur_index = i
            next_index = i + 1
            vec1 = [kps[cur_index][X] - start_point[X], kps[cur_index][Y] - start_point[Y]]
            vec2 = [kps[next_index][X] - start_point[X], kps[next_index][Y] - start_point[Y]]
            area += abs(0.5 * (vec1[X] * vec2[Y] - vec1[Y] * vec2[X]))
        return area

    def isBorder(self):
        is_border = False
        for crease in self.crease:
            if crease.getType() == BORDER:
                is_border = True
                break
        return is_border
    
    def getCrease(self):
        return self.crease
    
    def getBorderCrease(self):
        creases = []
        for crease in self.crease:
            if crease.getType() == BORDER:
                creases.append(deepcopy(crease))
        return creases
    
    def getMaxX(self):
        max_x = max([ele[X] for ele in self.seq_point])
        return max_x

    def getMaxY(self):
        max_y = max([ele[Y] for ele in self.seq_point])
        return max_y

    def getMinX(self):
        min_x = min([ele[X] for ele in self.seq_point])
        return min_x

    def getMinY(self):
        min_y = min([ele[Y] for ele in self.seq_point])
        return min_y
    
    def getAABB(self):
        min_x = self.getMinX()
        min_y = self.getMinY()
        max_x = self.getMaxX()
        max_y = self.getMaxY()
        return [[min_x, min_y], [max_x, max_y]]

    def getSeqPoint(self):
        self.seq_point.clear()
        for ele in self.crease:
            self.seq_point.append([ele[0][0], ele[0][1], 0])
        return self.seq_point
    
    def getCenter(self):
        seq_point = np.array(self.getSeqPoint())
        kp_len = len(seq_point)
        if kp_len == 3:
            return (seq_point[0] + seq_point[1] + seq_point[2]) / 3
        else:
            center_tri = [(seq_point[0] + seq_point[i] + seq_point[i + 1]) / 3 for i in range(1, kp_len - 1)]
            area_tri = [
                abs(0.5 * ((seq_point[i][X] - seq_point[0][X]) * (seq_point[i + 1][Y] - seq_point[0][Y]) - (seq_point[i][Y] - seq_point[0][Y]) * (seq_point[i + 1][X] - seq_point[0][X]))) for i in range(1, kp_len - 1)
            ]

            center = sum([center_tri[k] * area_tri[k] for k in range(kp_len - 2)]) / sum(area_tri)
            return center

class TSAPoint:
    def __init__(self) -> None:
        self.point = np.array([0.0, 0.0, 0.0])
        self.point_type = 'A' # A: border, B: origami
        self.id = 0
        self.dir = 1 # 1 means up to down, -1 means down to up

class TSAString:
    def __init__(self) -> None:
        self.width = 1.0
        self.height = 1.2
        self.start_point = np.array([0.0, 0.0, 0.0])
        self.end_point = np.array([0.0, 0.0, 0.0])
        self.type = BOTTOM
        self.start_type = FREE
        self.end_type = FREE
        self.ids = [] # start, end
        self.id_types = [] # start, end
        self.id = -1
    
    def setStringWidth(self, width: float):
        self.width = width

    def setStringHeight(self, height: float):
        self.height = height

    def setStringKeyPoint(self, start, end):
        self.start_point = np.array(start)
        self.end_point = np.array(end)

    def setStartType(self, type):
        self.start_type = type

    def setEndType(self, type):
        self.end_type = type

    def getDirectionVector(self):
        return self.end_point - self.start_point

    def generatePointWithResolution(self, resolution):
        vec = self.getDirectionVector()
        vec_length = np.linalg.norm(vec)
        if self.type != PASS:
            angle = np.arctan(vec[Y] / vec[X])
            if vec[X] < 0.0:
                angle += math.pi
            face_angle = angle - math.pi / 2.0
        start_point = self.start_point
        point_list = []
        # if end_point[Y] == start_point[Y]:
        #     new_first_point = np.array([start_point[X], start_point[Y] + self.width / 2.0, start_point[Z]])
        # else:
        #     K = (end_point[Z] - start_point[Z]) / (end_point[Y] - start_point[Y])
        #     new_first_point = np.array([start_point[X], start_point[Y] + self.width / 2.0 * K/(np.sqrt(K^2 + 1)), start_point[Z] + self.width / 2.0 /np.sqrt(K^2 + 1)])
        # point_list.append(new_first_point)
        if self.type != PASS: # K=0
            for i in range(resolution):
                point_list.append([start_point[X] + self.width / 2.0 * np.cos(2.0 * i / resolution * np.pi) * np.cos(face_angle) - vec[X]/vec_length*self.width / 2.0, 
                                            start_point[Y] + self.width / 2.0 * np.cos(2.0 * i / resolution * np.pi) * np.sin(face_angle) - vec[Y]/vec_length*self.width / 2.0,
                                            start_point[Z] + self.width / 2.0 * np.sin(2.0 * i / resolution * np.pi)])
        else:
            for i in range(resolution):
                point_list.append([start_point[X] + self.width / 2.0 * np.cos(-2.0 * i / resolution * np.pi), 
                                            start_point[Y] + self.width / 2.0 * np.sin(-2.0 * i / resolution * np.pi),
                                            start_point[Z] - self.width / 2.0])
        upper_point_list = []
        for ele in point_list:
            upper_point_list.append(ele + vec + vec/vec_length*self.width)

        return point_list, upper_point_list

class Node:
    def __init__(self, state, maximum_child, done, action_id, parent=None) -> None:
        # xn + sqrt(2ln(N) / v)
        self.state = np.array(state, dtype=np.uint8)
        self.visits = 1
        self.reward = 0.0
        # self.state = state
        self.maximum_child = maximum_child
        self.action_id = action_id
        self.done = done
        self.children = []
        self.parent = parent

    def initializeTree(self):
        self.visits = 1
        self.reward = 0.0
        if not self.done:
            for c in self.children:
                c.initializeTree()

    def addChild(self, child_state, maximum_child, done, action_id):
        self.children.append(Node(child_state, maximum_child, done, action_id, self))
    
    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fullyExpanded(self):
        return len(self.children) == self.maximum_child

    def existBestChild(self, standard):
        exist = False
        for c in self.children:
            if c.done and c.reward <= standard:
                continue
            elif c.done and c.visits > 1 and c.reward / (c.visits - 1) <= standard:
                continue
            elif c.children == None:
                continue
            else:
                exist = True
                break
        return exist

    def existChildWithAction(self, action_id):
        exist = False
        child = None
        for c in self.children:
            if c.action_id == action_id:
                exist = True
                child = c
                break
        return exist, child
    
    def bestChild(self, scalar, standard):
        best_score = -math.inf
        best_children = []
        for c in self.children:
            if c.done and c.reward <= standard:
                continue
            elif c.done and c.visits > 1 and c.reward / (c.visits - 1) <= standard:
                continue
            elif c.children == None:
                continue
            if c.visits != 1:
                exploit = c.reward / (c.visits - 1)
            else:
                exploit = 0.
            explore = math.sqrt(2. * math.log(self.visits) / float(c.visits))
            score = exploit + scalar * explore
            if score == best_score:
                best_children.append(c)
            elif score > best_score:
                best_children = [c]
                best_score = score
        if len(best_children) == 0:
            raise NotImplementedError
        best_child = np.random.choice(best_children)
        action_id = best_child.action_id
        return best_child, action_id

    def __repr__(self) -> str:
        # return f"Visits: {self.visits}, Reward: {self.reward}, Done: {self.done}"
        if self.parent == None:
            return f"{self.state}"
        else:
            if self.visits == 1:
                return f"{self.state}, 0"
            else:
                return f"{self.state}, {self.reward / (self.visits - 1) + math.sqrt(2. * math.log(self.parent.visits) / float(self.visits))}"
    
if __name__ == "__main__":
    # tsa = TSAString()
    # tsa.setStringKeyPoint([0.0, 0.0, 0.0], [10.0, 10.0, 0])
    # a, b = tsa.generatePointWithResolution(4)
    # c = 1
    unit = Unit()
    unit.addCrease(Crease([0., 0.], [10., 0.], BORDER))
    unit.addCrease(Crease([10., 0.], [0., 10.], BORDER))
    unit.addCrease(Crease([0., 10.], [0., 0.], BORDER))
    center = unit.getCenter()
    unit.setupMass(1)
    a = 1
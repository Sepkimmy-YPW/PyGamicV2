# import numpy as np
import math
from operator import eq
from copy import deepcopy

from utils import *

class StlMaker:
    def __init__(self) -> None:
        # unit plane
        self.unit_list = []
        # unit bias list
        self.unit_bias_list = []
        # modified unit plane
        self.modified_unit_list = []
        # triangle
        self.tri_list = []
        # crease plane
        self.valid_crease_list = []
        # triangle of crease
        self.crease_tri_list = []
        # crease after bias
        self.additional_crease_list = []
        # board triangle list
        self.board_tri_list = []
        # hole of unit 
        self.unit_hole_list = []
        # hole of connection
        self.connection_hole_list = []
        # stiffness of crease
        self.hard_crease_index = []
        # string list
        self.string_list = []
        self.string_tri_list = []
        # pillar list
        self.pillar_unit_list = []
        self.pillar_tri_list = []

        self.unit_hole_size = 0.1
        self.unit_hole_resolution = 3
        self.symmetry_flag = False

        self.connection_hole_size = 1.25

        self.string_width = 2.0

        self.min_bias = 0.12

        self.height = 2.0
        self.crease_height = 0.2
        self.bias = 3.0
        self.board_height = 0.2
        self.method = "upper_bias"
        self.hole_width_size_percent = 0.5
        self.hole_length_size_percent = 0.8
        self.s = ''
        self.print_accuracy = 0.2
        self.db_enable = False

        self.thin_mode = False

        self.layer = 3

        self.unit_height = 2.0

        self.using_modified_unit = False

        self.enable_difference = 0 # 0, 1, 2

        self.asym = False

        self.only_two_sides = False

    def clearValidCrease(self):
        self.valid_crease_list.clear()

    def clear(self):
        self.unit_list.clear()
        self.modified_unit_list.clear()
        self.tri_list.clear()
        self.board_tri_list.clear()
        self.unit_hole_list.clear()
        self.connection_hole_list.clear()
        self.string_list.clear()
        self.string_tri_list.clear()
        self.pillar_tri_list.clear()
        self.pillar_unit_list.clear()

    def clearCrease(self):
        self.valid_crease_list.clear()
        self.crease_tri_list.clear()
        self.additional_crease_list.clear()

    def clearAdditionalCrease(self):
        self.additional_crease_list.clear()

    def setPrintAccuracy(self, acc):
        self.print_accuracy = acc
        self.base_inner_bias = 4. * self.print_accuracy
        
    def setThinMode(self, mode):
        self.thin_mode = mode

    def setHeight(self, height):
        self.height = height

    def setBias(self, bias):
        self.bias = bias
        
    def setAsym(self, asym):
        self.asym = asym

    def setOnlyTwoSides(self, flag):
        self.only_two_sides = flag

    def setLayerOfCrease(self, layer):
        self.layer = layer

    def setBoardHeight(self, height):
        self.board_height = height

    def setMethod(self, method):
        self.method = method
    
    def setHoleWidth(self, hole_width):
        self.hole_width_size_percent = hole_width

    def setHoleLength(self, hole_length):
        self.hole_length_size_percent = hole_length

    def setUnitHoles(self, hole_kps):
        self.unit_hole_list = hole_kps

    def setConnectionHoles(self, hole_kps):
        self.connection_hole_list = hole_kps

    def setUnitHoleSize(self, size):
        self.unit_hole_size = size

    def setConnectionHoleSize(self, size):
        self.connection_hole_size = size

    def setUnitHoleResolution(self, resolution):
        self.unit_hole_resolution = resolution

    def setDbEnable(self, flag):
        self.db_enable = flag

    def setHardCrease(self, hard_crease_index):
        self.hard_crease_index = hard_crease_index

    def setStringWidth(self, width):
        self.string_width = width

    def setUnitBias(self, bias_list):
        self.unit_bias_list = bias_list

    def addOrigamiUnit(self, crease_list: list):
        unit = Unit()
        for crease in crease_list:
            unit.addCrease(crease)
        self.unit_list.append(unit)
        self.tri_list.append(None)
    
    def enableUsingModifiedUnit(self):
        self.using_modified_unit = True

    def disableUsingModifiedUnit(self):
        self.using_modified_unit = False

    def addPackedOrigamiUnit(self, unit: Unit):
        self.unit_list.append(unit)
        self.tri_list.append(None)

    def addPackedOrigamiModifiedUnit(self, unit: Unit):
        self.modified_unit_list.append(unit)
        # share the same tri_list with normal unit

    def addValidCreases(self, creases: list):
        for i in range(len(creases)):
            crease = creases[i]
            crease_type = crease.getType()
            if not (crease[END][Y] == crease[START][Y] and crease[END][X] == crease[START][X]) and (crease_type == VALLEY or crease_type == MOUNTAIN):
                same_crease_flag = False
                for ele in self.valid_crease_list:
                    if sameCrease(ele, crease):
                        same_crease_flag = True
                        break
                if not same_crease_flag:
                    crease.setOriginIndex(i)
                    crease.setIndex(len(self.valid_crease_list))
                    self.valid_crease_list.append(crease)
                    self.crease_tri_list.append(None)

    def size(self):
        return len(self.unit_list)

    def validCreaseSize(self):
        return len(self.valid_crease_list)
    
    def getTriangle(self, v1, v2, v3):
        vex1 = v1
        vex2 = v2
        vex3 = v3
        vector1 = [vex2[i] - vex1[i] for i in range(0, 3)]
        vector2 = [vex3[i] - vex1[i] for i in range(0, 3)]
        normal = [
            vector1[Y] * vector2[Z] - vector2[Y] * vector1[Z],
            vector1[Z] * vector2[X] - vector2[Z] * vector1[X],
            vector1[X] * vector2[Y] - vector2[X] * vector1[Y]
        ]
        divider = math.sqrt(normal[X] ** 2 + normal[Y] ** 2 + normal[Z] ** 2)
        try:
            unified_normal = [
                normal[X] / divider,
                normal[Y] / divider,
                normal[Z] / divider
            ]
        except:
            raise TypeError
        return [
            unified_normal, 
            [vex1, vex2, vex3]
        ]

    def pointInUnit(self, point):
        for unit_id in range(len(self.unit_list)):
            unit = self.unit_list[unit_id]
            kps = unit.getSeqPoint()
            if pointInPolygon(point, kps):
                return unit_id
        return None
    
    def getUnit(self, unit_id):
        return self.unit_list[unit_id]
    
    def getUnitWithKps(self, unit_id):
        return self.unit_list[unit_id].getSeqPoint()
    
    def calculateTriPlaneForString(self):
        s = self.string_list
        h = self.height + 10 * self.print_accuracy + s[-1].width
        tris = []
        for ele in s:
            if ele.type == BOTTOM:
                ele.start_point[Z] = 0.0
                ele.end_point[Z] = 0.0
            elif ele.type == TOP:
                ele.start_point[Z] = h
                ele.end_point[Z] = h
            else:
                ele.start_point[Z] = 0.0
                ele.end_point[Z] = h
            kps, upper_kps = ele.generatePointWithResolution(self.unit_hole_resolution)
            kp_num = len(kps)
            #bottom
            for cur in range(1, kp_num - 1):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[0], kps[cur], kps[next_ele])
                tris.append(ans1)
            #around
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[next_ele], kps[cur], upper_kps[cur])
                ans2 = self.getTriangle(upper_kps[next_ele], kps[next_ele], upper_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #upper
            for cur in range(1, kp_num - 1):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(upper_kps[0], upper_kps[next_ele], upper_kps[cur])
                tris.append(ans1)
        self.string_tri_list = tris

    def getAdditionalLineForUnit(self, unit_id, upper_x_bound=math.inf, lower_x_bound=-math.inf):
        unit = self.unit_list[unit_id]
        kps = unit.getSeqPoint()
        kp_num = len(kps)

        #around
        k_b = []
        upper_kps = []
        for i in range(0, kp_num):
            k_b.append([unit.crease[i].k(), unit.crease[i].b()])
        for cur in range(0, kp_num):
            next_ele = (cur + kp_num - 1) % kp_num
            crease_type = unit.crease[cur].getType()
            
            if self.enable_difference == 1:
                if crease_type == VALLEY:
                    bias = self.bias / 2.5
                elif crease_type == BORDER:
                    bias = 1e-3
                else:
                    bias = self.bias * 1.0
            elif self.enable_difference == 2:
                if crease_type == MOUNTAIN:
                    bias = self.bias / 2.5
                elif crease_type == BORDER:
                    bias = 1e-3
                else:
                    bias = self.bias * 1.0
            else:
                if crease_type == BORDER:
                    bias = 1e-3
                else:
                    bias = self.bias * 1.0
                
            if k_b[cur][0] == math.inf:
                b1_modified = [bias + k_b[cur][1], -bias + k_b[cur][1]]
            else:
                b1_modified = [bias * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1], -bias * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1]]
            
            crease_type = unit.crease[next_ele].getType()
            if self.enable_difference == 1:
                if crease_type == VALLEY:
                    bias = self.bias / 2.5
                elif crease_type == BORDER:
                    bias = 1e-3
                else:
                    bias = self.bias * 1.0
            elif self.enable_difference == 2:
                if crease_type == MOUNTAIN:
                    bias = self.bias / 2.5
                elif crease_type == BORDER:
                    bias = 1e-3
                else:
                    bias = self.bias * 1.0
            else:
                if crease_type == BORDER:
                    bias = 1e-3
                else:
                    bias = self.bias * 1.0

            if k_b[next_ele][0] == math.inf:
                b2_modified = [bias + k_b[next_ele][1], -bias + k_b[next_ele][1]]
            else:
                b2_modified = [bias * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1], -bias * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1]]
            ps = [
                calculateIntersection([k_b[cur][0], b1_modified[0]], [k_b[next_ele][0], b2_modified[0]]),
                calculateIntersection([k_b[cur][0], b1_modified[1]], [k_b[next_ele][0], b2_modified[0]]),
                calculateIntersection([k_b[cur][0], b1_modified[0]], [k_b[next_ele][0], b2_modified[1]]),
                calculateIntersection([k_b[cur][0], b1_modified[1]], [k_b[next_ele][0], b2_modified[1]]),
            ]
            finded = False
            for p in ps:
                angle = 0.0
                for kp_id in range(0, kp_num):
                    next_kp_id = (kp_id + 1) % kp_num
                    angle += calculateAngle(p, kps[kp_id], kps[next_kp_id])
                if abs(angle - 2 * math.pi) < 1e-5:
                    upper_kps.append(p + [self.height])
                    finded = True
                    break
            # intersection happen
            if not finded:
                next_ele = (cur + kp_num - 1) % kp_num
                if k_b[cur][0] == math.inf:
                    epsilon_b1_modified = [1e-3 + k_b[cur][1], -1e-3 + k_b[cur][1]]
                else:
                    epsilon_b1_modified = [1e-3 * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1], -1e-3 * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1]]
                if k_b[next_ele][0] == math.inf:
                    epsilon_b2_modified = [1e-3 + k_b[next_ele][1], -1e-3 + k_b[next_ele][1]]
                else:
                    epsilon_b2_modified = [1e-3 * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1], -1e-3 * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1]]
                epsilon_ps = [
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[0]], [k_b[next_ele][0], epsilon_b2_modified[0]]),
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[1]], [k_b[next_ele][0], epsilon_b2_modified[0]]),
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[0]], [k_b[next_ele][0], epsilon_b2_modified[1]]),
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[1]], [k_b[next_ele][0], epsilon_b2_modified[1]]),
                ]
                for id in range(0, len(epsilon_ps)):
                    angle = 0.0
                    for kp_id in range(0, kp_num):
                        next_kp_id = (kp_id + 1) % kp_num
                        angle += calculateAngle(epsilon_ps[id], kps[kp_id], kps[next_kp_id])
                    if abs(angle - 2 * math.pi) < 1e-5:
                        upper_kps.append(ps[id] + [self.height])
                        finded = True
                        break

        # check the self-intersection
        problem_point_id = []
        while(1):
            intersection = False
            upper_kps_num = len(upper_kps)
            problem_point_num = len(problem_point_id)
            accumulate_bonus_of_kps = 0
            for k in range(upper_kps_num):
                if (k + accumulate_bonus_of_kps) in problem_point_id:
                    accumulate_bonus_of_kps += 1
                previous_id = (k + upper_kps_num - 1) % upper_kps_num
                next_id = (k + 1) % upper_kps_num
                next_next_id = (k + 2) % upper_kps_num
                crease1 = Crease(upper_kps[k], upper_kps[previous_id], BORDER)
                crease2 = Crease(upper_kps[next_id], upper_kps[next_next_id], BORDER)
                dir_upper_kp = upper_kps[next_id][0] - upper_kps[k][0]
                dir_kp = kps[(k + 1 + accumulate_bonus_of_kps) % kp_num][0] - kps[(k + accumulate_bonus_of_kps) % kp_num][0]
                p = calculateIntersectionWithinCrease(crease1, crease2)
                # if p != None and dir_kp * dir_upper_kp < 0:
                if dir_kp * dir_upper_kp < -1e-5:
                    problem_point_id.append(k + problem_point_num)
                    intersection = True
                    del upper_kps[k]
                    if next_id == 0:
                        del upper_kps[next_id]
                        upper_kps.insert(next_id, p + [self.height])
                    else:
                        del upper_kps[k]
                        upper_kps.insert(k, p + [self.height])
                    break
            if not intersection: break

        #additional crease
        for cur in range(0, upper_kps_num):
            next_ele = (cur + 1) % upper_kps_num
            crease_type = unit.getCrease()[cur].getType()
            # c1 = Crease(upper_kps[cur], upper_kps[next_ele], BORDER)
            # c2 = Crease([lower_x_bound, -math.inf], [lower_x_bound, math.inf], BORDER)
            # p = calculateIntersectionWithinCrease(c1, c2)
            # if p != None:
            #     if upper_kps[cur][X] <= lower_x_bound:
            #         upper_kps[cur][X] = p[X]
            #         upper_kps[cur][Y] = p[Y]
            #     if upper_kps[next_ele][X] <= lower_x_bound:
            #         upper_kps[next_ele][X] = p[X]
            #         upper_kps[next_ele][Y] = p[Y]
            # else:
            #     c3 = Crease([upper_x_bound, -math.inf], [upper_x_bound, math.inf], BORDER)
            #     p = calculateIntersectionWithinCrease(c1, c3)
            #     if p != None:
            #         if upper_kps[cur][X] >= upper_x_bound:
            #             upper_kps[cur][X] = p[X]
            #             upper_kps[cur][Y] = p[Y]
            #         if upper_kps[next_ele][X] >= upper_x_bound:
            #             upper_kps[next_ele][X] = p[X]
                        # upper_kps[next_ele][Y] = p[Y]
            
            if (upper_kps[cur][X] - lower_x_bound) * (upper_kps[next_ele][X] - lower_x_bound) < -1e-5:
                if (upper_kps[cur][X] < lower_x_bound):
                    percent = (lower_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[cur][X] = lower_x_bound
                    upper_kps[cur][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])
                else:
                    percent = (lower_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[next_ele][X] = lower_x_bound
                    upper_kps[next_ele][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])
            elif (upper_kps[cur][X] - upper_x_bound) * (upper_kps[next_ele][X] - upper_x_bound) < -1e-5:
                if (upper_kps[cur][X] > upper_x_bound):
                    percent = (upper_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[cur][X] = upper_x_bound
                    upper_kps[cur][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])
                else:
                    percent = (upper_x_bound - upper_kps[cur][X]) / (upper_kps[next_ele][X] - upper_kps[cur][X])
                    upper_kps[next_ele][X] = upper_x_bound
                    upper_kps[next_ele][Y] = upper_kps[cur][Y] + percent * (upper_kps[next_ele][Y] - upper_kps[cur][Y])

        cur = 0
        next_ele = 1
        for i in range(0, upper_kps_num):
            crease_type = unit.getCrease()[cur].getType()
            if (upper_kps[cur][X] >= lower_x_bound and upper_kps[next_ele][X] >= lower_x_bound) and (upper_kps[cur][X] <= upper_x_bound and upper_kps[next_ele][X] <= upper_x_bound):
                self.additional_crease_list.append(Crease(
                    [upper_kps[cur][X], upper_kps[cur][Y]],
                    [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
                ))
                cur = next_ele
                next_ele = (next_ele + 1) % upper_kps_num
            else:
                next_ele = (next_ele + 1) % upper_kps_num
                if upper_kps[next_ele][X] <= lower_x_bound and upper_kps[next_ele][X] > upper_kps[cur][X]:
                    cur += 1
            # if (upper_kps[cur][X] < lower_x_bound and upper_kps[next_ele][X] < lower_x_bound) or (upper_kps[cur][X] > upper_x_bound and upper_kps[next_ele][X] > upper_x_bound):
            #     continue
            # upper_kps[cur][X] = min(max(upper_kps[cur][X], lower_x_bound), upper_x_bound)
            # upper_kps[next_ele][X] = min(max(upper_kps[next_ele][X], lower_x_bound), upper_x_bound)
            # if not (upper_kps[cur][X] == upper_kps[next_ele][X] and upper_kps[cur][Y] == upper_kps[next_ele][Y]):
            #     self.additional_crease_list.append(Crease(
            #         [upper_kps[cur][X], upper_kps[cur][Y]],
            #         [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
            #     ))

    def getAdditionalLineForAllUnit(self, upper_x_bound=math.inf, lower_x_bound=-math.inf):
        for i in range(len(self.unit_list)):
            self.getAdditionalLineForUnit(i, upper_x_bound, lower_x_bound)
        return self.additional_crease_list

    def getCreaseDrawing(self, crease_id):
        lines = []
        crease = self.valid_crease_list[crease_id]
        crease_type = crease.getType()
        k_standard = crease.k()
        # b_standard = crease.b()
        mid_point = crease.getMidPoint()

        if self.enable_difference == 1:
            if crease_type == VALLEY:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0
        elif self.enable_difference == 2:
            if crease_type == MOUNTAIN:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0
        else:
            if crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0

        if k_standard == math.inf:
            mid_point_bias1 = [
                mid_point[X] - bias,
                mid_point[Y],
            ]
            mid_point_bias2 = [
                mid_point[X] + bias,
                mid_point[Y],
            ]
        else:
            mid_point_bias1 = [
                mid_point[X] - k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] + 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]
            mid_point_bias2 = [
                mid_point[X] + k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] - 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]

        correct_crease = []
        for additional_crease in self.additional_crease_list:
            if pointOnCrease(mid_point_bias1, additional_crease):
                correct_crease.append(additional_crease)
                continue
            if pointOnCrease(mid_point_bias2, additional_crease):
                correct_crease.append(additional_crease)
                continue
        if len(correct_crease) == 2: #symmetric
            lines = [
                correct_crease[0], correct_crease[1],
                Crease(correct_crease[0][START], correct_crease[1][END], BORDER),
                Crease(correct_crease[1][START], correct_crease[0][END], BORDER)
            ]
            return lines
        else:
            return []

    def getBorderCreaseUnit(self):
        border_crease = []
        for unit in self.unit_list:
            border_crease += unit.getBorderCrease()
        unit = Unit()
        unit.addCrease(deepcopy(border_crease[0]))
        next_start_point = deepcopy(border_crease[0][END])
        del(border_crease[0])
        while len(border_crease):
            find = False
            for i in range(len(border_crease)):
                chosen_crease = border_crease[i]
                chosen_start = chosen_crease[START]
                chosen_end = chosen_crease[END]
                if distance(chosen_start, next_start_point) < 1e-5:
                    next_start_point = deepcopy(chosen_crease[END])
                    unit.addCrease(deepcopy(chosen_crease))
                    del(border_crease[i])
                    find = True
                    break
                if distance(chosen_end, next_start_point) < 1e-5:
                    chosen_crease = chosen_crease.getReverse()
                    next_start_point = deepcopy(chosen_crease[END])
                    unit.addCrease(deepcopy(chosen_crease))
                    del(border_crease[i])
                    find = True
                    break
            if not find:
                raise TypeError
        return unit

    def getInnerBiasUnitList(self):
        special_point_list = []
        for i in range(len(self.unit_list)):
            unit = self.unit_list[i]
            new_kps, _ = self.calculateInnerBiasAndSettingHeight(unit, i, self.bias, 0.0, MIDDLE, border_penalty=self.bias - 3.0 * self.base_inner_bias if self.method == 'symmetry' else 0.)
            reverse_kps = []
            for i in range(0, -len(new_kps), -1):
                reverse_kps.append(new_kps[i])
            special_point_list.append(reverse_kps)
        return special_point_list

    def getSpecialListAndPillar(self, bias):
        self.special_point_list = []
        self.pillar_unit_list = []
        for i in range(len(self.unit_list)):
            unit = self.unit_list[i]
            # exist_modify = False
            # for j in range(len(unit.getCrease())):
            #     if self.unit_bias_list[i][j] != None:
            #         exist_modify = True
            #         break
            
            original_kps, problem_id = self.calculateInnerBiasAndSettingHeight(unit, i, self.print_accuracy * 3., 0.0, MIDDLE, border_penalty=self.bias - 3.0 * self.base_inner_bias if self.method == 'symmetry' else 0., accumulation=True)
            new_kps, _ = self.calculateInnerBiasAndSettingHeight(unit, i, self.bias, 0.0, MIDDLE, border_penalty=self.bias - 3.0 * self.base_inner_bias if self.method == 'symmetry' else 0., accumulation=True)
            reverse_kps = []
            for k in range(0, -len(new_kps), -1):
                reverse_kps.append(new_kps[k])
            if 1:
                self.special_point_list.append(reverse_kps)
            # else:
            #     self.special_point_list.append([])

            pillar_for_unit = []
            pillar_resolution = 3

            accumulate_index = 0
            creases = unit.getCrease()
            for j in range(len(creases)):
                # if self.unit_bias_list[i][j] != None:
                #     if j in problem_id:
                #         accumulate_index += 1
                #         continue
                #     continue
                crease = creases[j]
                # next_crease = creases[(j + 1) % len(creases)]
                length = crease.getLength()
                # next_length = next_crease.getLength()

                if j in problem_id:
                    accumulate_index += 1
                    continue

                mid_point = [(original_kps[j - accumulate_index][X] + original_kps[(j - accumulate_index + 1) % len(original_kps)][X]) / 2.0, 
                            (original_kps[j - accumulate_index][Y] + original_kps[(j - accumulate_index + 1) % len(original_kps)][Y]) / 2.0]
                
                quad_back_point = [(3. * original_kps[j - accumulate_index][X] + original_kps[(j - accumulate_index + 1) % len(original_kps)][X]) / 4.0, 
                                    (3. * original_kps[j - accumulate_index][Y] + original_kps[(j - accumulate_index + 1) % len(original_kps)][Y]) / 4.0]
                
                quad_front_point = [(original_kps[j - accumulate_index][X] + 3. * original_kps[(j - accumulate_index + 1) % len(original_kps)][X]) / 4.0, 
                                    (original_kps[j - accumulate_index][Y] + 3. * original_kps[(j - accumulate_index + 1) % len(original_kps)][Y]) / 4.0]
                
                direction = crease.getDirection()
                normal = crease.getNormal()
                # next_direction = next_crease.getDirection()

                if 24. * bias < length:
                    #middle pillar
                    
                    forward_point = [mid_point[X] + bias * direction[X], mid_point[Y] + bias * direction[Y], 0.0]
                    backward_point = [mid_point[X] - bias * direction[X], mid_point[Y] - bias * direction[Y], 0.0]
                    normal_point = [mid_point[X] + bias * normal[X], mid_point[Y] + bias * normal[Y], 0.0]

                    new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                    pillar_for_unit.append(new_pillar_point)

                    # quad_back
                    forward_point = [quad_back_point[X] + bias * direction[X], quad_back_point[Y] + bias * direction[Y], 0.0]
                    backward_point = [quad_back_point[X] - bias * direction[X], quad_back_point[Y] - bias * direction[Y], 0.0]
                    normal_point = [quad_back_point[X] + bias * normal[X], quad_back_point[Y] + bias * normal[Y], 0.0]

                    new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                    pillar_for_unit.append(new_pillar_point)

                    # quad_front
                    forward_point = [quad_front_point[X] + bias * direction[X], quad_front_point[Y] + bias * direction[Y], 0.0]
                    backward_point = [quad_front_point[X] - bias * direction[X], quad_front_point[Y] - bias * direction[Y], 0.0]
                    normal_point = [quad_front_point[X] + bias * normal[X], quad_front_point[Y] + bias * normal[Y], 0.0]

                    new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                    pillar_for_unit.append(new_pillar_point)
                
                elif 12. * bias < length:
                    # quad_back
                    forward_point = [quad_back_point[X] + bias * direction[X], quad_back_point[Y] + bias * direction[Y], 0.0]
                    backward_point = [quad_back_point[X] - bias * direction[X], quad_back_point[Y] - bias * direction[Y], 0.0]
                    normal_point = [quad_back_point[X] + bias * normal[X], quad_back_point[Y] + bias * normal[Y], 0.0]

                    new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                    pillar_for_unit.append(new_pillar_point)

                    # quad_front
                    forward_point = [quad_front_point[X] + bias * direction[X], quad_front_point[Y] + bias * direction[Y], 0.0]
                    backward_point = [quad_front_point[X] - bias * direction[X], quad_front_point[Y] - bias * direction[Y], 0.0]
                    normal_point = [quad_front_point[X] + bias * normal[X], quad_front_point[Y] + bias * normal[Y], 0.0]

                    new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                    pillar_for_unit.append(new_pillar_point)
                
                elif 6. * bias < length:
                    #middle pillar
                    
                    forward_point = [mid_point[X] + bias * direction[X], mid_point[Y] + bias * direction[Y], 0.0]
                    backward_point = [mid_point[X] - bias * direction[X], mid_point[Y] - bias * direction[Y], 0.0]
                    normal_point = [mid_point[X] + bias * normal[X], mid_point[Y] + bias * normal[Y], 0.0]

                    new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
                    pillar_for_unit.append(new_pillar_point)
            #     if 10. * bias < length:
            #         #side pillar
                    
            #         # quad_back
            #         forward_point = [quad_back_point[X] + bias * direction[X], quad_back_point[Y] + bias * direction[Y], 0.0]
            #         backward_point = [quad_back_point[X] - bias * direction[X], quad_back_point[Y] - bias * direction[Y], 0.0]
            #         normal_point = [quad_back_point[X] + bias * normal[X], quad_back_point[Y] + bias * normal[Y], 0.0]

            #         new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
            #         pillar_for_unit.append(new_pillar_point)

            #         # quad_front
            #         forward_point = [quad_front_point[X] + bias * direction[X], quad_front_point[Y] + bias * direction[Y], 0.0]
            #         backward_point = [quad_front_point[X] - bias * direction[X], quad_front_point[Y] - bias * direction[Y], 0.0]
            #         normal_point = [quad_front_point[X] + bias * normal[X], quad_front_point[Y] + bias * normal[Y], 0.0]

            #         new_pillar_point = deepcopy([forward_point, backward_point, normal_point])
            #         pillar_for_unit.append(new_pillar_point)

                # mid_point = [original_kps[(j - accumulate_index + 1) % len(original_kps)][X], 
                #             original_kps[(j - accumulate_index + 1) % len(original_kps)][Y], 0.0]
                # alpha = math.acos(-direction[X] * next_direction[X] - direction[Y] * next_direction[Y])
                # side_bias_max = 2. * bias / math.sin(alpha)
                # calculated_bias = [0., 0.]

                # if ((4. * side_bias_max < length and crease.getType() != BORDER) or (2.5 * side_bias_max < length and crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # strong
                #     calculated_bias[0] = side_bias_max
                
                # elif ((8. * bias < length and crease.getType() != BORDER) or (5. * bias < length and crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # mixed
                #     calculated_bias[0] = 2. * bias if crease.getType() != BORDER else length / 3.

                # else:
                #     # enable_side_pillar = 1 # weak
                #     calculated_bias[0] = length / 4.
                
                # if ((4. * side_bias_max < next_length and next_crease.getType() != BORDER) or (2.5 * side_bias_max < next_length and next_crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # strong
                #     calculated_bias[1] = side_bias_max
                
                # elif ((8. * bias < next_length and next_crease.getType() != BORDER) or (5. * bias < next_length and next_crease.getType() == BORDER)):
                #     # enable_side_pillar = 1 # mixed
                #     calculated_bias[1] = 2. * bias if next_crease.getType() != BORDER else next_length / 3.

                # else:
                #     # enable_side_pillar = 1 # weak
                #     calculated_bias[1] = next_length / 4.

                # # if (enable_side_pillar):
                #     #side pillar

                # forward_point = [mid_point[X] + calculated_bias[1] * next_direction[X], mid_point[Y] + calculated_bias[1] * next_direction[Y], 0.0]
                # backward_point = [mid_point[X] - calculated_bias[0] * direction[X], mid_point[Y] - calculated_bias[0] * direction[Y], 0.0]

                # new_pillar_point = deepcopy([forward_point, mid_point, backward_point])
                # pillar_for_unit.append(new_pillar_point)

            self.pillar_unit_list.append(pillar_for_unit)
            # self.pillar_unit_list.append([])

    def calculateTriPlaneForCreaseUsingBindingMethod(self, base_height=0, upper_height=None):
        # unit = self.getBorderCreaseUnit()
        if upper_height == None:
            upper_height = self.board_height
        self.getSpecialListAndPillar(self.bias / 2.0)
        tris = []
        for i in range(len(self.unit_list)):
            unit = self.unit_list[i]
            special_point_list = self.special_point_list[i]
            pillars = self.pillar_unit_list[i]
            tris += self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =None, 
                upper_bias          =0, 
                down_bias           =0, 
                base_height         =base_height,
                upper_height        =upper_height, 
                add_hole            =True, 
                another_points_list =pillars + [special_point_list],
                additional_crease   =False,
                side_tri            =BORDER
            )
        return tris

    def calculateTriPlaneForCrease(self, crease_id, base_height=0, upper_height=None):
        if crease_id in self.hard_crease_index:
            base_height = -self.height / 2.0 + self.print_accuracy
            upper_height = self.height + self.board_height - 2 * self.print_accuracy

        if upper_height == None:
            upper_height = self.board_height
        tris = []
        crease = self.valid_crease_list[crease_id]
        crease_type = crease.getType()
        k_standard = crease.k()
        # b_standard = crease.b()
        mid_point = crease.getMidPoint()

        if self.enable_difference == 1:
            if crease_type == VALLEY:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0
        elif self.enable_difference == 2:
            if crease_type == MOUNTAIN:
                bias = self.bias / 2.5
            elif crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0
        else:
            if crease_type == BORDER:
                bias = 1e-3
            else:
                bias = self.bias * 1.0

        if k_standard == math.inf:
            mid_point_bias1 = [
                mid_point[X] - bias,
                mid_point[Y],
            ]
            mid_point_bias2 = [
                mid_point[X] + bias,
                mid_point[Y],
            ]
        else:
            mid_point_bias1 = [
                mid_point[X] - k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] + 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]
            mid_point_bias2 = [
                mid_point[X] + k_standard / math.sqrt(k_standard ** 2 + 1) * bias,
                mid_point[Y] - 1.0 / math.sqrt(k_standard ** 2 + 1) * bias,
            ]
        correct_crease = []
        for additional_crease in self.additional_crease_list:
            if pointOnCrease(mid_point_bias1, additional_crease):
                correct_crease.append(additional_crease)
                continue
            if pointOnCrease(mid_point_bias2, additional_crease):
                correct_crease.append(additional_crease)
                continue
        if len(correct_crease) == 2: #symmetric
            kps = [
                correct_crease[0][END] + [base_height],
                correct_crease[0][START] + [base_height],
                correct_crease[1][END] + [base_height],
                correct_crease[1][START] + [base_height]
            ]
            kp_num = len(kps)
            p_width = (1 - self.hole_width_size_percent) / 2
            p_length = (1 - self.hole_length_size_percent) / 2
            inner_kps = []
            #inner points
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                previous_ele = (cur + kp_num - 1) % kp_num
                method = cur % 2
                if method == 0:
                    kp = [kps[cur][X], kps[cur][Y], kps[cur][Z]]
                    delta_x = p_width * (kps[previous_ele][X] - kps[cur][X]) + p_length * (kps[next_ele][X] - kps[cur][X])
                    delta_y = p_width * (kps[previous_ele][Y] - kps[cur][Y]) + p_length * (kps[next_ele][Y] - kps[cur][Y])
                    kp[X] += delta_x
                    kp[Y] += delta_y
                    inner_kps.append(kp)
                else:
                    kp = [kps[cur][X], kps[cur][Y], kps[cur][Z]]
                    delta_x = p_length * (kps[previous_ele][X] - kps[cur][X]) + p_width * (kps[next_ele][X] - kps[cur][X])
                    delta_y = p_length * (kps[previous_ele][Y] - kps[cur][Y]) + p_width * (kps[next_ele][Y] - kps[cur][Y])
                    kp[X] += delta_x
                    kp[Y] += delta_y
                    inner_kps.append(kp)
            upper_kps = [[kps[x][X], kps[x][Y], kps[x][Z] + upper_height] for x in range(kp_num)]
            upper_inner_kps = [[inner_kps[x][X], inner_kps[x][Y], inner_kps[x][Z] + upper_height] for x in range(kp_num)]
            #bottom
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[cur], kps[next_ele], inner_kps[cur])
                ans2 = self.getTriangle(kps[next_ele], inner_kps[next_ele], inner_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #around
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(kps[next_ele], kps[cur], upper_kps[cur])
                ans2 = self.getTriangle(upper_kps[next_ele], kps[next_ele], upper_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #inner
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(inner_kps[cur], inner_kps[next_ele], upper_inner_kps[cur])
                ans2 = self.getTriangle(inner_kps[next_ele], upper_inner_kps[next_ele], upper_inner_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            #upper
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], upper_inner_kps[cur])
                ans2 = self.getTriangle(upper_inner_kps[next_ele], upper_kps[next_ele], upper_inner_kps[cur])
                tris.append(ans1)
                tris.append(ans2)
            self.crease_tri_list[crease_id] = tris
        # else:
        #     a = 1

    def calculateTriPlaneForAllCrease(self):
        for i in range(len(self.valid_crease_list)):
            self.calculateTriPlaneForCrease(i)

    def calculateDrawingForAllCrease(self):
        all_lines = []
        for i in range(len(self.valid_crease_list)):
            all_lines += self.getCreaseDrawing(i)
        return all_lines

    def calculateTriPlaneWithHole(self, kps, another_points_list, dir):
        kp_num = len(kps)
        tris = []
        all_lines = []
        for i in range(kp_num):
            next_id = (i + 1) % kp_num
            all_lines.append(Crease(
                kps[i], kps[next_id], BORDER
            ))
        for ele in another_points_list:
            for i in range(len(ele)):
                next_id = (i + 1) % len(ele)
                all_lines.append(Crease(
                    ele[i], ele[next_id], BORDER
                ))
        # all lines
        for i in range(len(another_points_list)):
            for point in another_points_list[i]:
                for kp in kps:
                    line1 = Crease(
                        point, kp, BORDER
                    )
                    intersection = False
                    for line2 in all_lines:
                        p = calculateIntersectionWithinCrease(line1, line2, strict_flag=True)
                        if p != None:
                            intersection = True
                            break
                    if not intersection:
                        all_lines.append(line1)
                for j in range(i + 1, len(another_points_list)):
                    for new_point in another_points_list[j]:
                        line1 = Crease(
                            point, new_point, BORDER
                        )
                        intersection = False
                        for line2 in all_lines:
                            p = calculateIntersectionWithinCrease(line1, line2, strict_flag=True)
                            if p != None:
                                intersection = True
                                break
                        if not intersection:
                            all_lines.append(line1)
        # all tris
        # seq point
        seq_kps = []
        for kp in kps:
            seq_kps.append(kp)
        for ele in another_points_list:
            for point in ele:
                seq_kps.append(point)
        for i in range(len(seq_kps)):
            point = seq_kps[i]
            end_point = []
            for line1 in all_lines:
                if distance(line1[START], point) < 1e-5:
                    end_point.append(line1[END])
            for end in end_point:
                start_point = []
                for line2 in all_lines:
                    if distance(line2[END], end) < 1e-5:
                        start_point.append(line2[START])
                for start in start_point:
                    for line3 in all_lines:
                        if (distance(point, line3[START]) < 1e-5 and distance(start, line3[END]) < 1e-5) or \
                            (distance(point, line3[END]) < 1e-5 and distance(start, line3[START]) < 1e-5):
                            position = seq_kps.index(start)
                            point_inside = False
                            for kp in seq_kps:
                                if pointInPolygon(kp, [point, end, start], return_min_distance=True) > 1e-5:
                                    point_inside = True
                                    break
                            if position > i and (not point_inside):
                                if dir > 0:
                                    ans = self.getTriangle(point, end, start)
                                    if ans[0][Z] < 0:
                                        ans = self.getTriangle(point, start, end)
                                else:
                                    ans = self.getTriangle(point, end, start)
                                    if ans[0][Z] > 0:
                                        ans = self.getTriangle(point, start, end)
                                tris.append(ans)
                            else:
                                break
        return tris

    def calculateInnerBiasAndSettingHeight(self, unit, unit_id, bias, height, side, enable_strong_modify=False, border_penalty=0.0, accumulation=False):
        kps = unit.getSeqPoint()
        kp_num = len(kps)
        k_b = []
        upper_kps = []
        bias_list = []
        for i in range(len(unit.getCrease())):
            ele = unit.getCrease()[i]
            if ele.getType() == BORDER:
                bias_list.append(bias - border_penalty + 1e-3)
            else:
                if side == UP:
                    if self.asym and ele.getType() == MOUNTAIN:
                        bias_list.append(self.min_bias)

                    else:
                        if unit_id != -1 and self.unit_bias_list[unit_id][i] != None and \
                            ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)): # condition to modify
                            if accumulation:
                                bias_list.append(self.unit_bias_list[unit_id][i] + bias)
                            else:
                                bias_list.append(self.unit_bias_list[unit_id][i])
                        else:
                            if self.enable_difference == 1: #valley small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                                else:
                                    bias_list.append(bias)
                            elif self.enable_difference == 2: #mountain small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias)
                                else:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                            else:
                                bias_list.append(bias)
                elif side == DOWN:
                    if self.asym and ele.getType() == VALLEY:
                        bias_list.append(self.min_bias)

                    else:
                        if unit_id != -1 and self.unit_bias_list[unit_id][i] != None and \
                            ((not self.only_two_sides) or (self.only_two_sides and enable_strong_modify)): # condition to modify
                            if accumulation:
                                bias_list.append(self.unit_bias_list[unit_id][i] + bias)
                            else:
                                bias_list.append(self.unit_bias_list[unit_id][i])
                        else:
                            if self.enable_difference == 1: #valley small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias)
                                else:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                            elif self.enable_difference == 2: #mountain small
                                if ele.getType() == VALLEY:
                                    bias_list.append(bias + 3 * self.print_accuracy)
                                else:
                                    bias_list.append(bias)
                            else:
                                bias_list.append(bias)
                else:
                    bias_list.append(bias)
        if unit.connection != None and enable_strong_modify:
            bias_list[unit.connection_number] = math.atan(unit.connection) * self.unit_height + self.min_bias

        for i in range(0, kp_num):
            k_b.append([unit.crease[i].k(), unit.crease[i].b()])
        for cur in range(0, kp_num):
            next_ele = (cur + kp_num - 1) % kp_num
            if k_b[cur][0] == math.inf:
                b1_modified = [bias_list[cur] + k_b[cur][1], -bias_list[cur] + k_b[cur][1]]
            else:
                b1_modified = [bias_list[cur] * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1], -bias_list[cur] * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1]]
            if k_b[next_ele][0] == math.inf:
                b2_modified = [bias_list[next_ele] + k_b[next_ele][1], -bias_list[next_ele] + k_b[next_ele][1]]
            else:
                b2_modified = [bias_list[next_ele] * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1], -bias_list[next_ele] * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1]]
            ps = [
                calculateIntersection([k_b[cur][0], b1_modified[0]], [k_b[next_ele][0], b2_modified[0]]),
                calculateIntersection([k_b[cur][0], b1_modified[1]], [k_b[next_ele][0], b2_modified[0]]),
                calculateIntersection([k_b[cur][0], b1_modified[0]], [k_b[next_ele][0], b2_modified[1]]),
                calculateIntersection([k_b[cur][0], b1_modified[1]], [k_b[next_ele][0], b2_modified[1]]),
            ]
            finded = False
            for p in ps:
                angle = 0.0
                for kp_id in range(0, kp_num):
                    next_kp_id = (kp_id + 1) % kp_num
                    angle += calculateAngle(p, kps[kp_id], kps[next_kp_id])
                if abs(angle - 2 * math.pi) < 1e-5:
                    upper_kps.append(p + [height])
                    finded = True
                    break
            # intersection happen
            if not finded:
                next_ele = (cur + + kp_num - 1) % kp_num
                if k_b[cur][0] == math.inf:
                    epsilon_b1_modified = [1e-3 + k_b[cur][1], -1e-3 + k_b[cur][1]]
                else:
                    epsilon_b1_modified = [1e-3 * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1], -1e-3 * math.sqrt(k_b[cur][0] ** 2 + 1) + k_b[cur][1]]
                if k_b[next_ele][0] == math.inf:
                    epsilon_b2_modified = [1e-3 + k_b[next_ele][1], -1e-3 + k_b[next_ele][1]]
                else:
                    epsilon_b2_modified = [1e-3 * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1], -1e-3 * math.sqrt(k_b[next_ele][0] ** 2 + 1) + k_b[next_ele][1]]
                epsilon_ps = [
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[0]], [k_b[next_ele][0], epsilon_b2_modified[0]]),
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[1]], [k_b[next_ele][0], epsilon_b2_modified[0]]),
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[0]], [k_b[next_ele][0], epsilon_b2_modified[1]]),
                    calculateIntersection([k_b[cur][0], epsilon_b1_modified[1]], [k_b[next_ele][0], epsilon_b2_modified[1]]),
                ]
                for id in range(0, len(epsilon_ps)):
                    angle = 0.0
                    for kp_id in range(0, kp_num):
                        next_kp_id = (kp_id + 1) % kp_num
                        angle += calculateAngle(epsilon_ps[id], kps[kp_id], kps[next_kp_id])
                    if abs(angle - 2 * math.pi) < 1e-5:
                        upper_kps.append(ps[id] + [height])
                        finded = True
                        break

        # check the self-intersection
        problem_point_id = []
        while(1):
            intersection = False
            upper_kps_num = len(upper_kps)
            problem_point_num = len(problem_point_id)
            accumulate_bonus_of_kps = 0
            for k in range(upper_kps_num):
                if k in problem_point_id:
                    accumulate_bonus_of_kps += 1
                previous_id = (k - 1 + upper_kps_num) % upper_kps_num
                next_id = (k + 1) % upper_kps_num
                next_next_id = (k + 2) % upper_kps_num
                crease1 = Crease(upper_kps[k], upper_kps[previous_id], BORDER)
                crease2 = Crease(upper_kps[next_id], upper_kps[next_next_id], BORDER)
                dir_upper_kp = upper_kps[next_id][0] - upper_kps[k][0]
                dir_kp = kps[(k + 1 + accumulate_bonus_of_kps) % kp_num][0] - kps[(k + accumulate_bonus_of_kps) % kp_num][0]
                p = calculateIntersectionWithinCrease(crease1, crease2)
                # if p != None and dir_kp * dir_upper_kp < 0:
                if dir_kp * dir_upper_kp < -1e-5:
                    problem_point_id.append(k + problem_point_num)
                    intersection = True
                    del upper_kps[k]
                    if next_id == 0:
                        del upper_kps[next_id]
                        upper_kps.insert(next_id, p + [height])
                    else:
                        del upper_kps[k]
                        upper_kps.insert(k, p + [height])
                    break
            if not intersection: break
        return upper_kps, problem_point_id

    def calculateTriPlaneWithBiasAndHeight(self, unit, unit_id, upper_bias, down_bias, base_height, upper_height, add_hole=False, another_points_list=None, additional_crease=True, penalty=None, side_tri=None, accumulation=False, bottom_tri=True, upper_tri=True):
        if type(unit) == Unit:
            standard_kps = unit.getSeqPoint()
        else:
            standard_kps = unit
        #judge the side
        side = MIDDLE
        if down_bias < upper_bias:
            down_strong_modify = False
            upper_strong_modify = True
            side = UP
        elif down_bias > upper_bias:
            down_strong_modify = True
            upper_strong_modify = False
            side = DOWN
        else:
            down_strong_modify = True
            upper_strong_modify = True
        for ele in another_points_list:
            for point in ele:
                point[Z] = base_height
        if down_bias > 0:
            if penalty == None:
                bottom_kps, bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias, base_height, side, down_strong_modify, down_bias, accumulation)
            else:
                bottom_kps, bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias, base_height, side, down_strong_modify, down_bias - penalty, accumulation)
        elif down_bias == 0:
            bottom_kps = deepcopy(standard_kps)
            for ele in bottom_kps:
                ele[Z] = base_height
            bottom_problem_id = []
        else:
            return #error
        if upper_bias > 0:
            if penalty == None:
                upper_kps, upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias, upper_height, side, upper_strong_modify, upper_bias, accumulation)
            else:
                upper_kps, upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias, upper_height, side, upper_strong_modify, upper_bias - penalty, accumulation)
        elif upper_bias == 0:
            upper_kps = deepcopy(standard_kps)
            for ele in upper_kps:
                ele[Z] = upper_height
            upper_problem_id = []
        else:
            return #error
        kp_num = len(bottom_kps)
        upper_kp_num = len(upper_kps)
        tris = []
        # bottom
        if bottom_tri:
            if not add_hole:
                for cur in range(1, kp_num - 1):
                    vex1 = bottom_kps[0]
                    vex2 = bottom_kps[cur]
                    vex3 = bottom_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex2, vex3)
                    tris.append(ans)
            else:
                tris += self.calculateTriPlaneWithHole(bottom_kps, another_points_list, -1)
        #around
        if upper_strong_modify:
            forward_step = 0
            upper_cur = 0
            for cur in range(0, kp_num):
                side_enable = True
                if type(unit) == Unit:
                    crease_type = unit.getCrease()[cur].getType()
                    if side_tri != None and crease_type != side_tri:
                        side_enable = False

                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = cur + forward_step
                if real_id in upper_problem_id:
                    if real_id not in bottom_problem_id:
                        if side_enable:
                            ans = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if side_enable:
                            ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                            tris.append(ans1)
                            tris.append(ans2)
                        upper_cur = (upper_cur + 1) % upper_kp_num
                else:
                    if side_enable:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    upper_cur = (upper_cur + 1) % upper_kp_num
        else:
            forward_step = 0
            cur = 0
            for upper_cur in range(0, upper_kp_num):
                side_enable = True
                if type(unit) == Unit:
                    crease_type = unit.getCrease()[cur].getType()
                    if side_tri != None and crease_type != side_tri:
                        side_enable = False

                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = upper_cur + forward_step
                if real_id in bottom_problem_id:
                    if real_id not in upper_problem_id:
                        if side_enable:
                            ans = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if side_enable:
                            ans1 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_next_ele])
                            tris.append(ans1)
                            tris.append(ans2)
                        cur = (cur + 1) % kp_num
                else:
                    if side_enable:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    cur = (cur + 1) % kp_num
        #upper
        upper_another_points_list = deepcopy(another_points_list)
        for ele in upper_another_points_list:
            for point in ele:
                point[Z] = upper_height
        if upper_tri:
            if not add_hole:
                for cur in range(1, upper_kp_num - 1):
                    vex1 = upper_kps[0]
                    vex2 = upper_kps[cur]
                    vex3 = upper_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex3, vex2)
                    tris.append(ans)
            else:
                tris += self.calculateTriPlaneWithHole(upper_kps, upper_another_points_list, 1)
        #hole
        if add_hole:
            for id in range(len(another_points_list)):
                for cur in range(len(another_points_list[id])):
                    next_ele = (cur + 1) % len(another_points_list[id])
                    ans1 = self.getTriangle(another_points_list[id][cur], upper_another_points_list[id][cur], another_points_list[id][next_ele])
                    ans2 = self.getTriangle(another_points_list[id][next_ele], upper_another_points_list[id][cur], upper_another_points_list[id][next_ele])
                    tris.append(ans1)
                    tris.append(ans2)

        if additional_crease:
            #additional crease
            for cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % upper_kp_num
                crease_type = unit.getCrease()[cur].getType()
                self.additional_crease_list.append(Crease(
                    [upper_kps[cur][X], upper_kps[cur][Y]],
                    [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
                ))
        return tris
    
    def calculateTriPlaneWithBiasAndHeightWithInner(self, unit, unit_id, upper_bias, down_bias, base_height, upper_height, add_hole=False, another_points_list=None, additional_crease=True, inner_bias=0.0):
        standard_kps = unit.getSeqPoint()
        side = MIDDLE
        if down_bias < upper_bias:
            down_strong_modify = False
            upper_strong_modify = True
            side = UP
        elif down_bias > upper_bias:
            down_strong_modify = True
            upper_strong_modify = False
            side = DOWN
        else:
            down_strong_modify = True
            upper_strong_modify = True
        for ele in another_points_list:
            for point in ele:
                point[Z] = base_height
        if down_bias > 0:
            bottom_kps, bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias, base_height, side, down_strong_modify, down_bias)
            if down_strong_modify:
                inner_bottom_kps, inner_bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias + 3 * inner_bias, base_height + self.layer * self.print_accuracy, side, down_strong_modify, down_bias)
            else:
                inner_bottom_kps, inner_bottom_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, down_bias + 3 * inner_bias, base_height, side, down_strong_modify, down_bias)
        elif down_bias == 0:
            bottom_kps = deepcopy(standard_kps)
            for ele in bottom_kps:
                ele[Z] = base_height
            bottom_problem_id = []
        else:
            return #error
        if upper_bias > 0:
            upper_kps, upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias, upper_height, side, upper_strong_modify, upper_bias)
            if upper_strong_modify:
                inner_upper_kps, inner_upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias + 3 * inner_bias, upper_height - self.layer * self.print_accuracy, side, upper_strong_modify, upper_bias)
            else:
                inner_upper_kps, inner_upper_problem_id = self.calculateInnerBiasAndSettingHeight(unit, unit_id, upper_bias + 3 * inner_bias, upper_height, side, upper_strong_modify, upper_bias)
        elif upper_bias == 0:
            upper_kps = deepcopy(standard_kps)
            for ele in upper_kps:
                ele[Z] = upper_height
            upper_problem_id = []
        else:
            return #error
        kp_num = len(bottom_kps)
        inner_kp_num = len(inner_bottom_kps)
        upper_kp_num = len(upper_kps)
        inner_upper_kp_num = len(inner_upper_kps)
        tris = []
        # bottom
        if not add_hole:
            if down_strong_modify:
                for cur in range(1, kp_num - 1):
                    vex1 = bottom_kps[0]
                    vex2 = bottom_kps[cur]
                    vex3 = bottom_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex2, vex3)
                    tris.append(ans)
            
                for cur in range(1, inner_kp_num - 1):
                    vex1 = inner_bottom_kps[0]
                    vex2 = inner_bottom_kps[cur]
                    vex3 = inner_bottom_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex3, vex2)
                    tris.append(ans)
        else:
            if down_strong_modify:
                tris += self.calculateTriPlaneWithHole(bottom_kps, another_points_list, -1)

                inner_another_points_list = deepcopy(another_points_list)
                for ele in inner_another_points_list:
                    for point in ele:
                        point[Z] = base_height + self.layer * self.print_accuracy
                tris += self.calculateTriPlaneWithHole(inner_bottom_kps, inner_another_points_list, 1)
        #around
        # if kp_num >= upper_kp_num:
        if upper_strong_modify:
            forward_step = 0
            upper_cur = 0
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()

                if real_id in upper_problem_id:
                    if real_id not in bottom_problem_id:
                        if 1:
                            ans = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])  
                            tris.append(ans1)
                            tris.append(ans2)
                        upper_cur = upper_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    upper_cur = upper_next_ele
        else:
            forward_step = 0
            cur = 0
            for upper_cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % kp_num
                upper_next_ele = (upper_cur + 1) % upper_kp_num
                real_id = upper_cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in bottom_problem_id:
                    if real_id not in upper_problem_id:
                        if 1:
                            ans = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[cur], upper_kps[upper_cur])
                            ans2 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_next_ele])
                            tris.append(ans1)
                            tris.append(ans2)
                        cur = next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], upper_kps[upper_cur])
                        ans2 = self.getTriangle(upper_kps[upper_next_ele], bottom_kps[next_ele], upper_kps[upper_cur])
                        tris.append(ans1)
                        tris.append(ans2)
                    cur = next_ele
        if upper_strong_modify:
            forward_step = 0
            upper_cur = 0
            for cur in range(0, inner_kp_num):
                next_ele = (cur + 1) % inner_kp_num
                upper_next_ele = (upper_cur + 1) % inner_upper_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_upper_problem_id:
                    if real_id not in inner_bottom_problem_id:
                        if 1:
                            ans = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            ans2 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[next_ele])
                            tris.append(ans1)
                            tris.append(ans2)
                        upper_cur = upper_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                        ans2 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[next_ele])
                        tris.append(ans1)
                        tris.append(ans2)
                    upper_cur = upper_next_ele
        else:
            forward_step = 0
            cur = 0
            for upper_cur in range(0, inner_upper_kp_num):
                next_ele = (cur + 1) % inner_kp_num
                upper_next_ele = (upper_cur + 1) % inner_upper_kp_num
                real_id = upper_cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_bottom_problem_id:
                    if real_id not in inner_upper_problem_id:
                        if 1:
                            ans = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                            ans2 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_next_ele], inner_bottom_kps[cur])
                            tris.append(ans1)
                            tris.append(ans2)
                        cur = next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(inner_bottom_kps[next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[cur])
                        ans2 = self.getTriangle(inner_upper_kps[upper_next_ele], inner_upper_kps[upper_cur], inner_bottom_kps[next_ele])
                        tris.append(ans1)
                        tris.append(ans2)
                    cur = next_ele
        #upper
        if not add_hole:
            if upper_strong_modify:
                for cur in range(1, upper_kp_num - 1):
                    vex1 = upper_kps[0]
                    vex2 = upper_kps[cur]
                    vex3 = upper_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex3, vex2)
                    tris.append(ans)
            
                for cur in range(1, inner_upper_kp_num - 1):
                    vex1 = inner_upper_kps[0]
                    vex2 = inner_upper_kps[cur]
                    vex3 = inner_upper_kps[cur + 1]
                    ans = self.getTriangle(vex1, vex2, vex3)
                    tris.append(ans)
        else:
            if upper_strong_modify:
                upper_another_points_list = deepcopy(another_points_list)
                for ele in upper_another_points_list:
                    for point in ele:
                        point[Z] = upper_height
                tris += self.calculateTriPlaneWithHole(upper_kps, upper_another_points_list, 1)

                inner_upper_another_points_list = deepcopy(another_points_list)
                for ele in inner_upper_another_points_list:
                    for point in ele:
                        point[Z] = upper_height - 3.0 * self.print_accuracy
                tris += self.calculateTriPlaneWithHole(inner_upper_kps, inner_upper_another_points_list, -1)
        #hole
        if add_hole:
            if down_strong_modify:
                for id in range(len(another_points_list)):
                    for cur in range(self.unit_hole_resolution):
                        next_ele = (cur + 1) % self.unit_hole_resolution
                        ans1 = self.getTriangle(another_points_list[id][cur], inner_another_points_list[id][cur], another_points_list[id][next_ele])
                        ans2 = self.getTriangle(another_points_list[id][next_ele], inner_another_points_list[id][cur], inner_another_points_list[id][next_ele])
                        tris.append(ans1)
                        tris.append(ans2)
            if upper_strong_modify:
                for id in range(len(another_points_list)):
                    for cur in range(self.unit_hole_resolution):
                        next_ele = (cur + 1) % self.unit_hole_resolution
                        ans1 = self.getTriangle(inner_upper_another_points_list[id][cur], upper_another_points_list[id][cur], inner_upper_another_points_list[id][next_ele])
                        ans2 = self.getTriangle(inner_upper_another_points_list[id][next_ele], upper_another_points_list[id][cur], upper_another_points_list[id][next_ele])
                        tris.append(ans1)
                        tris.append(ans2)

        #connection
        if upper_strong_modify:
            forward_step = 0
            inner_cur = 0
            for cur in range(0, kp_num):
                next_ele = (cur + 1) % kp_num
                inner_next_ele = (inner_cur + 1) % inner_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_bottom_problem_id:
                    if real_id not in bottom_problem_id:
                        if 1:
                            ans = self.getTriangle(bottom_kps[next_ele], inner_bottom_kps[inner_cur], bottom_kps[cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(bottom_kps[next_ele], inner_bottom_kps[inner_cur], bottom_kps[cur])
                            ans2 = self.getTriangle(inner_bottom_kps[inner_next_ele], inner_bottom_kps[inner_cur], bottom_kps[next_ele])  
                            tris.append(ans1)
                            tris.append(ans2)
                        inner_cur = inner_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(bottom_kps[next_ele], inner_bottom_kps[inner_cur], bottom_kps[cur])
                        ans2 = self.getTriangle(inner_bottom_kps[inner_next_ele], inner_bottom_kps[inner_cur], bottom_kps[next_ele])  
                        tris.append(ans1)
                        tris.append(ans2)
                    inner_cur = inner_next_ele
        else:
            forward_step = 0
            inner_cur = 0
            for cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % upper_kp_num
                inner_next_ele = (inner_cur + 1) % inner_upper_kp_num
                real_id = cur + forward_step
                crease_type = unit.getCrease()[real_id].getType()
                if real_id in inner_upper_problem_id:
                    if real_id not in upper_problem_id:
                        if 1:
                            ans = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
                            tris.append(ans)
                    else:
                        forward_step += 1
                        if 1:
                            ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
                            ans2 = self.getTriangle(inner_upper_kps[inner_next_ele], upper_kps[next_ele], inner_upper_kps[inner_cur])  
                            tris.append(ans1)
                            tris.append(ans2)
                        inner_cur = inner_next_ele
                else:
                    if 1:
                        ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
                        ans2 = self.getTriangle(inner_upper_kps[inner_next_ele], upper_kps[next_ele], inner_upper_kps[inner_cur])  
                        tris.append(ans1)
                        tris.append(ans2)
                    inner_cur = inner_next_ele
        
        # #repair
        # if upper_strong_modify:
        #     forward_step = 0
        #     inner_cur = 0
        #     for cur in range(0, upper_kp_num):
        #         next_ele = (cur + 1) % upper_kp_num
        #         inner_next_ele = (inner_cur + 1) % inner_upper_kp_num
        #         real_id = cur + forward_step
        #         crease_type = unit.getCrease()[real_id].getType()
        #         if real_id in inner_upper_problem_id:
        #             if real_id not in upper_problem_id:
        #                 if crease_type == VALLEY:
        #                     ans = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
        #                     tris.append(ans)
        #             else:
        #                 forward_step += 1
        #                 if crease_type == VALLEY:
        #                     ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
        #                     ans2 = self.getTriangle(inner_upper_kps[inner_next_ele], upper_kps[next_ele], inner_upper_kps[inner_cur])  
        #                     tris.append(ans1)
        #                     tris.append(ans2)
        #                 inner_cur = inner_next_ele
        #         else:
        #             if crease_type == VALLEY:
        #                 ans1 = self.getTriangle(upper_kps[next_ele], upper_kps[cur], inner_upper_kps[inner_cur])
        #                 ans2 = self.getTriangle(inner_upper_kps[inner_next_ele], upper_kps[next_ele], inner_upper_kps[inner_cur])  
        #                 tris.append(ans1)
        #                 tris.append(ans2)
        #             inner_cur = inner_next_ele
        # else:
        #     forward_step = 0
        #     inner_cur = 0
        #     for cur in range(0, kp_num):
        #         next_ele = (cur + 1) % kp_num
        #         inner_next_ele = (inner_cur + 1) % inner_kp_num
        #         real_id = cur + forward_step
        #         crease_type = unit.getCrease()[real_id].getType()
        #         if real_id in inner_bottom_problem_id:
        #             if real_id not in bottom_problem_id:
        #                 if crease_type == MOUNTAIN:
        #                     ans = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], inner_bottom_kps[inner_cur])
        #                     tris.append(ans)
        #             else:
        #                 forward_step += 1
        #                 if crease_type == MOUNTAIN:
        #                     ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], inner_bottom_kps[inner_cur])
        #                     ans2 = self.getTriangle(inner_bottom_kps[inner_next_ele], bottom_kps[next_ele], inner_bottom_kps[inner_cur])  
        #                     tris.append(ans1)
        #                     tris.append(ans2)
        #                 inner_cur = inner_next_ele
        #         else:
        #             if crease_type == MOUNTAIN:
        #                 ans1 = self.getTriangle(bottom_kps[next_ele], bottom_kps[cur], inner_bottom_kps[inner_cur])
        #                 ans2 = self.getTriangle(inner_bottom_kps[inner_next_ele], bottom_kps[next_ele], inner_bottom_kps[inner_cur])  
        #                 tris.append(ans1)
        #                 tris.append(ans2)
        #             inner_cur = inner_next_ele

        if additional_crease:
            #additional crease
            for cur in range(0, upper_kp_num):
                next_ele = (cur + 1) % upper_kp_num
                crease_type = unit.getCrease()[cur].getType()
                self.additional_crease_list.append(Crease(
                    [upper_kps[cur][X], upper_kps[cur][Y]],
                    [upper_kps[next_ele][X], upper_kps[next_ele][Y]], crease_type
                ))
        return tris

    def calculateTriPlaneForUnit(self, unit_id, inner=False):
        unit = self.unit_list[unit_id]
        add_hole = False
        another_points_list = []
        for ele in self.unit_hole_list:
            if ele[1] == unit_id:
                add_hole = True
                center = ele[0] + [0.0]
                points = generatePolygonByCenter(center, self.unit_hole_size, self.unit_hole_resolution)
                another_points_list.append(points)
        
        if self.method == "upper_bias":
            self.symmetry_flag = False
            self.unit_height = self.height
            self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =unit_id, 
                upper_bias          =self.bias, 
                down_bias           =self.min_bias, 
                base_height         =0,
                upper_height        =self.height, 
                add_hole            =add_hole, 
                another_points_list =another_points_list,
                additional_crease   =False
            )
        elif self.method == "both_bias":
            self.symmetry_flag = False
            self.unit_height = self.height
            self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =unit_id, 
                upper_bias          =self.bias, 
                down_bias           =self.bias, 
                base_height         =0,
                upper_height        =self.height, 
                add_hole            =add_hole, 
                another_points_list =another_points_list,
                additional_crease   =False
            )
        elif self.method == "symmetry":
            self.symmetry_flag = True
            self.unit_height = self.height / 2
            # add connection holes
            connection_left_holes = []
            connection_right_holes = []
            # judge if there is modified unit
            if self.using_modified_unit:
                try:
                    modified_unit = self.modified_unit_list[unit_id]
                    having_modified = True
                except:
                    having_modified = False
                    pass
                # add additional hole
                for ele in self.connection_hole_list:
                    if ele[1] == unit_id:
                        center = ele[0] + [0.0]
                        points = generatePolygonByCenter(center, self.connection_hole_size, self.unit_hole_resolution)
                        if ele[-1] == LEFT:
                            connection_left_holes.append(points)
                        else:
                            connection_right_holes.append(points)

            if len(connection_left_holes) > 0:
                add_hole = True

            if self.db_enable:
                if inner:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeightWithInner(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias, 
                        down_bias           =self.bias, 
                        base_height         =0,
                        upper_height        =self.height / 2, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_left_holes,
                        additional_crease   =False,
                        inner_bias = self.base_inner_bias
                    )
                else:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias + 3.0 * self.base_inner_bias, 
                        down_bias           =self.bias + 3.0 * self.base_inner_bias, 
                        base_height         =self.layer * self.print_accuracy,
                        upper_height        =self.height / 2, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_left_holes,
                        additional_crease   =False,
                        penalty             =3.0 * self.base_inner_bias
                    )
                if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                    add_hole = True
                else:
                    add_hole = False

                if not self.using_modified_unit:
                    if inner:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeightWithInner(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias, 
                            down_bias           =self.min_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            inner_bias = self.base_inner_bias
                        )
                    else:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias + 3.0 * self.base_inner_bias, 
                            down_bias           =self.min_bias + 3.0 * self.base_inner_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height - self.layer * self.print_accuracy, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            penalty             =3.0 * self.base_inner_bias
                        )
                elif self.using_modified_unit and having_modified:
                    if inner:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeightWithInner(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias, 
                            down_bias           =self.min_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            inner_bias = self.base_inner_bias
                        )
                    else:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =self.bias + 3.0 * self.base_inner_bias, 
                            down_bias           =self.min_bias + 3.0 * self.base_inner_bias, 
                            base_height         =self.height / 2 + self.board_height,
                            upper_height        =self.height + self.board_height - self.layer * self.print_accuracy, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                            penalty             =3.0 * self.base_inner_bias
                        )
            else:
                self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                    unit                =unit, 
                    unit_id             =unit_id, 
                    upper_bias          =self.min_bias, 
                    down_bias           =self.bias, 
                    base_height         =0,
                    upper_height        =self.height / 2, 
                    add_hole            =add_hole, 
                    another_points_list =deepcopy(another_points_list)+connection_left_holes,
                    additional_crease   =False
                )

                if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                    add_hole = True
                else:
                    add_hole = False

                if not self.using_modified_unit:
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2 + self.board_height,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False
                    )
                    
                elif self.using_modified_unit and having_modified:
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =modified_unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2 + self.board_height,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False
                    )
        elif self.method == 'binding':
            self.symmetry_flag = True
            self.unit_height = self.height / 2
            # add connection holes
            connection_left_holes = []
            connection_right_holes = []
            # judge if there is modified unit
            if self.using_modified_unit:
                try:
                    modified_unit = self.modified_unit_list[unit_id]
                    having_modified = True
                except:
                    having_modified = False
                    pass
                # add additional hole
                for ele in self.connection_hole_list:
                    if ele[1] == unit_id:
                        center = ele[0] + [0.0]
                        points = generatePolygonByCenter(center, self.connection_hole_size, self.unit_hole_resolution)
                        if ele[-1] == LEFT:
                            connection_left_holes.append(points)
                        else:
                            connection_right_holes.append(points)

            if len(connection_left_holes) > 0:
                add_hole = True

            if not self.thin_mode:
                # if self.db_enable:
                self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                    unit                =unit, 
                    unit_id             =unit_id, 
                    upper_bias          =self.min_bias, 
                    down_bias           =self.bias, 
                    base_height         =0,
                    upper_height        =self.height / 2. - self.print_accuracy / 2.0, 
                    add_hole            =add_hole, 
                    another_points_list =deepcopy(another_points_list)+connection_left_holes,
                    additional_crease   =False,
                    upper_tri           =False
                )

                self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                    unit                =unit, 
                    unit_id             =unit_id, 
                    upper_bias          =self.min_bias, 
                    down_bias           =self.min_bias, 
                    base_height         =self.height / 2. - self.print_accuracy / 2.0,
                    upper_height        =self.height / 2., 
                    add_hole            =add_hole, 
                    another_points_list =deepcopy(another_points_list)+connection_left_holes,
                    additional_crease   =False,
                    bottom_tri          =False
                )

                if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                    add_hole = True
                else:
                    add_hole = False

                if not self.using_modified_unit:
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2. + self.board_height,
                        upper_height        =self.height / 2. + self.board_height + self.print_accuracy / 2.0, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        upper_tri           =False
                    )

                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2. + self.board_height + self.print_accuracy / 2.0,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        bottom_tri           =False
                    )

                elif self.using_modified_unit and having_modified:
                    # self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                    #     unit                =modified_unit, 
                    #     unit_id             =unit_id, 
                    #     upper_bias          =self.bias, 
                    #     down_bias           =self.min_bias, 
                    #     base_height         =self.height / 2. + self.board_height,
                    #     upper_height        =self.height + self.board_height, 
                    #     add_hole            =add_hole, 
                    #     another_points_list =deepcopy(another_points_list)+connection_right_holes,
                    #     additional_crease   =False,
                    # )
                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =modified_unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.min_bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2. + self.board_height,
                        upper_height        =self.height / 2. + self.board_height + self.print_accuracy / 2.0, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        upper_tri           =False
                    )

                    self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                        unit                =modified_unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.bias, 
                        down_bias           =self.min_bias, 
                        base_height         =self.height / 2. + self.board_height + self.print_accuracy / 2.0,
                        upper_height        =self.height + self.board_height, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                        bottom_tri           =False
                    )
                # else:
                #     self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                #         unit                =unit, 
                #         unit_id             =unit_id, 
                #         upper_bias          =self.min_bias, 
                #         down_bias           =self.bias, 
                #         base_height         =0,
                #         upper_height        =self.height / 2., 
                #         add_hole            =add_hole, 
                #         another_points_list =deepcopy(another_points_list)+connection_left_holes,
                #         additional_crease   =False
                #     )

                #     if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                #         add_hole = True
                #     else:
                #         add_hole = False

                #     if not self.using_modified_unit:
                #         self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                #             unit                =unit, 
                #             unit_id             =unit_id, 
                #             upper_bias          =self.bias, 
                #             down_bias           =self.min_bias, 
                #             base_height         =self.height / 2. + self.board_height,
                #             upper_height        =self.height + self.board_height, 
                #             add_hole            =add_hole, 
                #             another_points_list =deepcopy(another_points_list)+connection_right_holes,
                #             additional_crease   =False
                #         )
                        
                #     elif self.using_modified_unit and having_modified:
                #         self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                #             unit                =modified_unit, 
                #             unit_id             =unit_id, 
                #             upper_bias          =self.bias, 
                #             down_bias           =self.min_bias, 
                #             base_height         =self.height / 2. + self.board_height,
                #             upper_height        =self.height + self.board_height, 
                #             add_hole            =add_hole, 
                #             another_points_list =deepcopy(another_points_list)+connection_right_holes,
                #             additional_crease   =False
                #         )
            else:
                if self.db_enable:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.print_accuracy, 
                        down_bias           =8.0 * self.print_accuracy, 
                        base_height         =0,
                        upper_height        =3.0 * self.print_accuracy, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_right_holes,
                        additional_crease   =False,
                    )

                    if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                        add_hole = True
                    else:
                        add_hole = False

                    if not self.using_modified_unit:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                        )
                    elif self.using_modified_unit and having_modified:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False,
                        )
                else:
                    self.tri_list[unit_id] = self.calculateTriPlaneWithBiasAndHeight(
                        unit                =unit, 
                        unit_id             =unit_id, 
                        upper_bias          =self.print_accuracy, 
                        down_bias           =8.0 * self.print_accuracy, 
                        base_height         =0,
                        upper_height        =self.print_accuracy * 3.0, 
                        add_hole            =add_hole, 
                        another_points_list =deepcopy(another_points_list)+connection_left_holes,
                        additional_crease   =False
                    )

                    if len(connection_right_holes) > 0 or len(another_points_list) > 0:
                        add_hole = True
                    else:
                        add_hole = False

                    if not self.using_modified_unit:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False
                        )
                        
                    elif self.using_modified_unit and having_modified:
                        self.tri_list[unit_id] += self.calculateTriPlaneWithBiasAndHeight(
                            unit                =modified_unit, 
                            unit_id             =unit_id, 
                            upper_bias          =8.0 * self.print_accuracy, 
                            down_bias           =self.print_accuracy, 
                            base_height         =self.print_accuracy * 3.0 + self.board_height,
                            upper_height        =self.print_accuracy * 6.0 + self.board_height, 
                            add_hole            =add_hole, 
                            another_points_list =deepcopy(another_points_list)+connection_right_holes,
                            additional_crease   =False
                        )
    
    def calculateTriPlaneForBoard(self, unit_id, base_height=0, upper_height=None):
        if upper_height == None:
            upper_height = self.board_height
        unit = self.unit_list[unit_id]

        # exist_modify = False
        # for j in range(len(unit.getCrease())):
        #     if self.unit_bias_list[unit_id][j] != None:
        #         exist_modify = True
        #         break
        
        add_hole = False
        another_points_list = []
        for ele in self.unit_hole_list:
            if ele[1] == unit_id:
                add_hole = True
                center = ele[0] + [0.0]
                points = generatePolygonByCenter(center, self.unit_hole_size, self.unit_hole_resolution)
                another_points_list.append(points)

        # add additional hole
        connection_left_holes = []
        for ele in self.connection_hole_list:
            if ele[1] == unit_id:
                center = ele[0] + [0.0]
                points = generatePolygonByCenter(center, self.connection_hole_size, self.unit_hole_resolution)
                if ele[-1] == LEFT:
                    connection_left_holes.append(points)

        if len(connection_left_holes) > 0:
            add_hole = True

        if 1:
            self.board_tri_list += self.calculateTriPlaneWithBiasAndHeight(
                unit                =unit, 
                unit_id             =unit_id, 
                upper_bias          =self.bias, 
                down_bias           =self.bias, 
                base_height         =base_height,
                upper_height        =upper_height, 
                add_hole            =add_hole, 
                another_points_list =another_points_list+connection_left_holes,
                additional_crease   =False,
                penalty=3.0 * self.base_inner_bias if self.method == 'symmetry' else self.bias,
                accumulation        =True
            )
    
    def calculateTriPlaneForPillar(self, base_height=0, upper_height=None):
        if upper_height == None:
            upper_height = self.board_height

        # self.board_tri_list.clear()
        for i in range(len(self.pillar_unit_list)):
            pillars = self.pillar_unit_list[i]
            for pillar in pillars:
                reverse_pillar = []
                for j in range(0, -len(pillar), -1):
                    reverse_pillar.append(pillar[j])
                self.board_tri_list += self.calculateTriPlaneWithBiasAndHeight(
                    unit                =reverse_pillar, 
                    unit_id             =-1,
                    upper_bias          =0, 
                    down_bias           =0, 
                    base_height         =base_height,
                    upper_height        =upper_height, 
                    add_hole            =False, 
                    another_points_list =[],
                    additional_crease   =False,
                    penalty=3.0 * self.base_inner_bias if self.method == 'symmetry' else self.bias
                )
        
    def calculateTriPlaneForAllUnit(self, inner=False):
        for i in range(len(self.unit_list)):
            self.calculateTriPlaneForUnit(i, inner)    

    def generateBoard(self):
        for unit_id in range(len(self.unit_list)):
            self.calculateTriPlaneForBoard(unit_id)
        if self.method == "binding":
            self.calculateTriPlaneForPillar()
            
    def addSpace(self, number):
        for i in range(0, number):
            self.s += ' '

    def addInfoToStlFile(self, tris):
        space_num = 1
        for ele in tris:
            self.addSpace(space_num)
            self.s += 'facet normal ' + str(ele[0][0]) + ' ' + str(ele[0][1]) + ' ' + str(ele[0][2]) + '\n'
            space_num += 1
            self.addSpace(space_num)
            self.s += 'outer loop\n'
            space_num += 1
            for i in range(0, 3):
                self.addSpace(space_num)
                self.s += 'vertex ' + str(ele[1][i][0]) + ' ' + str(ele[1][i][1]) + ' ' + str(ele[1][i][2]) + '\n'
            space_num -= 1
            self.addSpace(space_num)
            self.s += 'endloop\n'
            space_num -= 1
            self.addSpace(space_num)
            self.s += 'endfacet\n'

    def outputUnitStl(self, unit_id, filepath):
        self.s = 'solid PyGamic generated __Unit_' + str(unit_id) + '__ SLA File\n'
        tris = self.tri_list[unit_id]
        self.addInfoToStlFile(tris)
        self.s += 'endsolid\n'

        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputAllStl(self, filepath):
        self.s = 'solid PyGamic generated __All_Units__ SLA File\n'
        for i in range(len(self.unit_list)):
            tris = self.tri_list[i]
            self.addInfoToStlFile(tris)
        self.s += 'endsolid\n'
        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputCreaseStl(self, crease_id, filepath):
        self.s = 'solid PyGamic generated __Crease__ SLA File\n'
        tris = self.crease_tri_list[crease_id]
        self.addInfoToStlFile(tris)
        self.s += 'endsolid\n'

        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputAllCreaseStl(self, filepath):
        self.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
        for i in range(len(self.valid_crease_list)):
            tris = self.crease_tri_list[i]
            if tris != None:
                self.addInfoToStlFile(tris)
        self.s += 'endsolid\n'
        with open(filepath, 'w') as f:
            f.write(self.s)

    def outputBoardStl(self, filepath):
        self.s = 'solid PyGamic generated __Board__ SLA File\n'
        self.addInfoToStlFile(self.board_tri_list)
        self.s += 'endsolid\n'
        with open(filepath, 'w') as f:
            f.write(self.s)
    
    def outputStringStl(self, filepath):
        self.s = 'solid PyGamic generated __String__ SLA File\n'
        self.addInfoToStlFile(self.string_tri_list)
        self.s += 'endsolid\n'
        with open(filepath, 'w') as f:
            f.write(self.s)
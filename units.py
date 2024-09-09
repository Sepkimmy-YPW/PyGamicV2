# import numpy as np
import math
from utils import *
from copy import deepcopy
# import heapq

class Miura:
    def __init__(self, origin, main_length, width, alpha, function_type, entry_flag, direction, trans_origin, trans_length, border_start, data=None) -> None:
        if data == None:
            self.origin = origin
            self.main_length = main_length
            self.width = width
            self.alpha = alpha
            self.function_type = function_type
            self.entry_flag = entry_flag
            self.direction = direction
            self.trans_origin = trans_origin
            self.trans_length = trans_length
            self.border_start = border_start
        else:
            self.origin = data[0]
            self.main_length = data[1]
            self.width = data[2]
            self.alpha = data[3]
            self.function_type = data[4]
            self.entry_flag = data[5]
            self.direction = data[6]
            self.trans_origin = data[7]
            self.trans_length = data[8]
            self.border_start = data[9]
        self.up_line = EMPTY
        self.down_line = EMPTY
        self.entity_line = SHOW_COLOR
        self.kp = []
        self.line = []
        self.line_connect_to_body = []
        self.left_percent = 0.0
        self.right_percent = 1.0
        self.border_left_percent = 0.0
        self.border_right_percent = 1.0
        self.main_line_left_percent = 0.0
        self.main_line_right_percent = 1.0
        self.most_left = origin[0]
        self.crease_y_length = width / 2
        self.crease_x_length = self.crease_y_length / math.tan(self.alpha)
        self.modify_flag = False
        self.enable_connect_to_body = False
        self.connect_direction = None
        self.same_flag = [True, True, True] #01 123 46/57
        if function_type == ACTIVE_MIURA:
            self.border_end = self.origin[0] + self.main_length - self.crease_x_length
            if self.border_start > self.border_end:
                self.border_start = self.border_end
        else:
            self.border_end = self.origin[0] + self.main_length + self.crease_x_length
            if self.border_start > self.border_end:
                self.border_start = self.border_end

    def getFunctionType(self):
        return self.function_type
        
    def setMainLength(self, main_length):
        self.main_length = main_length

    def setUpBorder(self, up_line):
        self.up_line = up_line

    def setDownBorder(self, down_line):
        self.down_line = down_line

    def setBorder(self, up_line, down_line):
        self.up_line = up_line
        self.down_line = down_line
    
    def setEntityColor(self, color):
        self.entity_line = color

    def setLeftAndRightPercent(self, left_percent=None, right_percent=None):
        if left_percent != None:
            self.left_percent = left_percent
        if right_percent != None:
            self.right_percent = right_percent
    
    def setLeftAndRightPercentOfMainLine(self, left_percent=None, right_percent=None):
        if left_percent != None:
            self.main_line_left_percent = left_percent
        if right_percent != None:
            self.main_line_right_percent = right_percent

    def setLeftAndRightPercentOfBorder(self, left_percent, right_percent):
        self.border_left_percent = left_percent
        self.border_right_percent = right_percent

    def getMainLine(self):
        crease_y_length = self.crease_y_length
        crease_x_length = self.crease_x_length
        if crease_x_length < 1e-8:
            crease_x_length = 0.0
        if self.function_type == ACTIVE_MIURA:
            return [
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length - crease_x_length, self.origin[1]),
                    BORDER,
                    False
                ), 
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length - crease_x_length, self.origin[1] + self.width),
                    BORDER,
                    False
                )
            ]
        else:
            return [
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length + crease_x_length, self.origin[1]),
                    BORDER,
                    False
                ), 
                Crease(
                    (self.origin[0] + self.main_length, self.origin[1] + crease_y_length),
                    (self.origin[0] + self.main_length + crease_x_length, self.origin[1] + self.width),
                    BORDER,
                    False
                )
            ]

    def getBorderXPoint(self):
        return self.border_end

    def getNextBorderStartXPoint(self):
        if self.border_start > self.border_end:
            return self.border_start
        else:
            return self.border_end

    def setBorderXPoint(self, x):
        if x < self.border_end:
            self.modify_flag = True
            self.border_end = x
        
    def getEndPoint(self):
        if self.modify_flag:
            return self.border_end
        else:
            return self.border_start

    def setEnableConnectionToBody(self, flag: bool, direction):
        self.enable_connect_to_body = flag
        self.connect_direction = direction

    def getKeypoint(self):
        self.kp.clear()
        # 1
        self.kp.append((self.origin[0], self.origin[1] + self.width / 2))
        crease_y_length = self.crease_y_length
        crease_x_length = self.crease_x_length
        if crease_x_length < 1e-8:
            crease_x_length = 0.0
        if self.function_type == ACTIVE_MIURA:
            # 3
            flag = self.main_length - (1 - self.main_line_right_percent) * crease_x_length
            if flag <= 0.0:
                self.kp.append((self.origin[0], self.origin[1] + crease_y_length))
            else:
                self.kp.append((self.origin[0] + flag, self.origin[1] + crease_y_length))
                self.same_flag[0] = False
            self.kp.append((self.origin[0] + self.main_length - (1 - self.right_percent) * crease_x_length, self.origin[1] + self.right_percent * crease_y_length))
            self.kp.append((self.origin[0] + self.main_length - (1 - self.right_percent) * crease_x_length, self.origin[1] + (2 - self.right_percent) * crease_y_length))
            if self.right_percent != 1:
                self.same_flag[1] = False
            # 2/2
            self.kp.append((self.origin[0] + self.main_length - crease_x_length * (1 - self.left_percent), self.origin[1] + crease_y_length * self.left_percent))
            self.kp.append((self.origin[0] + self.main_length - crease_x_length * (1 - self.left_percent), self.origin[1] + crease_y_length * (2 - self.left_percent)))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1]))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1] + self.width))
            if self.left_percent != 0:
                self.same_flag[2] = False
        else:
            # 3
            flag = self.main_length + self.main_line_left_percent * crease_x_length
            if flag <= 1e-5:
                self.kp.append((self.origin[0], self.origin[1] + crease_y_length))
            else:
                self.kp.append((self.origin[0] + flag, self.origin[1] + crease_y_length))
                self.same_flag[0] = False
            self.kp.append((self.origin[0] + self.main_length + self.left_percent * crease_x_length, self.origin[1] + (1 - self.left_percent) * crease_y_length))
            self.kp.append((self.origin[0] + self.main_length + self.left_percent * crease_x_length, self.origin[1] + (1 + self.left_percent) * crease_y_length))
            if self.left_percent != 0:
                self.same_flag[1] = False
            # 2/2
            self.kp.append((self.origin[0] + self.main_length + crease_x_length * self.right_percent, self.origin[1] + crease_y_length * (1 - self.right_percent)))
            self.kp.append((self.origin[0] + self.main_length + crease_x_length * self.right_percent, self.origin[1] + crease_y_length * (1 + self.right_percent)))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1]))
            self.kp.append((max(self.border_start, self.border_end), self.origin[1] + self.width))
            if self.right_percent != 1:
                self.same_flag[2] = False

        return self.kp

    def getLine(self):
        self.line.clear()
        self.line_connect_to_body.clear()
        kp = self.getKeypoint()
        show_color = True
        if (self.entity_line == BLACK_COLOR):
            show_color = False
        if self.entry_flag == V:
            flag = VALLEY
            op_flag = MOUNTAIN
        else:
            flag = MOUNTAIN
            op_flag = VALLEY
        # if (self.main_length != 0):
        self.line.append(Crease(kp[0], kp[1], flag))
        self.line.append(Crease(kp[1], kp[2], BORDER, show_color))
        self.line.append(Crease(kp[1], kp[3], BORDER, show_color))
        if (self.function_type == ACTIVE_MIURA):
            self.line.append(Crease(kp[2], kp[4], op_flag, show_color))
            self.line.append(Crease(kp[3], kp[5], op_flag, show_color))
        else:
            self.line.append(Crease(kp[2], kp[4], flag, show_color))
            self.line.append(Crease(kp[3], kp[5], flag, show_color))
        self.line.append(Crease(kp[4], kp[-2], BORDER, show_color))
        self.line.append(Crease(kp[5], kp[-1], BORDER, show_color))
        
        if (self.up_line == HAVE_BORDER):
            if self.border_end > self.border_start:
                if (self.enable_connect_to_body and self.connect_direction == UP):
                    line = Crease(
                        [self.border_end, kp[-1][1]], 
                        [self.border_start, kp[-1][1]], 
                        op_flag
                    )
                    self.line.append(line)
                    self.line_connect_to_body.append(line)
                else:
                    self.line.append(Crease(
                        [self.border_end, kp[-1][1]],
                        [self.border_start, kp[-1][1]], 
                        BORDER
                    ))
        elif (self.up_line == HAVE_CONNECTION):
            if self.border_end > self.border_start:
                self.line.append(Crease(
                    [self.border_end, kp[-1][1]],
                    [self.border_start, kp[-1][1]], 
                    op_flag
                ))
        if (self.down_line == HAVE_BORDER):
            if self.border_end > self.border_start:
                if (self.enable_connect_to_body and self.connect_direction == DOWN):
                    line = Crease(
                        [self.border_end, kp[-2][1]], 
                        [self.border_start, kp[-2][1]], 
                        op_flag
                    )
                    self.line.append(line)
                    self.line_connect_to_body.append(line)
                else:
                    self.line.append(Crease(
                        [self.border_end, kp[-2][1]], 
                        [self.border_start, kp[-2][1]], 
                        BORDER
                    ))
        elif (self.down_line == HAVE_CONNECTION):
            if self.border_end > self.border_start:
                self.line.append(Crease(
                    [self.border_end, kp[-2][1]], 
                    [self.border_start, kp[-2][1]], 
                    op_flag
                ))

        return self.line
    
    def getLineConnectToBody(self):
        return self.line_connect_to_body

    def getSameFlagList(self):
        return self.same_flag

    def getTransition(self):
        tmp_origin = deepcopy(self.origin)
        if self.main_length > 0.0:
            tmp_origin[0] += self.main_length
        tmp_trans_origin = deepcopy(self.trans_origin)
        tmp_trans_origin[0] += self.trans_length
        tmp_entry_flag = not deepcopy(self.entry_flag)
        return tmp_origin, tmp_trans_origin, tmp_entry_flag

class LeanMiura:
    def __init__(self, unit_width, entry_flag, tsp, copy_time, half_flag, 
                 stretch_length, connection_flag, con_left_length, con_right_length, connection_hole_size) -> None:
        self.unit_width = unit_width
        self.entry_flag = entry_flag
        self.tsp = tsp
        self.copy_time = copy_time
        self.half_flag = half_flag
        self.stretch_length = stretch_length
        self.connection_flag = connection_flag
        self.con_left_length = con_left_length
        self.con_right_length = con_right_length
        self.connection_hole_size = connection_hole_size

        self.enable_global_modify = False

        self.global_left_bias = 0
        self.global_right_bias = 0

        self.origami_length = 0
        self.origami_width = 0

        # Left part of lean-Miura
        self.left_kp = []
        self.left_line = []

        # Right part of lean-Miura
        self.right_kp = []
        self.right_line = []

        self.middle_line = []

        self.left_unit = []
        self.right_unit = []
        self.middle_unit = []

        self.connection_left_unit = []
        self.connection_right_unit = []

        half_unit_width = self.unit_width / 2

        # define bias and get length of origami
        if self.half_flag == NO_HALF:
            if self.connection_flag:
                self.global_left_bias = self.con_left_length + half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            else:
                self.global_left_bias = half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            self.origami_length = self.stretch_length + 2 * half_unit_width * self.copy_time
        elif self.half_flag == LEFT_HALF:
            if self.connection_flag:
                self.global_left_bias = self.con_left_length + half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            else:
                self.global_left_bias = half_unit_width * self.copy_time
                self.global_right_bias = self.global_left_bias + self.stretch_length
            self.origami_length = self.stretch_length + half_unit_width * self.copy_time
        elif self.half_flag == RIGHT_HALF:
            self.global_left_bias = 0
            self.global_right_bias = self.stretch_length
            self.origami_length = self.stretch_length + half_unit_width * self.copy_time
        
        self.origami_width = self.unit_width * self.copy_time

    def clearUnit(self):
        self.left_unit.clear()
        self.right_unit.clear()
        self.middle_unit.clear()
        self.connection_left_unit.clear()
        self.connection_right_unit.clear()

    def setUnitWidth(self, unit_width):
        self.unit_width = unit_width

    def setEntryFlag(self, entry_flag):
        self.entry_flag = entry_flag

    def setTransitionStartPoint(self, tsp):
        self.tsp = tsp

    def setCopyTime(self, copy_time):
        self.copy_time = copy_time

    def setStretchLength(self, stretch_length):
        self.stretch_length = stretch_length

    def setConnectionFlag(self, connection_flag):
        self.connection_flag = connection_flag

    def setConnectionLeftLength(self, con_left_length):
        self.con_left_length = con_left_length

    def setConnectionRightLength(self, con_right_length):
        self.con_right_length = con_right_length

    def setHalfFlag(self, half_flag):
        self.half_flag = half_flag

    def setEnableModify(self, enabled: bool):
        self.enable_global_modify = enabled

    def getOrigamiInfo(self):
        return self.origami_length, self.origami_width
    
    def getHalfFlag(self):
        return self.half_flag
    
    def addLeftKp(self, kp):
        self.left_kp.append([kp[X] + self.tsp[X] + self.global_left_bias, kp[Y] + self.tsp[Y]])

    def addRightKp(self, kp):
        self.right_kp.append([kp[X] + self.tsp[X] + self.global_right_bias, kp[Y] + self.tsp[Y]])

    def addLeftLine(self, kp1, kp2, type_crease, show_color=True):
        self.left_line.append(Crease(kp1, kp2, type_crease, show_color))

    def addRightLine(self, kp1, kp2, type_crease, show_color=True):
        self.right_line.append(Crease(kp1, kp2, type_crease, show_color))

    def getKeyPoint(self):
        half_unit_width = self.unit_width / 2
 
        self.origami_width = self.unit_width * self.copy_time
        # add kps
        #1 lean direction
        for i in range(self.copy_time):
            self.addLeftKp([0 - half_unit_width * i, 0 + half_unit_width * i])
            self.addLeftKp([0 - half_unit_width * i, self.origami_width - half_unit_width * i])
            self.addRightKp([0 + half_unit_width * i, 0 + half_unit_width * i])
            self.addRightKp([0 + half_unit_width * i, self.origami_width - half_unit_width * i])
        left_x = -half_unit_width * self.copy_time
        right_x = half_unit_width * self.copy_time
        y = half_unit_width * self.copy_time
        self.addLeftKp([left_x, y])
        self.addRightKp([right_x, y])
        #2 outborder
        for i in range(1, self.copy_time + 1):
            self.addLeftKp([left_x, y - half_unit_width * i])
            self.addLeftKp([left_x, y + half_unit_width * i])
            self.addRightKp([right_x, y - half_unit_width * i])
            self.addRightKp([right_x, y + half_unit_width * i])
        #3 connection
        if self.connection_flag:
            left_x = -half_unit_width * self.copy_time - self.con_left_length
            right_x = half_unit_width * self.copy_time + self.con_right_length
            self.addLeftKp([left_x, y])
            self.addRightKp([right_x, y])
            for i in range(1, self.copy_time + 1):
                self.addLeftKp([left_x, y - half_unit_width * i])
                self.addLeftKp([left_x, y + half_unit_width * i])
                self.addRightKp([right_x, y - half_unit_width * i])
                self.addRightKp([right_x, y + half_unit_width * i])
        
        # LEFT_HALF
        if self.half_flag == LEFT_HALF:
            return self.left_kp
            
        # RIGHT_HALF
        if self.half_flag == RIGHT_HALF:
            return self.right_kp
        
        # NO_HALF
        return self.left_kp + self.right_kp

    def getLine(self):
        # get flag
        if self.entry_flag == V:
            flag = VALLEY
            op_flag = MOUNTAIN
        else:
            flag = MOUNTAIN
            op_flag = VALLEY

        #1 vertical
        for i in range(self.copy_time):
            id = self.copy_time - i
            if id % 2:
                f = op_flag
            else:
                f = flag
            self.addLeftLine(self.left_kp[2 * i], self.left_kp[2 * i + 1], f)
            self.addRightLine(self.right_kp[2 * i], self.right_kp[2 * i + 1], f)

        #2 45deg
        for i in range(self.copy_time):
            id = self.copy_time - i
            if id % 2:
                f = flag
            else:
                f = op_flag
            self.addLeftLine(self.left_kp[2 * i], self.left_kp[2 * i + 2], f)
            self.addLeftLine(self.left_kp[2 * i + 1], self.left_kp[2 * i + 3 if i < self.copy_time - 1 else 2 * i + 2], f)
            self.addRightLine(self.right_kp[2 * i], self.right_kp[2 * i + 2], f)
            self.addRightLine(self.right_kp[2 * i + 1], self.right_kp[2 * i + 3 if i < self.copy_time - 1 else 2 * i + 2], f)

        #inner horizontal
        if self.connection_flag:
            show_color = True
        else:
            show_color = False
        
        center_id = 2 * self.copy_time
        for i in range(1, self.copy_time + 1):
            if i == self.copy_time:
                f = BORDER
            elif i % 2:
                f = op_flag
            else:
                f = flag
            self.addLeftLine(self.left_kp[center_id - 2 * i], self.left_kp[center_id - 1 + 2 * i], f)
            self.addLeftLine(self.left_kp[center_id - 2 * i + 1], self.left_kp[center_id + 2 * i], f)
            self.addRightLine(self.right_kp[center_id - 2 * i], self.right_kp[center_id - 1 + 2 * i], f)
            self.addRightLine(self.right_kp[center_id - 2 * i + 1], self.right_kp[center_id + 2 * i], f)
        
        #border
        self.addLeftLine(self.left_kp[center_id], self.left_kp[center_id + 1], op_flag, show_color)
        self.addLeftLine(self.left_kp[center_id], self.left_kp[center_id + 2], op_flag, show_color)
        self.addRightLine(self.right_kp[center_id], self.right_kp[center_id + 1], op_flag, show_color)
        self.addRightLine(self.right_kp[center_id], self.right_kp[center_id + 2], op_flag, show_color)
        
        for i in range(1, self.copy_time):
            if i % 2:
                f = flag
            else:
                f = op_flag
            self.addLeftLine( self.left_kp [center_id - 1 + 2 * i], self.left_kp [center_id + 1 + 2 * i], f, show_color)
            self.addLeftLine( self.left_kp [center_id + 2 * i],     self.left_kp [center_id + 2 + 2 * i], f, show_color)
            self.addRightLine(self.right_kp[center_id - 1 + 2 * i], self.right_kp[center_id + 1 + 2 * i], f, show_color)
            self.addRightLine(self.right_kp[center_id + 2 * i],     self.right_kp[center_id + 2 + 2 * i], f, show_color)
    
        #connection
        if self.connection_flag:
            connection_center_id = 4 * self.copy_time + 1
            self.addLeftLine(self.left_kp[center_id], self.left_kp[connection_center_id], flag, show_color)
            self.addRightLine(self.right_kp[center_id], self.right_kp[connection_center_id], flag, show_color)
            for i in range(1, self.copy_time + 1):
                if i == self.copy_time:
                    f = BORDER
                elif i % 2:
                    f = op_flag
                else:
                    f = flag
                self.addLeftLine(self.left_kp[center_id - 1 + 2 * i], self.left_kp[connection_center_id - 1 + 2 * i], f)
                self.addLeftLine(self.left_kp[center_id + 2 * i], self.left_kp[connection_center_id + 2 * i], f)
                self.addRightLine(self.right_kp[center_id - 1 + 2 * i], self.right_kp[connection_center_id - 1 + 2 * i], f)
                self.addRightLine(self.right_kp[center_id + 2 * i], self.right_kp[connection_center_id + 2 * i], f)
            #connection border
            self.addLeftLine(self.left_kp[connection_center_id], self.left_kp[connection_center_id + 1], BORDER)
            self.addLeftLine(self.left_kp[connection_center_id], self.left_kp[connection_center_id + 2], BORDER)
            self.addRightLine(self.right_kp[connection_center_id], self.right_kp[connection_center_id + 1], BORDER)
            self.addRightLine(self.right_kp[connection_center_id], self.right_kp[connection_center_id + 2], BORDER)
            for i in range(1, self.copy_time):
                self.addLeftLine( self.left_kp [connection_center_id - 1 + 2 * i], self.left_kp [connection_center_id + 1 + 2 * i], BORDER)
                self.addLeftLine( self.left_kp [connection_center_id + 2 * i],     self.left_kp [connection_center_id + 2 + 2 * i], BORDER)
                self.addRightLine(self.right_kp[connection_center_id - 1 + 2 * i], self.right_kp[connection_center_id + 1 + 2 * i], BORDER)
                self.addRightLine(self.right_kp[connection_center_id + 2 * i],     self.right_kp[connection_center_id + 2 + 2 * i], BORDER)
        #middle
        if self.stretch_length > 0:
            self.middle_line.append(Crease(self.left_kp[0], self.right_kp[0], BORDER))
            self.middle_line.append(Crease(self.left_kp[1], self.right_kp[1], BORDER))
            
            if self.half_flag == LEFT_HALF:
                self.middle_line.append(Crease(self.right_kp[0], self.right_kp[1], BORDER))
            
            if self.half_flag == RIGHT_HALF:
                self.middle_line.append(Crease(self.left_kp[0], self.left_kp[1], BORDER))

        else:
            if self.half_flag == LEFT_HALF:
                self.left_line[0].crease_type = BORDER
            
            if self.half_flag == RIGHT_HALF:
                self.right_line[0].crease_type = BORDER

        if self.half_flag == LEFT_HALF:
            return self.left_line + self.middle_line
        
        if self.half_flag == RIGHT_HALF:
            return self.right_line + self.middle_line
        
        return self.left_line + self.right_line + self.middle_line
    
    def makeUnits(self, kp_id: list, ret=False):
        ul = Unit()
        ur = Unit()
        kp_len = len(kp_id)

        i = 0
        j = 0
        while i < kp_len:
            next_i = (i + 1) % kp_len
            next_j = (j - 1 + kp_len) % kp_len

            # left kp
            lstart = self.left_kp[kp_id[i]]
            lend = self.left_kp[kp_id[next_i]]
            # right kp
            rstart = self.right_kp[kp_id[j]]
            rend = self.right_kp[kp_id[next_j]]

            left_crease_type = BORDER
            for line in self.left_line:
                if line.pointIsStartAndEnd(lstart, lend):
                    left_crease_type = line.getType()
                    break

            right_crease_type = BORDER
            for line in self.right_line:
                if line.pointIsStartAndEnd(rstart, rend):
                    right_crease_type = line.getType()
                    break

            ul.addCrease(Crease(lstart, lend, left_crease_type))
            ur.addCrease(Crease(rstart, rend, right_crease_type))

            i += 1
            j -= 1

        if not ret:
            self.left_unit.append(ul)
            self.right_unit.append(ur)
        else:
            return ul, ur

    def addConnectionUnit(self):
        if self.connection_flag:
            part_horizontal = 2 * self.copy_time
            part_connection = 4 * self.copy_time + 1

            ul, ur = self.makeUnits([part_connection, part_horizontal, part_horizontal + 1, part_connection + 1], True)
            self.connection_left_unit.append(ul)
            self.connection_right_unit.append(ur)
            ul, ur = self.makeUnits([part_connection, part_connection + 2, part_horizontal + 2, part_horizontal], True)
            self.connection_left_unit.append(ul)
            self.connection_right_unit.append(ur)
            for i in range(1, self.copy_time):
                ul, ur = self.makeUnits([part_connection + 2 * i - 1, part_horizontal + 2 * i - 1, part_horizontal + 2 * i + 1, part_connection + 2 * i + 1], True)
                self.connection_left_unit.append(ul)
                self.connection_right_unit.append(ur)
                ul, ur = self.makeUnits([part_connection + 2 * i, part_connection + 2 * i + 2, part_horizontal + 2 * i + 2, part_horizontal + 2 * i], True)
                self.connection_left_unit.append(ul)
                self.connection_right_unit.append(ur)

    def getUnits(self, connection=False):
        self.clearUnit()

        part_horizontal = 2 * self.copy_time
        # part_connection = 4 * self.copy_time + 1

        #1 vertical
        for i in range(self.copy_time - 1):
            self.makeUnits([2 * i, 2 * i + 2, 2 * i + 3, 2 * i + 1])

        self.makeUnits([part_horizontal - 2, part_horizontal, part_horizontal - 1])

        #2 horizontal
        self.makeUnits([part_horizontal - 2, part_horizontal + 1, part_horizontal])
        self.makeUnits([part_horizontal - 1, part_horizontal, part_horizontal + 2])

        for i in range(1, self.copy_time):
            self.makeUnits([part_horizontal - 2 * i, part_horizontal - 2 * i - 2, part_horizontal + 2 * i + 1, part_horizontal + 2 * i - 1])
            self.makeUnits([part_horizontal - 2 * i - 1, part_horizontal - 2 * i + 1, part_horizontal + 2 * i, part_horizontal + 2 * i + 2])

        if self.stretch_length > 0:
            mid_unit = Unit()
            if self.entry_flag == V:
                flag = MOUNTAIN
            else:
                flag = VALLEY

            mid_unit.addCrease(Crease(self.right_kp[0], self.left_kp[0], BORDER))
            if self.half_flag == RIGHT_HALF:
                mid_unit.addCrease(Crease(self.left_kp[0], self.left_kp[1], BORDER))
            else:
                mid_unit.addCrease(Crease(self.left_kp[0], self.left_kp[1], flag))
            mid_unit.addCrease(Crease(self.left_kp[1], self.right_kp[1], BORDER))
            if self.half_flag == LEFT_HALF:
                mid_unit.addCrease(Crease(self.right_kp[1], self.right_kp[0], BORDER))
            else:
                mid_unit.addCrease(Crease(self.right_kp[1], self.right_kp[0], flag))
            self.middle_unit.append(mid_unit)

        if connection:
            self.addConnectionUnit()

        if self.half_flag == LEFT_HALF:
            return self.left_unit + self.middle_unit + self.connection_left_unit
    
        if self.half_flag == RIGHT_HALF:
            return self.right_unit + self.middle_unit + self.connection_right_unit
        
        return self.left_unit + self.right_unit + self.middle_unit + self.connection_left_unit + self.connection_right_unit

def getUnitWithinMiura(miura1: Miura, miura2: Miura):
    type1 = miura1.getFunctionType()
    type2 = miura2.getFunctionType()
    same_flag1 = miura1.getSameFlagList()
    same_flag2 = miura2.getSameFlagList()
    kp1 = miura1.getKeypoint()
    kp2 = miura2.getKeypoint()
    line1 = miura1.getLine()
    line2 = miura2.getLine()
    unit1 = Unit()
    unit2 = Unit()
    if type2 == ACTIVE_MIURA:
        if type1 == PASSIVE_MIURA:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            if not same_flag2[1]:
                unit1.addCrease(line2[1])
                unit2.addCrease(Crease(
                    line2[2][END],
                    line2[2][START],
                    line2[2].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                    line2[6][END],
                    line2[6][START],
                    line2[6].getType()
                ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
            if not same_flag1[1]:
                unit1.addCrease(Crease(
                    line1[1][1],
                    line1[1][0],
                    line1[1].getType()
                ))
                unit2.addCrease(line1[2])
        else:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            if not same_flag2[1]:
                unit1.addCrease(line2[1])
                unit2.addCrease(Crease(
                    line2[2][END],
                    line2[2][START],
                    line2[2].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                        line2[6][END],
                        line2[6][START],
                        line2[6].getType()
                    ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
    else:
        if type1 == PASSIVE_MIURA:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                        line2[6][END],
                        line2[6][START],
                        line2[6].getType()
                    ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
        else:
            if not same_flag2[0]:
                unit1.addCrease(line2[0])
                unit2.addCrease(Crease(
                    line2[0][END],
                    line2[0][START],
                    line2[0].getType()
                ))
            unit1.addCrease(line2[3])
            unit2.addCrease(Crease(
                line2[4][END],
                line2[4][START],
                line2[4].getType()
            ))
            if not same_flag2[2]:
                if kp2[6][0] < kp2[4][0]:
                    unit1.addCrease(line2[5])
                    unit2.addCrease(Crease(
                        line2[6][END],
                        line2[6][START],
                        line2[6].getType()
                    ))
            if kp2[6][0] <= kp2[4][0]:
                unit1.addCrease(line2[8])
                unit2.addCrease(Crease(
                    line2[7][END],
                    line2[7][START],
                    line2[7].getType()
                ))
                if not same_flag1[2]:
                    unit1.addCrease(Crease(
                        line1[5][1],
                        line1[5][0],
                        line1[5].getType()
                    ))
                    unit2.addCrease(line1[6])
            unit1.addCrease(Crease(
                line1[3][1],
                line1[3][0],
                line1[3].getType()
            ))
            unit2.addCrease(line1[4])
    unit2_modified = Unit()
    creases = unit2.getCrease()
    unit2_length = len(creases)
    for i in range(0, -unit2_length, -1):
        unit2_modified.addCrease(creases[i])
    return unit1, unit2_modified

class UnitPackParser:
    def __init__(self, tsp, kps, lines, lines_type) -> None:
        self.tsp = tsp
        self.maximum_finding_level = 8

        self.new_kps = []
        self.new_lines = []

        for ele in kps:
            self.new_kps.append([ele[X] + tsp[X], ele[Y] + tsp[Y]])

        for i in range(len(lines)):
            ele = lines[i]
            self.new_lines.append(Crease(
                [ele[START][X] + tsp[X], ele[START][Y] + tsp[Y]],
                [ele[END][X] + tsp[X], ele[END][Y] + tsp[Y]],
                lines_type[i]
            ))

    def setMaximumNumberOfEdgeInAllUnit(self, n):
        self.maximum_finding_level = n

    def getMaxDistance(self):
        max_x = max([self.new_kps[i][X] for i in range(len(self.new_kps))])
        min_x = min([self.new_kps[i][X] for i in range(len(self.new_kps))])
        max_y = max([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        min_y = min([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        return max(max_x - min_x, max_y - min_y)
    
    def getTotalBias(self):
        units = self.getUnits()
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
    
    def getKeyPoint(self):
        return self.new_kps

    def getLine(self):
        return self.new_lines

    def getUnits(self):
        index_list = []

        use_time_list = [(1 if ele.getType() == BORDER else 2) for ele in self.new_lines]

        not_consider_list = []

        self.new_units = []

        for edge_max in range(3, self.maximum_finding_level + 1):
            for i in range(len(self.new_lines)):
                index = [[i]]
                queue = [(i, 1, START)]
                start_point = self.new_lines[i][START]

                while len(queue) > 0:
                    id = queue[-1][0]
                    level = queue[-1][1]
                    before = queue[-1][2]
                    sub_index = index[-1]

                    del(queue[-1])
                    del(index[-1])

                    if before == START:
                        end = self.new_lines[id][END]
                    else:
                        end = self.new_lines[id][START]

                    if level >= edge_max:
                        if distance(start_point, end) < 1e-5:
                            temp = []
                            for existing_index in index_list:
                                temp_index = sorted(deepcopy(existing_index))
                                temp.append(temp_index)
                            current_index = sorted(deepcopy(sub_index))
                            if current_index not in temp:
                                # not include other creases
                                u = Unit()
                                crease = self.new_lines[sub_index[0]]
                                next_start = crease[END]
                                u.addCrease(crease)

                                number_of_crease = len(sub_index)
                                
                                for i in range(1, number_of_crease):
                                    crease = self.new_lines[sub_index[i]]
                                    if distance(crease[START], next_start) < 1e-5:
                                        next_start = crease[END]
                                        u.addCrease(crease)
                                    else:
                                        next_start = crease[START]
                                        u.addCrease(crease.getReverse())

                                seq_point = u.getSeqPoint()
                                kp_in_polygon = False
                                for line in self.new_lines:
                                    if pointInPolygon(line.getMidPoint(), seq_point, return_min_distance=True) > 1e-5:
                                        kp_in_polygon = True    

                                if not kp_in_polygon:
                                    for ele in current_index:
                                        if use_time_list[ele] >= 2:
                                            use_time_list[ele] -= 1
                                        elif use_time_list[ele] == 1:
                                            use_time_list[ele] -= 1
                                    index_list.append(deepcopy(sub_index))

                                    creases = u.getCrease()
                                    if crossProduct(creases[0], creases[1]) < 0:
                                        self.new_units.append(u)

                                    else:
                                        unit_modified = Unit()
                                        unit_length = len(creases)
                                        for i in range(0, -unit_length, -1):
                                            unit_modified.addCrease(creases[i].getReverse())
                                        self.new_units.append(unit_modified)
                        continue

                    for j in range(len(self.new_lines)):
                        if (j not in sub_index) and (j not in not_consider_list):
                            if distance(self.new_lines[j][START], end) < 1e-2:
                                queue.insert(0, (j, level + 1, START))
                                index.insert(0, sub_index + [j])
                            elif distance(self.new_lines[j][END], end) < 1e-2:
                                queue.insert(0, (j, level + 1, END))
                                index.insert(0, sub_index + [j])
            
            for i in range(len(use_time_list)):
                if use_time_list[i] <= 0:
                    not_consider_list.append(i)

        return self.new_units

class UnitPackParserReverse:
    def __init__(self, tsp, kps, lines, lines_type) -> None:
        self.tsp = tsp
        self.maximum_finding_level = 6

        self.new_kps = []
        self.new_lines = []

        for ele in kps:
            self.new_kps.append([ele[X] + tsp[X], ele[Y] + tsp[Y]])

        for i in range(len(lines)):
            ele = lines[i]
            self.new_lines.append(Crease(
                [ele[START][X] + tsp[X], ele[START][Y] + tsp[Y]],
                [ele[END][X] + tsp[X], ele[END][Y] + tsp[Y]],
                lines_type[i]
            ))

    def getKeyPoint(self):
        return self.new_kps
    
    def getMaxDistance(self):
        max_x = max([self.new_kps[i][X] for i in range(len(self.new_kps))])
        min_x = min([self.new_kps[i][X] for i in range(len(self.new_kps))])
        max_y = max([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        min_y = min([self.new_kps[i][Y] for i in range(len(self.new_kps))])
        return max(max_x - min_x, max_y - min_y), max_x - min_x, max_y - min_y
    
    def getTotalBias(self, units=None):
        if units == None:
            unit = self.getUnits()
        else:
            unit = units
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

    def getLine(self):
        return self.new_lines

    def setMaximumNumberOfEdgeInAllUnit(self, n):
        self.maximum_finding_level = n
        
    def getUnits(self):
        index_list = []

        use_time_list = [(1 if ele.getType() == BORDER else 2) for ele in self.new_lines]

        not_consider_list = []

        self.new_units = []

        for edge_max in range(3, self.maximum_finding_level + 1):
            for i in range(len(self.new_lines)):
                index = [[i]]
                queue = [(i, 1, START)]
                start_point = self.new_lines[i][START]

                while len(queue) > 0:
                    id = queue[-1][0]
                    level = queue[-1][1]
                    before = queue[-1][2]
                    sub_index = index[-1]

                    del(queue[-1])
                    del(index[-1])

                    if before == START:
                        end = self.new_lines[id][END]
                    else:
                        end = self.new_lines[id][START]

                    if level >= edge_max:
                        if distance(start_point, end) < 1e-5:
                            temp = []
                            for existing_index in index_list:
                                temp_index = sorted(deepcopy(existing_index))
                                temp.append(temp_index)
                            current_index = sorted(deepcopy(sub_index))
                            if current_index not in temp:
                                # no other creases
                                u = Unit()
                                crease = self.new_lines[sub_index[0]]
                                next_start = crease[END]
                                u.addCrease(crease)

                                number_of_crease = len(sub_index)
                                
                                for i in range(1, number_of_crease):
                                    crease = self.new_lines[sub_index[i]]
                                    if distance(crease[START], next_start) < 1e-5:
                                        next_start = crease[END]
                                        u.addCrease(crease)
                                    else:
                                        next_start = crease[START]
                                        u.addCrease(crease.getReverse())

                                seq_point = u.getSeqPoint()
                                kp_in_polygon = False
                                for line in self.new_lines:
                                    if pointInPolygon(line.getMidPoint(), seq_point, return_min_distance=True) > 1e-5:
                                        kp_in_polygon = True
                                if not kp_in_polygon:
                                    for ele in current_index:
                                        if use_time_list[ele] >= 2:
                                            use_time_list[ele] -= 1
                                        elif use_time_list[ele] == 1:
                                            use_time_list[ele] -= 1
                                    index_list.append(deepcopy(sub_index))

                                    creases = u.getCrease()

                                    if crossProduct(creases[0], creases[1]) > 0:
                                        self.new_units.append(u)

                                    else:
                                        unit_modified = Unit()
                                        unit_length = len(creases)
                                        for i in range(0, -unit_length, -1):
                                            unit_modified.addCrease(creases[i].getReverse())
                                        self.new_units.append(unit_modified)
                        continue

                    for j in range(len(self.new_lines)):
                        if (j not in sub_index) and (j not in not_consider_list):
                            if distance(self.new_lines[j][START], end) < 1e-5:
                                queue.insert(0, (j, level + 1, START))
                                index.insert(0, sub_index + [j])
                            elif distance(self.new_lines[j][END], end) < 1e-5:
                                queue.insert(0, (j, level + 1, END))
                                index.insert(0, sub_index + [j])
            for i in range(len(use_time_list)):
                if use_time_list[i] <= 0:
                    not_consider_list.append(i)

        return self.new_units

class Body:
    def __init__(self, main_area_point_list) -> None:
        self.kp = main_area_point_list
        self.line = []

    def getKeyPoint(self):
        return self.kp
    
    def getLine(self):
        self.line.clear()
        kp_length = len(self.kp)
        for i in range(kp_length - 1):
            self.line.append(Crease(self.kp[i], self.kp[i + 1], BORDER))
        self.line.append(Crease(self.kp[kp_length - 1], self.kp[0], BORDER))
        return self.line

class TreeBasedOrigamiGraph:
    def __init__(self, kps, lines) -> None:
        self.kps = kps
        self.lines = lines

        self.vertices = []
        self.edges = []
        self.units = []

    def calculateTreeBasedGraph(self):
        self.vertices.clear()
        self.edges.clear()
        self.units.clear()

        for kp in self.kps:
            self.vertices.append(Vertex(kp))

        for line_i in range(len(self.lines)):
            line = self.lines[line_i]
            line.visited = False
            start_point = line[START]
            end_point = line[END]
            crease_type = line.getType()
            match_start = False
            match_end = False
            for i in range(len(self.kps)):
                kp = self.kps[i]
                if not match_start and samePoint(start_point, kp, 2):
                    match_start = True
                    line.start_index = i
                    self.vertices[i].dn += 1
                    self.vertices[i].connection_index.append(line_i)
                    if crease_type == BORDER:
                        self.vertices[i].is_border_node = True
                if not match_end and samePoint(end_point, kp, 2):
                    match_end = True
                    line.end_index = i
                    self.vertices[i].dn += 1
                    self.vertices[i].connection_index.append(line_i)
                    if crease_type == BORDER:
                        self.vertices[i].is_border_node = True
                if match_start and match_end:
                    break

        for vertex in self.vertices:
            # vertex = Vertex()
            if not vertex.is_border_node and vertex.dn == 4:
                creases = [self.lines[index] for index in vertex.connection_index]
                crease_types = [creases[i].getType() for i in range(4)]
                same_type = False
                diff_type = False
                for i in range(1, 4):
                    if crease_types[0] == crease_types[i]:
                        same_type = True
                        if diff_type:
                            i -= 1
                            break
                    else:
                        diff_type = True
                        if same_type:
                            break
                if same_type:
                    only_crease_index = vertex.connection_index[i]
                else:
                    only_crease_index = vertex.connection_index[0]

                angles = []
                betas = [0, .0, .0, .0]
                for i in range(4):
                    if vertex.connection_index[i] != only_crease_index:
                        angle = angleBetweenCreases(self.lines[only_crease_index], creases[i])
                        angles.append([angle, vertex.connection_index[i]])
                
                angles = sorted(deepcopy(angles), key=lambda x: x[0])
                if angles[1][0] >= 0:
                    betas[1] = angles[1][0]
                    betas[0] = -angles[0][0]
                    betas[3] = angles[2][0] - angles[1][0]
                    betas[2] = 2 * math.pi - betas[0] - betas[1] - betas[3]
                else:
                    betas[1] = angles[2][0]
                    betas[0] = -angles[1][0]
                    betas[2] = angles[1][0] - angles[0][0]
                    betas[3] = 2 * math.pi - betas[0] - betas[1] - betas[2]
                
                # kind
                level = 0.
                coeff = 1.

                up_level = False

                if betas[0] + betas[1] - math.pi < -1e-5:
                    new_coeff = (math.sin(betas[0]) + math.sin(betas[1])) / math.sin(betas[0] + betas[1])
                elif betas[0] + betas[1] - math.pi < 1e-5:
                    new_coeff = 1.
                    up_level = True
                else:
                    raise TypeError
                
                # C180
                if abs(betas[0] + betas[3] - math.pi) < 1e-5 and abs(betas[1] + betas[2] - math.pi) < 1e-5:
                    if not up_level:
                        # give information to vertex
                        if angles[1][0] >= 0:
                            index_second_crease_1 = angles[0][1]
                            index_second_crease_2 = angles[1][1]
                        else:
                            index_second_crease_1 = angles[1][1]
                            index_second_crease_2 = angles[2][1]
                        vertex.level_list = [level for i in range(4)]
                        vertex.coeff_list = [coeff for i in range(4)]
                        vertex.coeff_list[vertex.connection_index.index(index_second_crease_1)] = new_coeff * coeff
                        vertex.coeff_list[vertex.connection_index.index(index_second_crease_2)] = new_coeff * coeff
                    else:
                        # give information to vertex
                        if angles[1][0] >= 0:
                            index_second_crease_1 = angles[0][1]
                            index_second_crease_2 = angles[1][1]
                        else:
                            index_second_crease_1 = angles[1][1]
                            index_second_crease_2 = angles[2][1]
                        vertex.level_list = [level for i in range(4)]
                        vertex.coeff_list = [coeff for i in range(4)]
                        vertex.level_list[vertex.connection_index.index(index_second_crease_1)] += 1
                        vertex.level_list[vertex.connection_index.index(index_second_crease_2)] += 1

        # give information to crease
        visited = len(self.lines)

        # remove borders
        for i in range(len(self.lines)):
            line = self.lines[i]
            if line.getType() == BORDER:
                visited -= 1

        backup_visited = visited
        backup_initial_crease_id = []

        # calculate level and coeff for creases
        while visited:
            crease_id = []

            # randomly select a crease
            for i in range(len(self.lines)):
                line = self.lines[i]
                if line.getType() != BORDER and not line.visited:
                    line.visited = True
                    line.level = 0
                    line.coeff = 1.
                    line.undefined = False
                    visited -= 1
                    crease_id.append(i)
                    break
            
            previous_min_level = 0
            while len(crease_id):
                index = 0
                min_level = self.lines[crease_id[0]].level
                min_index = 0
                min_problem = False

                for index in range(1, len(crease_id)):
                    if self.lines[crease_id[index]].level < min_level:
                        min_index = index
                        min_level = self.lines[crease_id[index]].level
                        if min_level < previous_min_level:
                            previous_min_level = min_level
                            backup_initial_crease_id.append(crease_id[index])
                            crease_id = deepcopy(backup_initial_crease_id)
                            min_problem = True
                            break
                
                previous_min_level = min_level
                if min_problem:
                    for line_i in self.lines:
                        line_i.visited = False
                    visited = backup_visited
                    continue
                    
                crease_first_id = crease_id[min_index]
                del(crease_id[min_index])
                kp1_id = self.lines[crease_first_id].start_index
                kp2_id = self.lines[crease_first_id].end_index
                kp_list = [kp1_id, kp2_id]
                for ele in kp_list:
                    vertex = self.vertices[ele]
                    if not vertex.is_border_node and vertex.dn == 4:
                        # find itself
                        position = vertex.connection_index.index(crease_first_id)
                        for i in range(4):
                            if i != position and not self.lines[vertex.connection_index[i]].visited:
                                divider = vertex.coeff_list[i] / vertex.coeff_list[position]
                                adder = vertex.level_list[i] - vertex.level_list[position]
                                new_crease_id = vertex.connection_index[i]
                                if adder != 0:
                                    self.lines[new_crease_id].level = int(self.lines[crease_first_id].level + adder)
                                    self.lines[new_crease_id].recover_level = self.lines[new_crease_id].level
                                    self.lines[new_crease_id].coeff = 1.
                                    self.lines[new_crease_id].visited = True
                                    self.lines[new_crease_id].undefined = True
                                else:
                                    self.lines[new_crease_id].level = int(self.lines[crease_first_id].level)
                                    self.lines[new_crease_id].recover_level = self.lines[new_crease_id].level
                                    self.lines[new_crease_id].coeff = self.lines[crease_first_id].coeff * divider
                                    self.lines[new_crease_id].visited = True
                                    self.lines[new_crease_id].undefined = self.lines[crease_first_id].undefined
                                visited -= 1
                                crease_id.append(new_crease_id)
                            elif i != position and self.lines[vertex.connection_index[i]].visited:
                                #validate
                                divider = vertex.coeff_list[i] / vertex.coeff_list[position]
                                adder = vertex.level_list[i] - vertex.level_list[position]
                                new_crease_id = vertex.connection_index[i]

                                # judge level
                                new_level = int(self.lines[crease_first_id].level + adder)
                                # initial
                                if new_level != self.lines[new_crease_id].level:
                                    # record conflict
                                    if new_level < self.lines[new_crease_id].level:
                                        crease_id.append(vertex.connection_index[i])
                                        break
                                    else:
                                        if adder >= 0:
                                            self.lines[new_crease_id].level = new_level
                                            self.lines[new_crease_id].recover_level = self.lines[new_crease_id].level
                                            crease_id.append(vertex.connection_index[i])

                                new_coeff = self.lines[crease_first_id].coeff * divider
                                # initial
                                if new_coeff != self.lines[new_crease_id].coeff:
                                    # record conflict
                                    pass
                                self.lines[new_crease_id].undefined = False
        

                        # for i in range(4):
                        #     self.lines[vertex.connection_index[i]].level = vertex.level_list[i]
                        #     self.lines[vertex.connection_index[i]].coeff = vertex.coeff_list[i]
                        #     self.lines[vertex.connection_index[i]].visited = True

        self.sequence_max_level = self.lines[0].level
        self.sequence_min_level = self.lines[0].level
        for i in range(len(self.lines)):
            level = self.lines[i].level
            if level > self.sequence_max_level:
                self.sequence_max_level = level
            elif level < self.sequence_min_level:
                self.sequence_min_level = level
        
        for level in range(int(self.sequence_min_level), int(self.sequence_max_level + 1)):
            for i in range(len(self.lines)):
                if self.lines[i].level == level:
                    break
            self.min_coeff = self.lines[i].coeff
            # get min coeff
            for j in range(i, len(self.lines)):
                if self.lines[j].level == level and self.lines[j].coeff < self.min_coeff:
                    self.min_coeff = self.lines[j].coeff
            for k in range(len(self.lines)):
                if self.lines[k].level == level:
                    self.lines[k].coeff /= self.min_coeff

                
                





    

    
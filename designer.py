import math
import numpy as np
from desc import *
from units import *


class OrigamiDesigner:
    def __init__(self, src, paper_info, design_info) -> None:
        self.src = src
        self.paper_info = paper_info
        self.design_info = design_info
        self.design_data = []
        self.k_design_data = []
        self.transition_initial_point = [0, 0]
        self.entry_flag = V
        self.stretch_left_length = 0
        self.stretch_right_length = 0

        self.additional_line_source = []

    def setTransitionStartPoint(self, tsp):
        self.transition_initial_point = tsp

    def getTransitionStartPoint(self):
        return self.transition_initial_point
    
    def setPaperInfo(self, paper_info):
        self.paper_info = paper_info

    def setDesignInfo(self, design_info):
        self.design_info = design_info

    def setEntryFlag(self, entry_flag):
        self.entry_flag = entry_flag

    def setSource(self, src):
        self.src = src
    
    def setLeftStretchLength(self, length):
        self.stretch_left_length = length

    def setRightStretchLength(self, length):
        self.stretch_right_length = length

    def clearAdditionalLine(self):
        self.additional_line_source.clear()
        
    def insertLineSource(self, line_source, index):
        self.additional_line_source.append([line_source, index])

    def getKLNumber(self):
        return len(self.src.lines)

    def bodyDesign(self):
        pass

    def getPaperData(self):
        if type(self.src) == KinematicLine:
            # not include connection
            length = sum([x[0] for x in self.src.lines])
            width = self.design_info['copy_time'] * self.paper_info['unit_width']
            return length, width
        elif type(self.src) == ModuleLeanMiura:
            length, width = self.design_data[0].getOrigamiInfo()
            return length, width
        elif type(self.src) == DxfDirectGrabber:
            max_y = max([self.src.kps[i][Y] for i in range(len(self.src.kps))])
            max_x = max([self.src.kps[i][X] for i in range(len(self.src.kps))])
            min_y = min([self.src.kps[i][Y] for i in range(len(self.src.kps))])
            min_x = min([self.src.kps[i][X] for i in range(len(self.src.kps))])
            return max_x - min_x, max_y - min_y

    def getGeometryData(self):
        type_src = type(self.src)
        if type_src == KinematicLine:
            desc = deepcopy(self.src.lines)
            k_design_data = []
            # entry_flag = self.entry_flag

            design_path_length = len(desc)
            k_desc = deepcopy(desc)

            if k_desc[0][1] > 0:
                k_design_data.append([k_desc[0][0], k_desc[0][1] / 2, 1])
            else:
                k_design_data.append([k_desc[0][0], k_desc[0][1] / 2, 0])

            for i in range(1, design_path_length):
                length = k_desc[i][0]
                alpha = (k_desc[i][1] - k_desc[i - 1][1]) / 2
                # if((alpha > 0 and entry_flag != V) or (alpha < 0 and entry_flag == V)):
                #     k_design_data.append([length, alpha, 1])
                # elif((alpha < 0 and entry_flag != V) or (alpha > 0 and entry_flag == V)):
                #     k_design_data.append([length, alpha, 0])
                if alpha > 0:
                    k_design_data.append([length, alpha, 1])
                else:
                    k_design_data.append([length, -alpha, 0])
            
            return k_design_data
    
    def parseData(self):
        type_src = type(self.src)
        if type_src == KinematicLine:
            desc = deepcopy(self.src.lines)
            if len(self.additional_line_source):
                for i in range(len(self.additional_line_source)):
                    desc.insert(self.additional_line_source[i][-1] + i, self.additional_line_source[i][0])
            copy_time = self.design_info['copy_time']
            unit_width = self.paper_info['unit_width']
            fp = self.design_info['folding_percent']
            self.design_data.clear()
            for k in range(copy_time):
                k_design_data = []
                point = [self.transition_initial_point[0], self.transition_initial_point[1] + unit_width * k]
                transition_point = [self.transition_initial_point[0], self.transition_initial_point[1] + unit_width / 4 + unit_width * k]
                entry_flag = self.entry_flag
                border_start_x = self.transition_initial_point[0]
                k_design_data.append(Miura(
                    point, 
                    0, 
                    self.paper_info['unit_width'], 
                    1.57, 
                    ACTIVE_MIURA, 
                    entry_flag, 
                    0, 
                    transition_point, 
                    0,
                    border_start_x
                ))
                if(k == 0):
                    k_design_data[-1].setBorder(up_line=HAVE_CONNECTION, down_line=HAVE_BORDER)
                if(k == copy_time - 1):
                    k_design_data[-1].setUpBorder(up_line=HAVE_BORDER)
                    if copy_time != 1:
                        k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                if(k and k != copy_time - 1):
                    k_design_data[-1].setUpBorder(up_line=HAVE_CONNECTION)
                    k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                k_design_data[-1].setEntityColor(BLACK_COLOR)
                border_start_x = k_design_data[-1].getBorderXPoint()
                point, transition_point, entry_flag = k_design_data[-1].getTransition()
                design_path_length = len(desc)
                k_desc = deepcopy(desc)
                for i in range(design_path_length - 1):
                    length = k_desc[i][0]
                    # alpha = (k_desc[i + 1][1] - k_desc[i][1]) / 2
                    alpha = math.atan(math.tan((k_desc[i + 1][1] - k_desc[i][1]) / 2) / math.sqrt(1 - fp**2))
                    if abs(alpha) < 0.1:
                        k_desc[i + 1][0] += length
                        continue
                    if((alpha > 0 and entry_flag != V) or (alpha < 0 and entry_flag == V)):
                        last_main_crease = k_design_data[-1].getMainLine()[1]
                        k_design_data.append(Miura(
                            point,
                            length - point[0] + transition_point[0] + unit_width / 4 / math.tan(abs(alpha)), 
                            self.paper_info['unit_width'], 
                            abs(alpha), 
                            ACTIVE_MIURA, 
                            entry_flag, 
                            0, 
                            transition_point, 
                            length,
                            border_start_x
                        ))
                        if(k == 0):
                            k_design_data[-1].setBorder(up_line=HAVE_CONNECTION, down_line=HAVE_BORDER)
                        if(k == copy_time - 1):
                            k_design_data[-1].setUpBorder(up_line=HAVE_BORDER)
                            if copy_time != 1:
                                k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                        if(k and k != copy_time - 1):
                            k_design_data[-1].setUpBorder(up_line=HAVE_CONNECTION)
                            k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                        k_design_data[-2].setBorderXPoint(k_design_data[-1].getBorderXPoint())
                        border_start_x = k_design_data[-1].getNextBorderStartXPoint()
                        point, transition_point, entry_flag = k_design_data[-1].getTransition()
                        current_main_crease = k_design_data[-1].getMainLine()[1]
                        percent = calculatePercent(last_main_crease, current_main_crease)
                        if percent != None:
                            if k_design_data[-2].getFunctionType() == PASSIVE_MIURA:
                                k_design_data[-2].setLeftAndRightPercent(left_percent=None, right_percent=percent_limit(percent[-1][0]))
                                k_design_data[-1].setLeftAndRightPercent(left_percent=percent_limit(1 - percent[-1][1]), right_percent=None)
                            else:
                                k_design_data[-2].setLeftAndRightPercent(left_percent=percent_limit(1 - percent[-1][0]), right_percent=None)
                                k_design_data[-1].setLeftAndRightPercent(left_percent=percent_limit(1 - percent[-1][1]), right_percent=None)
                    elif((alpha < 0 and entry_flag != V) or (alpha > 0 and entry_flag == V)):
                        last_main_crease = k_design_data[-1].getMainLine()[1]
                        k_design_data.append(Miura(
                            point,
                            length - point[0] + transition_point[0] - unit_width / 4 / math.tan(abs(alpha)), 
                            self.paper_info['unit_width'], 
                            abs(alpha), 
                            PASSIVE_MIURA, 
                            entry_flag, 
                            0, 
                            transition_point, 
                            length,
                            border_start_x
                        ))
                        if(k == 0):
                            k_design_data[-1].setBorder(up_line=HAVE_CONNECTION, down_line=HAVE_BORDER)
                        if(k == copy_time - 1):
                            k_design_data[-1].setUpBorder(up_line=HAVE_BORDER)
                            if copy_time != 1:
                                k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                        if(k and k != copy_time - 1):
                            k_design_data[-1].setUpBorder(up_line=HAVE_CONNECTION)
                            k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                        k_design_data[-2].setBorderXPoint(k_design_data[-1].getBorderXPoint())
                        border_start_x = k_design_data[-1].getNextBorderStartXPoint()
                        point, transition_point, entry_flag = k_design_data[-1].getTransition()
                        current_main_crease = k_design_data[-1].getMainLine()[1]
                        percent = calculatePercent(current_main_crease, last_main_crease)
                        if percent != None:
                            if k_design_data[-2].getFunctionType() == ACTIVE_MIURA:
                                k_design_data[-2].setLeftAndRightPercentOfMainLine(left_percent=None, right_percent=(1 - percent[0][1]))
                                k_design_data[-1].setLeftAndRightPercentOfMainLine(left_percent=percent[2][0], right_percent=None)
                                k_design_data[-2].setLeftAndRightPercent(left_percent=None, right_percent=percent_limit(1 - percent[-1][1]))
                                k_design_data[-1].setLeftAndRightPercent(left_percent=percent_limit(percent[-1][0]), right_percent=None)
                            else:
                                k_design_data[-2].setLeftAndRightPercentOfMainLine(left_percent=(1 - percent[0][1]), right_percent=None)
                                k_design_data[-1].setLeftAndRightPercentOfMainLine(left_percent=percent[2][0], right_percent=None)
                                k_design_data[-2].setLeftAndRightPercent(left_percent=percent_limit(percent[-1][1]), right_percent=None)
                                k_design_data[-1].setLeftAndRightPercent(left_percent=percent_limit(percent[-1][0]), right_percent=None)
                last_main_crease = k_design_data[-1].getMainLine()[1]
                k_design_data.append(Miura(
                    point,
                    k_desc[-1][0] - point[0] + transition_point[0], 
                    self.paper_info['unit_width'], 
                    1.57, 
                    ACTIVE_MIURA, 
                    entry_flag, 
                    0, 
                    transition_point, 
                    0,
                    border_start_x
                ))
                k_design_data[-1].setEntityColor(BLACK_COLOR)
                if(k == 0):
                    k_design_data[-1].setBorder(up_line=HAVE_CONNECTION, down_line=HAVE_BORDER)
                if(k == copy_time - 1):
                    k_design_data[-1].setUpBorder(up_line=HAVE_BORDER)
                    if copy_time != 1:
                        k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                if(k and k != copy_time - 1):
                    k_design_data[-1].setUpBorder(up_line=HAVE_CONNECTION)
                    k_design_data[-1].setDownBorder(down_line=HAVE_CONNECTION)
                k_design_data[-2].setBorderXPoint(k_design_data[-1].getBorderXPoint())
                border_start_x = k_design_data[-1].getNextBorderStartXPoint()
                point, transition_point, entry_flag = k_design_data[-1].getTransition()
                current_main_crease = k_design_data[-1].getMainLine()[1]
                percent = calculatePercent(last_main_crease, current_main_crease)
                if percent != None:
                    k_design_data[-2].setLeftAndRightPercent(left_percent=None, right_percent=percent_limit(percent[-1][0]))
                    k_design_data[-1].setLeftAndRightPercent(left_percent=percent_limit(1 - percent[-1][1]), right_percent=None)
                self.design_data += deepcopy(k_design_data)
                self.k_design_data = k_design_data

        elif type_src == ModuleLeanMiura:
            desc = deepcopy(self.src)
            self.design_data.clear()
            self.design_data.append(LeanMiura(
                unit_width=desc.unit_width,
                entry_flag=desc.entry_flag,
                tsp=desc.tsp,
                copy_time=desc.copy_time,
                half_flag=desc.half_flag,
                stretch_length=desc.stretch_length,
                connection_flag=desc.connection_flag,
                con_left_length=desc.con_left_length,
                con_right_length=desc.con_right_length,
                connection_hole_size=desc.connection_hole_size
            ))

        elif type_src == DxfDirectGrabber:
            desc = deepcopy(self.src)
            self.design_data.clear()
            self.design_data.append(UnitPackParser(
                tsp=self.transition_initial_point,
                kps=desc.kps,
                lines=desc.lines,
                lines_type=desc.lines_type
            ))

    def getDesignData(self):
        return self.design_data
    
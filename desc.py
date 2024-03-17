import math
import numpy as np
from new_module_dialog import NewModuleDialog, LEFT_HALF, RIGHT_HALF, NO_HALF
from PyQt5.QtWidgets import QInputDialog
import re

from utils import MOUNTAIN, VALLEY, BORDER
import dxfgrabber

class KinematicLine:
    def __init__(self) -> None:
        self.lines = []

    def append(self, data):
        self.lines.append(data)

    def getData(self):
        return self.lines

class UnitPack:
    def __init__(self) -> None:
        self.unit_point_list = []
        self.tsp = []

    def setTsp(self):
        self.tsp = self.unit_point_list[-1][0]

    def addPointGroup(self, unit):
        self.unit_point_list.append(unit)
    
    def getLength(self):
        return len(self.unit_point_list)
    
    def raiseDialog(self, parent):
        ok = True
        unit = []
        while ok:
            text, ok = QInputDialog.getText(parent, "Enter a 2D point axis", "2D point axis splited with ','")
            if not ok:
                break
            ans = re.findall(r"\d+\.?\d*", text)
            if len(ans) == 2:
                x = ans[0]
                y = ans[1]
                unit.append([x, y])
        if len(unit) >= 3:
            self.unit_point_list.append(unit)
            self.setTsp()


class ModuleLeanMiura:
    def __init__(self, data=[]) -> None:
        self.unit_width = None
        self.entry_flag = None
        self.copy_time = None
        self.half_flag = None
        self.stretch_length = None
        self.connection_flag = None
        self.con_left_length = None
        self.con_right_length = None
        self.connection_hole_size = None
        self.modify_stretch_flag = None

        self.tsp = [0, 0]

        self.enable_global_modify = None

        self.initial_flag = False

    def initialize(self, unit_width, entry_flag, copy_time, stretch_length, connection_flag, con_left_length, con_right_length, con_radius, half_flag, tsp, enabled):
        self.initial_flag = True
        self.setUnitWidth(unit_width)
        self.setEntryFlag(entry_flag)
        self.setCopyTime(copy_time)
        self.setStretchLength(stretch_length)
        self.setConnectionFlag(connection_flag)
        self.setConnectionLeftLength(con_left_length)
        self.setConnectionRightLength(con_right_length)
        self.setConnectionHoleSize(con_radius)
        self.setHalfFlag(half_flag)
        self.setEnableModify(enabled)
        self.setTsp(tsp)
        
    def getData(self):
        return {
            "half_type": self.half_flag,
            "stretch_length": self.stretch_length
        }
    
    def setUnitWidth(self, unit_width):
        self.unit_width = unit_width

    def setEntryFlag(self, entry_flag):
        self.entry_flag = entry_flag

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

    def setConnectionHoleSize(self, con_radius):
        self.connection_hole_size = con_radius

    def setHalfFlag(self, half_flag):
        self.half_flag = half_flag

    def setEnableModify(self, enabled: bool):
        self.enable_global_modify = enabled
    
    def setTsp(self, tsp):
        self.tsp = tsp

    def raiseDialog(self, parent):
        dialog = NewModuleDialog(parent=parent)
        if self.initial_flag:
            dialog.doubleSpinBox_unit_width.setValue(self.unit_width)
            dialog.spinBox_copy_time.setValue(self.copy_time)
            dialog.horizontalSlider_entry_flag.setValue(self.entry_flag)
            dialog.checkBox_enable_connection.setChecked(self.connection_flag)
            dialog.doubleSpinBox_con_left.setValue(self.con_left_length)
            dialog.doubleSpinBox_con_right.setValue(self.con_right_length)
            dialog.doubleSpinBox_con_radius.setValue(self.connection_hole_size)
            if self.half_flag == LEFT_HALF:
                dialog.radioButton_left_half.setChecked(True)
                dialog.radioButton_right_half.setChecked(False)
                dialog.radioButton_all.setChecked(False)
            if self.half_flag == RIGHT_HALF:
                dialog.radioButton_left_half.setChecked(False)
                dialog.radioButton_right_half.setChecked(True)
                dialog.radioButton_all.setChecked(False)
            if self.half_flag == NO_HALF:
                dialog.radioButton_left_half.setChecked(False)
                dialog.radioButton_right_half.setChecked(False)
                dialog.radioButton_all.setChecked(True)
            dialog.doubleSpinBox_stretch_length.setValue(self.stretch_length)
            dialog.doubleSpinBox_tspx.setValue(self.tsp[0])
            dialog.doubleSpinBox_tspy.setValue(self.tsp[1])
        if dialog.exec_():
            pass
        if not dialog.getOK():
            return
        else:
            self.initialize(
                unit_width          =dialog.getUnitWidth(),
                copy_time           =dialog.getCopyTime(),
                entry_flag          =dialog.getEntryFlag(),
                stretch_length      =dialog.getStretchLength(),
                connection_flag     =dialog.getEnableConnection(),
                con_left_length     =dialog.getConnectionLeftLength(),
                con_right_length    =dialog.getConnectionRightLength(),
                con_radius          =dialog.getConnectionRadius(),
                half_flag           =dialog.getHalfFlag(),
                tsp                 =dialog.getTsp(),
                enabled             =dialog.getUsingGlobalData()
            )
        dialog.destroy()

class DxfDirectGrabber:
    def __init__(self) -> None:
        self.lines = []
        self.lines_type = []
        self.kps = []

    def readFile(self, path):
        # 读取dxf文件
        self.kps.clear()
        self.lines.clear()
        self.lines_type.clear()
        dxf = dxfgrabber.readfile(path)

        # 遍历所有实体
        for entity in dxf.entities:
            # 如果是线段
            if entity.dxftype == 'LINE':
                # 获取线段的两个端点
                start_point = entity.start
                end_point = entity.end
                # 获取线段所在的图层
                layer = entity.layer
                # 获取线段的颜色
                color = entity.color

                duplicate = False
                for i in range(len(self.kps)):
                    if (start_point[0] - self.kps[i][0]) ** 2 + (start_point[1] - self.kps[i][1]) ** 2 < 1e-10:
                        duplicate = True
                        break
                
                if duplicate:
                    start_point = self.kps[i]
                else:
                    self.kps.append(start_point)

                duplicate = False
                for i in range(len(self.kps)):
                    if (end_point[0] - self.kps[i][0]) ** 2 + (end_point[1] - self.kps[i][1]) ** 2 < 1e-10:
                        duplicate = True
                        break
                
                if duplicate:
                    end_point = self.kps[i]
                else:
                    self.kps.append(end_point)

                self.lines.append([start_point, end_point])
                if layer == 'Mountain' or color == 1:
                    self.lines_type.append(MOUNTAIN)
                elif layer == 'Valley' or color == 5:
                    self.lines_type.append(VALLEY)
                else:
                    self.lines_type.append(BORDER)
    
    def getData(self):
        return {
            "kps": self.kps,
            "lines": self.lines,
            "lines_type": self.lines_type
        }
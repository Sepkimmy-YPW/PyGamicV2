# Std API import
import sys
import os
import math
import numpy as np
import json
# import multiprocessing
# import matplotlib as mpl
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add GUI python file into our application
sys.path.append('./gui')

# Qt API import
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog, QInputDialog, QMessageBox
from PyQt5.QtGui import QPainter, QColor, QPixmap, QPen, QCursor, QPolygon
from PyQt5.QtCore import QObject, Qt, QTimer, QPoint, QThread, pyqtSignal
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

# Window/Dialog sub-module import
from gui.Ui_window import Ui_MainWindow
from tm_window import TmWindow
from stl_dialog import StlSettingDialog
from pref_pack import PreferencePackWindow

# Sub-module import

# --Description module for describing the source of design process-- #
from desc import *
# --Designer module for designing origami from source-- #
from designer import *
# --Units module for representing origami units like Miura and LeanMiura-- #
from units import *
# --Utils module for providing necessary mathematical tools for application-- #
from utils import *
# --Dxftool module for outputing 2D drawing file of the origami-- #
from dxftool import *
# --Cdftool module for doing computational design -- #
from cdftool import *
# --Stltool module for outputing 3D model file of the origami-- #
from stltool2 import StlMaker

# from phys_sim9 import OrigamiSimulator as OrigamiSimulatorE
FOLDING_MAXIMUM = 0.95

# Window for user to design origami
class Mainwindow(Ui_MainWindow, QMainWindow):
    def __init__(self, parent=None) -> None:
        """
        @ function: Program initialization
        @ version: 0.1111
        @ developer: py
        @ progress: on road
        @ date: 20230415
        @ spec: Add notes
        """
        # Set up for parent and set up for UI
        super(Mainwindow, self).__init__(parent)
        self.setupUi(self)

        # Define action-function conncetion for application
        # 1 Action slot
        self.defineAction()
        # 2 Button slot
        self.defineButton()
        # 3 Spinbox slot
        self.defineSpinbox()
        
        # --Define flags for application-- #
        self.INITIAL_STATE = 0x0001
        self.IMPORT_SUCCESS = 0x0002
        self.DESIGN_FINISH = 0x0004
        self.DESIGN_ERROR = 0x0008
        self.OUTPUT_FINISH = 0x0010

        # --Define const value for application-- #
        self.KL_JSON = 0x0001
        self.PACKED_DATA = 0x0002
        self.THREADING_METHOD = 0x0004

        self.A4_length = 296
        self.A4_width = 210
        self.A4_half_length = 148
        self.A4_half_width = 105

        self.pixmap_length = 750
        self.pixmap_width = 450
        self.draw_panel_x_bias = 169
        self.draw_panel_y_bias = 62

        # --Define parameters for application-- #
        # Axis parameters
        self.cursor_x = 0
        self.cursor_y = 0
        self.real_x = 0
        self.real_y = 0
        self.initial_x = 0          # record initial value of cursor[X]
        self.initial_y = 0          # record initial value of cursor[Y]
        self.enable_moving = False  # represent that whether to enable recording the axis of cursor

        # Current file information
        self.file_type = None
        self.file_path = None
        self.string_file_path = None

        # Origami information
        self.origami_length = 0
        self.origami_width = 0
        self.origami_info = []  # Type:[origin_point, length, width, origami_type]
        self.origami_number = 0 # Number of all design results
        self.unit_number = []

        # The amplitude of operation (1-180)
        self.operation_amp = 1

        # Inner value got from Ui widgets
        self.unit_width         = self.spinbox_crease_width.value() # mm
        self.bias_max           = self.unit_width / 6.0
        self.copy_time          = self.spinbox_copy_time.value()
        self.entry_flag         = self.slider_flag.value()
        self.add_hole_mode      = self.checkBox_add_hole_mode.isChecked()
        self.add_string_mode    = self.checkBox_add_string_mode.isChecked()
        self.hole_size          = self.spinbox_hole_size.value()
        self.hole_resolution    = self.spinbox_resolution.value()
        self.state              = self.INITIAL_STATE

        # Storage information for designer
        self.storage = []
        self.rotation = []
        self.paper_info = {
            "unit_width": self.unit_width
        }
        self.design_info = {
            "copy_time": 1,
            "folding_percent": 0.0
        }

        # Pixmap information for drawing origami design result
        self.half_pixmap_length     = self.pixmap_length / 2    #375
        self.half_pixmap_width      = self.pixmap_width / 2     #225
        self.pixmap                 = QPixmap(self.pixmap_length, self.pixmap_width)
        self.A4_pixmap              = QPixmap(self.A4_length, self.A4_width)
        self.pixmap.fill(QColor(255, 255, 255))
        self.A4_pixmap.fill(QColor(255, 255, 255))

        # Set painter and painting device
        self.painter = QPainter(self.pixmap)
        self.painter.end()

        # Realworld coordinate information
        self.kps = []                           # keypoints
        self.lines = []                         # key lines
        self.units = []                         # origami units, type: [Unit...]
        self.additional_lines = []              # additional line for warning of the add-hole operation
        self.hole_kps = []                      # hole keypoints
        self.connection_hole_kps = []           # connection hole keypoints 
        self.crease_lines = []                  # crease line for output crease dxf file
        self.backup_connection_hole_kps = []    # back up the connection keypoint for exporting stl
        self.strings = []                       # TSA strings
        self.unit_bias_list = []
        self.backup_unit_bias_list = []
        self.enable_read_list_from_backup = False
    
        # Pixelworld coordinate information
        self.pixel_kps = []                     # keypoints
        self.pixel_lines = []                   # key lines
        self.pixel_additional_lines = []        # additional line for warning of the add-hole operation
        self.pixel_hole_kps = []                # hole keypoints
        self.pixel_connection_hole_kps = []     # connection hole keypoints 
        self.pixel_string_kps = []

        # Parameters for axisConverter()
        self.pixel_bias = [25, 25]                  # transform 2D: transform[X, Y]
        self.old_pixel_bias_x = self.pixel_bias[0]  # old bias[X]
        self.old_pixel_bias_y = self.pixel_bias[1]  # old bias[Y]
        self.pixel_scale = [
            13.0, 11.6, 10, 9.0, 7.8, 6.6, 5.5, 5, 3.8, 3.1, 2.5, 2.0, 1.6, 1.3, 1.1, 1.0, 
            0.93, 0.87, 0.81, 0.76, 0.71, 0.66, 0.62, 0.58, 0.54, 0.51, 0.48, 0.45, 0.42, 0.4, 0.2, 0.1, 0.05, 0.03, 0.01
        ]                                           # scale factor list
        self.pixel_scale_ranking = 14               # scale level referencing pixel_scale list
        self.current_pixel_scale = 1.0              # value got from pixel_scale[scale_ranking]
        self.pixel_scale_min_ranking = len(self.pixel_scale)

        # Window update method
        self.show_square = None
        self.show_process = True
        self.enable_design = True

        # Start QT-Thread flag
        self.enable_output_stl = False
        self.enable_cdf_curve_fitting = False
        self.enable_phys_data_collecting = False
        self.enable_mcts = False

        # Define QT-Thread object which is belong to the application
        self.stl_output_thread = None
        self.cdf_curve_fitting_thread = None
        self.phys_data_collecting_thread = None
        self.mcts_thread = None

        # Dxf output setting
        self.dxf_split_flag = False

        # Stl output information of file name
        self.output_stl_file_path = ""
        self.output_stl_crease_flag = ""
        self.output_stl_board_flag = ""

        # Additional line information
        self.bias_val = 0.3
        self.show_additional_crease = True # True if add-hole mode is enabled

        # Curve information for TM-Window
        self.x_list = []
        self.y_list = []
        self.dir_list = []
        self.curve_name = None # we generate corresponding file name of CDF result
        
        # Tools
        self.designer               = OrigamiDesigner(src=None, paper_info=None, design_info=None)
        self.dxf_writer             = OrigamiToDxfConverter()
        self.stl_writer             = StlMaker()
        self.additional_line_maker  = StlMaker() # this maker only generate additional lines, not for 3D model output
        self.pref_pack_window       = PreferencePackWindow(parent=self)
        self.string_generator       = TSAString() 

        # connection information
        self.enable_connection  = self.checkBox_connection.isChecked()
        self.connection_radius  = self.spinbox_connection_radius.value()
        self.con_left_length    = self.spinbox_con_left_length.value()
        self.con_right_length   = self.spinbox_con_right_length.value()
        self.add_bias_flag = [] # list of bool ([bool...]) of whether to add width-bias for every origami design result

        # Parameters of hole information we choose by double-clicking a hole
        self.choose_hole_flag = False
        self.choose_hole_id = 0
        self.choose_hole_index = 0

        # Parameters of origami information we choose by double-clicking a origami
        self.choose_origami_flag = False
        self.choose_origami_index = 0

        self.choose_unit_id = -1
        self.choose_crease_id = -1
        self.choose_kl_id = -1
        self.choose_line_id = -1
        self.choose_crease_sequence_id = -1
        self.choose_string_id = -1
        self.expert_mode = False
        self.edit_kl_mode = False
        self.edit_sequence_mode = False

        # Parameters of hard crease
        self.hard_crease_index = []

        # Parameters of string start&end
        self.exist_string_start = False
        self.string_start_point = []
        self.string_type = BOTTOM

        self.a_string = [] # one string
        self.string_total_information = [] # list of list
        
        # Use unified initialization for application
        self.pref_pack_window.readFile()                        # import setting(or preference)
        self.pref_pack = self.pref_pack_window.getPrefPack()    # get preference pack from pref_window
        self.limitation = self.pref_pack_window.getLimitation() # get CDF-Limitation pack from pref_window
        self.setupTimer()                                       # enable timer to update information of application
        self.updateState("Ready", self.state)                   # update state bar 1
        self.updateMessage("Ready")                             # update state bar 2(message bar)
        self.setFocusPolicy(Qt.StrongFocus)                     # strong focus 

        self.widget.setVisible(False)
        self.widget_edit_kl.setVisible(False)
        self.widget_edit_sequence.setVisible(False)

        self.P_candidate = []
        self.P_candidate_connection_index = []

        self.show_index = False

        self.full_description_mode = True

    def addConnectionHole(self): # real axis
        self.connection_hole_kps.clear()
        col_num = self.copy_time
        for j in range(len(self.origami_info)):
            if type(self.origami_info[j][-1]) == KinematicLine:
                origami = self.origami_info[j]
                all_unit_number_of_origami = self.unit_number[j]
                copy_time = self.copy_time
                row_num = int(all_unit_number_of_origami / copy_time)
                start_unit_id = sum([0] + [self.unit_number[x] for x in range(j)])
                for i in range(col_num):
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + 0.5 * self.con_left_length,
                            origami[0][Y] + self.unit_width / 4 + self.unit_width * i
                        ],
                        start_unit_id + i * row_num, True, LEFT #left
                    ])
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + 0.5 * self.con_left_length,
                            origami[0][Y] + self.unit_width * 3 / 4 + self.unit_width * i
                        ],
                        start_unit_id + i * row_num + 1, True, LEFT #left
                    ])
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + self.con_left_length + origami[1] + 1 / 2 * self.con_right_length,
                            origami[0][Y] + self.unit_width / 4 + self.unit_width * i
                        ],
                        start_unit_id + row_num - 2 + i * row_num, True, RIGHT #right
                    ])
                    self.connection_hole_kps.append([
                        [
                            origami[0][X] + self.con_left_length + origami[1] + 1 / 2 * self.con_right_length,
                            origami[0][Y] + self.unit_width * 3 / 4 + self.unit_width * i
                        ],
                        start_unit_id + row_num - 1 + i * row_num, True, RIGHT #right
                    ])
            elif type(self.origami_info[j][-1]) == ModuleLeanMiura:
                origami = self.origami_info[j]
                copy_time = origami[-1].copy_time
                if origami[-1].half_flag != NO_HALF:
                    start_unit_id = sum([0] + [self.unit_number[x] for x in range(j)]) + self.unit_number[j] - 2 * copy_time
                    # Get the global bias
                    if origami[-1].half_flag == LEFT_HALF:
                        initial_x = origami[0][X] + origami[-1].con_left_length / 2
                        initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    elif origami[-1].half_flag == RIGHT_HALF:
                        initial_x = origami[0][X] + origami[-1].unit_width * copy_time / 2 + origami[-1].stretch_length + origami[-1].con_right_length / 2
                        initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    f = LEFT # all at left board
                    # Add connection hole
                    for i in range(copy_time):
                        index = 1 + 2 * i
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y - index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index - 1, True, f
                        ])
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y + index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index, True, f
                        ])
                else:
                    start_unit_id = sum([0] + [self.unit_number[x] for x in range(j)]) + self.unit_number[j] - 4 * copy_time
                    initial_x = origami[0][X] + origami[-1].con_left_length / 2
                    initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    # Add connection hole
                    for i in range(copy_time):
                        index = 1 + 2 * i
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y - index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index - 1, True, LEFT
                        ])
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y + index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index, True, LEFT
                        ])
                    # Add connection hole
                    initial_x = origami[0][X] + origami[-1].con_left_length + origami[-1].unit_width * copy_time + origami[-1].stretch_length + origami[-1].con_right_length / 2
                    initial_y = origami[0][Y] + origami[-1].unit_width * copy_time / 2
                    for i in range(copy_time):
                        index = 1 + 2 * i
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y - index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index - 1 + 2 * copy_time, True, LEFT
                        ])
                        self.connection_hole_kps.append([
                            [
                                initial_x,
                                initial_y + index / 4 * origami[-1].unit_width
                            ],
                            start_unit_id + index + 2 * copy_time, True, LEFT
                        ])
                        
        self.backup_connection_hole_kps = deepcopy(self.connection_hole_kps)

    def addCrease(self, creases):
        for new_crease in creases:
            if new_crease.getLength() < 1e-5:
                continue
            same_crease = False
            for crease in self.lines:
                if sameCrease(new_crease, crease):
                    # same kp
                    same_crease = True
                    break
            if not same_crease:
                self.lines.append(new_crease)

    def addHoleToUnit(self, pixel_x, pixel_y):
        """
        @ function: Convert some axis fro
        m pixel to real, and add hole to unit
        @ version: 0.11
        @ developer: py
        @ progress: finish
        @ date: 20230305
        @ spec: None
        """
        real_x = (pixel_x - self.pixel_bias[0]) / self.current_pixel_scale
        real_y = (pixel_y - self.pixel_bias[1]) / self.current_pixel_scale
        unit_id = self.pointInUnit([real_x, real_y])
        if unit_id != None:
            # point, id, valid
            self.hole_kps.append([[real_x, real_y], unit_id, True])
            self.updateMessage("Successfully add hole on unit " + str(unit_id) + ". Total hole number: " + str(len(self.hole_kps)) + " ...")
        else:
            self.updateMessage("Failed to add hole, check if it is inside some unit...", "WARNING")

    def addHoleToUnitUsingRealAxis(self, real_x, real_y):
        """
        @ function: add hole to unit
        @ version: 0.11
        @ developer: py
        @ progress: finish
        @ date: 20230305
        @ spec: None
        """
        unit_id = self.pointInUnit([real_x, real_y])
        if unit_id != None:
            # point, id, valid
            self.hole_kps.append([[real_x, real_y], unit_id, True])
            self.updateMessage("Total: " + str(len(self.hole_kps)) + " Successfully add hole on unit " + str(unit_id) + " ...")
        else:
            self.updateMessage("Failed to add hole, check if it is inside some unit...", "WARNING")    

    def addKp(self, kps):
        for new_kp in kps:
            same_kp = False
            for kp in self.kps:
                if distance(kp, new_kp) < 1e-5:
                    # same kp
                    same_kp = True
                    break
            if not same_kp:
                self.kps.append(new_kp)

    def addLeanMiuraStorage(self):
        lean_miura_storage = ModuleLeanMiura()
        lean_miura_storage.raiseDialog(self)
        if lean_miura_storage.initial_flag:
            # success to add leanmiura
            self.updateMessage("Success to add LeanMiura")
            self.enable_design = True
            tsp = lean_miura_storage.tsp
            new_storage = [
                tsp, 
                lean_miura_storage,
                False
            ]
            self.storage.append(new_storage)
            self.rotation.append(0.0)
            self.add_bias_flag.append(False)

    def addMiuraStorage(self):
        path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a json file with kl specification", 
                ".", 
                "Json files (*.json);;All Files (*.*)"
            )
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            with open(path, 'r', encoding='utf-8') as fw:
                input_json = json.load(fw)
            origin_list = input_json['origin']
            add_width_flag_list = input_json['add_width']
            origin_list_length = len(origin_list)
            for i in range(origin_list_length):
                origin = origin_list[i]
                test, ok = QInputDialog.getText(self, "Reset transition start point", "enter tsp splited with ','")
                if ok:
                    ans = re.findall(r"\d+\.?\d*", test)
                    if len(ans) >= 2:
                        origin[0] = ans[0]
                        origin[1] = ans[1]
                add_width_flag = add_width_flag_list[i]
                kl = KinematicLine()
                for j in range(len(input_json['kl'][i])):
                    element = input_json['kl'][i][j]
                    kl.append(element)
                self.storage.append([origin, kl, add_width_flag])
                self.rotation.append(0.0)
                self.add_bias_flag.append(False)
            self.updateState("Succeeded to load Miura json file", self.state)
            self.enable_design = True

    def addPassStringUsingRealAxis(self, real_x, real_y):
        pass

    def addTSACandidators(self):
        content, ok = QInputDialog.getText(self, "TSA Candidators Input", "Please input the axis of TSA candidator: ")
        if ok:
            ans = re.findall(r"\d+\.?\d*", content)
            if len(ans) != 3:
                self.updateState("Failed to add candidator, make sure that you input x, y and z of the axis", self.state)
            else:
                connection, ok = QInputDialog.getInt(self, "Candidator Connection: ", "Please select a unit for connection: ", -1, -1, len(self.units) - 1, 1)
                if ok:
                    self.P_candidate.append([float(ans[0]), float(ans[1]), float(ans[2])])
                    self.P_candidate_connection_index.append(connection)
                    self.updateState(f"Succeed to add candidator {[float(ans[0]), float(ans[1]), float(ans[2])]} connected to {connection}", self.state)

    def addTsaAPoint(self):
        if len(self.P_candidate):
            index, ok = QInputDialog.getInt(self, "TSA Input: ", "Please input TSA A point ID: ", 0, 0, len(self.P_candidate) - 1, 1)
            if ok:
                self.addStringPoint(self.P_candidate[index][X], self.P_candidate[index][Y], index)
        else:
            self.updateState(f"Please add TSA A candidators first", self.state)
    
    def addTsaAPointWithResolutionValue(self, resolution_value):
        origami_size = [self.origami_length, self.origami_width]
        cal_x = self.pref_pack["tsa_radius"] * math.cos(resolution_value / self.pref_pack["tsa_resolution"] * 2 * math.pi) + origami_size[X] / 2.0
        cal_y = self.pref_pack["tsa_radius"] * math.sin(resolution_value / self.pref_pack["tsa_resolution"] * 2 * math.pi) + origami_size[Y] / 2.0
        self.addStringPoint(cal_x, cal_y, resolution_value)

    def addStringPoint(self, x, y, id=0, id_type='A', reverse=False):
        if self.exist_string_start:
            if reverse:
                end_point = [x, y, 0.0]
                self.strings.append(TSAString())
                self.strings[-1].type = self.string_type
                self.strings[-1].setStringKeyPoint(self.string_start_point, end_point)
                self.strings[-1].id = len(self.string_total_information)

                self.strings.append(TSAString())
                self.strings[-1].type = PASS
                self.strings[-1].setStringKeyPoint(end_point, end_point)
                self.strings[-1].id = len(self.string_total_information)

                tsa_point = TSAPoint()
                tsa_point.point = np.array(end_point)
                tsa_point.point_type = id_type
                tsa_point.id = id
                tsa_point.dir = self.string_type - 1
                self.a_string.append(tsa_point)

                if self.string_type == BOTTOM:
                    self.string_type = TOP
                    self.updateMessage("Enable passing from bottom to top. Add 2 strings, currently the type is top...")
                else:
                    self.string_type = BOTTOM
                    self.updateMessage("Enable passing from top to bottom. Add 2 strings, currently the type is bottom...")

                self.string_start_point = end_point
            else:
                end_point = [x, y, 0.0]
                self.strings.append(TSAString())
                self.strings[-1].type = self.string_type
                self.strings[-1].setStringKeyPoint(self.string_start_point, end_point)
                self.strings[-1].id = len(self.string_total_information)

                tsa_point = TSAPoint()
                tsa_point.point = np.array(end_point)
                tsa_point.point_type = id_type
                tsa_point.id = id
                tsa_point.dir = self.string_type - 1
                self.a_string.append(tsa_point)

                if self.string_type == BOTTOM:
                    self.updateMessage("Add 1 strings, currently the type is still bottom...")
                else:
                    self.updateMessage("Add 1 strings, currently the type is still top...")

                self.string_start_point = end_point
        else:
            if reverse:
                if self.string_type == BOTTOM:
                    self.string_type = TOP
                    self.updateMessage("Change the z-axis of the string, currently the type is top...")
                else:
                    self.string_type = BOTTOM
                    self.updateMessage("Change the z-axis of the string, currently the type is bottom...")
            else:
                self.string_start_point = [x, y, 0.0]
                self.updateMessage("Record the start point of the string...")

                tsa_point = TSAPoint()
                tsa_point.point = np.array([x, y, 0.0])
                tsa_point.point_type = id_type
                tsa_point.id = id
                tsa_point.dir = self.string_type - 1
                self.a_string.append(tsa_point)

                self.exist_string_start = True  

    def addUnitPackStorage(self):
        unit_pack_storage = UnitPack()
        unit_pack_storage.raiseDialog(self)
        if unit_pack_storage.getLength():
            # success to add leanmiura
            self.updateMessage("Success to add UnitPack")
            tsp = unit_pack_storage.tsp
            self.enable_design = True
            new_storage = [
                tsp, 
                unit_pack_storage,
                False
            ]
            self.storage.append(new_storage)
            self.rotation.append(0.0)
            self.add_bias_flag.append(False)

    def axisConverter(self):
        """
        @ function: Convert axis from real to pixel
        @ version: 0.11
        @ developer: py
        @ progress: finish
        @ date: 20230107
        @ spec: Add scale standard
        """
        self.pixel_kps.clear()
        self.pixel_lines.clear()
        self.pixel_hole_kps.clear()
        self.pixel_additional_lines.clear()
        self.pixel_connection_hole_kps.clear()
        self.pixel_string_kps.clear()

        # keypoints
        for kp in self.kps:
            self.pixel_kps.append(self.toPixel(kp))
        
        # lines
        for line in self.lines:
            self.pixel_lines.append(Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType(), hard=line.hard
            ))

        # additional lines
        for line in self.additional_lines:
            self.pixel_additional_lines.append(Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType()
            ))

        # hole keypoints
        for kp in self.hole_kps:
            self.pixel_hole_kps.append([self.toPixel(kp[0]), kp[1], kp[2]])

        # connection hole keypoints
        for kp in self.connection_hole_kps:
            self.pixel_connection_hole_kps.append([self.toPixel(kp[0]), kp[1], kp[2]]
            )

        # string keypoints
        for s in self.strings:
            self.pixel_string_kps.append([self.toPixel(s.start_point), self.toPixel(s.end_point),
                s.type, s.width, s.id
            ])

    def calculateSequence(self):
        try:
            tb = TreeBasedOrigamiGraph(self.kps, self.lines)
            tb.calculateTreeBasedGraph()
            if self.edit_sequence_mode:
                self.chooseCreaseSequence(self.choose_crease_sequence_id)
            self.updateMessage("Succeed to calculate sequence")
        except:
            self.updateMessage("Failed to calculate sequence")

    def cdfCurveFitting(self):
        if self.limitation["match_mode"] < 3:
            if len(self.x_list) == 0:
                self.updateMessage("No curve has been imported, please import a curve file first...")
                return
            if self.limitation["direction_enable"] and len(self.dir_list) == 0:
                self.updateMessage("No direction curve has been imported but direction match enabled, please import direction first...")
                return
        if self.enable_output_stl:
            self.updateMessage("A stl file is being outputed, please wait...")
            return
        if self.enable_cdf_curve_fitting:
            self.updateMessage("A cdf process is running, please wait...")
            return
        if self.enable_phys_data_collecting:
            self.updateMessage("A physical simulation is running, please wait...")
            return
        if self.enable_mcts:
            self.updateMessage("A MCTS Searching process is running, please wait...")
            return
        
        self.enable_cdf_curve_fitting = True
        if self.limitation["match_mode"] < 3:
            self.cdf_curve_fitting_thread = CdfCurveFittingThread(
                curve_name  =self.curve_name,
                pref_pack   =self.limitation,
                curve_x     =self.x_list,
                curve_y     =self.y_list,
                curve_dir   =self.dir_list if self.limitation["direction_enable"] else None
            )
        elif self.limitation["match_mode"] == 3:
            self.cdf_curve_fitting_thread = CdfCurveFittingThread(
                curve_name  ="exoskeleton",
                pref_pack   =self.limitation,
                curve_x     =None,
                curve_y     =None,
                curve_dir   =None
            )
        elif self.limitation["match_mode"] == 4:
            self.cdf_curve_fitting_thread = CdfCurveFittingThread(
                curve_name  ="zerodistance",
                pref_pack   =self.limitation,
                curve_x     =None,
                curve_y     =None,
                curve_dir   =None
            )
        self.cdf_curve_fitting_thread._emit.connect(self.drawProcess)
        self.cdf_curve_fitting_thread.start()

    def changeCoeff(self):
        self.lines[self.choose_crease_sequence_id].coeff = self.doubleSpinBox_coeff_value.value()

    def changeCopyTime(self):
        self.copy_time = self.spinbox_copy_time.value()
        self.design_info['copy_time'] = self.copy_time
        self.enable_design = True

    def changeConLeftLength(self):
        """
        @ function: Change stretch length of left side of the origami
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230314
        @ spec: None
        """
        new_length = self.spinbox_con_left_length.value()
        bias_length = len(self.add_bias_flag)
        
        for kp in self.hole_kps:
            kp[0][X] -= self.con_left_length
            kp[0][X] += new_length

        for i in range(bias_length):
            if self.add_bias_flag[i]:
                src = self.origami_info[i][-1]
                if type(src) == ModuleLeanMiura:
                    self.storage[i][1].stretch_length -= self.con_left_length
                    self.storage[i][1].stretch_length += new_length
        self.con_left_length = self.spinbox_con_left_length.value()
        self.spinbox_connection_radius.setMaximum(min(self.con_right_length, self.con_left_length) / 4)
        self.enable_design = True
        
    def changeConnectionRadius(self):
        self.connection_radius = self.spinbox_connection_radius.value()
        self.enable_design = True

    def changeConRightLength(self):
        self.con_right_length = self.spinbox_con_right_length.value()
        self.spinbox_connection_radius.setMaximum(min(self.con_right_length, self.con_left_length) / 4)
        self.enable_design = True

    def changeCreaseWidth(self):
        self.unit_width = self.spinbox_crease_width.value()
        self.paper_info['unit_width'] = self.unit_width
        self.enable_design = True

    def changeFlag(self):
        self.entry_flag = self.slider_flag.value()
        self.enable_design = True

    def changeFoldingPercent(self):
        val = self.horizontal_folding_slider.value()
        self.operation_amp = self.horizontal_folding_slider.value()
        self.design_info["folding_percent"] = val / 1000.0
        self.label_folding_percent.setText(str(round(100.0 - self.design_info["folding_percent"] * 100.0, 1)) + "%")
        self.label_folding_percent.setGeometry(int(180 + 0.74 * val), 494, 51, 16)
        self.enable_design = True

    def changeHard(self):
        previous = self.lines[self.choose_crease_sequence_id].hard
        if previous:
            self.lines[self.choose_crease_sequence_id].hard = False
            self.radioButton_hard_crease.setChecked(False)
        else:
            self.lines[self.choose_crease_sequence_id].hard = True
            self.radioButton_hard_crease.setChecked(True)
        self.enable_design

    def changeHardAngle(self):
        self.lines[self.choose_crease_sequence_id].hard_angle = self.doubleSpinBox_hard_angle_value.value() * math.pi / 180.0

    def changeHoleSize(self):
        self.hole_size = self.spinbox_hole_size.value()

    def changeHoleResolution(self):
        self.hole_resolution = self.spinbox_resolution.value()
    
    def changeKlLength(self):
        self.storage[self.choose_kl_id][DATA][self.choose_line_id][0] = self.doubleSpinBox_length.value()
        self.enable_design = True

    def changeKlSectorAngle(self):
        self.storage[self.choose_kl_id][DATA][self.choose_line_id][1] = self.doubleSpinBox_sector_angle.value() / 180.0 * math.pi
        self.enable_design = True

    def changeLevel(self):
        self.lines[self.choose_crease_sequence_id].level = self.spinBox_level_value.value()

    def changeRecoverLevel(self):
        self.lines[self.choose_crease_sequence_id].recover_level = self.spinBox_recover_level_value.value()

    def changeUnitBias(self):
        self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] = self.doubleSpinBox_expert_mode.value()

    def checkHoleKpIsValid(self):
        for ele in self.hole_kps:
            try:
                in_unit_id = ele[1]
                # find which origami is falls
                origami_id = -1
                all_unit = 0
                while all_unit <= in_unit_id:
                    all_unit += self.unit_number[origami_id + 1]
                    origami_id += 1
                origami = self.origami_info[origami_id]
                lower_x_bound = origami[0][X] + self.con_left_length
                upper_x_bound = origami[0][X] + origami[1] + self.con_left_length
                unit = self.units[in_unit_id].getSeqPoint()
                if self.enable_connection:
                    min_dis = pointInPolygon(ele[0], unit, return_min_distance=True, lower_x_bound=lower_x_bound, upper_x_bound=upper_x_bound)
                else:
                    min_dis = pointInPolygon(ele[0], unit, return_min_distance=True)
                if min_dis and min_dis > self.hole_size:
                    ele[2] = True
                else:
                    ele[2] = False
            except:
                ele[2] = False

    def chooseKl(self, id):
        self.choose_kl_id = id
        self.choose_line_id = 0
        self.label_current_kl_number.setText(str(id))
        self.label_current_line_number.setText("0")
        self.doubleSpinBox_length.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][0])
        self.doubleSpinBox_sector_angle.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][1] * 180.0 / math.pi)

    def chooseKlLine(self, id):
        self.choose_line_id = id
        self.label_current_line_number.setText(str(id))
        self.doubleSpinBox_length.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][0])
        self.doubleSpinBox_sector_angle.setValue(self.storage[self.choose_kl_id][DATA][self.choose_line_id][1] * 180.0 / math.pi)

    def chooseNextKl(self):
        for i in range(self.choose_kl_id + 1, len(self.storage)):
            if type(self.storage[i][DATA]) == KinematicLine:
                self.chooseKl(i)
                break

    def chooseNextKlLine(self):
        line_number = len(self.storage[self.choose_kl_id][DATA])
        choose_line_id = (self.choose_line_id + 1) % line_number
        self.chooseKlLine(choose_line_id)

    def choosePreviousKl(self):
        for i in range(self.choose_kl_id - 1, -1, -1):
            if type(self.storage[i][DATA]) == KinematicLine:
                self.chooseKl(i)
                break

    def choosePreviousKlLine(self):
        line_number = len(self.storage[self.choose_kl_id][DATA])
        choose_line_id = (self.choose_line_id - 1 + line_number) % line_number
        self.chooseKlLine(choose_line_id)

    def chooseUnit(self, id):
        self.choose_unit_id = id
        self.choose_crease_id = 0
        self.label_current_unit_number.setText(str(id))
        self.label_current_crease_number.setText("0")
        if self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] == None:
            self.radioButton_use_default.setChecked(True)
            self.radioButton_expert_mode.setChecked(False)
            self.doubleSpinBox_expert_mode.setVisible(False)
        else:
            self.radioButton_use_default.setChecked(False)
            self.radioButton_expert_mode.setChecked(True)
            self.doubleSpinBox_expert_mode.setVisible(True)
            self.doubleSpinBox_expert_mode.setValue(self.unit_bias_list[self.choose_unit_id][self.choose_crease_id])

    def chooseCreaseSequence(self, id):
        self.choose_crease_sequence_id = id
        self.label_crease_sequence_id.setText(str(id))
        self.spinBox_level_value.setValue(self.lines[id].level)
        self.doubleSpinBox_coeff_value.setValue(self.lines[id].coeff)
        self.spinBox_recover_level_value.setValue(self.lines[id].recover_level)
        self.radioButton_hard_crease.setChecked(self.lines[id].hard)
        self.doubleSpinBox_hard_angle_value.setValue(self.lines[id].hard_angle * 180.0 / math.pi)

    def chooseCrease(self, id):
        self.choose_crease_id = id
        self.label_current_crease_number.setText(str(id))
        if self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] == None:
            self.radioButton_use_default.setChecked(True)
            self.radioButton_expert_mode.setChecked(False)
            self.doubleSpinBox_expert_mode.setVisible(False)
        else:
            self.radioButton_use_default.setChecked(False)
            self.radioButton_expert_mode.setChecked(True)
            self.doubleSpinBox_expert_mode.setVisible(True)
            self.doubleSpinBox_expert_mode.setMinimum(self.pref_pack["print_accuracy"])
            self.doubleSpinBox_expert_mode.setMaximum(self.unit_width / 6.0)
            self.doubleSpinBox_expert_mode.setValue(self.unit_bias_list[self.choose_unit_id][self.choose_crease_id])

    def chooseNextUnit(self):
        unit_number = len(self.units)
        choose_unit_id = (self.choose_unit_id + 1) % unit_number
        self.chooseUnit(choose_unit_id)

    def chooseNextCrease(self):
        crease_number = len(self.units[self.choose_unit_id].getCrease())
        choose_crease_id = (self.choose_crease_id + 1) % crease_number
        self.chooseCrease(choose_crease_id)
    
    def chooseNextCreaseSequence(self):
        id = (self.choose_crease_sequence_id + 1) % len(self.lines)
        self.chooseCreaseSequence(id)

    def choosePreviousUnit(self):
        unit_number = len(self.units)
        choose_unit_id = (self.choose_unit_id - 1 + unit_number) % unit_number
        self.chooseUnit(choose_unit_id)

    def choosePreviousCrease(self):
        crease_number = len(self.units[self.choose_unit_id].getCrease())
        choose_crease_id = (self.choose_crease_id - 1 + crease_number) % crease_number
        self.chooseCrease(choose_crease_id)

    def choosePreviousCreaseSequence(self):
        id = (self.choose_crease_sequence_id - 1 + len(self.lines)) % len(self.lines)
        self.chooseCreaseSequence(id)

    def closeEvent(self, event):
        self.stopThread()
        self.timer.stop()
        self.deleteLater()

    def defineAction(self):
        """
        @ function: define actions and connections with specific method
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230415
        @ spec: None
        """
        self.actionNew_file.triggered.connect(self.newFile)
        self.actionImport.triggered.connect(self.importKL)
        self.actionPrint_P.triggered.connect(self.printOrigami)
        self.actionAs_Dxf.triggered.connect(self.exportAsDxf)
        self.actionAs_Stl.triggered.connect(self.exportAsStl)
        self.actionAll_As_Stl.triggered.connect(self.exportAllAsStl)
        self.actionAs_Split_Dxf.triggered.connect(self.exportAsSplitDxf)
        self.actionSettings.triggered.connect(self.setting)
        self.actionSave_result.triggered.connect(self.saveResult)
        self.actionOpen_file_O.triggered.connect(self.openFile)
        self.actionLeanMiura.triggered.connect(self.addLeanMiuraStorage)
        self.actionMiura.triggered.connect(self.addMiuraStorage)
        self.actionTransition_T.triggered.connect(self.showTG)
        self.actionView_Curve.triggered.connect(self.showCurve)
        self.actionCDF_Curve_Fitting_F.triggered.connect(self.cdfCurveFitting)
        self.actionStop_Thread_S.triggered.connect(self.stopThread)
        self.actionImport_dxf.triggered.connect(self.importDxf)
        self.actionAdd_Holes.triggered.connect(self.oneClickAddHoles)
        self.actionPhysical_Simulation_P.triggered.connect(self.physicalSimulation)
        self.actionImport_Directions_D.triggered.connect(self.showDirection)
        self.actionAdd_TSA_A_point.triggered.connect(self.addTsaAPoint)
        self.actionCollect_Physical_Data_C.triggered.connect(self.physicalDataCollecting)
        self.actionPlot_Physical_Data.triggered.connect(self.plotJson)
        self.actionPlot_Evolution_Data.triggered.connect(self.plotEvolutionJson)
        self.actionExplicit_Simulation_E.triggered.connect(self.physicalSimulationExplicit)
        self.actionExpert_Mode_E.triggered.connect(self.expertModeEnable)
        self.actionEdit_kl_E.triggered.connect(self.editKl)
        self.actionCalculate_Sequence.triggered.connect(self.calculateSequence)
        self.actionAs_Full_description_Data.triggered.connect(self.exportDescriptionData)
        self.actionEdit_Sequence_S.triggered.connect(self.editSequence)
        self.actionImport_string_path.triggered.connect(self.importStringPath)
        self.actionAdd_TSA_A_candidators.triggered.connect(self.addTSACandidators)
        self.actionShow_Index.triggered.connect(self.showIndex)
        self.actionDelete_TSA_A_Candidators.triggered.connect(self.deleteTSACandidators)

    def defineButton(self):
        """
        @ function: define buttons and connections with specific method
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230415
        @ spec: None
        """
        self.button_design.clicked.connect(self.onDesign)
        self.button_threading_design.clicked.connect(self.onDesignThreadingMethod)
        self.button_reset_view.clicked.connect(self.resetView)
        self.radiobutton_A4.clicked.connect(self.showA4Square)
        self.radiobutton_none.clicked.connect(self.showNone)
        self.checkBox_add_hole_mode.clicked.connect(self.onAddHoleMode)
        self.checkBox_connection.clicked.connect(self.onAddConnection)
        self.checkBox_add_string_mode.clicked.connect(self.onAddStringMode)
        self.pushButton_next_unit.clicked.connect(self.chooseNextUnit)
        self.pushButton_next_crease.clicked.connect(self.chooseNextCrease)
        self.pushButton_previous_unit.clicked.connect(self.choosePreviousUnit)
        self.pushButton_previous_crease.clicked.connect(self.choosePreviousCrease)
        self.radioButton_use_default.clicked.connect(self.setBiasAsDefault)
        self.radioButton_expert_mode.clicked.connect(self.setBiasAsExpertModified)
        self.pushButton_next_line.clicked.connect(self.chooseNextKlLine)
        self.pushButton_previous_line.clicked.connect(self.choosePreviousKlLine)
        self.pushButton_previous_kl.clicked.connect(self.choosePreviousKl)
        self.pushButton_next_kl.clicked.connect(self.chooseNextKl)
        self.pushButton_previous_crease_sequence.clicked.connect(self.choosePreviousCreaseSequence)
        self.pushButton_next_crease_sequence.clicked.connect(self.chooseNextCreaseSequence)
        self.radioButton_hard_crease.clicked.connect(self.changeHard)

    def defineSpinbox(self):
        """
        @ function: define spinboxes and connections with specific method
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230415
        @ spec: None
        """
        self.spinbox_crease_width.valueChanged.connect(self.changeCreaseWidth)
        self.spinbox_copy_time.valueChanged.connect(self.changeCopyTime)
        self.spinbox_hole_size.valueChanged.connect(self.changeHoleSize)
        self.spinbox_resolution.valueChanged.connect(self.changeHoleResolution)
        self.spinbox_connection_radius.valueChanged.connect(self.changeConnectionRadius)
        self.spinbox_con_left_length.valueChanged.connect(self.changeConLeftLength)
        self.spinbox_con_right_length.valueChanged.connect(self.changeConRightLength)
        self.horizontal_folding_slider.valueChanged.connect(self.changeFoldingPercent)
        self.slider_flag.valueChanged.connect(self.changeFlag)
        self.doubleSpinBox_expert_mode.valueChanged.connect(self.changeUnitBias)
        self.doubleSpinBox_length.valueChanged.connect(self.changeKlLength)
        self.doubleSpinBox_sector_angle.valueChanged.connect(self.changeKlSectorAngle)
        self.spinBox_level_value.valueChanged.connect(self.changeLevel)
        self.doubleSpinBox_coeff_value.valueChanged.connect(self.changeCoeff)
        self.spinBox_recover_level_value.valueChanged.connect(self.changeRecoverLevel)
        self.doubleSpinBox_hard_angle_value.valueChanged.connect(self.changeHardAngle)

    def deleteTSACandidators(self):
        items = []
        # List all P_CANDIDATE
        for i in range(len(self.P_candidate)):
            items.append(f"Candidator {str(i)}, Axis: [{self.P_candidate[i][X]}, {self.P_candidate[i][Y]}, {self.P_candidate[i][Z]}], connecting to {self.P_candidate_connection_index[i]}")
        selected_item, ok = QInputDialog.getItem(self, "Select Miura Item", "Select a Miura combo:", items)
        # If press ok
        if ok:
            index = items.index(selected_item)
            del(self.P_candidate[index])
            del(self.P_candidate_connection_index[index])
            self.updateState(f"Succeed to delete Candidator {index}", self.state)

    def design(self):
        """
        @ function: Design origami crease
        @ version: 0.11
        @ developer: py
        @ progress: on road
        @ spec: Cancel try/except
        """
        self.kps.clear()
        self.lines.clear()
        self.additional_lines.clear()
        self.crease_lines.clear()
        self.connection_hole_kps.clear()
        self.units.clear()
        self.unit_bias_list.clear()

        body_line = []
        # whether to get additional line
        self.additional_line_maker.clearValidCrease()
        if self.add_hole_mode:
            self.additional_line_maker.clear()
            self.additional_line_maker.clearCrease()   
            self.additional_line_maker.setBias(self.bias_val)
            self.additional_line_maker.enable_difference = self.pref_pack["additional_line_option"]
        # Clear origami info 
        self.origami_info.clear()
        self.origami_number = len(self.storage)
        self.unit_number.clear()
        try:
            # for all storage
            for i in range(self.origami_number):
                r = R(self.rotation[i])
                src = self.storage[i][1]
                # ----- Set the transition start point for designer ----- #
                # ----- START ----- #
                if(self.storage[i][2]):
                    self.designer.setTransitionStartPoint([
                        self.storage[i][0][0], 
                        self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                    ])
                else:
                    self.designer.setTransitionStartPoint(self.storage[i][0])
                # ----- END ----- #

                # ----- Set the source for designer ----- #
                # ----- START ----- #
                self.designer.setSource(src)
                # ----- END ----- #

                # If LeanMiura is using global data
                # ----- START ----- #
                if type(src) == ModuleLeanMiura:
                    if src.enable_global_modify:
                        src.unit_width = self.unit_width
                        src.copy_time = self.copy_time
                        src.entry_flag = self.entry_flag if self.copy_time % 2 else not self.entry_flag
                        src.connection_flag = self.enable_connection
                        src.con_left_length = self.con_left_length
                        src.con_right_length = self.con_right_length
                        src.connection_hole_size = self.connection_radius
                # ----- END ----- #

                # Clear data of additional_line_maker to generate new additional line
                # ----- START ----- #
                self.additional_line_maker.clear() 
                # ----- END ----- #

                # Set connection at left and right, and change hole position
                self.designer.clearAdditionalLine()
                # ----- START ----- #
                if type(src) == KinematicLine:
                    if self.enable_connection:
                        self.designer.insertLineSource([self.con_left_length, self.designer.src.lines[0][1]], 0)
                        self.designer.insertLineSource([self.con_right_length, self.designer.src.lines[-1][1]], self.designer.getKLNumber())
                        if not self.add_bias_flag[i]:
                            for kp in self.hole_kps:
                                kp[0][X] += self.con_left_length
                            self.add_bias_flag[i] = True
                    else:
                        if self.add_bias_flag[i]:
                            for kp in self.hole_kps:
                                kp[0][X] -= self.con_left_length
                            self.add_bias_flag[i] = False
                elif type(src) == ModuleLeanMiura:
                    if self.enable_connection:
                        if not self.add_bias_flag[i]:
                            if self.storage[i][1].half_flag == RIGHT_HALF:
                                # We have modify the stretch line of LeanMiura
                                self.storage[i][1].stretch_length += src.con_left_length
                                self.storage[i][1].modify_stretch_flag = True
                                self.add_bias_flag[i] = True
                    else:
                        if self.add_bias_flag[i]:
                            if self.storage[i][1].half_flag == RIGHT_HALF:
                                self.storage[i][1].stretch_length -= src.con_left_length
                                self.storage[i][1].modify_stretch_flag = False
                                self.add_bias_flag[i] = False
                # ----- END ----- #

                # Set the data of designer to make new origami
                # ----- START ----- #
                self.designer.setPaperInfo(self.paper_info)
                self.designer.setDesignInfo(self.design_info)
                self.designer.setEntryFlag(self.entry_flag)
                # ----- END ----- #
                
                # <<<<< Before design <<<<< #

                # -------- Design -------- #
                # ----- START ----- #
                self.designer.parseData()
                # ----- END ----- #
                # -------- Design -------- #

                # >>>>> After design >>>>> #

                # Get design data
                # ----- START ----- #
                self.origami_length, self.origami_width = self.designer.getPaperData() # not include connection
                # Set tsp, length, width and type
                self.origami_info.append([self.designer.getTransitionStartPoint(), self.origami_length, self.origami_width, self.storage[i][1]])
                data = self.designer.getDesignData()
                # ----- END ----- #

                # Get kp, line, unit
                # ----- START ----- #
                # LeanMiura Module, only one result at data[0]
                if type(src) == ModuleLeanMiura: 
                    # Pull out the modify stretch length
                    if src.modify_stretch_flag:
                        self.origami_info[i][1] -= src.con_left_length
                    kp = data[0].getKeyPoint()
                    line = data[0].getLine()
                    new_kp_list = []
                    new_line_list = []
                    if abs(self.rotation[i]) > 1e-5:
                        tsp = np.array(self.storage[i])
                        for ele in kp:
                            new_kp = (r @ np.array([kp[X] - self.storage[i][X], kp[Y] - self.storage[i][Y]]) + tsp).tolist()
                            new_kp_list.append(new_kp)
                        self.kps += new_kp_list
                        for ele in line:
                            new_start = (r @ np.array(ele[START]) + tsp).tolist()
                            new_end = (r @ np.array(ele[END]) + tsp).tolist()
                            new_line = Crease(new_start, new_end, ele.getType())
                            new_line_list.append(new_line)
                        self.lines += new_line_list
                    else:
                        self.addKp(kp)
                        self.addCrease(line)
                    u = data[0].getUnits(connection=True)
                    self.unit_number.append(len(u))
                    if self.add_hole_mode:
                        for ele in u:
                            self.additional_line_maker.addPackedOrigamiUnit(ele)
                    else:
                        for ele in u:
                            self.units.append(ele)
                # Miura Module
                elif type(src) == KinematicLine:
                    data_length = len(data)
                    row_data_length = int(data_length / self.copy_time)
                    self.unit_number.append((row_data_length - 1) * self.copy_time * 2)
                    for j in range(self.copy_time):
                        for k in range(0, row_data_length):
                            unit_id = k + j * row_data_length
                            kp = data[unit_id].getKeypoint()
                            # self.addKp(kp)
                            self.addKp(kp)
                            line = data[unit_id].getLine()
                            record_body_line = data[unit_id].getLineConnectToBody()
                            # self.addCrease(line)
                            self.addCrease(line)
                            body_line += record_body_line
                            if self.add_hole_mode:
                                if k >= 1:
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])
                                    self.additional_line_maker.addPackedOrigamiUnit(u1)
                                    self.additional_line_maker.addPackedOrigamiUnit(u2)
                            else:
                                if k >= 1:
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])
                                    self.units.append(u1)
                                    self.units.append(u2)

                elif type(src) == DxfDirectGrabber:
                    kp = data[0].getKeyPoint()
                    line = data[0].getLine()
                    u = data[0].getUnits()

                    self.addKp(kp)
                    self.addCrease(line)
                    self.unit_number.append(len(u))
                    if self.add_hole_mode:
                        for ele in u:
                            self.additional_line_maker.addPackedOrigamiUnit(ele)
                    else:
                        for ele in u:
                            self.units.append(ele)

                self.additional_line_maker.addValidCreases(self.lines)
                if self.add_hole_mode:
                    self.units += self.additional_line_maker.unit_list
                if self.enable_read_list_from_backup:
                    self.unit_bias_list = deepcopy(self.backup_unit_bias_list)
                    self.enable_read_list_from_backup = False
                else:
                    self.unit_bias_list = [[None for j in range(len(unit.getCrease()))] for unit in self.units]
                if self.expert_mode:
                    self.chooseUnit(self.choose_unit_id)
                    self.chooseCrease(self.choose_crease_id)
                if self.edit_sequence_mode:
                    self.chooseCreaseSequence(self.choose_crease_sequence_id)
                # UnitPackParser
                
                
                # Collect units done
                # -----END----- #
                  
                # Get additional line based on units
                # -----START----- #           
                if self.add_hole_mode:
                    if self.enable_connection:
                        self.additional_lines += self.additional_line_maker.getAdditionalLineForAllUnit(
                            upper_x_bound=self.origami_info[i][1] + self.con_left_length + self.storage[i][0][0],
                            lower_x_bound=self.con_left_length + self.storage[i][0][0]
                        )
                    else:
                        self.additional_lines += self.additional_line_maker.getAdditionalLineForAllUnit()
                    self.crease_lines += self.additional_line_maker.calculateDrawingForAllCrease()
                # -----END----- #
                
                # Add connection hole
                # -----START----- #
                if self.enable_connection:
                    self.addConnectionHole()
                # -----END----- #

                # Set crease stiffness
                for index in self.hard_crease_index:
                    origin_index = self.additional_line_maker.valid_crease_list[index].origin_index
                    self.lines[origin_index].setHard(True)
                
            # paint result
            self.drawCreasePattern()
            if self.state == self.DESIGN_FINISH or self.state == self.DESIGN_ERROR:
                self.updateState("Design finished with success", self.DESIGN_FINISH)

        except Exception as e:
            self.updateState("Design failed, Adjusting...", self.DESIGN_ERROR, "ERROR")
            if self.bias_val > 0.3:
                self.bias_val -= 0.05
            else:
                self.spinbox_crease_width.setValue(self.unit_width - 0.25)
                self.bias_val = self.unit_width / 6.0
    
    def dictToStorage(self, input_json: json, json_type):
        """
        @ function: Turn input json file to application storage
        @ version: 0.11
        @ developer: py
        @ progress: on road
        @ spec: add packed-data parser
        """
        
        if json_type == self.KL_JSON:
            self.storage.clear()
            self.add_bias_flag.clear()
            self.file_type = "KL"
            origin_list = input_json['origin']
            add_width_flag_list = input_json['add_width']
            origin_list_length = len(origin_list)
            for i in range(origin_list_length):
                origin = origin_list[i]
                add_width_flag = add_width_flag_list[i]
                kl = KinematicLine()
                for j in range(len(input_json['kl'][i])):
                    element = input_json['kl'][i][j]
                    kl.append(element)
                self.storage.append([origin, kl, add_width_flag])
                self.rotation.append(0.0)
                self.add_bias_flag.append(False)
        
        elif json_type == self.PACKED_DATA:
            # We clear all storage to generate new one
            self.storage.clear()
            self.add_bias_flag.clear()
            # Get main data, hole data and settings
            data = input_json['origami']
            util = input_json['util']
            setting = input_json['setting']
            #1 Setup all the parameters in setting
            self.setGlobalParameter(
                unit_width          =setting["unit_width"],
                copy_time           =setting["copy_time"],
                entry_flag          =setting["entry_flag"],
                add_hole_mode       =setting["add_hole_mode"],
                hole_size           =setting["hole_size"],
                hole_resolution     =setting["hole_resolution"],
                enable_connection   =setting["enable_connection"],
                con_left_length     =setting["con_left_length"],
                con_right_length    =setting["con_right_length"],
                con_radius          =setting["con_radius"],
                bias_val            =setting["bias_val"]
            )
            #2 Setup all storage data in data
            for ele in data:
                self.file_type = ele["type"]
                origin = ele["tsp"]
                add_width_flag = ele["add_width"]
                if self.file_type == "kl":
                    kl = KinematicLine()
                    for j in range(len(ele['data'])):
                        element = ele['data'][j]
                        kl.append(element)
                    self.storage.append([origin, kl, add_width_flag])
                    self.rotation.append(0.0)
                    self.add_bias_flag.append(False)
                elif self.file_type == "leanMiura":
                    lean_miura_storage = ModuleLeanMiura()
                    lean_miura_storage.initialize(
                        unit_width          =self.unit_width,
                        copy_time           =self.copy_time,
                        entry_flag          =self.entry_flag,
                        stretch_length      =ele["data"]["stretch_length"],
                        connection_flag     =self.enable_connection,
                        con_left_length     =self.con_left_length,
                        con_right_length    =self.con_right_length,
                        con_radius          =self.connection_radius,
                        half_flag           =ele["data"]["half_type"],
                        tsp                 =origin,
                        enabled             =True
                    )
                    self.storage.append([origin, lean_miura_storage, add_width_flag])
                    self.rotation.append(0.0)
                    self.add_bias_flag.append(False)
                elif self.file_type == "dxf":
                    dxf_grabber = DxfDirectGrabber()
                    dxf_grabber.kps = ele["data"]["kps"]
                    dxf_grabber.lines = ele["data"]["lines"]
                    dxf_grabber.lines_type = ele["data"]["lines_type"]
                    self.storage.append([origin, dxf_grabber, add_width_flag])
                    self.rotation.append(0.0)
                    self.add_bias_flag.append(False)

            # Start design
            self.design()
            # End design

            #3 Add holes to origami in util
            self.hole_kps.clear()
            for ele in util["hole_axis"]:
                self.addHoleToUnitUsingRealAxis(ele[0][X], ele[0][Y])

            #4 Add strings
            try:
                self.P_candidate = input_json['P_candidators']["points"]
                self.P_candidate_connection_index = input_json['P_candidators']["connections"]
                self.string_total_information = []
                self.strings = []   
                string_type_list = input_json['strings']['type']
                string_id_list = input_json['strings']['id']
                string_reverse_list = input_json['strings']['reverse']
                for i in range(len(string_type_list)):
                    self.startAddString()
                    if string_reverse_list[i][0] == -1:
                        self.string_type = BOTTOM
                    else:
                        self.string_type = TOP
                    for j in range(len(string_type_list[i])):
                        if string_type_list[i][j] == 'A':
                            index = string_id_list[i][j]
                            self.addStringPoint(self.P_candidate[index][X], self.P_candidate[index][Y], index)
                        else:
                            unit_axis = self.units[string_id_list[i][j]].getCenter()
                            self.addStringPoint(unit_axis[X], unit_axis[Y], string_id_list[i][j], 'B', string_reverse_list[i][j])
                    self.endAddString()
            except:
                pass

            try:
                self.backup_unit_bias_list = input_json['unit_bias_list']
                self.enable_read_list_from_backup = True
            except:
                self.enable_read_list_from_backup = False

            self.updateState("Succeeded to load json file", self.IMPORT_SUCCESS)
            self.updateMessage("Previewing, click the \"design\" button to continue...")

        elif json_type == self.THREADING_METHOD:
            #4 Add strings
            # Parameters of string start&end
            self.exist_string_start = False
            self.string_start_point = []
            self.string_type = BOTTOM

            self.strings = []  
            self.a_string = [] # one string
            self.string_total_information = [] # list of list

            string_type_list = input_json['type']
            string_id_list = input_json['id']
            string_reverse_list = input_json['reverse']
            for i in range(len(string_type_list)):
                self.startAddString()
                if string_reverse_list[i][0] == -1:
                    self.string_type = BOTTOM
                else:
                    self.string_type = TOP
                for j in range(len(string_type_list[i])):
                    if string_type_list[i][j] == 'A':
                        self.addTsaAPointWithResolutionValue(string_id_list[i][j])
                    else:
                        unit_axis = self.units[string_id_list[i][j]].getCenter()
                        self.addStringPoint(unit_axis[X], unit_axis[Y], string_id_list[i][j], 'B', string_reverse_list[i][j])
                self.endAddString()

    def drawA4Pixmap(self):
        """
        @ function: Draw origami crease on A4
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ spec: update color and drawing method
        """
        line_weight = self.pref_pack["line_weight"]
        self.A4_pixmap = QPixmap(round(self.A4_length), round(self.A4_width))
        bias_x = round(self.half_pixmap_length - self.A4_half_length)
        bias_y = round(self.half_pixmap_width - self.A4_half_width)
        A4_pixel_kps = []
        A4_pixel_lines = []
        self.A4_pixmap.fill(QColor(255, 255, 255))
        painter = QPainter(self.A4_pixmap)
        for kp in self.pixel_kps:
            A4_pixel_kps.append([kp[0] - bias_x, kp[1] - bias_y])
        for line in self.pixel_lines:
            A4_pixel_lines.append(Crease(
                [(round(line[0][0] - bias_x)), (round(line[0][1] - bias_y))], 
                [(round(line[1][0] - bias_x)), (round(line[1][1] - bias_y))], 
                line.getType()
            ))
        for line in A4_pixel_lines:
            type_crease = line.getType()
            if type_crease == MOUNTAIN:
                painter.setPen(QPen(QColor(255, 0, 0), line_weight, Qt.SolidLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            elif type_crease == VALLEY:
                painter.setPen(QPen(QColor(0, 0, 255), line_weight, Qt.DashLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            elif type_crease == CUTTING:
                painter.setPen(QPen(QColor(0, 205, 102), line_weight, Qt.SolidLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
            else:
                painter.setPen(QPen(QColor(0, 0, 0), line_weight + 1, Qt.SolidLine))
                painter.drawLine(line[0][0], line[0][1], line[1][0], line[1][1])
        painter.setPen(QPen(QColor(0, 0, 0), line_weight + 1, Qt.SolidLine))
        if self.pref_pack["show_keypoint"]:
            for kp in A4_pixel_kps:
                painter.drawRect(kp[0] - 1, kp[1] - 1, 2, 2)
        painter.end()

    def drawCreasePattern(self):
        """
        @ function: Draw origami crease
        @ version: 0.1111
        @ developer: py
        @ progress: waiting
        @ date: 20230228
        @ spec: update color and drawing method
                add hole showing
        """
        line_weight = self.pref_pack["line_weight"]
        theme = self.pref_pack["theme"]

        self.painter.begin(self.pixmap)
        self.axisConverter()

        if self.expert_mode:
            unit = self.units[self.choose_unit_id]
            line = unit.getCrease()[self.choose_crease_id]
            highlight_crease = Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType(), hard=line.hard
            )
            highlight_unit = [Crease(
                self.toPixel(ele[START]), self.toPixel(ele[END]), 
                ele.getType(), hard=ele.hard
            ) for ele in unit.getCrease()]
        
        if self.edit_sequence_mode:
            line = self.lines[self.choose_crease_sequence_id]
            highlight_crease_sequence = Crease(
                self.toPixel(line[START]), self.toPixel(line[END]), 
                line.getType(), hard=line.hard
            )

        if theme == 0:
            self.pixmap.fill(QColor(255, 255, 255))
        else:
            self.pixmap.fill(QColor(0, 0, 0))
        if self.show_square == "A4":
            self.painter.fillRect(
                round(self.half_pixmap_length - self.A4_half_length), 
                round(self.half_pixmap_width - self.A4_half_width), 
                round(self.A4_length), 
                round(self.A4_width),
                QColor(135, 206, 250)
            )
        #draw lines
        for line in self.pixel_lines:
            type_crease = line.getType()
            if line.hard:
                self.painter.setPen(QPen(QColor(160, 160, 160), line_weight, Qt.SolidLine))
                self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
            else:
                if type_crease == MOUNTAIN:
                    self.painter.setPen(QPen(QColor(255, 0, 0), line_weight, Qt.SolidLine))
                    self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                elif type_crease == VALLEY:
                    self.painter.setPen(QPen(QColor(0, 0, 255), line_weight, Qt.DashLine))
                    self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                elif type_crease == CUTTING:
                    self.painter.setPen(QPen(QColor(0, 205, 102), line_weight, Qt.SolidLine))
                    self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                else:
                    if theme == 0:
                        self.painter.setPen(QPen(QColor(0, 0, 0), line_weight + 1, Qt.SolidLine))
                        self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
                    else:
                        self.painter.setPen(QPen(QColor(255, 255, 255), line_weight + 1, Qt.SolidLine))
                        self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
        # draw kps
        if self.pref_pack["show_keypoint"]:
            if theme == 0:
                self.painter.setPen(QPen(QColor(0, 0, 0), line_weight + 1, Qt.SolidLine))
                for kp in self.pixel_kps:
                    self.painter.drawRect(kp[X] - 2, kp[Y] - 2, 4, 4)
            else:
                self.painter.setPen(QPen(QColor(255, 255, 255), line_weight + 1, Qt.SolidLine))
                for kp in self.pixel_kps:
                    self.painter.drawRect(kp[X] - 2, kp[Y] - 2, 4, 4)
        # draw additional lines
        self.painter.setPen(QPen(QColor(255, 165, 0), round(line_weight / 2 + 0.5), Qt.DashDotDotLine))    
        for line in self.pixel_additional_lines:
            self.painter.drawLine(line[START][X], line[START][Y], line[END][X], line[END][Y])
        # draw holes
        for kp in self.pixel_hole_kps:
            if kp[2]:
                self.painter.setPen(QPen(QColor(36, 203, 105), round(line_weight / 2 + 0.5), Qt.SolidLine))
                self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.hole_size))
            else:
                self.painter.setPen(QPen(QColor(192, 192, 192), round(line_weight / 2 + 0.5), Qt.DashLine))
                self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.hole_size))
        # draw connection border
        if self.enable_connection:
            self.painter.setPen(QPen(QColor(155, 35, 155), round(line_weight / 2 + 0.5), Qt.DashLine))
            for ele in self.origami_info:
                pixel_border_left = (self.con_left_length + ele[0][X]) * self.current_pixel_scale + self.pixel_bias[X]
                pixel_border_right = (ele[1] + self.con_left_length + ele[0][X]) * self.current_pixel_scale + self.pixel_bias[X]
                pixel_border_up = (ele[2] + ele[0][Y]) * self.current_pixel_scale + self.pixel_bias[Y]
                pixel_border_down = ele[0][Y] * self.current_pixel_scale + self.pixel_bias[Y]
                self.painter.drawLine(
                    pixel_border_left, 
                    pixel_border_down,
                    pixel_border_left,
                    pixel_border_up
                )
                self.painter.drawLine(
                    pixel_border_right, 
                    pixel_border_down,
                    pixel_border_right,
                    pixel_border_up
                )
            # draw connection hole
            for kp in self.pixel_connection_hole_kps:
                if kp[2]:
                    self.painter.setPen(QPen(QColor(36, 203, 105), round(line_weight / 2 + 0.5), Qt.SolidLine))
                    self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.connection_radius))
                else:
                    self.painter.setPen(QPen(QColor(192, 192, 192), round(line_weight / 2 + 0.5), Qt.DashLine))
                    self.painter.drawPolygon(self.generatePolygonByCenter(kp[0], self.connection_radius))

        # draw strings
        color = QColor(190, 167, 219)
        for ele in self.pixel_string_kps:
            if ele[4] != self.choose_string_id:
                if ele[2] == BOTTOM:
                    self.painter.setPen(QPen(color, round(line_weight / 2 + 0.5), Qt.DashLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                elif ele[2] == TOP:
                    self.painter.setPen(QPen(color, round(line_weight / 2 + 0.5), Qt.SolidLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                else:
                    pixel_width = ele[3] / 2.0 * self.current_pixel_scale
                    self.painter.setPen(QPen(color, round(line_weight / 2 + 0.5), Qt.SolidLine))
                    self.painter.fillRect(int(ele[START][X] - pixel_width), int(ele[START][Y] - pixel_width), int(pixel_width * 2), int(pixel_width * 2), color)
        
        color = QColor(86, 10, 180)
        for ele in self.pixel_string_kps:
            if ele[4] == self.choose_string_id:
                if ele[2] == BOTTOM:
                    self.painter.setPen(QPen(color, round(line_weight / 2 + 0.5), Qt.DashLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                elif ele[2] == TOP:
                    self.painter.setPen(QPen(color, round(line_weight / 2 + 0.5), Qt.SolidLine))
                    self.painter.drawLine(ele[START][X], ele[START][Y], ele[END][X], ele[END][Y])
                else:
                    pixel_width = ele[3] / 2.0 * self.current_pixel_scale
                    self.painter.setPen(QPen(color, round(line_weight / 2 + 0.5), Qt.SolidLine))
                    self.painter.fillRect(int(ele[START][X] - pixel_width), int(ele[START][Y] - pixel_width), int(pixel_width * 2), int(pixel_width * 2), color)

        # draw highlight crease and unit
        if self.expert_mode:
            self.painter.setPen(QPen(QColor(36, 203, 105), round(line_weight), Qt.SolidLine))
            for crease in highlight_unit:
                self.painter.drawLine(crease[START][X], crease[START][Y], crease[END][X], crease[END][Y])
            self.painter.setPen(QPen(QColor(255, 223, 0), round(line_weight), Qt.SolidLine))
            self.painter.drawLine(highlight_crease[START][X], highlight_crease[START][Y], highlight_crease[END][X], highlight_crease[END][Y])
        
        if self.edit_sequence_mode:
            self.painter.setPen(QPen(QColor(255, 223, 0), round(line_weight), Qt.SolidLine))
            self.painter.drawLine(highlight_crease_sequence[START][X], highlight_crease_sequence[START][Y], highlight_crease_sequence[END][X], highlight_crease_sequence[END][Y])
        
        self.painter.setPen(QPen(QColor(0, 0, 0), round(line_weight), Qt.SolidLine))
        if self.show_index:
            for i in range(len(self.units)):
                pixel_center = self.toPixel(self.units[i].getCenter())   
                self.painter.drawText(QPoint(pixel_center[X], pixel_center[Y]), str(i))
            for i in range(len(self.P_candidate)):
                pixel_center = self.toPixel(self.P_candidate[i])
                center = QPoint(pixel_center[X], pixel_center[Y])
                self.painter.drawRect(pixel_center[X] - 2, pixel_center[Y] - 2, 4, 4)
                self.painter.drawText(center, str(i))

        self.draw_panel.setPixmap(self.pixmap)
        self.painter.end()

    def drawProcess(self, process=0.0):
        """
        @ function: draw process bar
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230312
        @ spec: None
        """
        if self.enable_output_stl:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_output_stl = False
                self.updateMessage("Succeed to export as stl at " + 
                                   self.output_stl_file_path + 
                                   self.output_stl_crease_flag + 
                                   self.output_stl_board_flag)
        elif self.enable_cdf_curve_fitting:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_cdf_curve_fitting = False
                self.updateMessage("Succeed to do cdf process")
        elif self.enable_phys_data_collecting:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_phys_data_collecting = False
                self.updateMessage("Succeed to collect physical simulation data")
        elif self.enable_mcts:
            self.process_bar.setValue(int(process * 100))
            if process == 1.0:
                self.enable_mcts = False
                self.updateMessage("Succeed to do MCTS")
            else:
                with open(os.path.join(self.string_file_path, "current.json"), 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dictToStorage(input_json, self.THREADING_METHOD)
                self.updateState("Current strings", self.DESIGN_FINISH)

        else:
            self.process_bar.setValue(int(process * 100))

    def editKl(self):
        if self.expert_mode:
            self.updateMessage("Please close the expert mode first...")
            return
        if self.edit_sequence_mode:
            self.updateMessage("Please close the edit sequence mode first...")
            return
        find_kl = False
        for i in range(len(self.storage)):
            if type(self.storage[i][1]) == KinematicLine:
                find_kl = True
                break
        if find_kl:
            self.chooseKl(i)
            self.chooseKlLine(0)
            self.widget_edit_kl.setVisible(True)
            self.edit_kl_mode = True
        else:
            self.edit_kl_mode = False
            self.widget_edit_kl.setVisible(False)
            self.updateMessage("No Kinematic Line are found.", "WARNING")

    def editSequence(self):
        if self.expert_mode:
            self.updateMessage("Please close the expert mode first...")
            return
        if self.edit_kl_mode:
            self.updateMessage("Please close the edit kl mode first...")
            return
        self.chooseCreaseSequence(0)
        self.edit_sequence_mode = True
        self.widget_edit_sequence.setVisible(True)
       
    def endAddString(self):
        self.actionAdd_TSA_A_point.setEnabled(False)
        self.updateMessage("Add-string mode disabled...")
        if len(self.a_string) >= 2:
            self.string_total_information.append(self.a_string)

    def expertModeEnable(self):
        if self.edit_kl_mode:
            self.updateMessage("Please close the edit kl mode first...")
            return
        if self.edit_sequence_mode:
            self.updateMessage("Please close the edit sequence mode first...")
            return
        if self.add_hole_mode:
            self.chooseUnit(0)
            self.chooseCrease(0)
            self.widget.setVisible(True)
            self.expert_mode = True
        else:
            self.expert_mode = False
            self.widget.setVisible(False)
            self.updateMessage("Please enable add-hole mode first.", "WARNING")

    def exportAsDxf(self, file_path):
        """
        @ function: export as dxf
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230312
        @ spec: None
        """
        if file_path == False:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Name Dxf file and specify a path", 
                ".", 
                "dxf files (*.dxf)"
            ) 
        if file_path == '':
            self.updateMessage("Cancel exporting as dxf")
        else:
            try:
                self.dxf_writer.setFileName(file_path)
                self.dxf_writer.ExportAsDxf(self.lines)
                self.updateMessage("Succeed to export as dxf at " + file_path)
            except:
                self.updateMessage("Failed to export dxf file, please check the permission..." + file_path)
    
    def exportAsSplitDxf(self):
        """
        @ function: export as dxf with boards and creases
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230416
        @ spec: None
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Name Dxf file and specify a path", 
            ".", 
            "dxf files (*.dxf)"
        ) 
        if file_path == '':
            self.updateMessage("Cancel exporting as dxf")
        else:
            try:
                self.dxf_writer.setFileName(file_path.split('.')[0] + '_board.dxf')
                self.dxf_writer.ExportAsDxf(self.additional_lines)
                self.dxf_writer.setFileName(file_path.split('.')[0] + '_crease.dxf')
                self.dxf_writer.ExportAsDxf(self.crease_lines)
                self.updateMessage("Succeed to export 2 files as dxf at " + file_path)
            except:
                self.updateMessage("Failed to export dxf file, please check the permission..." + file_path)

    def exportAllAsStl(self):
        """
        @ function: export all stl file
        @ version: 0.1
        @ developer: py
        @ progress: waiting
        @ date: 20230312
        @ spec: None
        """
        for ele in self.hole_kps:
            if not ele[2]:
                self.updateMessage("Failed to call stl output, check if any holes are invalid(in gray color)...")
                return
        if self.enable_output_stl:
            self.updateMessage("A stl file is being outputed, please wait...")
            return
        if self.enable_cdf_curve_fitting:
            self.updateMessage("A cdf process is running, please wait...")
            return
        if self.enable_phys_data_collecting:
            self.updateMessage("A physical simulation is running, please wait...")
            return
        if self.enable_mcts:
            self.updateMessage("A MCTS Searching process is running, please wait...")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Name Stl file and specify a path", 
            ".", 
            "stl files (*.stl)"
        ) 
        if file_path == '':
            self.updateMessage("Cancel exporting as stl")
        else:
            stl_dialog = StlSettingDialog(parent=self)
            stl_dialog.spinBox_unit_id.setEnabled(False)
            stl_dialog.setBiasAndLock(self.bias_val)
            if stl_dialog.exec_():
                pass
            if not stl_dialog.getOK():
                self.updateMessage("Cancel exporting as stl")
                return
            # STL settings 
            # -----START----- #
            height = stl_dialog.getHeight()
            bias = stl_dialog.getBias()
            method = stl_dialog.getMethod()
            if method == "symmetry" and self.pref_pack['enable_db_bind']:
                method = 'binding'

            self.enable_output_stl = True
            self.updateMessage("Stl file generating...")
            self.drawProcess(0.0)
            self.repaint()
            
            self.stl_writer.clear()
            self.stl_writer.clearCrease()

            self.stl_writer.setPrintAccuracy(self.pref_pack['print_accuracy'])
            self.stl_writer.setAsym(self.pref_pack["asym"])
            self.stl_writer.setOnlyTwoSides(self.pref_pack["only_two_sides"])

            if len(self.strings):
                self.stl_writer.string_width = self.strings[-1].width
                self.stl_writer.string_list = deepcopy(self.strings)

            self.stl_writer.setHeight(height)
            self.stl_writer.setBias(bias)
            self.stl_writer.min_bias = self.pref_pack['middle_bias']
            self.stl_writer.setMethod(method)
            self.stl_writer.setThinMode(self.pref_pack['thin_mode'])
            self.stl_writer.setDbEnable(self.pref_pack['enable_db'])
            self.stl_writer.setPillarDisable(self.pref_pack['disable_pillars'])
            self.stl_writer.border_nobias = not self.pref_pack['stl_asymmetry']
            connection_enabled = stl_dialog.getConnectionNeeded()
            board_enabled = stl_dialog.getBoardNeeded()

            self.stl_writer.setHoleWidth(stl_dialog.getHoleWidth())
            self.stl_writer.setHoleLength(stl_dialog.getHoleLength())

            self.stl_writer.setLayerOfCrease(self.pref_pack['layer'])
            board_height = stl_dialog.getBoardHeight()
            self.stl_writer.setBoardHeight(board_height)

            # set hole
            self.stl_writer.setUnitHoles(deepcopy(self.hole_kps))
            self.stl_writer.setUnitHoleSize(self.hole_size)
            self.stl_writer.setUnitHoleResolution(self.hole_resolution)

            self.stl_writer.disableUsingModifiedUnit()

            # set connection hole
            if self.enable_connection:
                self.stl_writer.setConnectionHoleSize(self.connection_radius)
                self.stl_writer.setConnectionHoles(self.backup_connection_hole_kps)
            
            # set crease stiffness
            self.stl_writer.setHardCrease(self.hard_crease_index)

            self.stl_writer.setUnitBias(self.unit_bias_list)

            stl_dialog.destroy()
            # -----END----- #

            # Global settings
            # -----START----- #
            if method == "symmetry":
                # Connection of crease
                connection_enabled = True
            # connection_angle = self.pref_pack["connection_angle"] * math.pi / 180.0
            # Copy designer
            designer = deepcopy(self.designer)
            # -----END----- #

            # for all storage
            # try:
            for i in range(self.origami_number):
                src = self.storage[i][1]
                # ----- Set the transition start point for designer ----- #
                # ----- START ----- #
                if(self.storage[i][2]):
                    designer.setTransitionStartPoint([
                        self.storage[i][0][0], 
                        self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                    ])
                else:
                    designer.setTransitionStartPoint(self.storage[i][0])
                # ----- END ----- #

                # ----- Set the source for designer ----- #
                # ----- START ----- #
                designer.setSource(src)
                # ----- END ----- #

                # If LeanMiura is using global data
                # ----- START ----- #
                if type(src) == ModuleLeanMiura:
                    if src.enable_global_modify:
                        src.unit_width = self.unit_width
                        src.copy_time = self.copy_time
                        src.entry_flag = self.entry_flag if self.copy_time % 2 else not self.entry_flag
                        src.connection_flag = self.enable_connection
                        src.con_left_length = self.con_left_length
                        src.con_right_length = self.con_right_length
                        src.connection_hole_size = self.connection_radius
                # ----- END ----- #

                # Clear additional line for designer
                # ----- START ----- #
                designer.clearAdditionalLine()
                # ----- END ----- #

                # Set the data of designer to make new origami
                # ----- START ----- #
                designer.setPaperInfo(self.paper_info)
                designer.setDesignInfo(self.design_info)
                designer.setEntryFlag(self.entry_flag)
                # ----- END ----- #

                # <<<<< Before design <<<<< #

                # -------- Design -------- #
                # ----- START ----- #
                if method != "symmetry": # 1 designer
                    designer.parseData()
                    # ----- END ----- #
                    # -------- Design -------- #

                    # >>>>> After design >>>>> #
                    # get design data
                    data = designer.getDesignData()

                    # Get unit
                    # -----START----- #
                    # LeanMiura Module, only one result at data[0]
                    if type(src) == ModuleLeanMiura: 
                        _ = data[0].getKeyPoint()
                        creases = data[0].getLine()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        u = data[0].getUnits(connection=True)
                        for ele in u:
                            self.stl_writer.addPackedOrigamiUnit(ele)

                    # Miura Module
                    elif type(src) == KinematicLine:
                        data_length = len(data)
                        row_data_length = int(data_length / self.copy_time)
                        for j in range(self.copy_time):
                            for k in range(0, row_data_length):
                                unit_id = k + j * row_data_length
                                creases = data[unit_id].getLine()
                                if connection_enabled:
                                    self.stl_writer.addValidCreases(creases)
                                if k >= 1:
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])
      
                                    self.stl_writer.addPackedOrigamiUnit(u1)
                                    self.stl_writer.addPackedOrigamiUnit(u2)
                    
                    elif type(src) == DxfDirectGrabber:
                        creases = data[0].getLine()
                        u = data[0].getUnits()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        if self.add_hole_mode:
                            for ele in u:
                                self.stl_writer.addPackedOrigamiUnit(ele)

                else: # 2 designer
                    other_designer = deepcopy(designer)
                    # first designer add left
                    if self.enable_connection:
                        if type(src) == KinematicLine:
                            designer.insertLineSource([self.con_left_length, designer.src.lines[0][1]], 0)
                            # second designer add right and set tsp
                            if(self.storage[i][2]):
                                other_designer.setTransitionStartPoint([
                                    self.storage[i][0][0] + self.con_left_length, 
                                    self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                                ])
                            else:
                                other_designer.setTransitionStartPoint([
                                    self.storage[i][0][0] + self.con_left_length,
                                    self.storage[i][0][1]
                                ])
                            other_designer.insertLineSource([self.con_right_length, designer.src.lines[-1][1]], designer.getKLNumber())
                        elif type(src) == ModuleLeanMiura:
                            if(self.storage[i][2]):
                                transition_initial_point = [
                                    self.storage[i][0][0] + self.con_left_length, 
                                    self.storage[i][0][1] + self.copy_time * self.paper_info['unit_width']
                                ]
                            else:
                                transition_initial_point = [
                                    self.storage[i][0][0] + self.con_left_length, 
                                    self.storage[i][0][1]
                                ]
                            other_designer.setTransitionStartPoint(transition_initial_point)
                            modified_lean_miura = deepcopy(src)
                            modified_lean_miura.tsp = transition_initial_point
                            modified_lean_miura.connection_flag = False
                            other_designer.setSource(modified_lean_miura)
                    
                    # -------- Design -------- #
                    # ----- START ----- #
                    # parse data at the same time
                    designer.parseData()
                    other_designer.parseData()
                    # ----- END ----- #
                    # -------- Design -------- #

                    # get the first designer data
                    data = designer.getDesignData()

                    # LeanMiura Module
                    if type(src) == ModuleLeanMiura: # 1 data
                        _ = data[0].getKeyPoint()
                        creases = data[0].getLine()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        u = data[0].getUnits(connection=True)
                        for ele in u:
                            self.stl_writer.addPackedOrigamiUnit(ele)
                    
                    # Miura module
                    elif type(src) == KinematicLine:    
                        data_length = len(data)
                        row_data_length = int(data_length / self.copy_time)
                        for j in range(self.copy_time):
                            for k in range(0, row_data_length):
                                unit_id = k + j * row_data_length
                                creases = data[unit_id].getLine()
                                if connection_enabled:
                                    self.stl_writer.addValidCreases(creases)
                                if k >= 1:
                                    # add to normal unit
                                    u1, u2 = getUnitWithinMiura(data[unit_id - 1], data[unit_id])

                                    self.stl_writer.addPackedOrigamiUnit(u1)
                                    self.stl_writer.addPackedOrigamiUnit(u2)
                    
                    elif type(src) == DxfDirectGrabber:
                        creases = data[0].getLine()
                        u = data[0].getUnits()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        if self.add_hole_mode:
                            for ele in u:
                                self.stl_writer.addPackedOrigamiUnit(ele)
                    
                    # get the second designer data
                    second_data = other_designer.getDesignData()

                    # LeanMiura Module
                    if type(src) == ModuleLeanMiura: # 1 data
                        _ = second_data[0].getKeyPoint()
                        creases = second_data[0].getLine()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        u = second_data[0].getUnits(connection=True)
                        for ele in u:
                            self.stl_writer.addPackedOrigamiModifiedUnit(ele)

                    # Miura module
                    elif type(src) == KinematicLine: 
                        data_length = len(second_data)
                        row_data_length = int(data_length / self.copy_time)
                        for j in range(self.copy_time):
                            for k in range(0, row_data_length):
                                unit_id = k + j * row_data_length
                                creases = second_data[unit_id].getLine()
                                if connection_enabled:
                                    self.stl_writer.addValidCreases(creases)
                                if k >= 1:
                                    # add to normal unit
                                    u1, u2 = getUnitWithinMiura(second_data[unit_id - 1], second_data[unit_id])
   
                                    self.stl_writer.addPackedOrigamiModifiedUnit(u1)
                                    self.stl_writer.addPackedOrigamiModifiedUnit(u2)
                    
                    elif type(src) == DxfDirectGrabber:
                        creases = second_data[0].getLine()
                        u = second_data[0].getUnits()
                        if connection_enabled:
                            self.stl_writer.addValidCreases(creases)
                        if self.add_hole_mode:
                            for ele in u:
                                self.stl_writer.addPackedOrigamiModifiedUnit(ele)

                    # can generate different up-down unit when symmetry
                    self.stl_writer.enableUsingModifiedUnit()

            if self.pref_pack['debug_mode']:
                #symmetry method
                if method == "symmetry":
                    if self.stl_writer.db_enable:
                        self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.pref_pack["print_accuracy"])
                        soft_file_path = file_path.split('.')[0] + '_S.stl'

                        self.stl_writer.calculateTriPlaneForAllUnit(inner=True)

                        self.stl_writer.outputAllStl(soft_file_path)
                        hard_file_path = file_path.split('.')[0] + '_H.stl'

                        self.stl_writer.calculateTriPlaneForAllUnit(inner=False)

                        self.stl_writer.outputAllStl(hard_file_path)
                    else:
                        self.stl_writer.setBoardHeight(self.pref_pack["print_accuracy"])

                        self.stl_writer.calculateTriPlaneForAllUnit()

                        self.stl_writer.outputAllStl(file_path)

                    if self.stl_writer.db_enable:
                        self.stl_writer.setHeight(2 * self.pref_pack["print_accuracy"])
                    else:
                        self.stl_writer.setHeight(self.pref_pack["print_accuracy"])
                    self.stl_writer.setBias(self.pref_pack["board_bias"])
                    self.stl_writer.setHoleWidth(0.001)
                    self.stl_writer.setHoleLength(0.001)
                    self.stl_writer.getAdditionalLineForAllUnit()
                    
                    crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'

                    self.stl_writer.calculateTriPlaneForAllCrease()

                    self.stl_writer.outputAllCreaseStl(crease_file_path)

                    board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                    self.stl_writer.generateBoard()
                    self.stl_writer.outputBoardStl(board_file_path)

                elif method == "binding":            
                    # set difference
                    self.stl_writer.enable_difference = self.pref_pack["additional_line_option"]

                    self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.stl_writer.print_accuracy)

                    hard_file_path = file_path.split('.')[0] + '.stl'

                    self.stl_writer.calculateTriPlaneForAllUnit(inner=False)

                    self.stl_writer.outputAllStl(hard_file_path)

                    # set difference
                    self.stl_writer.enable_difference = 0
                    self.stl_writer.setBias(self.pref_pack["board_bias"])
                    self.stl_writer.setHoleWidth(1e-5)
                    self.stl_writer.setHoleLength(1e-5)
                    self.stl_writer.getAdditionalLineForAllUnit()
                    
                    crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'

                    self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)

                    board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                    self.stl_writer.generateBoard()
                    self.stl_writer.outputBoardStl(board_file_path)

                    if (len(self.stl_writer.string_list)):
                        self.stl_writer.calculateTriPlaneForString()
                        string_file_path = file_path.split('.')[0] + '_string.stl'
                        self.stl_writer.outputStringStl(string_file_path)
                
                self.drawProcess(1.0)
                self.enable_output_stl = False
                return
            
            else:
                self.stl_output_thread = StlOutputThread(
                    self.stl_writer, 
                    self.show_process,
                    method,
                    connection_enabled,
                    board_enabled,
                    bias,
                    file_path,
                    pref_pack=self.pref_pack
                )
                self.stl_output_thread._emit.connect(self.drawProcess)
                self.stl_output_thread.start()
                self.output_stl_file_path = file_path
                if method == "upper_bias" or method == "both_bias":
                    self.output_stl_crease_flag = ""
                    if connection_enabled:
                        self.output_stl_crease_flag = "(+ *_crease.stl)"
                    self.output_stl_board_flag = ""
                    if board_enabled:
                        self.output_stl_crease_flag = "(+ *_board.stl)"
                elif method == "symmetry":
                    self.output_stl_crease_flag = "(+ *_midlayer_C.stl)"
                    self.output_stl_board_flag = "(+ *_midlayer_B.stl)"
                return

    def exportAsStl(self):
        return

    def exportDescriptionData(self, file_path):
        if file_path == False:
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Save the description data", 
                ".", 
                "json files (*.json)"
            ) 
        if file_path == '':
            self.updateMessage("Cancel exporting design result")
        else:
            # the units is compatible with the left-hand coodination
            s = {
                "kps": self.kps,
                "lines": [line.points for line in self.lines],
                "units": [unit.getSeqPoint() for unit in self.units],
                "line_features": [{
                    "type": line.getType(),
                    "level": line.level,
                    "coeff": line.coeff,
                    "recover_level": line.recover_level,
                    "hard": line.hard,
                    "hard_angle": line.hard_angle
                } for line in self.lines],
                "strings": {
                    "type": [[self.string_total_information[i][j].point_type for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "id": [[self.string_total_information[i][j].id for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "reverse": [[self.string_total_information[i][j].dir for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))]
                },
                "P_candidators": {
                    "points": self.P_candidate,
                    "connections": self.P_candidate_connection_index 
                }
            }

            with open(file_path, 'w', encoding="utf-8") as f:
                json.dump(s, f, indent=4)
            self.updateMessage("Succeed to save the description data at " + file_path)

    def generatePolygonByCenter(self, center, radius):
        """
        @ function: generate polygon for drawing
        @ version: 0.1
        @ developer: py
        @ progress: on road
        @ date: 20230228
        @ spec: None
        """
        points = []
        step = math.pi * 2 / self.hole_resolution
        for i in range(0, self.hole_resolution):
            points.append(
                QPoint(
                    int(center[0] + math.cos(i * step) * radius * self.current_pixel_scale), 
                    int(center[1] + math.sin(i * step) * radius * self.current_pixel_scale)
                )
            )
        polygon = QPolygon(points)
        return polygon
    
    def initialize(self):
        self.kps = []                           # keypoints
        self.lines = []                         # key lines
        self.units = []                         # origami units, type: [Unit...]
        self.additional_lines = []              # additional line for warning of the add-hole operation
        self.hole_kps = []                      # hole keypoints
        self.connection_hole_kps = []           # connection hole keypoints 
        self.crease_lines = []                  # crease line for output crease dxf file
        self.backup_connection_hole_kps = []    # back up the connection keypoint for exporting stl
        self.strings = []                       # TSA strings
        self.add_bias_flag = []
        self.a_string = []
        self.string_total_information = []
        self.P_candidate = []
        self.P_candidate_connection_index = []

        # Pack of add hole mode
        self.checkBox_add_hole_mode.setChecked(False)
        self.checkBox_add_string_mode.setChecked(False)
        self.actionAdd_TSA_A_point.setEnabled(False)

        # Pack of connection
        self.checkBox_connection.setChecked(False)

        self.enable_connection = False
        self.add_hole_mode = False
        self.add_string_mode = False
        self.expert_mode = False

    def importDxf(self, not_call_openfile_dialog=False, file_name=''):
        if not not_call_openfile_dialog:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a dxf file", 
                ".", 
                "Dxf files (*.dxf);;All Files (*.*)"
            )
        else:
            path = file_name
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            try:
                dxf = DxfDirectGrabber()
                dxf.readFile(path)
                self.storage.append([
                    [0, 0], dxf, False
                ])
                self.rotation.append(0.0)
                self.add_bias_flag.append(False)
                # self.hole_kps.clear()
                self.file_path = path
                self.updateState("Succeeded to load dxf file", self.IMPORT_SUCCESS)
                self.updateMessage("Previewing, click the \"design\" button to continue...")
                self.enable_design = True
            except:
                self.updateState("Failed to load dxf file, check if it is in correct format", self.state, "ERROR")

    def importKL(self, not_call_openfile_dialog=False, file_name=''):
        """
        @ function: Import kinematic lines for design
        @ version: 0.1
        @ developer: py
        @ progress: on road
        @ date: 20230305
        @ spec: add choose dialog
        """
        if not not_call_openfile_dialog:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a json file with kl specification", 
                ".", 
                "Json files (*.json);;All Files (*.*)"
            )
        else:
            path = file_name
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dictToStorage(input_json, self.KL_JSON)
                # self.hole_kps.clear()
                self.file_path = path
                self.enable_design = True
                self.updateState("Succeeded to load json file", self.IMPORT_SUCCESS)
                self.updateMessage("Previewing, click the \"design\" button to continue...")
            except:
                self.updateState("Failed to load json file, check if it is in correct format", self.state, "ERROR")
        
    def importStringPath(self, not_call_openfile_dialog=False, file_name=''):
        if not not_call_openfile_dialog:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "Choose a string path", 
                ".", 
                "Json files (*.json);;All Files (*.*)"
            )
        else:
            path = file_name
        if path == '':
            self.updateState("Cancel loading file", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dictToStorage(input_json, self.THREADING_METHOD)
                self.updateState("Current strings", self.DESIGN_FINISH)
            except:
                self.updateState("Failed to load json file, check if it is in correct format", self.state, "ERROR")

    def keyPressEvent(self, event):
        """
        @ function: exactly move the pixmap
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20230227
        @ spec: None
        """
        self.key = ''
        if self.state == self.DESIGN_FINISH:
            if event.key() == Qt.Key_Up:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Up"
                    self.pixel_bias[Y] -= 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][Y] -= 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][Y] -= 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Down:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Down"
                    self.pixel_bias[Y] += 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][Y] += 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][Y] += 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Left:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Left"
                    self.pixel_bias[X] -= 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][X] -= 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][X] -= 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Right:
                # self.enable_design = True
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+Right"
                    self.pixel_bias[X] += 1
                else:
                    if self.choose_hole_flag:
                        self.hole_kps[self.choose_hole_index][0][X] += 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current hole axis: " + str(self.hole_kps[self.choose_hole_index][0]))
                    elif self.choose_origami_flag:
                        self.storage[self.choose_origami_index][0][X] += 0.05 * (self.operation_amp + 1)
                        self.updateMessage("Current origami tsp: " + str(self.storage[self.choose_origami_index][0]))
                        self.enable_design = True
            elif event.key() == Qt.Key_Z:
                if event.modifiers() & Qt.ControlModifier: 
                    if len(self.hole_kps) > 0:
                        self.hole_kps.pop()
                        self.updateMessage("Delete the latest point...")
            elif event.key() == Qt.Key_Q:
                if self.choose_origami_flag: 
                    self.rotation[self.choose_origami_index] -= 1.0 / 180.0 * math.pi
                    self.enable_design = True
            elif event.key() == Qt.Key_E:
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+E"
                    self.exportAllAsStl()
                else:
                    if self.choose_origami_flag: 
                        self.rotation[self.choose_origami_index] += 1.0 / 180.0 * math.pi
                        self.enable_design = True
            elif event.key() == Qt.Key_S:
                if event.modifiers() & Qt.ShiftModifier:
                    self.key = "Shift+S"
                    self.saveResult()
            elif event.key() == Qt.Key_D:
                # self.enable_design = True
                if len(self.strings):
                    if self.strings[-1].type == PASS:
                        self.strings.pop()
                        self.string_type = self.strings[-1].type
                        self.strings.pop()
                    else:
                        self.string_type = self.strings[-1].type
                        self.strings.pop()
                            
                    if self.add_string_mode:
                        if len(self.a_string) > 2:
                            self.a_string.pop()
                            self.string_start_point = self.a_string[-1].point.tolist()
                        else:
                            self.exist_string_start = False
                            self.string_start_point = []
                            self.a_string = []
                            self.string_type = BOTTOM
                    else:
                        if len(self.string_total_information[-1]) > 2:
                            self.string_total_information[-1].pop()
                        else:
                            self.string_total_information.pop()
                    self.updateMessage("Delete the latest string...")
            elif event.key() == Qt.Key_Escape:
                # self.enable_design = True
                self.choose_hole_flag = False
                self.choose_origami_flag = False
                self.expert_mode = False
                self.edit_kl_mode = False
                self.edit_sequence_mode = False
                self.choose_unit_id = 0
                self.choose_crease_id = 0
                self.choose_kl_id = 0
                self.choose_line_id = 0
                self.choose_crease_sequence_id = 0
                self.choose_string_id = -1
                self.widget.setVisible(False)
                self.widget_edit_kl.setVisible(False)
                self.widget_edit_sequence.setVisible(False)
                self.updateMessage("Back to normal view...")
            elif event.key() == Qt.Key_Delete:
                # self.enable_design = True
                if self.choose_hole_flag:
                    del(self.hole_kps[self.choose_hole_index])
                    self.choose_hole_flag = False
                    self.updateMessage("Delete hole id: " + str(self.choose_hole_id) + ", back to normal view...")

    def mapFromPixelToReal(self, pixel_x, pixel_y):
        real_x = (pixel_x - self.pixel_bias[X]) / self.current_pixel_scale
        real_y = (pixel_y - self.pixel_bias[Y]) / self.current_pixel_scale
        return real_x, real_y

    def mapFromRealToPixel(self, real_x, real_y):
        pixel_x = real_x * self.current_pixel_scale + self.pixel_bias[X]
        pixel_y = real_y * self.current_pixel_scale + self.pixel_bias[Y]
        return pixel_x, pixel_y

    def mouseDoubleClickEvent(self, event):
        x = event.x()
        y = event.y()
        if event.button() == Qt.LeftButton:
            self.cursor_x = x - self.draw_panel_x_bias
            self.cursor_y = y - self.draw_panel_y_bias
            self.real_x, self.real_y = self.mapFromPixelToReal(
                self.cursor_x,
                self.cursor_y
            )
            for i in range(len(self.hole_kps)):
                if distance([self.real_x, self.real_y], self.hole_kps[i][0]) < self.hole_size:
                    self.choose_hole_flag = True
                    self.choose_origami_flag = False
                    self.choose_hole_id = self.hole_kps[i][1]
                    self.choose_hole_index = i
                    self.updateMessage("Choose hole index: " + str(i))
                    return

            for i in range(len(self.storage)):
                if distance([self.real_x, self.real_y], self.storage[i][0]) < self.pref_pack["line_weight"]:
                    if isinstance(self.storage[i][1], ModuleLeanMiura):
                        self.storage[i][1].raiseDialog(self)
                        self.storage[i][0] = self.storage[i][1].tsp
                        self.enable_design = True
                        return
                    if isinstance(self.storage[i][1], KinematicLine):
                        self.choose_origami_flag = True
                        self.choose_hole_flag = False
                        self.choose_origami_index = i
                        self.updateMessage("Choose origami index: " + str(i))
                        self.enable_design = True
                        return
                    if isinstance(self.storage[i][1], DxfDirectGrabber):
                        self.choose_origami_flag = True
                        self.choose_hole_flag = False
                        self.choose_origami_index = i
                        self.updateMessage("Choose origami index: " + str(i))
                        self.enable_design = True
                        return
                    
            valid_crease_list = self.additional_line_maker.valid_crease_list
            for i in range(len(valid_crease_list)):
                if pointOnCrease([self.real_x, self.real_y], valid_crease_list[i], 2):
                    if i not in self.hard_crease_index:
                        self.hard_crease_index.append(i)
                        self.enable_design = True
                    else:
                        del(self.hard_crease_index[self.hard_crease_index.index(i)])
                        self.enable_design = True
                    break
                
            for i in range(len(self.strings)):
                if self.strings[i].type != PASS:
                    if pointOnCrease([self.real_x, self.real_y], Crease(self.strings[i].start_point, self.strings[i].end_point, BORDER), 2):
                        self.choose_string_id = self.strings[i].id
                        break

    def mouseMoveEvent(self, event) -> None:
        """
        @ function: move the pixmap axis
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230107
        @ spec: None
        """
        if self.enable_moving:
            x = event.x()
            y = event.y()
            self.pixel_bias[0] = self.old_pixel_bias_x + x - self.initial_x
            self.pixel_bias[1] = self.old_pixel_bias_y + y - self.initial_y
            self.cursor_x = event.x() - self.draw_panel_x_bias
            self.cursor_y = event.y() - self.draw_panel_y_bias
            self.real_x, self.real_y = self.mapFromPixelToReal(
                self.cursor_x,
                self.cursor_y
            )

    def mousePressEvent(self, event) -> None:
        """
        @ function: move the pixmap axis
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230107
        @ spec: None
        """
        if self.state == self.DESIGN_FINISH:
            x = event.x()
            y = event.y()
            self.cursor_x = x - self.draw_panel_x_bias
            self.cursor_y = y - self.draw_panel_y_bias
            self.real_x, self.real_y = self.mapFromPixelToReal(
                self.cursor_x,
                self.cursor_y
            )
            if event.button() == Qt.RightButton:
                if self.add_hole_mode and not self.add_string_mode:
                    self.addHoleToUnit(self.cursor_x, self.cursor_y)
                elif self.add_string_mode:
                    unit_id = self.pointInUnit([self.real_x, self.real_y])
                    if unit_id != None:
                        self.addStringPoint(self.real_x, self.real_y, unit_id, 'B', True)
                    else:
                        self.updateMessage("Invalid threading method. The threading point is not in some unit.")
                else:
                    if (self.cursor_x > 0 and self.cursor_x < self.pixmap_length and self.cursor_y > 0 and self.cursor_y < self.pixmap_width):
                        self.initial_x = x
                        self.initial_y = y
                        self.old_pixel_bias_x = self.pixel_bias[0]
                        self.old_pixel_bias_y = self.pixel_bias[1]
                        self.enable_moving = True

            else:
                if self.add_string_mode:
                    unit_id = self.pointInUnit([self.real_x, self.real_y])
                    if unit_id != None:
                        self.addStringPoint(self.real_x, self.real_y, unit_id, 'B', False)
                    else:
                        self.updateMessage("Invalid threading method. The threading point is not in some unit.")
                else:
                    if (self.cursor_x > 0 and self.cursor_x < self.pixmap_length and self.cursor_y > 0 and self.cursor_y < self.pixmap_width):
                        self.initial_x = x
                        self.initial_y = y
                        self.old_pixel_bias_x = self.pixel_bias[0]
                        self.old_pixel_bias_y = self.pixel_bias[1]
                        self.enable_moving = True

    def mouseReleaseEvent(self, event) -> None:
        self.enable_moving = False

    def newFile(self):
        if len(self.storage) == 0:
            self.designer = OrigamiDesigner(src=[], paper_info=self.paper_info, design_info=self.design_info)
            self.updateState("Succeeded to create new file", self.IMPORT_SUCCESS)
            self.updateMessage("Please add some origami...")
            self.enable_design = True
            self.initialize()
        else:
            reply = QMessageBox.question(self, "New file option", "Do you want to replace current design?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.designer = OrigamiDesigner(src=[], paper_info=self.paper_info, design_info=self.design_info)
                self.storage.clear()
                self.updateState("Succeeded to create new file", self.IMPORT_SUCCESS)
                self.updateMessage("Please add some origami...")
                self.enable_design = True
                self.initialize()

    def onAddHoleMode(self):
        """
        @ function: add hole to units
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230227
        @ spec: None
        """
        self.add_hole_mode = self.checkBox_add_hole_mode.isChecked()
        if self.add_hole_mode:
            self.add_hole_mode = False
            self.bias_val, ok = QInputDialog.getDouble(self, "Bias setting", "Please set the bias", 0.01, 0.01, self.bias_max, decimals=2)
            if ok:
                self.add_hole_mode = True
                self.show_additional_crease = True
                self.updateMessage("Add-hole mode enabled...")
            else:
                self.add_hole_mode = False
                self.checkBox_add_hole_mode.setChecked(False)
        else:
            if self.expert_mode:
                self.expert_mode = False
                self.choose_unit_id = 0
                self.choose_crease_id = 0
                self.widget.setVisible(False)
            self.updateMessage("Add-hole mode disabled...")
        self.enable_design = True
    
    def onAddConnection(self):
        self.enable_connection = self.checkBox_connection.isChecked()
        self.enable_design = True

    def onAddStringMode(self):
        self.add_string_mode = self.checkBox_add_string_mode.isChecked()
        if self.add_string_mode:
            self.startAddString()
        else:
            self.endAddString()
        self.enable_design = True

    def onDesign(self):
        self.design()
        self.updateState("Design finished with success", self.DESIGN_FINISH)
        self.updateMessage("Design mode enabled...")

    def onDesignThreadingMethod(self):
        pass
        # if self.enable_output_stl:
        #     self.updateMessage("A stl file is being outputed, please wait...")
        #     return
        # if self.enable_cdf_curve_fitting:
        #     self.updateMessage("A cdf process is running, please wait...")
        #     return
        # if self.enable_phys_data_collecting:
        #     self.updateMessage("A physical simulation is running, please wait...")
        #     return
        # if self.enable_mcts:
        #     self.updateMessage("A MCTS Searching process is running, please wait...")
        #     return
        
        # file_path, _ = QFileDialog.getSaveFileName(
        #     self, 
        #     "Specify the filename of the simulation result"
        # ) 

        # if file_path == '':
        #     self.updateState("Cancel outputing mcts results.", self.state)
        #     return
        # else:
        #     if not os.path.exists(file_path):
        #         os.makedirs(file_path)
        # self.string_file_path = file_path

        # self.enable_mcts = True

        # batch_size = 4
        
        # self.exportAsDxf("./dxfResult/phys_sim.dxf")

        # max_edge = 0
        # for ele in self.units:
        #     edge = len(ele.crease)
        #     if edge > max_edge:
        #         max_edge = edge

        # # 获取点和线段信息
        # dxfg = DxfDirectGrabber()
        # dxfg.readFile("./dxfResult/phys_sim.dxf")

        # # 获取折纸单元信息
        # unit_parser = UnitPackParserReverse(
        #                 tsp=[0.0, 0.0],
        #                 kps=dxfg.kps,
        #                 lines=dxfg.lines,
        #                 lines_type=dxfg.lines_type
        #             )

        # unit_parser.setMaximumNumberOfEdgeInAllUnit(max_edge) #For every units, there exists at most 4 edges
        # input_units = unit_parser.getUnits()

        # # calculate max length of view
        # max_size, max_x, max_y = unit_parser.getMaxDistance()
        # total_bias = unit_parser.getTotalBias(units=input_units)

        # # new_lines = []
        # # for i in range(len(self.lines)):
        # #     line = self.lines[i]
        # #     if distance(line[START], line[END]) < 1e-5:
        # #         continue
        # #     not_duplicate = True
        # #     for j in range(i):
        # #         other_line = self.lines[j]
        # #         if (distance(line[START], other_line[START]) < 1e-5 and distance(line[END], other_line[END]) < 1e-5) or (distance(line[START], other_line[END]) < 1e-5 and distance(line[END], other_line[START]) < 1e-5):
        # #             not_duplicate = False
        # #             break
        # #     if not not_duplicate:
        # #         continue
        # #     new_lines.append(line)

        # # 构建蒙特卡洛搜索树
        # mcts = MCTS_Simplified(self.units, unit_parser.new_lines, self.kps, self.pref_pack['tsa_radius'], self.pref_pack["tsa_resolution"], origami_size = [self.origami_length, self.origami_width], string_number=2, generation=self.limitation["mcts_epoch"])
        
        # if not self.pref_pack['debug_mode']:
        #     self.mcts_thread = MCTSThread(mcts, batch_size, [self.origami_length, self.origami_width], self.pref_pack, self.limitation, self.units, max_edge, input_units, max_size, total_bias, file_path)
        #     self.mcts_thread._emit.connect(self.drawProcess)
        #     self.mcts_thread.start()
        #     return
        # else:
        #     scores = []
        #     for i in range(self.limitation["mcts_epoch"]):
        #         methods, initial_method = mcts.ask(batch_size, i)

        #         try:
        #             with open(os.path.join(self.file_path, "current.json"), 'w', encoding="utf-8") as f:
        #                 json.dump(initial_method[0], f, indent=4)
        #         except:
        #             pass

        #         print("Debug mode, using 1 process")
        #         reward_list = [0.0 for _ in range(len(methods))]

        #         ori_sim = OrigamiSimulator(use_gui=False)

        #         ori_sim.string_total_information = methodToTotalInformation(methods[0], mcts.P_points, mcts.O_points)
        #         ori_sim.pref_pack = self.pref_pack
        #         ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
        #         ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
        #         ori_sim.initializeRunning()

        #         for j in range(len(methods)):
        #             reward_list[j] = (len(methods[j]["id"][0]) + len(methods[j]["id"][1])) / 18. + (methods[j]["id"][0][0] + methods[j]["id"][1][0]) / 48.

        #         mcts.tell(reward_list)

        #         maximum_reward = max(reward_list)
        #         maximum_reward_index = reward_list.index(maximum_reward)
        #         scores.append(maximum_reward)

        #         print("Epoch: " + str(i) + ", Max Value: " + str(maximum_reward) + ", Total batch size: " + str(len(reward_list)))

        #         best_method = methods[maximum_reward_index]
        #         total_string = deepcopy(best_method)
        #         total_string["score"] = maximum_reward
                
        #         try:
        #             with open(os.path.join(file_path, "result_epoch_" + str(i) + "_score_" + str(round(maximum_reward, 2))) + ".json", 'w', encoding="utf-8") as f:
        #                 json.dump(total_string, f, indent=4)
        #         except:
        #             pass
                
        #         score_list = {
        #             "score": scores
        #         }

        #         try:
        #             with open(os.path.join(file_path, "score.json"), 'w', encoding="utf-8") as f:
        #                 json.dump(score_list, f, indent=4)
        #         except:
        #             pass
            
        #     print("END TRAINING!")
        #     self.enable_mcts = False

    def oneClickAddHoles(self):
        if self.add_hole_mode:
            for u in self.units:
                center = u.getCenter()
                self.addHoleToUnitUsingRealAxis(center[X], center[Y])
            self.updateMessage("Succeed in adding holes in the center of each unit...")
        else:
            self.updateMessage("Add-hole mode is not enabled, please enable add_hole mode...")

    def openFile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a design pack", 
            ".", 
            "Json files (*.json);;Txt files (*.txt);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel opening file", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dictToStorage(input_json, self.PACKED_DATA)
                self.updateState("Succeeded to load packed-data file", self.DESIGN_FINISH)
                self.enable_design = True
            except:
                self.updateState("Failed to open file, check if it is in correct format", self.state, "ERROR")

    def physicalDataCollecting(self):
        # --Physical Sim module for simulate the folding process of origami-- #
        if self.enable_output_stl:
            self.updateMessage("A stl file is being outputed, please wait...")
            return
        if self.enable_cdf_curve_fitting:
            self.updateMessage("A cdf process is running, please wait...")
            return
        if self.enable_phys_data_collecting:
            self.updateMessage("A physical simulation is running, please wait...")
            return
        if self.enable_mcts:
            self.updateMessage("A MCTS Searching process is running, please wait...")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Specify the filename of the outputed simulation result", 
            ".", 
            "json files (*.json)"
        ) 
        if file_path == '':
            self.updateMessage("Cancel exporting physical simulation result")
        else:
            self.enable_phys_data_collecting = True

            max_edge = 0
            for ele in self.units:
                edge = len(ele.crease)
                if edge > max_edge:
                    max_edge = edge

            if not self.full_description_mode:
                from phys_sim12 import OrigamiSimulator
                self.exportAsDxf("./dxfResult/phys_sim.dxf")

                # 获取点和线段信息
                dxfg = DxfDirectGrabber()
                dxfg.readFile("./dxfResult/phys_sim.dxf")

                # 获取折纸单元信息
                unit_parser = UnitPackParserReverse(
                                tsp=[0.0, 0.0],
                                kps=dxfg.kps,
                                lines=dxfg.lines,
                                lines_type=dxfg.lines_type
                            )

                unit_parser.setMaximumNumberOfEdgeInAllUnit(max_edge) #For every units, there exists at most 4 edges
                input_units = unit_parser.getUnits()

                # calculate max length of view
                max_size, max_x, max_y = unit_parser.getMaxDistance()
                total_bias = unit_parser.getTotalBias(units=input_units)

                ori_sim = OrigamiSimulator(use_gui=False)

                # ori_sim.string_total_information = deepcopy(self.string_total_information)
                ori_sim.pref_pack = self.pref_pack
                ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
                ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
                ori_sim.initializeRunning()
            else:
                from phys_sim17 import OrigamiSimulator
                self.exportDescriptionData('./descriptionData/phys_sim.json')
                ori_sim = OrigamiSimulator(use_gui=False, fast_simulation=self.pref_pack["fast_simulation_mode"])

                # ori_sim.string_total_information = deepcopy(self.string_total_information)
                ori_sim.start("phys_sim", max_edge, ori_sim.TSA_SIM)
                ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
                ori_sim.fast_simulation_mode = False
                ori_sim.initializeRunning()
            
            
            # ori_sim = OrigamiSimulator(use_gui=False, debug_mode=self.pref_pack["debug_mode"])
            # # set string parameters
            # # ori_sim.strings = deepcopy(self.strings)
            # ori_sim.string_total_information = deepcopy(self.string_total_information)

            # ori_sim.start("phys_sim", max_edge, ori_sim.TSA_SIM)
            # ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
            # ori_sim.initializeRunning()
            
            if not self.pref_pack["debug_mode"]:
                self.phys_data_collecting_thread = ThreadingMethodSearchingThread(file_path, ori_sim)
                self.phys_data_collecting_thread._emit.connect(self.drawProcess)
                self.phys_data_collecting_thread.start()
            else:
                step = 1
                while 1:
                    ori_sim.step()
                    
                    if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 500 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 200 and ori_sim.can_rotate):
                        if ori_sim.sim_mode == ori_sim.TSA_SIM and ori_sim.can_rotate:
                            if ori_sim.recorded_folding_percent[-1] > FOLDING_MAXIMUM:
                                break
                            else:
                                if ori_sim.recorded_folding_percent[-1] < np.mean(np.array(ori_sim.recorded_folding_percent[-51: -1])):
                                    break
                        else:
                            break
                    step += 1

                all_dis = {
                    "control_string_decrease": [],
                    "string_decrease_each": [],
                    "max_force": [],
                    "folding_percent": [],
                    "max_folding_percent": [],
                    "min_folding_percent": [],
                    "time": []
                }

                all_dis["control_string_decrease"] = ori_sim.recorded_string_decrease_length_control
                all_dis["string_decrease_each"] = ori_sim.recorded_string_decrease_length
                all_dis["folding_percent"] = ori_sim.recorded_folding_percent
                all_dis["max_folding_percent"] = ori_sim.recorded_maximum_folding_percent
                all_dis["min_folding_percent"] = ori_sim.recorded_minimum_folding_percent
                all_dis["max_force"] = ori_sim.recorded_max_force
                all_dis["time"] = ori_sim.recorded_t

                with open(file_path, 'w', encoding="utf-8") as f:
                    json.dump(all_dis, f, indent=4)
                self.drawProcess(1.0)
                self.enable_phys_data_collecting = False

    def physicalSimulation(self):
        max_edge = 0
        for ele in self.units:
            edge = len(ele.crease)
            if edge > max_edge:
                max_edge = edge
        if not self.full_description_mode:
            from phys_sim12 import OrigamiSimulator
            self.exportAsDxf("./dxfResult/phys_sim.dxf")

            ori_sim = OrigamiSimulator(use_gui=True, debug_mode=self.pref_pack['debug_mode'])
            # set string parameters
            # ori_sim.strings = deepcopy(self.strings)
            ori_sim.string_total_information = deepcopy(self.string_total_information)
            
            ori_sim.start("phys_sim", max_edge, ori_sim.FOLD_SIM)

            ori_sim.run()
        else:
            from phys_sim17 import OrigamiSimulator
            self.exportDescriptionData('./descriptionData/phys_sim.json')
            ori_sim = OrigamiSimulator(use_gui=True, debug_mode=self.pref_pack['debug_mode'], fast_simulation=self.pref_pack["fast_simulation_mode"])
            # ori_sim.string_total_information = deepcopy(self.string_total_information)
            ori_sim.start("phys_sim", max_edge, ori_sim.FOLD_SIM)
            ori_sim.run(False, False)

    def physicalSimulationExplicit(self):
        pass

    def plotJsonReadFile(self):
        path, _ = QFileDialog.getOpenFileName(
            None, 
            "Choose a score pack", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            return False, {}
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                
                return True, input_json
            except:
                pass

    def plotJson(self):
        plt.rcParams['font.sans-serif'] = 'Times new roman'
        fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 9))
        data_number = 0
        maximum_force = 0.
        while 1:
            ok, data = self.plotJsonReadFile()
            if not ok:
                break
            else:
                data_number += 1
                if data_number == 1:
                    rgb = (0.2, 0.01, 0.5)
                elif data_number == 2:
                    rgb = (0.5, 0.1, 0.6)
                elif data_number == 3:
                    rgb = (0.8, 0.3, 0.4)
                else:
                    rgb = (0.87, 0.6, 0.37)
                # 0-0 folding-percent with current_time
                ax[0][0].set_xmargin(0)
                ax[0][0].set_ymargin(0)
                ax[0][0].spines['top'].set_visible(False)
                ax[0][0].spines['right'].set_visible(False)

                x = data["time"]
                y = data["folding_percent"]
                y_upperbound = data["max_folding_percent"]
                y_lowerbound = data["min_folding_percent"]

                ax[0][0].plot(x, y, color=rgb, linewidth=2, label='Exp. ' + str(data_number))
                ax[0][0].fill_between(x, y_upperbound, y_lowerbound, alpha=0.3, facecolor=rgb, label='Bound - Exp. ' + str(data_number))

                ax[0][0].set_ylim(-1, 1)
                ax[0][0].set_xlabel("System time (s)", fontsize=13)
                ax[0][0].set_ylabel("Folding percent", fontsize=13)
                ax[0][0].tick_params(labelsize=13)

                ax[0][0].set_title("Folding percent - System time", fontsize=16)
                ax[0][0].legend()

                # 0-1 max_force with current_time
                ax[0][1].set_xmargin(0)
                ax[0][1].set_ymargin(0)
                ax[0][1].spines['top'].set_visible(False)
                ax[0][1].spines['right'].set_visible(False)

                x = data["time"]
                y = data["max_force"]

                ax[0][1].plot(x, y, color=rgb, linewidth=2, label='Exp. ' + str(data_number))

                if max(y) * 1.1 > maximum_force:
                    ax[0][1].set_ylim(0, max(y) * 1.1)
                    maximum_force = max(y) * 1.1
                ax[0][1].set_xlabel("System time (s)", fontsize=13)
                ax[0][1].set_ylabel("Maximum force (N)", fontsize=13)
                ax[0][1].tick_params(labelsize=13)

                ax[0][1].set_title("Maximum force - System time", fontsize=16)
                ax[0][1].legend()

                # 1-0 control_decrease with each_decrease
                ax[1][0].set_xmargin(0)
                ax[1][0].set_ymargin(0)
                ax[1][0].spines['top'].set_visible(False)
                ax[1][0].spines['right'].set_visible(False)

                x = data["time"]
                y_standard = data["control_string_decrease"]
                y_each = data["string_decrease_each"]

                ax[1][0].plot(x, y_standard, color=rgb, linewidth=2, label='Exp. ' + str(data_number))
                for ele in y_each:
                    ax[1][0].plot(x, ele, color=rgb, linewidth=1, linestyle='--')

                ax[1][0].set_ylim(0, y_standard[-1] * 1.1)
                ax[1][0].set_xlabel("System time (s)", fontsize=13)
                ax[1][0].set_ylabel("String length decrease (mm)", fontsize=13)
                ax[1][0].tick_params(labelsize=13)

                ax[1][0].set_title("String length decrease - System time", fontsize=16)
                ax[1][0].legend()

                # 1-1 Folding speed
                ax[1][1].set_xmargin(0)
                ax[1][1].set_ymargin(0)
                ax[1][1].spines['top'].set_visible(False)
                ax[1][1].spines['right'].set_visible(False)

                x = data["control_string_decrease"]
                y = data["folding_percent"]

                ax[1][1].plot(x, y, color=rgb, linewidth=2, label='Exp. ' + str(data_number))

                ax[1][1].set_ylim(-1, 1)
                ax[1][1].set_xlabel("String length decrease (mm)", fontsize=13)
                ax[1][1].set_ylabel("Folding percent", fontsize=13)
                ax[1][1].tick_params(labelsize=13)

                ax[1][1].set_title("Folding percent - String length decrease", fontsize=16)
                ax[1][1].legend()
            
            if data_number == 4:
                break

        fig.subplots_adjust(left=0.08, right=0.93, top=0.93, bottom=0.08, wspace=0.29, hspace=0.29)
        if data_number == 0:
            return

        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.ylim(0, 1)
        # plt.grid(axis='x')
        # plt.legend(handles=[l1, l2], labels=["Folding percent", "Max force (Normalized)"], loc="upper left", fontsize=16)
        # plt.ylim((0, 60))
        plt.show()

    def plotEvolutionJsonReadFile(self):
        path, _ = QFileDialog.getOpenFileName(
            None, 
            "Choose a score pack", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            return None, None, False
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                score_list = input_json["score"]
                list_length = len(score_list)
                x_list = [x for x in range(list_length)]
                return np.array(x_list), np.array(score_list), True
            except:
                pass

    def plotEvolutionJson(self):
        plt.rcParams['font.sans-serif'] = 'Times new roman'
        plt.figure()
        plt.title("Max Fitness Value - Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Fitness Value")
        length_list = []
        while 1:
            x, y, goon = self.plotEvolutionJsonReadFile()
            if not goon:
                break
            else:
                # plt.plot(x, 1.0 - y)
                plt.plot(x, y)
                length_list.append(len(x))
        if len(length_list) == 0:
            return
        max_length = max(length_list)
        step_number = int(max_length / 60) + 2
        labels = [str(i * 60) for i in range(step_number)]
        plt.xticks(range(0, step_number * 60, 60), labels=labels)
        plt.grid(axis='x')
        plt.ylim((0, 1))
        plt.show()

    def pointInUnit(self, point):
        length = len(self.units)
        for unit_id in range(length):
            unit = self.units[unit_id]
            kps = unit.getSeqPoint()
            if pointInPolygon(point, kps):
                return unit_id
        return None
    
    def printOrigami(self):
        """
        @ function: print origami
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220313
        @ spec: None
        """
        printer = QPrinter()
        print_dialog = QPrintDialog(printer, self)
        if (QDialog.Accepted == print_dialog.exec_()):
            if self.show_square == 'A4':
                self.drawA4Pixmap()
                painter = QPainter(printer)
                rect = painter.viewport()
                size = self.A4_pixmap.size()
                size.scale(rect.size(), Qt.KeepAspectRatio)
                painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
                painter.setWindow(self.A4_pixmap.rect())
                painter.drawImage(0, 0, self.A4_pixmap.toImage())
                painter.end()
                self.updateMessage("Succeed to print origami on A4-H...")
            elif self.show_square == None:
                painter = QPainter(printer)
                rect = painter.viewport()
                size = self.pixmap.size()
                size.scale(rect.size(), Qt.KeepAspectRatio)
                painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
                painter.setWindow(self.pixmap.rect())
                painter.drawImage(0, 0, self.pixmap.toImage())
                painter.end()
                self.updateMessage("Succeed to print origami for all screen...")   

    def resetView(self):
        self.pixel_bias = [25, 25]
        self.current_pixel_scale = 1.0
        self.pixel_scale_ranking = 14
        self.A4_length = 296
        self.A4_width = 210
        self.A4_half_length = 148
        self.A4_half_width = 105

    def saveResult(self):
        """
        @ function: Save packed result
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220313
        @ spec: None
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save the design result", 
            ".", 
            "json files (*.json)"
        ) 
        if file_path == '':
            self.updateMessage("Cancel exporting design result")
        else:
            s = {
                "origami": [],
                "unit_bias_list": self.unit_bias_list,
                "util": {
                    "hole_axis": [ele for ele in self.hole_kps]
                },
                "setting": {
                    "unit_width": self.unit_width,
                    "copy_time": self.copy_time,
                    "entry_flag": self.entry_flag,
                    "add_hole_mode": self.add_hole_mode,
                    "hole_size": self.hole_size,
                    "hole_resolution": self.hole_resolution,
                    "enable_connection": self.enable_connection,
                    "con_left_length": self.con_left_length,
                    "con_right_length": self.con_right_length,
                    "con_radius": self.connection_radius,
                    "bias_val": self.bias_val
                },
                "strings": {
                    "type": [[self.string_total_information[i][j].point_type for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "id": [[self.string_total_information[i][j].id for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))],
                    "reverse": [[self.string_total_information[i][j].dir for j in range(len(self.string_total_information[i]))] for i in range(len(self.string_total_information))]
                },
                "P_candidators": {
                    "points": self.P_candidate,
                    "connections": self.P_candidate_connection_index 
                }
            }
            for i in range(self.origami_number):
                if type(self.storage[i][1]) == KinematicLine:
                    ori_type = "kl"
                elif type(self.storage[i][1]) == ModuleLeanMiura:
                    ori_type = "leanMiura"
                elif type(self.storage[i][1]) == DxfDirectGrabber:
                    ori_type = "dxf"
                dic = {
                    "type": ori_type,
                    "tsp": self.storage[i][0],
                    "data": self.storage[i][1].getData(),
                    "add_width": self.storage[i][2]
                }
                s["origami"].append(dic)
                
            with open(file_path, 'w', encoding="utf-8") as f:
                json.dump(s, f, indent=4)
            self.updateMessage("Succeed to save result at " + file_path)

    def setBiasAsDefault(self):
        self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] = None
        self.doubleSpinBox_expert_mode.setVisible(False)

    def setBiasAsExpertModified(self):
        self.doubleSpinBox_expert_mode.setMinimum(self.pref_pack["print_accuracy"])
        self.doubleSpinBox_expert_mode.setMaximum(self.unit_width / 6.0)
        if self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] == None:
            self.doubleSpinBox_expert_mode.setValue(self.pref_pack["print_accuracy"])
            self.unit_bias_list[self.choose_unit_id][self.choose_crease_id] = self.pref_pack["print_accuracy"]
        else:
            self.doubleSpinBox_expert_mode.setValue(self.unit_bias_list[self.choose_unit_id][self.choose_crease_id])
        self.doubleSpinBox_expert_mode.setVisible(True)

    def setGlobalParameter(self, unit_width=None, copy_time=None, entry_flag=None, add_hole_mode=None, 
                           hole_size=None, hole_resolution=None, enable_connection=None, 
                           con_left_length=None, con_right_length=None, con_radius=None, bias_val=None):
        """
        @ function: Set global parameter of app
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220403
        @ spec: None
        """
        if unit_width != None:
            self.unit_width = unit_width
            self.paper_info['unit_width'] = unit_width
            self.spinbox_crease_width.setValue(unit_width)
        if copy_time != None:
            self.copy_time = copy_time
            self.design_info['copy_time'] = self.copy_time
            self.spinbox_copy_time.setValue(copy_time)
        if entry_flag != None:
            self.entry_flag = entry_flag
            self.slider_flag.setValue(entry_flag)
        if add_hole_mode != None:
            self.add_hole_mode = add_hole_mode
            if add_hole_mode:
                self.checkBox_add_hole_mode.setChecked(True)
            else:
                self.checkBox_add_hole_mode.setChecked(False)
        if hole_size != None:
            self.hole_size = hole_size
            self.spinbox_hole_size.setValue(hole_size)
        if hole_resolution != None:
            self.hole_resolution = hole_resolution
            self.spinbox_resolution.setValue(hole_resolution)
        if enable_connection != None:
            self.enable_connection = enable_connection
            if enable_connection:
                self.checkBox_connection.setChecked(True)
            else:
                self.checkBox_connection.setChecked(False)
        if con_left_length != None:
            self.con_left_length = con_left_length
            self.spinbox_con_left_length.setValue(con_left_length)
        if con_right_length != None:
            self.con_right_length = con_right_length
            self.spinbox_con_right_length.setValue(con_right_length)
        if con_radius != None:
            self.connection_radius = con_radius
            self.spinbox_connection_radius.setValue(con_radius)
        if bias_val != None:
            self.bias_val = bias_val

    def setting(self):
        """
        @ function: Setting for application
        @ version: 0.1
        @ developer: py
        @ progress: finish
        @ date: 20220313
        @ spec: None
        """
        self.pref_pack_window.readFile()
        self.pref_pack_window.show()
        if self.pref_pack_window.exec_():
            pass
        if self.pref_pack_window.ok:
            self.pref_pack = self.pref_pack_window.getPrefPack()
            self.limitation = self.pref_pack_window.getLimitation()
            self.updateMessage("Succeed to save settings...")
        else:
            self.updateMessage("Cancel saving settings...")

    def setupTimer(self):
        self.timer = QTimer(self)
        self.timer.start(33)
        self.timer.timeout.connect(self.updateWindow)

    def showA4Square(self):
        self.show_square = 'A4'

    def showCurve(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a curve json file", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel importing curve", self.state)
        else:
            try:
                self.curve_name = os.path.basename(path)
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.x_list = input_json["x"]
                self.y_list = input_json["y"]
                self.tm_window = TmWindow()
                self.tm_window.importXPoints(self.x_list)
                self.tm_window.importYPoints(self.y_list)
                self.tm_window.plot()
                self.tm_window.show()
            except:
                pass

    def showDirection(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Choose a curve json file", 
            ".", 
            "Json files (*.json);;All Files (*.*)"
        )
        if path == '':
            self.updateState("Cancel importing direction curve", self.state)
        else:
            try:
                with open(path, 'r', encoding='utf-8') as fw:
                    input_json = json.load(fw)
                self.dir_list = input_json["dir"]
                self.tm_window = TmWindow()
                self.tm_window.importXPoints(self.x_list)
                self.tm_window.importYPoints(self.y_list)
                self.tm_window.plot()
                self.tm_window.show()
            except:
                pass

    def showIndex(self):
        if self.show_index:
            self.show_index = False
            self.actionShow_Index.setText("Show Index")
        else:
            self.show_index = True
            self.actionShow_Index.setText("Hide Index")


    def showNone(self):
        self.show_square = None
        
    def showTG(self):
        items = []
        real_index_mapper = []
        # List all Miura combo
        for i in range(self.origami_number):
            type_src = type(self.storage[i][1])
            if type_src == KinematicLine:
                items.append("Miura Combo[" + str(len(items)) + "] - storage[" + str(i) + "]")
                real_index_mapper.append(i)
        selected_item, ok = QInputDialog.getItem(self, "Select Miura Item", "Select a Miura combo:", items)
        # If press ok
        if ok:
            index = real_index_mapper[items.index(selected_item)]
            designer = deepcopy(self.designer)
            designer.setSource(self.storage[index][1])
            data_list = designer.getGeometryData()
            self.tm_window = TmWindow()
            self.tm_window.importXPoints(self.x_list)
            self.tm_window.importYPoints(self.y_list)
            self.tm_window.getTm().setSource(data_list)
            self.tm_window.getTm().enable_hypar_connection = self.limitation["hypar_enable"]
            self.tm_window.startShow()

    def startAddString(self):
        self.actionAdd_TSA_A_point.setEnabled(True)
        self.exist_string_start = False
        self.string_start_point = []
        self.a_string = []
        self.string_type = BOTTOM
        self.updateMessage("Add-string mode enabled...")

    def stopThread(self):
        exist_thread = False
        if self.enable_cdf_curve_fitting and self.cdf_curve_fitting_thread:
            self.cdf_curve_fitting_thread.terminate()
            self.cdf_curve_fitting_thread = None
            self.enable_cdf_curve_fitting = False
            exist_thread = True
        if self.enable_output_stl and self.stl_output_thread:
            self.stl_output_thread.terminate()
            self.stl_output_thread = None
            self.enable_output_stl = False
            exist_thread = True
        if self.enable_phys_data_collecting and self.phys_data_collecting_thread:
            self.phys_data_collecting_thread.terminate()
            self.phys_data_collecting_thread = None
            self.enable_phys_data_collecting = False
            exist_thread = True
        if self.enable_mcts and self.mcts_thread:
            self.mcts_thread.terminate()
            self.mcts_thread = None
            self.enable_mcts = False
            exist_thread = True

        if exist_thread:
            self.drawProcess(1.0)
            self.updateMessage("Stop all threads running in application...")
        else:
            self.updateMessage("No threads are running in application...")

    def toPixel(self, x: list):
        return [round(x[X] * self.current_pixel_scale + self.pixel_bias[X]), round(x[Y] * self.current_pixel_scale + self.pixel_bias[Y])]
    
    def updateMessage(self, message, message_type="MESSAGE"):
        self.label_message.setText(message_type + "::" + message)
        if not self.enable_output_stl and not self.enable_cdf_curve_fitting and not self.enable_phys_data_collecting and not self.enable_mcts:
            self.drawProcess(1.0)

    def updateState(self, state, type_state, message_type="INFO"):
        """
        @ function: Update state of application
        @ version: 0.1111
        @ developer: py
        @ progress: finish
        @ date: 20220302
        @ spec: add print origami, add adding holes
        """
        self.state = type_state
        if type_state == self.INITIAL_STATE:
            # No input, No design

            # Pack of add hole mode
            self.checkBox_add_hole_mode.setEnabled(False)
            self.checkBox_add_string_mode.setEnabled(False)
            self.spinbox_hole_size.setEnabled(False)
            self.spinbox_resolution.setEnabled(False)

            # Pack of main params
            self.spinbox_crease_width.setEnabled(False)
            self.spinbox_copy_time.setEnabled(False)
            self.slider_flag.setEnabled(False)
            self.horizontal_folding_slider.setEnabled(False)

            # Pack of design button
            self.button_design.setEnabled(False)
            self.button_threading_design.setEnabled(False)

            # Pack of actions
            self.actionSave_result.setEnabled(False)
            self.actionPrint_P.setEnabled(False)
            self.menuExport_E.setEnabled(False)
            self.actionLeanMiura.setEnabled(False)
            self.actionTransition_T.setEnabled(False)
            self.actionPhysical_Simulation_P.setEnabled(False)
            self.actionCollect_Physical_Data_C.setEnabled(False)
            self.actionAdd_Holes.setEnabled(False)
            self.actionAdd_TSA_A_point.setEnabled(False)
            self.actionExplicit_Simulation_E.setEnabled(False)
            self.actionExpert_Mode_E.setEnabled(False)
            self.actionEdit_kl_E.setEnabled(False)
            self.actionCalculate_Sequence.setEnabled(False)
            self.actionEdit_Sequence_S.setEnabled(False)
            self.actionImport_string_path.setEnabled(False)
            self.actionAdd_TSA_A_candidators.setEnabled(False)
            self.actionDelete_TSA_A_Candidators.setEnabled(False)

            # Pack of connection
            self.checkBox_connection.setEnabled(False)
            self.spinbox_connection_radius.setEnabled(False)
            self.spinbox_con_left_length.setEnabled(False)
            self.spinbox_con_right_length.setEnabled(False)
            

            # Modify the state label
            self.label_state.setText(message_type + "::" + state)
        elif type_state == self.IMPORT_SUCCESS:
            # User can design when in this state

            # Pack of add hole mode
            self.checkBox_add_hole_mode.setEnabled(False)
            self.checkBox_add_string_mode.setEnabled(False)
            self.spinbox_hole_size.setEnabled(False)
            self.spinbox_resolution.setEnabled(False)

            # Pack of main params
            self.spinbox_crease_width.setEnabled(True)
            self.spinbox_copy_time.setEnabled(True)
            self.slider_flag.setEnabled(True)
            self.horizontal_folding_slider.setEnabled(True)

            # Pack of design button
            self.button_design.setEnabled(True)
            self.button_threading_design.setEnabled(True)

            # Pack of actions
            self.actionSave_result.setEnabled(False)
            self.actionPrint_P.setEnabled(False)
            self.menuExport_E.setEnabled(False)
            self.actionLeanMiura.setEnabled(False)
            self.actionTransition_T.setEnabled(False)
            self.actionPhysical_Simulation_P.setEnabled(False)
            self.actionCollect_Physical_Data_C.setEnabled(False)
            self.actionAdd_Holes.setEnabled(False)
            self.actionExplicit_Simulation_E.setEnabled(False)
            self.actionExpert_Mode_E.setEnabled(False)
            self.actionEdit_kl_E.setEnabled(False)
            self.actionCalculate_Sequence.setEnabled(False)
            self.actionEdit_Sequence_S.setEnabled(False)
            self.actionImport_string_path.setEnabled(False)
            self.actionAdd_TSA_A_candidators.setEnabled(False)
            self.actionDelete_TSA_A_Candidators.setEnabled(False)

            # Pack of connection
            self.checkBox_connection.setEnabled(False)
            self.spinbox_connection_radius.setEnabled(False)
            self.spinbox_con_left_length.setEnabled(False)
            self.spinbox_con_right_length.setEnabled(False)

            # Modify the state label
            self.label_state.setText(message_type + "::" + state)
        elif type_state == self.DESIGN_FINISH:
            # User can design and add holes when in this state
            
            # Pack of add hole mode
            self.checkBox_add_hole_mode.setEnabled(True)
            self.checkBox_add_string_mode.setEnabled(True)
            self.spinbox_hole_size.setEnabled(True)
            self.spinbox_resolution.setEnabled(True)

            # Pack of main params
            self.spinbox_crease_width.setEnabled(True)
            self.spinbox_copy_time.setEnabled(True)
            self.slider_flag.setEnabled(True)
            self.horizontal_folding_slider.setEnabled(True)

            # Pack of design button
            self.button_design.setEnabled(True)
            self.button_threading_design.setEnabled(True)

            # Pack of actions
            self.actionSave_result.setEnabled(True)
            self.actionPrint_P.setEnabled(True)
            self.menuExport_E.setEnabled(True)
            self.actionLeanMiura.setEnabled(True)
            self.actionTransition_T.setEnabled(True)
            self.actionPhysical_Simulation_P.setEnabled(True)
            self.actionCollect_Physical_Data_C.setEnabled(True)
            self.actionAdd_Holes.setEnabled(True)
            self.actionExplicit_Simulation_E.setEnabled(True)
            self.actionExpert_Mode_E.setEnabled(True)
            self.actionEdit_kl_E.setEnabled(True)
            self.actionCalculate_Sequence.setEnabled(True)
            self.actionEdit_Sequence_S.setEnabled(True)
            self.actionImport_string_path.setEnabled(True)
            self.actionAdd_TSA_A_candidators.setEnabled(True)
            if len(self.P_candidate):
                self.actionDelete_TSA_A_Candidators.setEnabled(True)
            else:
                self.actionDelete_TSA_A_Candidators.setEnabled(False)

            # Pack of connection
            self.checkBox_connection.setEnabled(True)
            self.spinbox_connection_radius.setEnabled(True)
            self.spinbox_con_left_length.setEnabled(True)
            self.spinbox_con_right_length.setEnabled(True)

            # Modify the state label
            self.label_state.setText(message_type + "::" + state)
        elif type_state == self.DESIGN_ERROR:
            self.label_state.setText(message_type + "::" + state)
        elif type_state == self.OUTPUT_FINISH:
            self.label_state.setText(message_type + "::" + state)
        else:
            self.label_state.setText("ERROR::Unknown state for application")

    def updateWindow(self):
        """
        @ function: Update window in 30Hz
        @ version: 0.111
        @ developer: py
        @ progress: on road
        @ date: 20230107
        @ spec: add zoom and axis bias show /0.111
        """
        try:
            self.label_zoom.setText("Zoom scale: " + str(self.current_pixel_scale))
            self.label_axis_bias.setText("(" + str(self.pixel_bias[0]) + ", " + str(self.pixel_bias[1]) + ")")
            self.bias_max = self.unit_width / 6.0
            self.label_bias_val.setText(str(np.round(self.bias_val, 2)))
            self.checkHoleKpIsValid()
            if self.enable_moving:
                if self.pref_pack["cursor_axis_mode"] == "pixel_axis":
                    self.label_cursor_axis.setText("(" + str(self.cursor_x) + ", " + str(self.cursor_y) + ")")
                else:
                    self.label_cursor_axis.setText("(" + str(np.round(self.real_x, 1)) + ", " + str(np.round(self.real_y, 1)) + ")")
        except:
            pass
        if self.state:
            if self.enable_design:
                self.design()
                self.enable_design = False
            else:
                # paint result
                self.drawCreasePattern()

    def wheelEvent(self, event) -> None:
        """
        @ function: Zoom in/out
        @ version: 0.1
        @ developer: py
        @ progress: on road
        @ date: 20230107
        @ spec: None
        """
        if (self.state == self.DESIGN_FINISH):
            focus = self.draw_panel.mapFromGlobal(QCursor.pos())
            x = focus.x()
            y = focus.y()
            if x > 0 and x < self.pixmap_length and y > 0 and y < self.pixmap_width:
                if event.angleDelta().y() > 0:
                    if self.pixel_scale_ranking > 0:
                        self.pixel_scale_ranking -= 1
                        old_scale = self.current_pixel_scale
                        self.current_pixel_scale = self.pixel_scale[self.pixel_scale_ranking]
                        self.pixel_bias[0] = int(x - (x - self.pixel_bias[0]) / old_scale * self.current_pixel_scale)
                        self.pixel_bias[1] = int(y - (y - self.pixel_bias[1]) / old_scale * self.current_pixel_scale)
                        self.A4_length = self.A4_length / old_scale * self.current_pixel_scale
                        self.A4_width = self.A4_width / old_scale * self.current_pixel_scale
                        self.A4_half_length = self.A4_half_length / old_scale * self.current_pixel_scale
                        self.A4_half_width = self.A4_half_width / old_scale * self.current_pixel_scale
                else:
                    if self.pixel_scale_ranking < self.pixel_scale_min_ranking - 1:
                        self.pixel_scale_ranking += 1
                        old_scale = self.current_pixel_scale
                        self.current_pixel_scale = self.pixel_scale[self.pixel_scale_ranking]
                        self.pixel_bias[0] = int(x - (x - self.pixel_bias[0]) / old_scale * self.current_pixel_scale)
                        self.pixel_bias[1] = int(y - (y - self.pixel_bias[1]) / old_scale * self.current_pixel_scale)
                        self.A4_length = self.A4_length / old_scale * self.current_pixel_scale
                        self.A4_width = self.A4_width / old_scale * self.current_pixel_scale
                        self.A4_half_length = self.A4_half_length / old_scale * self.current_pixel_scale
                        self.A4_half_width = self.A4_half_width / old_scale * self.current_pixel_scale

# def workerMultisim(mlist, method, pointer, pref_pack, max_edge, input_units, max_size, total_bias):
#     string_total_information = method

#     ori_sim = OrigamiSimulator(use_gui=False)

#     ori_sim.string_total_information = string_total_information
#     ori_sim.pref_pack = pref_pack

#     ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
#     ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
#     ori_sim.initializeRunning()
    
#     # while ori_sim.dead_count < 500:
#     #     ori_sim.step()
#     #     if ori_sim.folding_angle_reach_pi[0] or math.isnan(ori_sim.total_energy[0]) or (ori_sim.current_t > 4.0 and not ori_sim.can_rotate) or (ori_sim.current_t > 20.0 and ori_sim.can_rotate):
#     #         break
#     while 1:
#         ori_sim.step()
#         if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 500 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 200 and ori_sim.can_rotate):
#             break
        
#     if not ori_sim.can_rotate:
#         print("Batch: " + str(pointer) + ", Value: Error Actuation")
#         mlist[pointer] = 0.0
#     else:
#         folding_percent = ori_sim.recorded_folding_percent[-1]
#         folding_speed = (ori_sim.recorded_folding_percent[-1] - ori_sim.recorded_folding_percent[0]) / (ori_sim.recorded_t[-1] - ori_sim.recorded_t[0])
#         value = folding_speed * folding_percent

#         print("Batch: " + str(pointer) + ", Value: " + str(value))
        
#         mlist[pointer] = value

# A thread for cdf-curve-fitting
class CdfCurveFittingThread(QThread):
    _emit = pyqtSignal(float)

    def __init__(self, pref_pack, curve_name: str, curve_x, curve_y, curve_dir, algorithm='es') -> None:
        super().__init__()
        self.pref_pack = pref_pack
        self.curve_name = curve_name.split('.')[0]
        self.x = curve_x
        self.y = curve_y
        self.dir = curve_dir
        self.algorithm = algorithm
        self.storage_number = pref_pack["storage"]
        self.batch_size = pref_pack["batch_size"]
        self.process_num = pref_pack["process_number"]
        self.match_mode = pref_pack["match_mode"]
        self.discrete_resolution = pref_pack["discrete_resolution"]
        self.hypar_enable = pref_pack["hypar_enable"]
        self.direction_enable = pref_pack["direction_enable"]

        self.exo_angle1 = pref_pack["exo_angle1"]
        self.exo_angle2 = pref_pack["exo_angle2"]
        self.exo_X = pref_pack["exo_X"]
        self.exo_Y = pref_pack["exo_Y"]
        self.exo_theta = pref_pack["exo_theta"]

        self.BEST_MATCH = 0
        self.STRICT_MATCH = 1
        self.DISCRETE_MATCH = 2
        self.EXO_MATCH = 3
        self.ZERO_MATCH = 4
    
    def run(self):
        step_min = self.pref_pack["row_number"][0]
        step_max = self.pref_pack["row_number"][1]

        process_num = self.process_num

        cfh = CurveFittingHelper()
        if self.x != None:
            cfh.setGoalList([[self.x[i], self.y[i]] for i in range(len(self.x))])
        
        else:
            if self.match_mode == self.EXO_MATCH:
                cfh.setExoGoal(self.exo_X, self.exo_Y, self.exo_theta * math.pi / 180.0)

        if self.direction_enable:
            cfh.setDirectionGoalList(self.dir)

        tm = TransitionModel()
        tm.enable_hypar_connection = self.pref_pack["hypar_enable"]
        
        total_step = (step_max - step_min + 1) * self.pref_pack["generation"] * 10
        update_scale_step = self.pref_pack["generation"] / 20
        output_best_flag  = self.pref_pack["generation"] / 20
        change_mode_flag  = self.pref_pack["generation"] / 2

        s = {
            "origin": [
                [0.0, 0.0]
            ],
            "kl": [
            
            ],
            "add_width":[
                False
            ],
            "score": 0.0
        }

        score_dist = {
            "score": []
        }
        
        for i in range(int(step_min), int(step_max + 1)):
            for time in range(10):
                score_list = []
                current_best_sub_scores = []
                score = 0.0
                early_stop = 0

                if self.algorithm == 'es':
                    algo = ES(self.storage_number)
                    algo.mode("cmaes")
                    algo.initialize(self.batch_size * process_num, 2 * i)
                elif self.algorithm == 'ga':
                    algo = GA(self.storage_number)
                    algo.initialize(self.batch_size * process_num, 2 * i)
                mapper = []
                for j in range(i):
                    mapper.append([self.pref_pack["unit_length"][0], self.pref_pack["unit_length"][1]])
                    mapper.append([
                        [-self.pref_pack["miura_angle"][1] * math.pi / 180.0, -self.pref_pack["miura_angle"][0] * math.pi / 180.0],
                        [self.pref_pack["miura_angle"][0] * math.pi / 180.0, self.pref_pack["miura_angle"][1] * math.pi / 180.0]
                    ])
                algo.setMapper(mapper)
                
                while(algo.iteration < self.pref_pack["generation"]):
                    reward_list = []

                    self._emit.emit((((i - step_min) * 10 + time) * self.pref_pack["generation"] + algo.iteration) / total_step)
                    data = algo.ask()

                    if process_num == 1:
                        for pack in data:
                            k_data = pack.data
 
                            tm.setSource([
                                [k_data[2 * k], abs(k_data[2 * k + 1]), 0 if k_data[2 * k + 1] < 0 else 1] for k in range(i)
                            ])

                            if (self.match_mode == self.BEST_MATCH or self.match_mode == self.STRICT_MATCH):
                                all_ef, all_ef_dir = tm.getAllEndEffector()
                                cfh.setOriginList(all_ef)
                                if self.match_mode == self.BEST_MATCH:
                                    p = cfh.bestMatch() 
                                else:
                                    p = cfh.strictMatch() * cfh.distanceMatch()
                                if self.direction_enable:
                                    cfh.setDirectionOriginList(all_ef_dir)
                                    p *= cfh.directionMatch(len(all_ef_dir))
                                reward_list.append(p)

                            elif (self.match_mode == self.EXO_MATCH):
                                all_ef = []
                                all_ef_dir = []
                                tm.main_folding_angle = self.exo_angle1 * math.pi / 180.0
                                tl = tm.getTransitionLines()
                                all_ef.append(deepcopy(tm.end_ef))
                                all_ef_dir.append(deepcopy(tm.end_ef_dir))
                                intersect1 = calculateIntersection(tl)

                                tm.main_folding_angle = self.exo_angle2 * math.pi / 180.0
                                tl = tm.getTransitionLines()
                                all_ef.append(deepcopy(tm.end_ef))
                                all_ef_dir.append(deepcopy(tm.end_ef_dir))
                                intersect2 = calculateIntersection(tl)

                                cfh.setOriginList(all_ef)
                                cfh.setDirectionOriginList(all_ef_dir)

                                p = cfh.exoMatch(intersect1, intersect2)
                                reward_list.append(p)

                            elif (self.match_mode == self.ZERO_MATCH):
                                part_ef, _ = tm.getPartEndEffector(45)
                                cfh.setOriginList(part_ef)
                                cfh.setIntersectionTime(tm.self_intersection_number)
                                p = cfh.zeroMatch()
                                reward_list.append(p)

                            else:
                                part_ef, _ = tm.getPartEndEffector(45)
                                cfh.setOriginList(part_ef)
                                cfh.setIntersectionTime(tm.self_intersection_number)
                                p, sub_scores = cfh.partMatch(self.discrete_resolution)
                                reward_list.append(p)
                                if p > score:
                                    current_best_sub_scores = sub_scores
   
                        algo.evaluate(reward_list)
                    
                    new_score = algo.getCurrentBest()
                    
                    if new_score > score:
                        score = new_score
                        early_stop = 0
                    else:
                        early_stop += 1

                    score_list.append(score)

                    if early_stop >= 100:
                        print("Early stop at ITER: " + str(algo.iteration))
                        design_result = deepcopy(algo.storage[-1])
                        kl = []
                        data_result = []
                        alpha = 0.0
                        for k in range(i):
                            data = design_result.data[2*k: 2*k+2] #length, angle and up/down
                            length = data[0] * (self.pref_pack["unit_length"][1] - self.pref_pack["unit_length"][0]) + self.pref_pack["unit_length"][0]
                            if data[1] < 0.5:
                                angle = (2 * data[1] * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) - self.pref_pack["miura_angle"][1]) * math.pi / 180.0
                            else:
                                angle = ((2 * data[1] - 1) * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) + self.pref_pack["miura_angle"][0]) * math.pi / 180.0
                            alpha += angle * 2
                            kl.append([length, alpha])
                            data_result.append([length, abs(angle), 0 if angle < 0 else 1])
                        s["kl"] = [kl]
                        s["score"] = design_result.score
                        score_dist["score"] = deepcopy(score_list)
                        
                        folder_result = './cdfResult/' + self.curve_name + '/result'
                        result_filename = 'row' + str(i) + '-' + str(time) + '-' + str(algo.iteration) + '.json'
                        if not os.path.exists(folder_result):
                            os.makedirs(folder_result)
                        with open(os.path.join(folder_result, result_filename), 'w', encoding="utf-8") as f:
                            json.dump(s, f, indent=4)

                        data_json = {
                            "data": data_result
                        }
                        folder_score = './cdfResult/' + self.curve_name + '/score'
                        score_filename = 'score' + str(i) + '-' + str(time) + '-' + str(algo.iteration) + '.json'
                        if not os.path.exists(folder_score):
                            os.makedirs(folder_score)
                        with open(os.path.join(folder_score, score_filename), 'w', encoding="utf-8") as f:
                            json.dump(score_dist, f, indent=4)

                        folder_data = './cdfResult/' + self.curve_name + '/data'
                        data_filename = 'data' + str(i) + '-' + str(time) + '-' + str(algo.iteration) + '.json'
                        if not os.path.exists(folder_data):
                            os.makedirs(folder_data)
                        with open(os.path.join(folder_data, data_filename), 'w', encoding="utf-8") as f:
                            json.dump(data_json, f, indent=4)
                        break

                    print(str(cfh.intersection_time) + " ES Counter: " + str(early_stop) + " (" + algo.sample_mode + ")" + "ITER: " + str(algo.iteration) + "    Current Best Score: " + str(np.round(score, 6)) + ' ' + str(current_best_sub_scores))
                    if self.algorithm == "es":
                        if algo.iteration % update_scale_step == 0:
                            algo.updateScale(0.25 * (1 - algo.iteration / self.pref_pack["generation"]) ** 2)

                        if algo.iteration % output_best_flag == 0:
                            design_result = deepcopy(algo.storage[-1])
                            kl = []
                            data_result = []
                            alpha = 0.0
                            for k in range(i):
                                data = design_result.data[2*k: 2*k+2] #length, angle and up/down
                                length = data[0] * (self.pref_pack["unit_length"][1] - self.pref_pack["unit_length"][0]) + self.pref_pack["unit_length"][0]
                                if data[1] < 0.5:
                                    angle = (2 * data[1] * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) - self.pref_pack["miura_angle"][1]) * math.pi / 180.0
                                else:
                                    angle = ((2 * data[1] - 1) * (self.pref_pack["miura_angle"][1] - self.pref_pack["miura_angle"][0]) + self.pref_pack["miura_angle"][0]) * math.pi / 180.0
                                alpha += angle * 2
                                kl.append([length, alpha])
                                data_result.append([length, abs(angle), 0 if angle < 0 else 1])
                            s["kl"] = [kl]
                            s["score"] = design_result.score
                            score_dist["score"] = deepcopy(score_list)

                            folder_result = './cdfResult/' + self.curve_name + '/result'
                            result_filename = 'row' + str(i) + '-' + str(time) + '-' + str(algo.iteration) + '.json'
                            if not os.path.exists(folder_result):
                                os.makedirs(folder_result)
                            with open(os.path.join(folder_result, result_filename), 'w', encoding="utf-8") as f:
                                json.dump(s, f, indent=4)

                            data_json = {
                                "data": data_result
                            }
                            folder_score = './cdfResult/' + self.curve_name + '/score'
                            score_filename = 'score' + str(i) + '-' + str(time) + '-' + str(algo.iteration) + '.json'
                            if not os.path.exists(folder_score):
                                os.makedirs(folder_score)
                            with open(os.path.join(folder_score, score_filename), 'w', encoding="utf-8") as f:
                                json.dump(score_dist, f, indent=4)
                            
                            folder_data = './cdfResult/' + self.curve_name + '/data'
                            data_filename = 'data' + str(i) + '-' + str(time) + '-' + str(algo.iteration) + '.json'
                            if not os.path.exists(folder_data):
                                os.makedirs(folder_data)
                            with open(os.path.join(folder_data, data_filename), 'w', encoding="utf-8") as f:
                                json.dump(data_json, f, indent=4)

        self._emit.emit(1.0)

# A thread for generating stl file
# The time is very long, so keep it inside the thread
class StlOutputThread(QThread):
    _emit = pyqtSignal(float)
 
    def __init__(
            self, 
            stl_writer, 
            show_process, 
            method,
            connection_enabled,
            board_enabled,
            bias,
            filepath,
            pref_pack):
        super().__init__()
        self.stl_writer = stl_writer
        self.show_process = show_process
        self.method = method
        self.connection_enabled = connection_enabled
        self.board_enabled = board_enabled
        self.bias = bias
        self.file_path = filepath
        self.pref_pack = pref_pack
 
    def run(self):
        show_process = self.show_process
        method = self.method
        connection_enabled = self.connection_enabled
        board_enabled = self.board_enabled
        bias = self.bias
        file_path = self.file_path
        pref_pack = self.pref_pack

        try:
            # normal method
            if method == "upper_bias" or method == "both_bias":
                # generate unit
                if show_process:
                    unit_size = self.stl_writer.size()
                    if connection_enabled and board_enabled:
                        divide = 2.4
                    elif connection_enabled and not board_enabled:
                        divide = 1.2
                    elif not connection_enabled and board_enabled:
                        divide = 2
                    else:
                        divide = 1.0
                    for i in range(unit_size):
                        self.stl_writer.calculateTriPlaneForUnit(i)
                        self._emit.emit(i / unit_size / divide * 2 / 3)
                else:
                    self.stl_writer.calculateTriPlaneForAllUnit()
                # output unit stl
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                    for i in range(unit_size):
                        tris = self.stl_writer.tri_list[i]
                        self.stl_writer.addInfoToStlFile(tris)
                        self._emit.emit(2 / divide / 3 + i / unit_size / divide / 3)
                    self.stl_writer.s += 'endsolid\n'
                    with open(file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                else:
                    self.stl_writer.outputAllStl()

                # generate crease
                connection_flag = ""
                if show_process:
                    crease_size = self.stl_writer.validCreaseSize()
                    if board_enabled:
                        base = 0.417
                    else:
                        base = 0.834
                if connection_enabled:
                    crease_file_path = file_path.split('.')[0] + '_crease.stl'
                    if show_process:
                        for i in range(crease_size):
                            self.stl_writer.calculateTriPlaneForCrease(i)          
                            self._emit.emit(base + i / crease_size * 0.083)
                    else:
                        self.stl_writer.calculateTriPlaneForAllCrease()
                    # output crease stl
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                        for i in range(crease_size):
                            tris = self.stl_writer.crease_tri_list[i]
                            if tris != None:
                                self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(base + 0.083 + i / crease_size * 0.083)
                        self.stl_writer.s += 'endsolid\n'
                        with open(crease_file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllCreaseStl(crease_file_path)
                    connection_flag = "(+ *_crease.stl)"

                # generate board
                if show_process:
                    if connection_flag:
                        head = 0.584
                    else:
                        head = 0.5
                if board_enabled:
                    board_file_path = file_path.split('.')[0] + '_board.stl'
                    if show_process:
                        for i in range(unit_size):
                            self.stl_writer.calculateTriPlaneForBoard(i)
                            self._emit.emit(head + i / unit_size * (1 - head))
                    else:
                        self.stl_writer.generateBoard()
                    self.stl_writer.outputBoardStl(board_file_path)

            #symmetry method
            elif method == "symmetry":
                unit_size = self.stl_writer.size()               
                # set difference
                self.stl_writer.enable_difference = self.pref_pack["additional_line_option"]

                if self.stl_writer.db_enable:
                    self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.stl_writer.print_accuracy)
                    soft_file_path = file_path.split('.')[0] + '_S.stl'
                    if show_process:
                        for unit_id in range(unit_size):
                            self.stl_writer.calculateTriPlaneForUnit(unit_id, inner=True)
                            self._emit.emit(unit_id / unit_size * 0.133)
                    else:
                        self.stl_writer.calculateTriPlaneForAllUnit(inner=True)
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                        for i in range(unit_size):
                            tris = self.stl_writer.tri_list[i]
                            self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(0.133 + i / unit_size * 0.067)
                        self.stl_writer.s += 'endsolid\n'
                        with open(soft_file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllStl(soft_file_path)
                    hard_file_path = file_path.split('.')[0] + '.stl'
                    if show_process:
                        for unit_id in range(unit_size):
                            self.stl_writer.calculateTriPlaneForUnit(unit_id, inner=False)
                            self._emit.emit(0.2 + unit_id / unit_size * 0.133)
                    else:
                        self.stl_writer.calculateTriPlaneForAllUnit(inner=False)
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                        for i in range(unit_size):
                            tris = self.stl_writer.tri_list[i]
                            self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(0.333 + i / unit_size * 0.067)
                        self.stl_writer.s += 'endsolid\n'
                        with open(hard_file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllStl(hard_file_path)
                else:
                    self.stl_writer.setBoardHeight(3.0 * pref_pack["print_accuracy"])
                    if show_process:
                        for unit_id in range(unit_size):
                            self.stl_writer.calculateTriPlaneForUnit(unit_id)
                            self._emit.emit(unit_id / unit_size * 0.266)
                    else:
                        self.stl_writer.calculateTriPlaneForAllUnit()
                    if show_process:
                        self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                        for i in range(unit_size):
                            tris = self.stl_writer.tri_list[i]
                            self.stl_writer.addInfoToStlFile(tris)
                            self._emit.emit(0.266 + i / unit_size * 0.133)
                        self.stl_writer.s += 'endsolid\n'
                        with open(file_path, 'w') as f:
                            f.write(self.stl_writer.s)
                    else:
                        self.stl_writer.outputAllStl(file_path)

                connection_flag = ""
 
                # set difference
                self.stl_writer.enable_difference = 0
                self.stl_writer.setBias(pref_pack["board_bias"])
                self.stl_writer.setHoleWidth(1e-5)
                self.stl_writer.setHoleLength(1e-5)
                self.stl_writer.getAdditionalLineForAllUnit()
                crease_size = self.stl_writer.validCreaseSize()
                
                crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                    self._emit.emit(0.5)
                else:
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                if show_process:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                    self._emit.emit(0.6)
                else:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                # if show_process:
                #     self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                #     for i in range(crease_size):
                #         self.stl_writer.calculateTriPlaneForCrease(i)
                #         self._emit.emit(0.4 + i / crease_size * 0.1)
                # else:
                #     self.stl_writer.calculateTriPlaneForAllCrease()
                # if show_process:
                #     for i in range(crease_size):
                #         tris = self.stl_writer.crease_tri_list[i]
                #         if tris != None:
                #             self.stl_writer.addInfoToStlFile(tris)
                #         self._emit.emit(0.5 + i / crease_size * 0.1)
                #     self.stl_writer.s += 'endsolid\n'
                #     with open(crease_file_path, 'w') as f:
                #         f.write(self.stl_writer.s)
                # else:
                #     self.stl_writer.outputAllCreaseStl(crease_file_path)
                connection_flag = "(+ *_midlayer_C.stl)"

                board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                if show_process:
                    for i in range(unit_size):
                        self.stl_writer.calculateTriPlaneForBoard(i)
                        self._emit.emit(0.6 + i / unit_size * 0.4)
                else:
                    self.stl_writer.generateBoard()
                self.stl_writer.outputBoardStl(board_file_path)

                if (len(self.stl_writer.string_list)):
                    self.stl_writer.calculateTriPlaneForString()
                    string_file_path = file_path.split('.')[0] + '_string.stl'
                    self.stl_writer.outputStringStl(string_file_path)
            elif method == 'binding':
                unit_size = self.stl_writer.size()               

                self.stl_writer.enable_difference = self.pref_pack["additional_line_option"]

                self.stl_writer.setBoardHeight(self.pref_pack["layer"] * self.stl_writer.print_accuracy)

                hard_file_path = file_path.split('.')[0] + '.stl'
                if show_process:
                    for unit_id in range(unit_size):
                        self.stl_writer.calculateTriPlaneForUnit(unit_id, inner=False)
                        self._emit.emit(unit_id / unit_size * 0.333)
                else:
                    self.stl_writer.calculateTriPlaneForAllUnit(inner=False)
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Units__ SLA File\n'
                    for i in range(unit_size):
                        tris = self.stl_writer.tri_list[i]
                        self.stl_writer.addInfoToStlFile(tris)
                        self._emit.emit(0.333 + i / unit_size * 0.067)
                    self.stl_writer.s += 'endsolid\n'
                    with open(hard_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                else:
                    self.stl_writer.outputAllStl(hard_file_path)

                connection_flag = ""

                # set difference
                self.stl_writer.enable_difference = 0
                self.stl_writer.setBias(pref_pack["board_bias"])
                self.stl_writer.setHoleWidth(1e-5)
                self.stl_writer.setHoleLength(1e-5)
                self.stl_writer.getAdditionalLineForAllUnit()
                crease_size = self.stl_writer.validCreaseSize()
                
                crease_file_path = file_path.split('.')[0] + '_midlayer_C.stl'
                if show_process:
                    self.stl_writer.s = 'solid PyGamic generated __All_Crease__ SLA File\n'
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                    self._emit.emit(0.5)
                else:
                    tris = self.stl_writer.calculateTriPlaneForCreaseUsingBindingMethod()
                if show_process:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                    self._emit.emit(0.6)
                else:
                    self.stl_writer.addInfoToStlFile(tris)
                    self.stl_writer.s += 'endsolid\n'
                    with open(crease_file_path, 'w') as f:
                        f.write(self.stl_writer.s)
                connection_flag = "(+ *_midlayer_C.stl)"

                board_file_path = file_path.split('.')[0] + '_midlayer_B.stl'

                if show_process:
                    for i in range(unit_size):
                        self.stl_writer.calculateTriPlaneForBoard(i)
                        self._emit.emit(0.6 + i / unit_size * 0.4)
                    self.stl_writer.calculateTriPlaneForPillar()
                else:
                    self.stl_writer.generateBoard()
                self.stl_writer.outputBoardStl(board_file_path)

                if (len(self.stl_writer.string_list)):
                    self.stl_writer.calculateTriPlaneForString()
                    string_file_path = file_path.split('.')[0] + '_string.stl'
                    self.stl_writer.outputStringStl(string_file_path)

            self._emit.emit(1.0)
        except Exception as e:
            print("No")
            self._emit.emit(1.0)

# class MCTSThread(QThread):
#     _emit = pyqtSignal(float)

#     def __init__(self, mcts, batch_size, origami_size, pref_pack, limitation, units, max_edge, input_units, max_size, total_bias, file_path) -> None:
#         super().__init__()
#         self.mcts = mcts
#         self.batch_size = batch_size
#         self.origami_size = origami_size
#         self.pref_pack = pref_pack
#         self.limitation = limitation
#         self.units = units
#         self.max_edge = max_edge
#         self.input_units = input_units
#         self.max_size = max_size
#         self.total_bias = total_bias
#         self.file_path = file_path

#     def run(self):
#         mcts = self.mcts
#         batch_size = self.batch_size
#         origami_size = self.origami_size
#         max_edge = self.max_edge
#         input_units = self.input_units
#         max_size = self.max_size
#         total_bias = self.total_bias

#         scores = []
#         for i in range(self.limitation["mcts_epoch"]):
#             methods, initial_method = mcts.ask(batch_size, i)

#             try:
#                 with open(os.path.join(self.file_path, "current.json"), 'w', encoding="utf-8") as f:
#                     json.dump(initial_method[0], f, indent=4)
#             except:
#                 pass
            
#             self._emit.emit(i / self.limitation["mcts_epoch"])

#             valid_number = len(methods)
#             if valid_number >= 1: 
#                 print("There are " + str(valid_number) + " valid cases, using multi-process technology")
#                 initial_fitness_list = []
#                 for j in range(len(methods)):
#                     initial_fitness_list.append(0.0)

#                 mlist = multiprocessing.Manager().list(initial_fitness_list)

#                 p_list = []

#                 pointer = 0

#                 while pointer < len(methods):
#                     p = multiprocessing.Process(target=workerMultisim, args=(
#                             mlist, methodToTotalInformation(methods[pointer], mcts.P_points, mcts.O_points), pointer,
#                             self.pref_pack, max_edge, input_units, max_size, total_bias
#                         )
#                     )
#                     p_list.append(p)
#                     pointer += 1
                
#                 process_id = 0

#                 total_process_number = 4

#                 current_process_number = 0

#                 while process_id < len(methods):
#                     while current_process_number < total_process_number:
#                         p_list[process_id].start()
#                         current_process_number += 1
#                         process_id += 1
#                         if current_process_number == total_process_number or process_id == len(methods):
#                             break

#                     while current_process_number > 0:
#                         p_list[process_id - current_process_number].join()
#                         current_process_number -= 1

#                 reward_list = list(mlist)

#             elif valid_number == 1:
#                 print("Only 1 valid case, using 1 process")
#                 reward_list = [0.0]

#                 ori_sim = OrigamiSimulator(use_gui=False)

#                 ori_sim.string_total_information = methodToTotalInformation(methods[0], mcts.P_points, mcts.O_points)
#                 ori_sim.pref_pack = self.pref_pack

#                 ori_sim.startOnlyTSA(input_units, max_size, total_bias, max_edge)
#                 ori_sim.enable_tsa_rotate = ori_sim.string_length_decrease_step
#                 ori_sim.initializeRunning()
                
#                 while 1:
#                     ori_sim.step()
#                     if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 500 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 200 and ori_sim.can_rotate):
#                         break
                    
#                 if not ori_sim.can_rotate:
#                     print("Epoch: " + str(i) + ", Batch: " + str(0) + ", Value: Error Actuation")
#                     reward_list[0] = 0.0
#                 else:
#                     folding_percent = ori_sim.recorded_folding_percent[-1]
#                     folding_speed = (ori_sim.recorded_folding_percent[-1] - ori_sim.recorded_folding_percent[0]) / (ori_sim.recorded_t[-1] - ori_sim.recorded_t[0])

#                     value = folding_speed * folding_percent
#                     print("Epoch: " + str(i) + ", Batch: " + str(j) + ", Value: " + str(value))
                    
#                     reward_list[0] = value

#             mcts.tell(reward_list)

#             maximum_reward = max(reward_list)
#             maximum_reward_index = reward_list.index(maximum_reward)
#             scores.append(maximum_reward)

#             print("Epoch: " + str(i) + ", Max Value: " + str(maximum_reward) + ", Total batch size: " + str(len(reward_list)))

#             best_method = methods[maximum_reward_index]
#             total_string = deepcopy(best_method)
#             total_string["score"] = maximum_reward

#             try:
#                 with open(os.path.join(self.file_path, "result_epoch_" + str(i) + "_score_" + str(round(maximum_reward, 2))) + ".json", 'w', encoding="utf-8") as f:
#                     json.dump(total_string, f, indent=4)
#             except:
#                 pass
            
#             score_list = {
#                 "score": scores
#             }

#             try:
#                 with open(os.path.join(self.file_path, "score.json"), 'w', encoding="utf-8") as f:
#                     json.dump(score_list, f, indent=4)
#             except:
#                 pass
        
#         self._emit.emit(1.0)
#         print("END TRAINING!")

class ThreadingMethodSearchingThread(QThread):
    _emit = pyqtSignal(float)
 
    def __init__(
            self, file_path, ori_simulator):
        super().__init__()
        self.file_path = file_path
        self.ori_simulator = ori_simulator

    def run(self):
        self._emit.emit(0.0)
        ori_sim = self.ori_simulator
        
        step = 1
        while 1:
            ori_sim.step()
                    
            if ori_sim.folding_angle_reach_pi[0] or (ori_sim.dead_count >= 100 and not ori_sim.can_rotate) or (ori_sim.dead_count >= 40 and ori_sim.can_rotate):
                if ori_sim.sim_mode == ori_sim.TSA_SIM and ori_sim.can_rotate:
                    if ori_sim.recorded_folding_percent[-1] > FOLDING_MAXIMUM:
                        break
                    else:
                        if ori_sim.recorded_folding_percent[-1] < np.mean(np.array(ori_sim.recorded_folding_percent[-51: -1])):
                            break
                else:
                    break
            step += 1
            self._emit.emit(ori_sim.folding_percent)

        all_dis = {
            "control_string_decrease": [],
            "string_decrease_each": [],
            "max_force": [],
            "folding_percent": [],
            "max_folding_percent": [],
            "min_folding_percent": [],
            "time": []
        }

        all_dis["control_string_decrease"] = ori_sim.recorded_string_decrease_length_control
        all_dis["string_decrease_each"] = ori_sim.recorded_string_decrease_length
        all_dis["folding_percent"] = ori_sim.recorded_folding_percent
        all_dis["max_folding_percent"] = ori_sim.recorded_maximum_folding_percent
        all_dis["min_folding_percent"] = ori_sim.recorded_minimum_folding_percent
        all_dis["max_force"] = ori_sim.recorded_max_force
        all_dis["time"] = ori_sim.recorded_t

        with open(self.file_path, 'w', encoding="utf-8") as f:
            json.dump(all_dis, f, indent=4)

        self._emit.emit(1.0)

# import numpy as np
# import math
import re
import sys

sys.path.append('./gui')

from PyQt5.QtWidgets import QDialog
from gui.Ui_pref_window import Ui_Settings

class PreferencePackWindow(Ui_Settings, QDialog):
    def __init__(self, parent=None) -> None:
        QDialog.__init__(self, parent)
        self.setupUi(self)

        self.buttonBox.accepted.connect(self.onAccepted)
        self.buttonBox.rejected.connect(self.onRejected)

        self.pref_pack = {
            "print_accuracy": 0.2,
            "split_distance": 1.0,
            "output_bc": True,
            "board_bias" : 1.4,
            "connection_angle": 0,
            "enable_db": False,
            "enable_db_bind": False,
            "additional_line_option": 0,
            "stl_asymmetry": True,
            "cursor_axis_mode": "pixel_axis",
            "line_weight": 3,
            "show_keypoint": True,
            "theme": 0,
            "tsa_radius": 100.00,
            "tsa_resolution": 72,
            "debug_mode": False,
            "thin_mode": False,
            "layer": 3
        }

        self.limitation = {
            "unit_length": [0, 200.0],
            "miura_angle": [15.0, 75.0],
            "row_number": [4, 5],
            "generation": 100000,
            "batch_size": 10,
            "process_number": 1,
            "storage": 100,
            "match_mode": "discrete",
            "discrete_resolution": 2,
            "hypar_enable": True,
            "direction_enable": False,
            "mcts_epoch": 100
        }

        self.ok = False
        self.default_path = "./setting/setting.log"

        self.readFile()
        self.setUiFromData()

    def getPrefPack(self):
        return self.pref_pack
    
    def getLimitation(self):
        return self.limitation
    
    def onAccepted(self):
        self.ok = True
        self.saveDataFromUi()
        self.saveFile()

    def onRejected(self):
        self.ok = False
    
    def setUiFromData(self):
        """
        @ function: carry data from inner space to Ui
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        #1
        self.doubleSpinBox_print_accuracy.setValue(self.pref_pack["print_accuracy"])
        #11
        self.doubleSpinBox_dxfsplit_dis.setValue(self.pref_pack["split_distance"])
        #2
        self.checkBox_output_bc.setChecked(self.pref_pack["output_bc"])
        #3
        self.doubleSpinBox_board_bias.setValue(self.pref_pack["board_bias"])
        #31
        self.doubleSpinBox_connection_angle.setValue(self.pref_pack["connection_angle"])
        #32
        self.checkBox_db.setChecked(self.pref_pack["enable_db"])
        #321
        self.checkBox_db_bind.setChecked(self.pref_pack["enable_db_bind"])
        #33
        if self.pref_pack["additional_line_option"] == 0:
            self.radioButton_nodiff.setChecked(True)
            self.radioButton_valleysmall.setChecked(False)
            self.radioButton_mountainsmall.setChecked(False)
        elif self.pref_pack["additional_line_option"] == 1:
            self.radioButton_nodiff.setChecked(False)
            self.radioButton_valleysmall.setChecked(True)
            self.radioButton_mountainsmall.setChecked(False)
        else:
            self.radioButton_nodiff.setChecked(False)
            self.radioButton_valleysmall.setChecked(False)
            self.radioButton_mountainsmall.setChecked(True)
        #34
        self.checkBox_stl_asymmetry.setChecked(self.pref_pack["stl_asymmetry"])
        #35
        self.checkBox_thin_mode.setChecked(self.pref_pack["thin_mode"])
        #36
        self.spinBox_layer.setValue(self.pref_pack["layer"])
        #4 We need to know which mode is
        cursor_axis_mode = self.pref_pack["cursor_axis_mode"]
        if cursor_axis_mode == "pixel_axis":
            self.radioButton_pixel_axis.setChecked(True)
            self.radioButton_real_axis.setChecked(False)
        elif cursor_axis_mode == "real_axis":
            self.radioButton_pixel_axis.setChecked(False)
            self.radioButton_real_axis.setChecked(True)
        #5
        self.spinBox_line_weight.setValue(self.pref_pack["line_weight"])
        #6
        self.checkBox_show_kp.setChecked(self.pref_pack["show_keypoint"])
        #7
        theme = self.pref_pack["theme"]
        if theme == 0:
            self.radioButton_light_theme.setChecked(True)
        else:
            self.radioButton_dark_theme.setChecked(True)
        #8
        self.doubleSpinBox_tsa_radius.setValue(self.pref_pack["tsa_radius"])
        #9
        self.spinBox_tsa_resolution.setValue(self.pref_pack["tsa_resolution"])
        #10
        self.checkBox_debug_mode.setChecked(self.pref_pack["debug_mode"])
        #LIMITATION
        #1
        self.doubleSpinBox_unit_length_min.setValue(self.limitation["unit_length"][0])
        self.doubleSpinBox_unit_length_max.setValue(self.limitation["unit_length"][1])
        #2
        self.doubleSpinBox_miura_angle_min.setValue(self.limitation["miura_angle"][0])
        self.doubleSpinBox_miura_angle_max.setValue(self.limitation["miura_angle"][1])
        #3
        self.spinBox_row_number_min.setValue(self.limitation["row_number"][0])
        self.spinBox_row_number_max.setValue(self.limitation["row_number"][1])
        #4
        self.spinBox_generation.setValue(self.limitation["generation"])
        #5
        self.spinBox_batch_size.setValue(self.limitation["batch_size"])
        #6
        self.spinBox_process_number.setValue(self.limitation["process_number"])
        #7
        self.spinBox_storage.setValue(self.limitation["storage"])
        #8
        if self.limitation["match_mode"] == 0:
            self.radioButton_bestmatch.setChecked(True)
            self.radioButton_strict_match.setChecked(False)
            self.radioButton_discretematch.setChecked(False)
        elif self.limitation["match_mode"] == 1:
            self.radioButton_bestmatch.setChecked(False)
            self.radioButton_strict_match.setChecked(True)
            self.radioButton_discretematch.setChecked(False)
        else:
            self.radioButton_bestmatch.setChecked(False)
            self.radioButton_strict_match.setChecked(False)
            self.radioButton_discretematch.setChecked(True)
        #9
        self.spinBox_discrete_resolution.setValue(self.limitation["discrete_resolution"])
        #10
        self.checkBox_hypar_enable.setChecked(self.limitation["hypar_enable"])
        #11
        self.checkBox_direction_enable_2.setChecked(self.limitation["direction_enable"])
        #12
        self.spinBox_mcts_epoch.setValue(self.limitation["mcts_epoch"])

    def saveDataFromUi(self):
        """
        @ function: carry data from Ui to inner space
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        #1
        self.pref_pack["print_accuracy"] = self.doubleSpinBox_print_accuracy.value()
        #11
        self.pref_pack["split_distance"] = self.doubleSpinBox_dxfsplit_dis.value()
        #2
        self.pref_pack["output_bc"] = self.checkBox_output_bc.isChecked()
        #3
        self.pref_pack["board_bias"] = self.doubleSpinBox_board_bias.value()
        #31
        self.pref_pack["connection_angle"] = self.doubleSpinBox_connection_angle.value()
        #32
        self.pref_pack["enable_db"] = self.checkBox_db.isChecked()
        #321
        self.pref_pack["enable_db_bind"] = self.checkBox_db_bind.isChecked()
        #33
        if self.radioButton_nodiff.isChecked():
            self.pref_pack["additional_line_option"] = 0
        elif self.radioButton_valleysmall.isChecked():
            self.pref_pack["additional_line_option"] = 1
        elif self.radioButton_mountainsmall.isChecked():
            self.pref_pack["additional_line_option"] = 2
        #34
        self.pref_pack["stl_asymmetry"] = self.checkBox_stl_asymmetry.isChecked()
        #35
        self.pref_pack["thin_mode"] = self.checkBox_thin_mode.isChecked()
        #36
        self.pref_pack["layer"] = self.spinBox_layer.value()
        #4 We need to know which radio button is checked
        cursor_axis_mode = ""
        if self.radioButton_pixel_axis.isChecked():
            cursor_axis_mode = "pixel_axis"
        elif self.radioButton_real_axis.isChecked():
            cursor_axis_mode = "real_axis"
        self.pref_pack["cursor_axis_mode"] = cursor_axis_mode
        #5
        self.pref_pack["line_weight"] = self.spinBox_line_weight.value()
        #6
        self.pref_pack["show_keypoint"] = self.checkBox_show_kp.isChecked()
        #7
        theme = 0
        if self.radioButton_light_theme.isChecked():
            theme = 0
        else:
            theme = 1
        self.pref_pack["theme"] = theme
        #8
        self.pref_pack["tsa_radius"] = self.doubleSpinBox_tsa_radius.value()
        #9
        self.pref_pack["tsa_resolution"] = self.spinBox_tsa_resolution.value()
        #10
        self.pref_pack["debug_mode"] = self.checkBox_debug_mode.isChecked()
        #LIMITATION
        #1
        self.limitation["unit_length"] = [
            self.doubleSpinBox_unit_length_min.value(), self.doubleSpinBox_unit_length_max.value()
        ]
        #2
        self.limitation["miura_angle"] = [
            self.doubleSpinBox_miura_angle_min.value(), self.doubleSpinBox_miura_angle_max.value()
        ]
        #3
        self.limitation["row_number"] = [
            self.spinBox_row_number_min.value(), self.spinBox_row_number_max.value()
        ]
        #4
        self.limitation["generation"] = self.spinBox_generation.value()
        #5
        self.limitation["batch_size"] = self.spinBox_batch_size.value()
        #6
        self.limitation["process_number"] = self.spinBox_process_number.value()
        #7
        self.limitation["storage"] = self.spinBox_storage.value()
        #8
        if self.radioButton_bestmatch.isChecked():
            self.limitation["match_mode"] = 0
        elif self.radioButton_strict_match.isChecked():
            self.limitation["match_mode"] = 1
        elif self.radioButton_discretematch.isChecked():
            self.limitation["match_mode"] = 2
        #9
        self.limitation["discrete_resolution"] = self.spinBox_discrete_resolution.value()
        #10
        self.limitation["hypar_enable"] = self.checkBox_hypar_enable.isChecked()
        #11
        self.limitation["direction_enable"] = self.checkBox_direction_enable_2.isChecked()
        #12
        self.limitation["mcts_epoch"] = self.spinBox_mcts_epoch.value()

    def saveFile(self):
        """
        @ function: save the setting file
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        file_path = self.default_path
        s = ""
        for key in self.pref_pack:
            s += "[" + key + "] = " + str(self.pref_pack[key]) + '\n'
        
        for key in self.limitation:
            if key == "generation":
                break
            s += "[" + key + "] = [" + str(self.limitation[key][0]) + ', ' + str(self.limitation[key][1]) + ']\n'   
        for ele in ["generation", "batch_size", "process_number", "storage", "match_mode", "discrete_resolution", "hypar_enable", "direction_enable", "mcts_epoch"]:
            s += "[" + ele + "] = " + str(self.limitation[ele]) + '\n'

        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(s)

    def readFile(self):
        """
        @ function: read the setting file
        @ version: 0.1
        @ developer: py
        @ progress: onroad
        @ date: 20230313
        @ spec: None
        """
        path = self.default_path
        try:
            with open(path, 'r', encoding='utf-8') as fw:
                s = fw.readlines()
                string_len = len(s)
                for i in range(string_len):
                    segment = s[i].split(' ', 2)
                    head = segment[0][1: -1]
                    content = segment[-1][0: -1]
                    if head not in [
                        "print_accuracy", "split_distance", "output_bc", "board_bias", 
                        "connection_angle", "enable_db", "enable_db_bind", "additional_line_option", "stl_asymmetry", "thin_mode", "layer",
                        "cursor_axis_mode", "line_weight", "show_keypoint", "theme", "tsa_radius", "tsa_resolution",
                        "unit_length", "miura_angle", "row_number", "generation", 
                        "batch_size", "process_number", "storage", "match_mode", "discrete_resolution",
                        "hypar_enable", "direction_enable", "debug_mode", "mcts_epoch"
                    ]:
                        raise TypeError
                    #1
                    if head == "print_accuracy":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = float(ans[0])
                            continue
                        else:
                            raise TypeError
                    #11
                    if head == "split_distance":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = float(ans[0])
                            continue
                        else:
                            raise TypeError
                    #2
                    elif head == "output_bc":
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #3
                    elif head == "board_bias":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = float(ans[0])
                            continue
                        else:
                            raise TypeError
                    #31
                    elif head == "connection_angle":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = float(ans[0])
                            continue
                        else:
                            raise TypeError
                    #32
                    elif head == "enable_db":
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #321
                    elif head == "enable_db_bind":
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #33
                    elif head == "additional_line_option":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #34
                    elif head == "stl_asymmetry":
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #35
                    elif head == "thin_mode":
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #36
                    elif head == "layer":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #4
                    elif head == "cursor_axis_mode":
                        self.pref_pack[head] = content
                    #5
                    elif head == "line_weight":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #6
                    elif head == "show_keypoint":
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #7
                    elif head == "theme":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #8
                    elif head == "tsa_radius":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = float(ans[0])
                            continue
                        else:
                            raise TypeError
                    #9
                    elif head == "tsa_resolution":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.pref_pack[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #10
                    elif head == "debug_mode":
                        if content == "True":
                            self.pref_pack[head] = True
                            continue
                        elif content == "False":
                            self.pref_pack[head] = False
                            continue
                        else:
                            raise TypeError
                    #LIMITATION
                    #1
                    elif head == "unit_length":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 2:
                            self.limitation[head] = [float(ans[0]), float(ans[1])]
                            continue
                        else:
                            raise TypeError
                    #2
                    elif head == "miura_angle":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 2:
                            self.limitation[head] = [float(ans[0]), float(ans[1])]
                            continue
                        else:
                            raise TypeError
                    #3
                    elif head == "row_number":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 2:
                            self.limitation[head] = [float(ans[0]), float(ans[1])]
                            continue
                        else:
                            raise TypeError
                    #4
                    elif head == "generation":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #5
                    elif head == "batch_size":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #6
                    elif head == "process_number":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #7
                    elif head == "storage":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #8
                    elif head == "match_mode":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #9
                    elif head == "discrete_resolution":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
                    #10
                    elif head == "hypar_enable":
                        if content == "True":
                            self.limitation[head] = True
                            continue
                        elif content == "False":
                            self.limitation[head] = False
                            continue
                        else:
                            raise TypeError
                    #11
                    elif head == "direction_enable":
                        if content == "True":
                            self.limitation[head] = True
                            continue
                        elif content == "False":
                            self.limitation[head] = False
                            continue
                        else:
                            raise TypeError
                    #12
                    elif head == "mcts_epoch":
                        ans = re.findall(r"\d+\.?\d*", content)
                        if len(ans) == 1:
                            self.limitation[head] = int(ans[0])
                            continue
                        else:
                            raise TypeError
        except:
            # No log file exists, we will create one
            self.saveFile()
        # Set Ui data
        self.setUiFromData()

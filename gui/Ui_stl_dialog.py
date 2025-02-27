# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\BaiduSyncdisk\2023-2024 Spring\11-1 SGLAB\pygamic\gui\stl_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        Dialog.resize(540, 360)
        Dialog.setMinimumSize(QtCore.QSize(540, 360))
        Dialog.setMaximumSize(QtCore.QSize(540, 360))
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(130, 310, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label_unit_height = QtWidgets.QLabel(Dialog)
        self.label_unit_height.setGeometry(QtCore.QRect(40, 30, 121, 21))
        self.label_unit_height.setObjectName("label_unit_height")
        self.label_unit_bias = QtWidgets.QLabel(Dialog)
        self.label_unit_bias.setGeometry(QtCore.QRect(40, 60, 121, 21))
        self.label_unit_bias.setObjectName("label_unit_bias")
        self.label_method = QtWidgets.QLabel(Dialog)
        self.label_method.setGeometry(QtCore.QRect(40, 90, 121, 21))
        self.label_method.setObjectName("label_method")
        self.radioButton_both_bias = QtWidgets.QRadioButton(Dialog)
        self.radioButton_both_bias.setGeometry(QtCore.QRect(100, 95, 89, 16))
        self.radioButton_both_bias.setObjectName("radioButton_both_bias")
        self.radioButton_upper_bias = QtWidgets.QRadioButton(Dialog)
        self.radioButton_upper_bias.setGeometry(QtCore.QRect(100, 120, 89, 16))
        self.radioButton_upper_bias.setChecked(False)
        self.radioButton_upper_bias.setObjectName("radioButton_upper_bias")
        self.label_unit_id = QtWidgets.QLabel(Dialog)
        self.label_unit_id.setGeometry(QtCore.QRect(40, 170, 121, 21))
        self.label_unit_id.setObjectName("label_unit_id")
        self.spinBox_unit_id = QtWidgets.QSpinBox(Dialog)
        self.spinBox_unit_id.setGeometry(QtCore.QRect(150, 170, 61, 22))
        self.spinBox_unit_id.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spinBox_unit_id.setObjectName("spinBox_unit_id")
        self.label_connections = QtWidgets.QLabel(Dialog)
        self.label_connections.setGeometry(QtCore.QRect(40, 200, 121, 21))
        self.label_connections.setObjectName("label_connections")
        self.checkBox_true = QtWidgets.QCheckBox(Dialog)
        self.checkBox_true.setGeometry(QtCore.QRect(170, 205, 71, 16))
        self.checkBox_true.setChecked(True)
        self.checkBox_true.setObjectName("checkBox_true")
        self.doubleSpinBox_unit_height = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox_unit_height.setGeometry(QtCore.QRect(150, 30, 62, 22))
        self.doubleSpinBox_unit_height.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_unit_height.setMinimum(0.1)
        self.doubleSpinBox_unit_height.setSingleStep(0.1)
        self.doubleSpinBox_unit_height.setProperty("value", 0.1)
        self.doubleSpinBox_unit_height.setObjectName("doubleSpinBox_unit_height")
        self.doubleSpinBox_unit_bias = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox_unit_bias.setGeometry(QtCore.QRect(150, 60, 62, 22))
        self.doubleSpinBox_unit_bias.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_unit_bias.setMinimum(0.1)
        self.doubleSpinBox_unit_bias.setSingleStep(0.1)
        self.doubleSpinBox_unit_bias.setObjectName("doubleSpinBox_unit_bias")
        self.label_hole_width = QtWidgets.QLabel(Dialog)
        self.label_hole_width.setGeometry(QtCore.QRect(260, 30, 151, 21))
        self.label_hole_width.setObjectName("label_hole_width")
        self.doubleSpinBox_hole_width = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox_hole_width.setGeometry(QtCore.QRect(410, 30, 62, 22))
        self.doubleSpinBox_hole_width.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_hole_width.setMinimum(0.01)
        self.doubleSpinBox_hole_width.setMaximum(0.99)
        self.doubleSpinBox_hole_width.setSingleStep(0.01)
        self.doubleSpinBox_hole_width.setProperty("value", 0.35)
        self.doubleSpinBox_hole_width.setObjectName("doubleSpinBox_hole_width")
        self.label_hole_length = QtWidgets.QLabel(Dialog)
        self.label_hole_length.setGeometry(QtCore.QRect(260, 60, 151, 21))
        self.label_hole_length.setObjectName("label_hole_length")
        self.doubleSpinBox_hole_length = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox_hole_length.setGeometry(QtCore.QRect(410, 60, 62, 22))
        self.doubleSpinBox_hole_length.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_hole_length.setMinimum(0.01)
        self.doubleSpinBox_hole_length.setMaximum(0.99)
        self.doubleSpinBox_hole_length.setSingleStep(0.01)
        self.doubleSpinBox_hole_length.setProperty("value", 0.8)
        self.doubleSpinBox_hole_length.setObjectName("doubleSpinBox_hole_length")
        self.pushButton_double_materials = QtWidgets.QPushButton(Dialog)
        self.pushButton_double_materials.setGeometry(QtCore.QRect(254, 183, 221, 31))
        self.pushButton_double_materials.setAutoRepeat(False)
        self.pushButton_double_materials.setAutoExclusive(False)
        self.pushButton_double_materials.setObjectName("pushButton_double_materials")
        self.pushButton_regular = QtWidgets.QPushButton(Dialog)
        self.pushButton_regular.setGeometry(QtCore.QRect(254, 138, 221, 31))
        self.pushButton_regular.setAutoRepeat(False)
        self.pushButton_regular.setAutoExclusive(False)
        self.pushButton_regular.setObjectName("pushButton_regular")
        self.label_board = QtWidgets.QLabel(Dialog)
        self.label_board.setGeometry(QtCore.QRect(40, 230, 121, 21))
        self.label_board.setObjectName("label_board")
        self.checkBox_board_true = QtWidgets.QCheckBox(Dialog)
        self.checkBox_board_true.setGeometry(QtCore.QRect(170, 234, 71, 16))
        self.checkBox_board_true.setChecked(True)
        self.checkBox_board_true.setObjectName("checkBox_board_true")
        self.label_board_height = QtWidgets.QLabel(Dialog)
        self.label_board_height.setGeometry(QtCore.QRect(40, 260, 121, 21))
        self.label_board_height.setObjectName("label_board_height")
        self.doubleSpinBox_board_height = QtWidgets.QDoubleSpinBox(Dialog)
        self.doubleSpinBox_board_height.setGeometry(QtCore.QRect(150, 260, 62, 22))
        self.doubleSpinBox_board_height.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.doubleSpinBox_board_height.setMinimum(0.1)
        self.doubleSpinBox_board_height.setSingleStep(0.1)
        self.doubleSpinBox_board_height.setObjectName("doubleSpinBox_board_height")
        self.radioButton_symmetry = QtWidgets.QRadioButton(Dialog)
        self.radioButton_symmetry.setGeometry(QtCore.QRect(100, 145, 89, 16))
        self.radioButton_symmetry.setChecked(True)
        self.radioButton_symmetry.setObjectName("radioButton_symmetry")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Stl Settings"))
        self.label_unit_height.setText(_translate("Dialog", "Unit height(mm):"))
        self.label_unit_bias.setText(_translate("Dialog", "Unit bias(mm):"))
        self.label_method.setText(_translate("Dialog", "method:"))
        self.radioButton_both_bias.setText(_translate("Dialog", "Both bias"))
        self.radioButton_upper_bias.setText(_translate("Dialog", "Upper bias"))
        self.label_unit_id.setText(_translate("Dialog", "Output unit Id:"))
        self.label_connections.setText(_translate("Dialog", "Output connections:"))
        self.checkBox_true.setText(_translate("Dialog", "True"))
        self.label_hole_width.setText(_translate("Dialog", "Connection hole width:"))
        self.label_hole_length.setText(_translate("Dialog", "Connection hole length:"))
        self.pushButton_double_materials.setText(_translate("Dialog", "Double materials recommand setting"))
        self.pushButton_regular.setText(_translate("Dialog", "Regular recommand setting"))
        self.label_board.setText(_translate("Dialog", "Output board:"))
        self.checkBox_board_true.setText(_translate("Dialog", "True"))
        self.label_board_height.setText(_translate("Dialog", "Board height(mm):"))
        self.radioButton_symmetry.setText(_translate("Dialog", "Up-Symmetry"))

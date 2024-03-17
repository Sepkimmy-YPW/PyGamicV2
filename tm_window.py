# import sys
import matplotlib
from matplotlib.patches import Rectangle
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from cdftool import *

class TmWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.tm = TransitionModel()
        self.folding_angle = 0
        self.enable_update_canvas = False
        self.setWindowTitle("Transition angle")

        # 创建一个垂直布局和一个QWidget小部件
        self.lay_out = QVBoxLayout()
        self.widget = QWidget(self)
        self.widget.setLayout(self.lay_out)
        self.setCentralWidget(self.widget)

        self.dynamic_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.dynamic_ax = self.dynamic_canvas.figure.subplots()
        self.lay_out.addWidget(self.dynamic_canvas)

        self.horizontal_folding_slider = QtWidgets.QSlider()
        self.horizontal_folding_slider.setGeometry(QtCore.QRect(165, 480, 751, 22))
        self.horizontal_folding_slider.setMaximum(180)
        self.horizontal_folding_slider.setProperty("value", 0)
        self.horizontal_folding_slider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontal_folding_slider.setObjectName("horizontal_folding_slider")
        self.lay_out.addWidget(self.horizontal_folding_slider)

        self.horizontal_folding_slider.valueChanged.connect(self.changeFoldingAngle)

        self.enable_show_process = False

        self._timer = self.dynamic_canvas.new_timer(33, [
            (self.updateCanvas, (), {})
        ])
        self._timer.start()

        self.x_axis = []
        self.y_axis = []

    def importXPoints(self, xpoints):
        self.x_axis = xpoints

    def importYPoints(self, ypoints):
        self.y_axis = ypoints

    def plot(self):
        self.dynamic_ax.clear()
        self.dynamic_ax.plot(self.x_axis[0: len(self.x_axis): len(self.x_axis) // 1] + [self.x_axis[-1]], self.y_axis[0: len(self.y_axis): len(self.y_axis) // 1] + [self.y_axis[-1]], marker='*', c='r', markersize=12)
        rect1 = Rectangle((140.0, 30.0), 110, 110, color='orange')
        rect2 = Rectangle((150.0, 40.0), 100, 100, color='grey')
        self.dynamic_ax.add_patch(rect1)
        self.dynamic_ax.add_patch(rect2)

        self.dynamic_ax.set_xlim(-50.0, 300.0)
        self.dynamic_ax.set_ylim(-30.0, 170.0)

        self.dynamic_ax.figure.canvas.draw()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Up:
            self.folding_angle += 1
            if self.folding_angle >= 180:
                self.folding_angle = 180
        elif event.key() == Qt.Key_Down:
            self.folding_angle -= 1
            if self.folding_angle <= 0:
                self.folding_angle = 0
        elif event.key() == Qt.Key_C:
            self.enable_show_process = not self.enable_show_process
    
    def getTm(self):
        return self.tm
    
    def updateCanvas(self):
        if self.enable_update_canvas:
            self.tm.setMainFoldingAngle(self.folding_angle * math.pi / 180.0)
            self.tl = self.tm.getTransitionLines()
            self.dynamic_ax.clear()
            # Get a AABB box
            min_x, max_x, min_y, max_y = self.tm.getAABB()

            if len(self.x_axis) > 0:
                # self.dynamic_ax.plot(self.x_axis, self.y_axis, marker='x', c='g')
                self.dynamic_ax.plot(self.x_axis[0: len(self.x_axis): len(self.x_axis) // 1] + [self.x_axis[-1]], self.y_axis[0: len(self.y_axis): len(self.y_axis) // 1] + [self.y_axis[-1]], marker='*', c='r', markersize=12)
                rect1 = Rectangle((140.0, 30.0), 1000, 1000, color='orange')
                rect2 = Rectangle((150.0, 40.0), 1000, 1000, color='grey')
                self.dynamic_ax.add_patch(rect1)
                self.dynamic_ax.add_patch(rect2)

            # 绘制所有线段
            for i in range(len(self.tl)):
                start, end = self.tl[i].getData()
                line, = self.dynamic_ax.plot([start[0], end[0]], [start[1], end[1]], linewidth=6)
                  
            if self.enable_show_process:
                self.dynamic_ax.plot([self.tm.all_end_ef[i][X] for i in range(len(self.tm.all_end_ef))], [self.tm.all_end_ef[i][Y] for i in range(len(self.tm.all_end_ef))])

            self.dynamic_ax.plot([self.tm.end_ef[0]], [self.tm.end_ef[1]], marker='o', c='r')

            self.dynamic_ax.set_xlim(min_x, max_x)
            self.dynamic_ax.set_ylim(min_y, max_y)
            self.dynamic_ax.figure.canvas.draw()
        
    def changeFoldingAngle(self):
        self.folding_angle = self.horizontal_folding_slider.value()

    def startShow(self):
        self.tm.setMainFoldingAngle(0)
        self.tl = self.tm.getTransitionLines()
        self.enable_update_canvas = True
        # self.tm.plotUi()
        self.show()

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() > 0:
            self.folding_angle += 1
            if self.folding_angle >= 180:
                self.folding_angle = 180
        else:
            self.folding_angle -= 1
            if self.folding_angle <= 0:
                self.folding_angle = 0
        self.horizontal_folding_slider.setValue(self.folding_angle)
    
    def closeEvent(self, event):
        self._timer.stop()
        self.deleteLater()

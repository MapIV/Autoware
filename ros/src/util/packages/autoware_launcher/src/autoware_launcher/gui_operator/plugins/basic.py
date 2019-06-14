# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

from autoware_launcher.core.launch import AwLaunchNode
import time

class QImage(QtWidgets.QLabel):
    def __init__(self, img_path, width=200, height=200):
        super(QImage, self).__init__()

        pixmap = QtGui.QPixmap(img_path)
        pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        self.setPixmap(pixmap)

class QImages(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    def __init__(self, img_list, size=(100, 100), default=0):
        super(QImages, self).__init__()

        self.cur_i = default
        self.pixamps = [QtGui.QPixmap(img) for img in img_list]

        self.update_img()
    
    # This function is called when mouse is pressed
    def mousePressEvent(self, event):
        self.clicked.emit()

    def update_img(self):
        self.setPixmap(self.pixamps[self.cur_i])

    def set_img(self, ind):
        if ind != self.cur_i and ind < len(self.pixamps):
            self.cur_i = ind
            self.update_img()

class QToggleImage(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    switchedOn = QtCore.pyqtSignal()
    switchedOff = QtCore.pyqtSignal()

    def __init__(self, off_img, on_img, size=(120, 120), default=0):
        super(QToggleImage, self).__init__()

        # size is (width, heigh)
        self._value = default
        self.off_img = QtGui.QPixmap(off_img).scaled(size[0], size[1], QtCore.Qt.KeepAspectRatio)
        self.on_img = QtGui.QPixmap(on_img).scaled(size[0], size[1], QtCore.Qt.KeepAspectRatio)

        self.update_img()
        self.clicked.connect(self.toggle)
    
    # This function is called when mouse is pressed
    def mousePressEvent(self, event):
        self.toggle()

    def update_img(self):
        if self._value == 0:
            self.setPixmap(self.off_img)
        else:
            self.setPixmap(self.on_img)

    def set_value(self, val):
        if val != self._value:
            self._value = val
            self.update_img()
            self.emit_switchedSignal()

    def emit_switchedSignal(self):
        if self._value == 1:
            self.switchedOn.emit()
        else:
            self.switchedOff.emit()

    def toggle(self):
        if self._value == 1:
            self.set_value(0)
        else:
            self.set_value(1)

    def isOn(self):
        if self._value == 1:
            return True
        else:
            return False


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class AwLabeledLineEdit(QtWidgets.QWidget):
    textChanged = QtCore.pyqtSignal(str)

    def __init__(self, pre_text='', post_text=''):
        super(AwLabeledLineEdit, self).__init__()

        self.pre_label = QtWidgets.QLabel()
        self.set_pre_label_text(pre_text)

        self.post_label = QtWidgets.QLabel()
        self.set_post_label_text(post_text)

        self.lineedit = QtWidgets.QLineEdit()
        self.lineedit.textChanged.connect(self.emit_textChanged)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.pre_label)
        layout.addWidget(self.lineedit)
        layout.addWidget(self.post_label)

        self.setLayout(layout)

    def set_pre_label_text(self, txt):
        self.pre_label.setText(txt)

    def set_post_label_text(self, txt):
        self.post_label.setText(txt)

    def set_text(self, text):
        self.lineedit.setText(str(text))

    def emit_textChanged(self, text):
        self.textChanged.emit(text)


class AwToggleSwitch(QtWidgets.QSlider):

    switchedOn = QtCore.pyqtSignal()
    switchedOff = QtCore.pyqtSignal()

    def __init__(self, default=0):
        QtWidgets.QSlider.__init__(self, QtCore.Qt.Horizontal)

        self.setMaximumWidth(70)
        self.setMinimum(0)
        self.setMaximum(1)
        self.setSliderPosition(default)

        self.sliderReleased.connect(self.toggle)
        self.valueChanged.connect(self.emit_switchedSignal)

    def toggle(self):
        if self.value() == 1:
            self.setSliderPosition(0)
            self.setValue(1)
        else:
            self.setSliderPosition(1)
            self.setValue(0)

    def emit_switchedSignal(self):
        if self.value() == 1:
            self.switchedOn.emit()
        else:
            self.switchedOff.emit()

    def isOn(self):
        if self.currentValue == 1:
            return True
        else:
            return False


# TODO modify update logic
class AwNodeListWidget(QtWidgets.QWidget):

    def __init__(self, target="", depth=1):
        super(AwNodeListWidget, self).__init__()
        self.nodes = []
        self.target = target.split("/")
        self.depth = depth

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

    def update_node_list(self, nodes):
        self.nodes = nodes
        self.remove_all_widgets()

        for n in self.nodes:
            widget = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(QtWidgets.QLabel(n["name"]))
            hbox.addWidget(QtWidgets.QLabel(n["status"]))
            widget.setLayout(hbox)

            self.layout().addWidget(widget)

    def remove_all_widgets(self):
        for i in reversed(range(self.layout().count())): 
            if self.layout().itemAt(i) is not None:
                self.layout().itemAt(i).widget().deleteLater()
    
    def is_target_node(self, name):
        ns = name.split("/")

        if len(ns) < len(self.target):
            return False
        
        if len(ns) > len(self.target) + self.depth:
            return False
        
        for i in range(len(self.target)):
            if ns[i] != self.target[i]:
                return False
        return True
    
    # this function is called by NodeStatusWatcher
    def node_status_updated(self, nodes):
        target_nodes = []
        for name, val in nodes.items():
            if not self.is_target_node(name):
                continue

            if val == AwLaunchNode.STOP:
                status = "Off"
            elif val == AwLaunchNode.EXEC:
                status = "On"
            elif val == AwLaunchNode.TERM:
                status = "Off"
            else:
                status = "N/A"
            
            target_nodes.append({"name": name, "status": status})
                
        # print(target_nodes)
        self.update_node_list(target_nodes)


class AwThread(QtCore.QThread):
    
    update = QtCore.pyqtSignal()
    
    def __init__(self, parent, period):
        super(AwThread, self).__init__(parent)
        self.period = period
 
    def run(self):
        while True:
            time.sleep(self.period)
            self.update.emit()

# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import time


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

    def __init__(self, default=0):
        super(AwNodeListWidget, self).__init__()
        self.nodes = []

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

    def update_node_list(self, nodes):
        self.nodes = nodes
        self.remove_all_widgets()

        for n in self.nodes:
            widget = QtWidgets.QWidget()
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(QtWidgets.QLabel(n.name))
            hbox.addWidget(QtWidgets.QLabel('On' if n.status else 'Off'))
            widget.setLayout(hbox)

            self.layout().addWidget(widget)

    def remove_all_widgets(self):
        for i in reversed(range(self.layout().count())): 
            self.layout().itemAt(i).widget().deleteLater()


class AwNode(object):
    def __init__(self, **kargs):
        self.name = kargs.get('name', 'node')
        self.status = kargs.get('status', True)

    # TODO
    def start(self):
        self.status = True

    # TODO
    def stop(self):
        self.status = False

class AwThread(QtCore.QThread):
    
    update = QtCore.pyqtSignal()
    
    def __init__(self, parent, period):
        super(AwThread, self).__init__(parent)
        self.period = period
 
    def run(self):
        while True:
            time.sleep(self.period)
            self.update.emit()
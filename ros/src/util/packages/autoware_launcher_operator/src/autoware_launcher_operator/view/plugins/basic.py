# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class AwToggleSwitch(QtWidgets.QSlider):

    switchedOn = QtCore.pyqtSignal()
    switchedOff = QtCore.pyqtSignal()

    def __init__(self, default=0):
        QtWidgets.QSlider.__init__(self, QtCore.Qt.Horizontal)

        self.setMaximumWidth(30)
        self.setMinimum(0)
        self.setMaximum(1)
        self.setSliderPosition(default)

        self.sliderReleased.connect(self.toggle)
        self.valueChanged.connect(self.emitSwitchedSignal)

    def toggle(self):
        if self.value() == 1:
            self.setValue(0)
        else:
            self.setValue(1)

    def emitSwitchedSignal(self):
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
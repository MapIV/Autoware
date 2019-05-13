# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


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

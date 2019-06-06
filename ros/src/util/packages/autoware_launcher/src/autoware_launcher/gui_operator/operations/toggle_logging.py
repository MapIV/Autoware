# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

# from ..plugins.basic import AwToggleSwitch
from ...core import myutils
from ..plugins.basic import QToggleImage


class AwToggleLoggingWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwToggleLoggingWidget, self).__init__()
        self.context = context

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel('Logging')
        layout.addWidget(self.label)

        # self.switch = AwToggleSwitch()
        self.switch = QToggleImage(myutils.package('resources/toggle_on.png'), myutils.package('resources/toggle_off.png'))
        self.switch.switchedOn.connect(self.switchedOn)
        self.switch.switchedOff.connect(self.switchedOff)
        layout.addWidget(self.switch)

        self.setLayout(layout)

    def switchedOn(self):
        print('logging switch on')

    def switchedOff(self):
        print('logging switch off')

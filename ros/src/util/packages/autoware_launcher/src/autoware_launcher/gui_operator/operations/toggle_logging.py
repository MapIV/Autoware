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

        self.logging_node = "logging_node"
        self.logging_topic = " /points_raw"
        self.logging_name = "autoware_bag"
        self.logging_path = self.context.userhome_path

        self.logging_start_proc = QtCore.QProcess(self)
        self.logging_stop_proc = QtCore.QProcess(self)

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel('Logging')
        layout.addWidget(self.label)

        # self.switch = AwToggleSwitch()
        self.switch = QToggleImage(myutils.package('resources/toggle_off.png'), myutils.package('resources/toggle_on.png'))
        self.switch.switchedOn.connect(self.switchedOn)
        self.switch.switchedOff.connect(self.switchedOff)
        layout.addWidget(self.switch)

        self.setLayout(layout)

    def switchedOn(self):
        print('logging start: ' + self.logging_path + "/" + self.logging_name)
        self.logging_start_proc.start('rosbag record -O {}/{} {} __name:={}'.format(self.logging_path, self.logging_name, self.logging_topic, self.logging_node))

    def switchedOff(self):
        print('logging stopped')
        self.logging_stop_proc.start("rosnode kill " + self.logging_node)
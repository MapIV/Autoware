# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from ...core import myutils
from ..plugins.basic import QToggleImage

class AwSimulationWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwSimulationWidget, self).__init__()
        self.context = context

        # button
        self.lgsvl_button = QToggleImage(myutils.package('resources/lgsvl_off.png'), myutils.package('resources/lgsvl_on.png'), size=(150, 150))
        self.lgsvl_button.switchedOn.connect(self.start_lgsvl)
        self.lgsvl_button.switchedOff.connect(self.stop_lgsvl)
        self.carla_button = QToggleImage(myutils.package('resources/carla_off.png'), myutils.package('resources/carla_on.png'), size=(150, 150))
        self.carla_button.switchedOn.connect(self.start_carla)
        self.carla_button.switchedOff.connect(self.stop_carla)
        self.gazebo_button = QToggleImage(myutils.package('resources/gazebo_off.png'), myutils.package('resources/gazebo_on.png'), size=(150, 150))
        self.gazebo_button.switchedOn.connect(self.start_gazebo)
        self.gazebo_button.switchedOff.connect(self.stop_gazebo)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.lgsvl_button)
        layout.addWidget(self.carla_button)
        layout.addWidget(self.gazebo_button)
        layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.setLayout(layout)

    def start_lgsvl(self):
        print('start lgsvl')

    def stop_lgsvl(self):
        print('stop lgsvl')

    def start_carla(self):
        print('start carla')

    def stop_carla(self):
        print('stop carla')

    def start_gazebo(self):
        print('start gazebo')

    def stop_gazebo(self):
        print('stop gazebo')

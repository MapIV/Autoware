# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from .real import AwRealSensorWidget
from .rosbag import AwRosbagSimulatorWidget
from .simulation import AwSimulationWidget


class AwMainViewWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwMainViewWidget, self).__init__()
        self.context = context

        # button
        # self.real_button = QtWidgets.QPushButton('REAL')
        # self.real_button.clicked.connect(self.select_real_mode)
        # self.rosbag_button = QtWidgets.QPushButton('ROSBAG')
        # self.rosbag_button.clicked.connect(self.select_rosbag_mode)
        # self.sim_button = QtWidgets.QPushButton('SIM')
        # self.sim_button.clicked.connect(self.select_sim_mode)

        # tabs
        self.real_tab = AwRealSensorWidget(context)
        self.rosbag_tab = AwRosbagSimulatorWidget(context)
        self.sim_tab = AwSimulationWidget(context)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.real_tab, 'REAL')
        self.tabs.addTab(self.rosbag_tab, 'ROSBAG')
        self.tabs.addTab(self.sim_tab, 'SIM')

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        
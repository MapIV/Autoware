# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwModeSelectWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwModeSelectWidget, self).__init__()
        self.context = context

        # button
        self.real_button = QtWidgets.QPushButton('REAL')
        self.real_button.clicked.connect(self.select_real_mode)
        self.rosbag_button = QtWidgets.QPushButton('ROSBAG')
        self.rosbag_button.clicked.connect(self.select_rosbag_mode)
        self.sim_button = QtWidgets.QPushButton('SIM')
        self.sim_button.clicked.connect(self.select_sim_mode)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.real_button)
        layout.addWidget(self.rosbag_button)
        layout.addWidget(self.sim_button)
        self.setLayout(layout)

    def select_real_mode(self):
        print('select real mode')

    def select_rosbag_mode(self):
        print('select rosbag mode')

    def select_sim_mode(self):
        print('select sim mode')

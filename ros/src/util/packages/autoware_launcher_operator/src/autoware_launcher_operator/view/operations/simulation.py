# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwSimulationWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwSimulationWidget, self).__init__()
        self.context = context

        # button
        self.lgsvl_button = QtWidgets.QPushButton('LGSVL')
        self.lgsvl_button.clicked.connect(self.push_lgsvl_btn)
        self.carla_button = QtWidgets.QPushButton('CARLA')
        self.carla_button.clicked.connect(self.push_carla_btn)
        self.gazebo_button = QtWidgets.QPushButton('GAZEBO')
        self.gazebo_button.clicked.connect(self.push_gazebo_btn)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.lgsvl_button)
        layout.addWidget(self.carla_button)
        layout.addWidget(self.gazebo_button)
        self.setLayout(layout)

    def push_lgsvl_btn(self):
        print('start lgsvl')

    def push_carla_btn(self):
        print('start carla')

    def push_gazebo_btn(self):
        print('start gazebo')

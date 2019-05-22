# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from ..plugins.basic import AwNodeListWidget, AwNode, QHLine


class AwRealSensorWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwRealSensorWidget, self).__init__()
        self.context = context

        # button
        self.run_sensing_btn = QtWidgets.QPushButton('Run Sensing')
        self.run_sensing_btn.clicked.connect(self.run_sensing_btn_clicked)
        self.exit_sensing_btn = QtWidgets.QPushButton('Exit Sensing')
        self.exit_sensing_btn.clicked.connect(self.exit_sensing_btn_clicked)
        self.engage_actuation_btn = QtWidgets.QPushButton('Engage Actuation')
        self.engage_actuation_btn.clicked.connect(self.engage_actuation_btn_clicked)
        self.disengage_actuation_btn = QtWidgets.QPushButton('Disegage Actuation')
        self.disengage_actuation_btn.clicked.connect(self.disengage_actuation_btn_clicked)

        # pull down menu
        self.sensing_profile_pdmenu = QtWidgets.QComboBox()
        self.set_sensing_profile_contents()
        self.sensing_profile_pdmenu.currentTextChanged.connect(self.on_sensing_profile_changed)
        self.actuation_profile_pdmenu = QtWidgets.QComboBox()
        self.set_actuation_profile_contents()
        self.actuation_profile_pdmenu.currentTextChanged.connect(self.on_actuation_profile_changed)

        #  node list
        self.sensing_nodes = []
        self.sensing_node_list = AwNodeListWidget()
        self.actuation_nodes = []
        self.actuation_node_list = AwNodeListWidget()

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.sensing_profile_pdmenu)
        sub_layout1 = QtWidgets.QHBoxLayout()
        sub_layout1.addWidget(self.run_sensing_btn)
        sub_layout1.addWidget(self.exit_sensing_btn)
        layout.addLayout(sub_layout1)
        layout.addWidget(self.sensing_node_list)
        layout.addWidget(QHLine())
        layout.addWidget(self.actuation_profile_pdmenu)
        sub_layout2 = QtWidgets.QHBoxLayout()
        sub_layout2.addWidget(self.engage_actuation_btn)
        sub_layout2.addWidget(self.disengage_actuation_btn)
        layout.addLayout(sub_layout2)
        layout.addWidget(self.actuation_node_list)
        self.setLayout(layout)

    def run_sensing_btn_clicked(self):
        print('run sensing')

        # TODO run launch file
        nodes = [
            AwNode(name='lidar', status=True),
            AwNode(name='camera', status=True),
            AwNode(name='camera_lidar', status=True),
            AwNode(name='multi_lidar', status=True),
            AwNode(name='lidar_preprocessor', status=True),
        ]
        self.update_sensing_node(nodes)

    def exit_sensing_btn_clicked(self):
        print('exit sensing')

        # TODO stop related nodes
        nodes = [
            AwNode(name='lidar', status=False),
            AwNode(name='camera', status=False),
            AwNode(name='camera_lidar', status=False),
            AwNode(name='multi_lidar', status=False),
            AwNode(name='lidar_preprocessor', status=False),
        ]
        self.update_sensing_node(nodes)

    def engage_actuation_btn_clicked(self):
        print('engage actuation')

        # TODO run launch file
        self.actuation_nodes = [
            AwNode(name='autonomoustuff', status=True),
        ]
        self.actuation_node_list.update_node_list(self.actuation_nodes)

    def disengage_actuation_btn_clicked(self):
        print('disengage_actuation')

        # TODO stop related nodes
        self.actuation_nodes = [
            AwNode(name='autonomoustuff', status=False),
        ]
        self.actuation_node_list.update_node_list(self.actuation_nodes)

    def set_sensing_profile_contents(self):
        self.sensing_profile_pdmenu.addItems(['dummy1', 'dummy2'])

    def on_sensing_profile_changed(self, text):
        print('select sensing profile: ' + text)

    def set_actuation_profile_contents(self):
        self.actuation_profile_pdmenu.addItems(['dummy1', 'dummy2'])

    def on_actuation_profile_changed(self, text):
        print('select actuation profile: ' + text)

    def update_sensing_node(self, nodes):
        self.sensing_nodes = nodes
        self.sensing_node_list.update_node_list(self.sensing_nodes)

    def update_actuation_node(self, nodes):
        self.actuation_nodes = nodes
        self.actuation_node_list.update_node_list(self.actuation_nodes)
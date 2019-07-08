# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from ..plugins.basic import AwNodeListWidget, QHLine


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
        self.sensing_node_list = AwNodeListWidget(target="root/sensing")
        self.context.register_node_status_watcher_client(self.sensing_node_list)
        self.actuation_node_list = AwNodeListWidget(target="root/actuation")
        self.context.register_node_status_watcher_client(self.actuation_node_list)

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
        print('run sensing: ' + self.context.sensing_profile)
        self.context.server.launch_node("root/sensing", True)

    def exit_sensing_btn_clicked(self):
        print('exit sensing')
        self.context.server.launch_node("root/sensing", False)

    def engage_actuation_btn_clicked(self):
        print('engage actuation')
        self.context.server.launch_node("root/vehicle", True)

    def disengage_actuation_btn_clicked(self):
        print('disengage actuation')
        self.context.server.launch_node("root/vehicle", False)

    def set_sensing_profile_contents(self):
        self.sensing_profile_pdmenu.addItems(self.context.sensing_profile_list)

        # load default sensing  
        dirpath = "operator/sensing/" + self.sensing_profile_pdmenu.currentText()
        self.context.server.load_profile_subtree(dirpath, "root/sensing")

    def on_sensing_profile_changed(self, text):
        print('select sensing profile: ' + text)
        self.context.set_sensing_profile(text)

    def set_actuation_profile_contents(self):
        # self.actuation_profile_pdmenu.addItems(['dummy1', 'dummy2'])
        self.actuation_profile_pdmenu.addItems(self.context.actuation_profile_list)

        # load default actuation profile
        dirpath = "operator/actuation/" + self.actuation_profile_pdmenu.currentText()
        self.context.server.load_profile_subtree(dirpath, "root/vehicle")

    def on_actuation_profile_changed(self, text):
        print('select actuation profile: ' + text)
        self.context.set_actuation_profile(text)

    def update_sensing_node(self, nodes):
        self.sensing_nodes = nodes
        self.sensing_node_list.update_node_list(self.sensing_nodes)

    def update_actuation_node(self, nodes):
        self.actuation_nodes = nodes
        self.actuation_node_list.update_node_list(self.actuation_nodes)
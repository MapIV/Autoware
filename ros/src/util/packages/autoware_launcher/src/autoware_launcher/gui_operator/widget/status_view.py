# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from .basic import NodeStatusList
from collections import OrderedDict
import os

class StatusViewWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(StatusViewWidget, self).__init__()
        self.context = context

        # map status
        self.map_text = QtWidgets.QLabel('Map')
        self.map_pbar = QtWidgets.QProgressBar()
        map_layout = QtWidgets.QHBoxLayout()
        map_layout.addWidget(self.map_text)
        map_layout.addWidget(self.map_pbar)

        # computing status
        self.computing_text = QtWidgets.QLabel('Computing')
        computing_nodes = OrderedDict()
        computing_nodes['root/localization'] = 'Localization'
        computing_nodes['root/detection'] = 'Detection'
        computing_nodes['root/perception/object/tracking'] = 'Tracking'
        computing_nodes['root/planning/prediction'] = 'Prediction'
        computing_nodes['root/planning/decision'] = 'Decision'
        computing_nodes['root/planning/mission'] = 'Mission'
        computing_nodes['root/planning/motion'] = 'Motion'
        computing_nodes['root/wheel'] = 'Wheel'
        self.computing_node_status = NodeStatusList(computing_nodes)
        self.context.register_node_status_watcher_client(self.computing_node_status)
        computing_layout = QtWidgets.QVBoxLayout()
        computing_layout.addWidget(self.computing_text)
        computing_layout.addWidget(self.computing_node_status)

        # sensing status
        self.sensing_text = QtWidgets.QLabel('Sensing')
        self.sensing_node_status = NodeStatusList('root/sensing', depth=1, ignore=['root/sensing'])
        self.context.register_node_status_watcher_client(self.sensing_node_status)
        sensing_layout = QtWidgets.QVBoxLayout()
        sensing_layout.addWidget(self.sensing_text)
        sensing_layout.addWidget(self.sensing_node_status)

        # Actuation status
        self.actuation_text = QtWidgets.QLabel('Actuation')
        actuation_nodes = {'root/actuation': 'autonomoustuff'}
        self.actuation_node_status = NodeStatusList(actuation_nodes)
        self.context.register_node_status_watcher_client(self.actuation_node_status)
        actuation_layout = QtWidgets.QVBoxLayout()
        actuation_layout.addWidget(self.actuation_text)
        actuation_layout.addWidget(self.actuation_node_status)

        # addjust widget size
        self.setMinimumWidth(350)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(map_layout)
        layout.addLayout(computing_layout)
        layout.addLayout(sensing_layout)
        layout.addLayout(actuation_layout)
        layout.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(layout)

        self.node_status_updated({})

    def node_status_updated(self, nodes):
        self.computing_node_status.node_status_updated(nodes)
        self.sensing_node_status.node_status_updated(nodes)
        self.actuation_node_status.node_status_updated(nodes)

    def set_map_progress(self, val):
        # val: 0 to 100
        self.map_pbar.setValue(val)
    
    def complete_map_progress(self):
        self.set_map_progress(100)

    def reset_map_progress(self):
        self.set_map_progress(0)
    
    def show_real_mode(self):
        self.map_text.show()
        self.map_pbar.show()
        self.computing_text.show()
        self.computing_node_status.show()
        self.sensing_text.show()
        self.sensing_node_status.show()
        self.actuation_text.show()
        self.actuation_node_status.show()

    def show_rosbag_mode(self):
        self.map_text.show()
        self.map_pbar.show()
        self.computing_text.show()
        self.computing_node_status.show()
        self.sensing_text.hide()
        self.sensing_node_status.hide()
        self.actuation_text.hide()
        self.actuation_node_status.hide()

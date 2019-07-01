# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from ..plugins.basic import AwNodeListWidget, AwNode


class AwComputingWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwComputingWidget, self).__init__()
        self.context = context

        # run computing button
        self.run_button = QtWidgets.QPushButton('Run Computing')
        self.run_button.clicked.connect(self.on_run_clicked)

        #  exit computing button
        self.exit_button = QtWidgets.QPushButton('Exit Computing')
        self.exit_button.clicked.connect(self.on_exit_clicked)

        #  node list
        self.nodes = []
        self.node_list = AwNodeListWidget()

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.run_button)
        layout.addWidget(self.exit_button)
        layout.addWidget(self.node_list)
        self.setLayout(layout)

    def on_run_clicked(self):
        print('run computing' + self.context.computing_profile)
        self.context.server.launch_node("root/computing", True)

        # TODO run launch file
        self.nodes = [
            AwNode(name='Localization', status=True),
            AwNode(name='Detection', status=True),
            AwNode(name='Tracking', status=True),
            AwNode(name='Pedestrian', status=True),
            AwNode(name='Decision', status=True),
        ]
        self.node_list.update_node_list(self.nodes)

    def on_exit_clicked(self):
        print('exit computing')
        self.context.server.launch_node("root/computing", False)

        # TODO stop related nodes
        for n in self.nodes:
            n.stop()
        self.node_list.update_node_list(self.nodes)

    # def on_cancel_clicked(self):
    #     print('on load map cancel clicked')
    #     self.set_progress(50)

    # TODO load computing profile
    def load_profile(self):
        print('load computing profile')


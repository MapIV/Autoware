# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from ..plugins.basic import AwNodeListWidget


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
        self.node_list = AwNodeListWidget(target="root/computing")
        self.context.register_node_status_watcher_client(self.node_list)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.run_button)
        layout.addWidget(self.exit_button)
        layout.addWidget(self.node_list)
        self.setLayout(layout)

    def on_run_clicked(self):
        print('run computing' + self.context.computing_profile)
        self.context.server.launch_node("root/computing", True)

    def on_exit_clicked(self):
        print('exit computing')
        self.context.server.launch_node("root/computing", False)

    # def on_cancel_clicked(self):
    #     print('on load map cancel clicked')
    #     self.set_progress(50)

    # TODO load computing profile
    def load_profile(self):
        print('load computing profile')


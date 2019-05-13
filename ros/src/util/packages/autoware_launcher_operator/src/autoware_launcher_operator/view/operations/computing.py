# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwComputingWidget(QtWidgets.QWidget):

    def __init__(self, guimgr):
        super(AwComputingWidget, self).__init__()
        self.load_profile()

        # run computing button
        self.run_button = QtWidgets.QPushButton('Run Computing')
        self.run_button.clicked.connect(self.on_run_clicked)

        #  exit computing button
        self.exit_button = QtWidgets.QPushButton('exit Map')
        self.exit_button.clicked.connect(self.on_exit_clicked)

        #  show nodes
        self.set_node_labels_layout()

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.run_button)
        layout.addWidget(self.exit_button)
        layout.addLayout(self.node_labels_layout)
        self.setLayout(layout)

    def on_run_clicked(self):
        print('on run computing clicked')

    def on_exit_clicked(self):
        print('on exit clicked')

    def on_cancel_clicked(self):
        print('on load map cancel clicked')
        self.set_progress(50)

    def load_profile(self):
        self.nodes = [
            Computing(name='Localization', status=True),
            Computing(name='Detection', status=True),
            Computing(name='Tracking', status=True),
            Computing(name='Pedestrian', status=True),
            Computing(name='Decision', status=True),
        ]

    def set_node_labels_layout(self):
        self.node_labels_layout = QtWidgets.QVBoxLayout()
        for n in self.nodes:
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(QtWidgets.QLabel(n.name))
            hbox.addWidget(QtWidgets.QLabel('On' if n.status else 'Off'))
            self.node_labels_layout.addLayout(hbox)


class Computing(object):
    def __init__(self, **kargs):
        self.name = kargs.get('name', 'computing')
        self.status = kargs.get('status', True)

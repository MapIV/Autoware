# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwLoadMapWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwLoadMapWidget, self).__init__()
        self.context = context

        # load map button
        self.load_button = QtWidgets.QPushButton('Load Map')
        self.load_button.clicked.connect(self.on_load_clicked)

        #  unmap button
        self.unload_button = QtWidgets.QPushButton('Unload Map')
        self.unload_button.clicked.connect(self.on_unload_clicked)

        # progress bar
        self.pbar = QtWidgets.QProgressBar()

        #  cancel button
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        sub_layout1 = QtWidgets.QHBoxLayout()
        sub_layout2 = QtWidgets.QHBoxLayout()

        sub_layout1.addWidget(self.load_button)
        sub_layout1.addWidget(self.unload_button)
        sub_layout2.addWidget(self.pbar)
        sub_layout2.addWidget(self.cancel_button)

        layout.addLayout(sub_layout1)
        layout.addLayout(sub_layout2)
        self.setLayout(layout)

    def on_load_clicked(self):
        print('load ' + self.context.map_profile)
        self.set_progress(100)

    def on_unload_clicked(self):
        print('on unload map clicked')
        self.set_progress(0)

    def on_cancel_clicked(self):
        print('on load map cancel clicked')
        self.set_progress(50)

    def set_progress(self, val):
        # val: 0 to 100
        self.pbar.setValue(val)

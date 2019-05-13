# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwLoadComputingProfileWidget(QtWidgets.QWidget):

    def __init__(self, guimgr):
        super(AwLoadComputingProfileWidget, self).__init__()

        # pull down menu
        self.pdMenu = QtWidgets.QComboBox()
        self.set_pdMenu_contents()
        self.pdMenu.currentTextChanged.connect(self.on_file_selected)

        # button
        self.button = QtWidgets.QPushButton('Create profile')
        self.button.clicked.connect(self.create_profile)

        # set layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.pdMenu)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def set_pdMenu_contents(self):
        self.pdMenu.addItems(['dummy1', 'dummy2'])

    def on_file_selected(self, text):
        print('select computing profile: ' + text)

    def create_profile(self):
        print('create computing profile')

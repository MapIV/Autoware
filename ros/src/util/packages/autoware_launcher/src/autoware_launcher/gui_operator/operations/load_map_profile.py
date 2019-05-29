# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwLoadMapProfileWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwLoadMapProfileWidget, self).__init__()
        self.context = context

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
        self.pdMenu.addItems(self.context.map_profile_list)

        # load default map
        dirpath = "operator/maps/" + self.pdMenu.currentText()
        self.context.server.load_profile_subtree(dirpath, "root/map")

    def on_file_selected(self, text):
        print('select map profile: ' + text)
        self.context.set_map_profile(text)

    def create_profile(self):
        print('create map profile')

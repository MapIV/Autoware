# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwMainViewWidget(QtWidgets.QWidget):

    def __init__(self, guimgr):
        super(AwMainViewWidget, self).__init__()

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel('Main View')
        layout.addWidget(self.label)

        self.setLayout(layout)

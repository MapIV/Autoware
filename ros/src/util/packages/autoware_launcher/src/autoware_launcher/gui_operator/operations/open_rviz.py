# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class AwOpenRvizWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwOpenRvizWidget, self).__init__()
        self.context = context

        self.button = QtWidgets.QPushButton('Open Rviz', self)
        self.button.clicked.connect(self.onclicked)

    def onclicked(self):
        print('open rviz')
        self.context.server.launch_node("root/visualization", True)

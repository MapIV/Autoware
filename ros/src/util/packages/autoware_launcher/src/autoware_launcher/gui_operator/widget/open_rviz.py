# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets


class OpenRvizWidget(QtWidgets.QPushButton):

    def __init__(self, context):
        super(OpenRvizWidget, self).__init__()
        self.context = context

        self.setText('Open Rviz')
        self.clicked.connect(self.on_clicked)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def on_clicked(self):
        print('open rviz')
        self.context.server.launch_node("root/visualization", True)

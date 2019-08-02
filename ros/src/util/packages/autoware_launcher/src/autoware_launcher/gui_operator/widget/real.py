# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from ..plugins.basic import AwNodeListWidget, QHLine
from .profile_runner_list import ProfileRunnerListWidget
from .open_rviz import OpenRvizWidget


class RealWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(RealWidget, self).__init__()
        self.context = context

        self.profile_runner_list = ProfileRunnerListWidget(self.context)
        self.profile_runner_list.append('Map', run_text='Load', stop_text='Unload')
        self.profile_runner_list.get('Map').set_dirpath('operator/maps')
        self.profile_runner_list.append('Sensing')
        self.profile_runner_list.append('Localization')
        self.profile_runner_list.append('Perception')
        self.profile_runner_list.append('Actuation')
        self.profile_runner_list.append('Planning')

        self.open_rviz = OpenRvizWidget(self.context)
        self.open_rviz.setMinimumHeight(50)

        # addjust widget size
        self.profile_runner_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.profile_runner_list)
        layout.addWidget(self.open_rviz)
        layout.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(layout)

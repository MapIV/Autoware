# from python_qt_binding import QtCore
# from python_qt_binding import QtGui
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

# from ..core import myutils
# from ..core import AwLaunchClientIF

from .operations import AwMainViewWidget
from .operations import AwModeSelectWidget
from .operations import AwLoadMapProfileWidget
from .operations import AwLoadComputingProfileWidget
from .operations import AwLoadMapWidget
from .operations import AwComputingWidget
from .operations import AwOpenRvizWidget
from .operations import AwToggleLoggingWidget
from .operations import AwToggleGatewayWidget

from ..context import Context


class AwFieldOperatorPanel(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwFieldOperatorPanel, self).__init__()
        self.context = context

        # self.awlogo = QtWidgets.QLabel()
        # pixmap = QtGui.QPixmap(myutils.package("resources/autoware_logo.png"))
        # self.awlogo.setPixmap(pixmap)
        # self.awlogo.setAlignment(QtCore.Qt.AlignCenter)
        # self.awlogo.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QGridLayout()

        self.main_view = AwMainViewWidget(self.context)
        self.mode_select = AwModeSelectWidget(self.context)
        self.load_map_profile = AwLoadMapProfileWidget(self.context)
        self.load_computing_profile = AwLoadComputingProfileWidget(self.context)
        self.load_map = AwLoadMapWidget(self.context)
        self.computing = AwComputingWidget(self.context)
        self.open_rviz = AwOpenRvizWidget(self.context)
        self.logging = AwToggleLoggingWidget(self.context)
        self.gateway = AwToggleGatewayWidget(self.context)

        # connect widgets
        self.mode_select.set_select_real_mode_callback(self.main_view.select_real)
        self.mode_select.set_select_rosbag_mode_callback(self.main_view.select_rosbag)
        self.mode_select.set_select_sim_mode_callback(self.main_view.select_sim)

        # layout.addWidget(self.awlogo,                  0, 0,  2,  4)
        layout.addWidget(self.mode_select,    2, 0, 11,  4)
        layout.addWidget(self.load_map_profile,       13, 0,  1,  8)
        layout.addWidget(self.load_computing_profile, 14, 0,  1,  8)
        layout.addWidget(self.main_view,           0, 4, 13,  4)
        layout.addWidget(self.load_map,                0, 8,  3,  4)
        layout.addWidget(self.computing,               3, 8,  10,  4)
        layout.addWidget(self.open_rviz,               13, 8,  1, 4)
        layout.addWidget(self.logging,                 14, 8,  2, 2)
        layout.addWidget(self.gateway,                 14, 10,  2, 2)

        self.setLayout(layout)
        # self.set_style("background-color:red;")
        # self.set_style("#baseWidget {border:2px solid black;}")

    def set_style(self, style_str):
        self.main_view.setStyleSheet(style_str)
        self.mode_select.setStyleSheet(style_str)
        self.load_map_profile.setStyleSheet(style_str)
        self.load_computing_profile.setStyleSheet(style_str)
        self.load_map.setStyleSheet(style_str)
        self.computing.setStyleSheet(style_str)
        self.open_rviz.setStyleSheet(style_str)
        self.logging.setStyleSheet(style_str)
        self.gateway.setStyleSheet(style_str)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = AwFieldOperatorPanel(Context())
    w.show()
    sys.exit(app.exec_())

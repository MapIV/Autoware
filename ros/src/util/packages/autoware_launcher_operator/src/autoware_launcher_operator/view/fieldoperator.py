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


class AwFieldOperatorPanel(QtWidgets.QWidget):

    def __init__(self, guimgr):
        super(AwFieldOperatorPanel, self).__init__()
        self.guimgr = guimgr

        # self.awlogo = QtWidgets.QLabel()
        # pixmap = QtGui.QPixmap(myutils.package("resources/autoware_logo.png"))
        # self.awlogo.setPixmap(pixmap)
        # self.awlogo.setAlignment(QtCore.Qt.AlignCenter)
        # self.awlogo.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QGridLayout()

        self.main_view = AwMainViewWidget(self.guimgr)
        self.mode_select = AwModeSelectWidget(self.guimgr)
        self.load_map_profile = AwLoadMapProfileWidget(self.guimgr)
        self.load_computing_profile = AwLoadComputingProfileWidget(self.guimgr)
        self.load_map = AwLoadMapWidget(self.guimgr)
        self.computing = AwComputingWidget(self.guimgr)
        self.open_rviz = AwOpenRvizWidget(self.guimgr)
        self.logging = AwToggleLoggingWidget(self.guimgr)
        self.gateway = AwToggleGatewayWidget(self.guimgr)

        # self.set_style("background-color:red;")
        # self.set_style("#baseWidget {border:2px solid black;}")

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
    w = AwFieldOperatorPanel('dummy')
    w.show()
    sys.exit(app.exec_())

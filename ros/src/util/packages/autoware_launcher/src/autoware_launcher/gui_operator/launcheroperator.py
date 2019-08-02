# from python_qt_binding import QtCore
# from python_qt_binding import QtGui
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

from autoware_launcher.core import myutils
from .widget import MainViewWidget
from .widget import StatusViewWidget
from .plugins.basic import QImage

from .context import Context


class LauncherOperatorPanel(QtWidgets.QWidget):

    def __init__(self, context):
        super(LauncherOperatorPanel, self).__init__()
        self.context = context

        self.awlogo = QImage(myutils.package("resources/autoware_logo.png"))
        self.main_view = MainViewWidget(self.context)
        self.status_view = StatusViewWidget(self.context)

        layout = QtWidgets.QGridLayout()
        # args of addWidget: widget, fromRow, fromColumn, rowSpan, columnSpan
        layout.addWidget(self.main_view, 0, 0, 10, 5)
        layout.addWidget(self.awlogo, 0, 5, 1, 5)
        layout.addWidget(self.status_view, 1, 5, 9, 5)

        self.setLayout(layout)

        # connect widgets
        self.main_view.real_tab.profile_runner_list.get('Map').set_run_callback(self.status_view.complete_map_progress)
        self.main_view.real_tab.profile_runner_list.get('Map').set_stop_callback(self.status_view.reset_map_progress)
        self.main_view.set_select_real_callback(self.status_view.show_real_mode)
        self.main_view.set_select_rosbag_callback(self.status_view.show_rosbag_mode)

        self.set_style()

    # TODO load css file
    def set_style(self):
        style = '''
            QPushButton{
                background: #164886; 
                color: #ffffff;
            }
            QPushButton:hover{
                background: #265896; 
                color: #ffffff;
            }
            QPushButton:pressed{
                background: #ffffff; 
                color: #000000;
            }
        '''
        self.setStyleSheet(style)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = LauncherOperatorPanel(Context())
    w.show()
    sys.exit(app.exec_())

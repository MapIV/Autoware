# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from .real import RealWidget
from .rosbag import RosbagWidget

class MainViewWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(MainViewWidget, self).__init__()
        self.context = context

        # tabs
        self.real_tab = RealWidget(context)
        self.rosbag_tab = RosbagWidget(context)
        self.select_real_callback = lambda *arg: arg
        self.select_rosbag_callback = lambda *arg: arg

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.real_tab, 'REAL')
        self.tabs.addTab(self.rosbag_tab, 'ROSBAG')
        self.tabs.currentChanged.connect(self.tab_changed)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def set_select_real_callback(self, func):
        self.select_real_callback = func
    
    def set_select_rosbag_callback(self, func):
        self.select_rosbag_callback = func
    
    def tab_changed(self, idx):
        if idx == 0:    # real_tab
            self.select_real_callback()
        elif idx == 1:    # rosbag_tab
            self.select_rosbag_callback()

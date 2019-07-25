# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from autoware_launcher.core import myutils
from autoware_launcher.core.launch import AwLaunchNode
from autoware_launcher.gui_operator.plugins.basic import QToggleImage


class AwToggleGatewayWidget(QtWidgets.QWidget):
    TARGET_NODE = 'root/vehicle'

    def __init__(self, context):
        super(AwToggleGatewayWidget, self).__init__()
        self.context = context

        layout = QtWidgets.QVBoxLayout()

        self.label = QtWidgets.QLabel('Gateway')
        layout.addWidget(self.label)

        self.switch = QToggleImage(myutils.package('resources/toggle_off.png'), myutils.package('resources/toggle_on.png'))
        self.switch.switchedOn.connect(self.switchedOn)
        self.switch.switchedOff.connect(self.switchedOff)
        layout.addWidget(self.switch)

        self.context.register_node_status_watcher_client(self)

        self.setLayout(layout)

    def switchedOn(self):
        print('gateway switch on')
        if self.context.node_status_watcher.nodes.get(self.TARGET_NODE, None) != AwLaunchNode.EXEC:
            self.context.server.launch_node(self.TARGET_NODE, True)

    def switchedOff(self):
        print('gateway switch off')
        if self.context.node_status_watcher.nodes.get(self.TARGET_NODE, None) != AwLaunchNode.STOP:
            self.context.server.launch_node(self.TARGET_NODE, False)
    
    # this function is called by NodeStatusWatcher
    def node_status_updated(self, nodes):
        if nodes.get(self.TARGET_NODE, None) == AwLaunchNode.STOP:
            self.switch.set_value(0)
        elif nodes.get(self.TARGET_NODE, None) == AwLaunchNode.EXEC:
            self.switch.set_value(1)



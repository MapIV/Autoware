# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from .profile_runner import ProfileRunnerWidget

class ProfileRunnerListWidget(QtWidgets.QWidget):
    def __init__(self, context):
        super(ProfileRunnerListWidget, self).__init__()
        self.context = context
        self.pr_list = []

        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)

    def append(self, name, run_text='Run', stop_text='Exit'):
        pr = ProfileRunnerWidget(self.context, name, run_text, stop_text)
        self.pr_list.append(pr)
        self.layout().addWidget(pr)
    
    def get(self, name):
        for pr in self.pr_list:
            if pr.name == name:
                return pr
        return None
# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import os


class ProfileRunnerWidget(QtWidgets.QWidget):

    def __init__(self, context, name, run_text='Run', stop_text='Exit'):
        super(ProfileRunnerWidget, self).__init__()
        self.context = context
        self.name = name
        self.dirpath = os.path.join(self.context.dirpath, name.lower())
        self.node = os.path.join(self.context.nodepath, name.lower())
        self.run_callback = lambda *arg: arg
        self.stop_callback = lambda *arg: arg
        self.running = False

        # name label
        self.name_label = QtWidgets.QLabel(name)

        # pull down menu
        self.pdmenu = QtWidgets.QComboBox()
        self.pdmenu.currentTextChanged.connect(self.on_profile_changed)

        # run button
        self.run_button = QtWidgets.QPushButton(run_text)
        self.run_button.clicked.connect(self.on_run_clicked)

        # stop button
        self.stop_button = QtWidgets.QPushButton(stop_text)
        self.stop_button.clicked.connect(self.on_stop_clicked)

        # addjust widget size
        self.name_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.pdmenu.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.run_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.stop_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # set layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.name_label, 1)
        layout.addWidget(self.pdmenu, 2)
        layout.addWidget(self.run_button, 1)
        layout.addWidget(self.stop_button, 1)
        self.setLayout(layout)

        # load profile list
        self.update_pdmenu()
        # self.update_buttons()
    
    def set_dirpath(self, val, update_gui=True):
        self.dirpath = val
        if update_gui:
            self.update_pdmenu()

    def set_node(self, val):
        self.node = val
    
    def set_run_callback(self, f):
        self.run_callback = f

    def set_stop_callback(self, f):
        self.stop_callback = f

    def update_pdmenu(self):
        self.context.load_profile_list(self.dirpath)
        self.pdmenu.clear()
        self.pdmenu.addItems(self.context.profile_list[self.dirpath])

    def on_profile_changed(self, text):
        print('select {} profile: {}'.format(self.name, text))
        self.context.set_selected_profile(self.dirpath, text)

    def on_run_clicked(self):
        print('run {}'.format(self.node))
        dirpath = os.path.join(self.dirpath, self.context.selected_profile[self.dirpath])
        self.context.server.load_profile_subtree(dirpath, self.node)
        self.context.server.launch_node(self.node, True)
        self.run_callback()

        self.running = True
        # self.update_buttons()

    def on_stop_clicked(self):
        if self.running:
            print('stop {}'.format(self.node))
            self.context.server.launch_node(self.node, False)
            self.stop_callback()

            self.running = False
            # self.update_buttons()
        
    def update_buttons(self):
        if self.running:
            self.run_button.setHidden(True)
            self.stop_button.setHidden(False)
        else:
            self.run_button.setHidden(False)
            self.stop_button.setHidden(True)

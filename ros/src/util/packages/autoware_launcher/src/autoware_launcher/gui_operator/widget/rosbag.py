# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from .profile_runner_list import ProfileRunnerListWidget
from .open_rviz import OpenRvizWidget
from .basic import LabeledLineEdit
from .basic import HLine

class RosbagWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(RosbagWidget, self).__init__()
        self.context = context

        self.rate = 1
        self.offset = 0

        self.rosbag_path = ""
        self.repeat_rosbag = False
        self.use_sim_time = True
        self.progress_rate = 0

        self.rosbag_info_proc = QtCore.QProcess(self)
        self.rosbag_play_proc = QtCore.QProcess(self)
        self.rosparam_use_sim_time_proc = QtCore.QProcess(self)

        self.rosbag_info_proc.finished.connect(self.rosbag_info_completed)
        self.rosbag_play_proc.finished.connect(self.rosbag_finished)

        # button
        self.open_rosbag_btn = QtWidgets.QPushButton('Open Rosbag')
        self.open_rosbag_btn.clicked.connect(self.open_rosbag_btn_clicked)
        self.play_btn = QtWidgets.QPushButton('Play')
        self.play_btn.clicked.connect(self.play_btn_clicked)
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_btn_clicked)
        self.pause_btn = QtWidgets.QPushButton('Pause')
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self.pause_btn_clicked)

        # text area
        self.textarea = QtWidgets.QPlainTextEdit()
        self.textarea.setReadOnly(True)
        self.set_text('rosbag info here')

        # line edit
        self.rate_lineedit = LabeledLineEdit('Rate', '(x)')
        self.rate_lineedit.textChanged.connect(self.rate_changed)
        self.rate_lineedit.set_text(self.rate)
        self.offset_lineedit = LabeledLineEdit('Offset', '(s)')
        self.offset_lineedit.textChanged.connect(self.offset_changed)
        self.offset_lineedit.set_text(self.offset)

        # progress bar
        self.pbar = QtWidgets.QProgressBar()
        self.set_progress(0)

        # check box
        self.repeat_cbox = QtWidgets.QCheckBox('Repeat')
        self.repeat_cbox.stateChanged.connect(self.update_repeat_status)
        self.use_sim_time_cbox = QtWidgets.QCheckBox('Use sim time')
        self.use_sim_time_cbox.stateChanged.connect(self.update_use_sim_time_status)

        # profile runner
        self.profile_runner_list = ProfileRunnerListWidget(self.context)
        self.profile_runner_list.append('Map', run_text='Load', stop_text='Unload')
        self.profile_runner_list.get('Map').set_dirpath('operator/maps/')
        self.profile_runner_list.append('Localization')
        self.profile_runner_list.append('Perception')
        self.profile_runner_list.append('Planning')

        # open rviz button
        self.open_rviz = OpenRvizWidget(self.context)

        # hline
        self.hline = HLine()

        # addjust widget size
        self.open_rosbag_btn.setMinimumHeight(50)
        self.textarea.setMaximumHeight(300)
        self.rate_lineedit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.offset_lineedit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.profile_runner_list.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.open_rviz.setMinimumHeight(50)

        # set layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.open_rosbag_btn)
        layout.addWidget(self.textarea)
        sub_layout1 = QtWidgets.QHBoxLayout()
        sub_layout1.addWidget(self.rate_lineedit)
        sub_layout1.addWidget(self.offset_lineedit)
        sub_layout2 = QtWidgets.QHBoxLayout()
        sub_layout2.addWidget(self.play_btn)
        sub_layout2.addWidget(self.stop_btn)
        sub_layout2.addWidget(self.pause_btn)
        sub_layout3 = QtWidgets.QHBoxLayout()
        sub_layout3.addWidget(self.repeat_cbox)
        sub_layout3.addWidget(self.use_sim_time_cbox)
        layout.addLayout(sub_layout1)
        layout.addLayout(sub_layout3)
        layout.addLayout(sub_layout2)
        layout.addWidget(self.pbar)
        layout.addWidget(self.hline)
        layout.addWidget(self.profile_runner_list)
        layout.addWidget(self.open_rviz)
        layout.setAlignment(QtCore.Qt.AlignTop)
        self.setLayout(layout)

        # initialize check box
        if self.repeat_rosbag:
            self.repeat_cbox.setChecked(QtCore.Qt.Checked)
        else:
            self.repeat_cbox.setChecked(QtCore.Qt.Unchecked)
        if self.use_sim_time:
            self.use_sim_time_cbox.setChecked(QtCore.Qt.Checked)
            self.update_use_sim_time_status(QtCore.Qt.Checked)
        else:
            self.use_sim_time_cbox.setChecked(QtCore.Qt.Unchecked)
            self.update_use_sim_time_status(QtCore.Qt.Unchecked)

    def open_rosbag_btn_clicked(self):
        filepath, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Select Rosbag File", self.context.userhome_path)
        if filepath:
            self.rosbag_path = filepath
            self.rosbag_info_proc.start("rosbag info " + self.rosbag_path)
            print('open rosbag file: ' + self.rosbag_path)

    def play_btn_clicked(self):
        xml = self.context.rosbag_play_xml
        # if self.repeat_rosbag:
        #     option = "--loop --clock" + " --rate=" + str(self.rate) + " --start=" + str(self.offset)
        # else:
        #     option = "--clock" + " --rate=" + str(self.rate) + " --start=" + str(self.offset)
        option = "--rate=" + str(self.rate) + " --start=" + str(self.offset)
        if self.repeat_rosbag:
            option += " --loop"
        if self.use_sim_time:
            option += " --clock"
        arg = self.rosbag_path
        self.rosbag_play_proc.start('roslaunch {} options:="{}" bagfile:={}'.format(xml, option, arg))
        self.rosbag_play_proc.processId()
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.set_progress(0)
        self.rosbag_play_proc.readyReadStandardOutput.connect(self.update_progress)

    def stop_btn_clicked(self):
        self.rosbag_play_proc.terminate()
        self.rosbag_finished()

    def pause_btn_clicked(self):
        self.rosbag_play_proc.write(" ")
    
    def rosbag_finished(self):
        self.progress_rate = 0
        self.set_progress(100)
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setChecked(False)

    def set_text(self, txt):
        self.textarea.setPlainText(txt)

    def set_progress(self, val):
        # val: 0 to 100
        self.pbar.setValue(val)

    def update_repeat_status(self, state):
        if state == QtCore.Qt.Checked:
            self.repeat_rosbag = True
        else:
            self.repeat_rosbag = False

    def update_use_sim_time_status(self, state):
        if state == QtCore.Qt.Checked:
            self.use_sim_time = True
            self.rosparam_use_sim_time_proc.start("rosparam set /use_sim_time true")
        else:
            self.use_sim_time = False
            self.rosparam_use_sim_time_proc.start("rosparam set /use_sim_time false")

    def rate_changed(self, val):
        self.rate = val

    def offset_changed(self, val):
        self.offset = val

    def rosbag_info_completed(self):
        stdout = self.rosbag_info_proc.readAllStandardOutput().data().decode('utf-8')
        stderr = self.rosbag_info_proc.readAllStandardOutput().data().decode('utf-8')
        self.set_text(stdout + stderr)

    def update_progress(self):
        text = self.rosbag_play_proc.readAllStandardOutput().data()

        # parse text to show progresss
        # text will be like " [RUNNING]  Bag Time: 1427157668.783592   Duration: 0.000000 / 479.163620"
        l = text.find("Duration: ")
        if l != -1:
            nums = text[l:].split()    # ["Duration:", "0.000000", "/", "479.163620", ...]
            self.progress_rate = 100 * float(nums[1])/float(nums[3])
            self.set_progress(self.progress_rate)

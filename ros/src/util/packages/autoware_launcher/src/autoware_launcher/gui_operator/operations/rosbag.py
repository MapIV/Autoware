# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from ..plugins.basic import AwLabeledLineEdit
from ..plugins.basic import AwThread

class AwRosbagSimulatorWidget(QtWidgets.QWidget):

    def __init__(self, context):
        super(AwRosbagSimulatorWidget, self).__init__()
        self.context = context

        self.rate = 1
        self.offset = 0

        self.rosbag_path = ""
        self.repeat_rosbag = False
        self.progress_rate = 0

        self.rosbag_info_proc = QtCore.QProcess(self)
        self.rosbag_play_proc = QtCore.QProcess(self)
        self.rosbag_progress_proc = QtCore.QProcess(self)
        self.rosbag_start_time_proc = QtCore.QProcess(self)
        self.rosbag_end_time_proc = QtCore.QProcess(self)

        self.rosbag_info_proc.finished.connect(self.rosbag_info_completed)
        self.rosbag_play_proc.finished.connect(self.rosbag_finished)
        self.rosbag_progress_proc.finished.connect(self.rosbag_progress_completed)
        self.rosbag_start_time_proc.finished.connect(self.rosbag_start_time_completed)
        self.rosbag_end_time_proc.finished.connect(self.rosbag_end_time_completed)

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
        self.rate_lineedit = AwLabeledLineEdit('Rate', '(x)')
        self.rate_lineedit.textChanged.connect(self.rate_changed)
        self.rate_lineedit.set_text(self.rate)
        self.offset_lineedit = AwLabeledLineEdit('Offset', '(s)')
        self.offset_lineedit.textChanged.connect(self.offset_changed)
        self.offset_lineedit.set_text(self.offset)

        # progress bar
        self.pbar = QtWidgets.QProgressBar()
        self.set_progress(0)
        self.thread = AwThread(self, period=2.0)
        self.thread.update.connect(self.progress_update)

        # check box
        self.checkbox = QtWidgets.QCheckBox('Repeat')
        self.checkbox.stateChanged.connect(self.update_repeat_status)

        # set layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.open_rosbag_btn)
        layout.addWidget(self.textarea)
        sub_layout1 = QtWidgets.QHBoxLayout()
        sub_layout1.addWidget(self.rate_lineedit)
        sub_layout1.addWidget(self.offset_lineedit)
        layout.addLayout(sub_layout1)
        sub_layout2 = QtWidgets.QHBoxLayout()
        sub_layout2.addWidget(self.play_btn)
        sub_layout2.addWidget(self.stop_btn)
        sub_layout2.addWidget(self.pause_btn)
        layout.addLayout(sub_layout2)
        layout.addWidget(self.pbar)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def open_rosbag_btn_clicked(self):
        filepath, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Select Rosbag File", self.context.userhome_path)
        if filepath:
            self.rosbag_path = filepath
            self.rosbag_info_proc.start("rosbag info " + self.rosbag_path)
            self.rosbag_start_time_proc.start("rosbag info -y --key=start " + self.rosbag_path)
            self.rosbag_end_time_proc.start("rosbag info -y --key=end " + self.rosbag_path)
            print('open rosbag file: ' + self.rosbag_path)

    def play_btn_clicked(self):
        xml = self.context.rosbag_play_xml      
        if self.repeat_rosbag:
            option = "--loop --clock" + " --rate=" + str(self.rate) + " --start=" + str(self.offset)
        else:
            option = "--clock" + " --rate=" + str(self.rate) + " --start=" + str(self.offset)
        arg = self.rosbag_path
        self.rosbag_play_proc.start('roslaunch {} options:="{}" bagfile:={}'.format(xml, option, arg))
        self.rosbag_play_proc.processId()
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.set_progress(0)
        self.thread.start()

    def stop_btn_clicked(self):
        self.rosbag_play_proc.terminate()
        self.rosbag_finished()

    def pause_btn_clicked(self):
        self.rosbag_play_proc.write(" ")
    
    def rosbag_finished(self):
        self.progress_rate = 0
        self.set_progress(100)
        self.thread.terminate()
        self.thread.wait()
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
        if state or state == QtCore.Qt.Checked:
            self.repeat_rosbag = True
        else:
            self.repeat_rosbag = False

    def rate_changed(self, val):
        self.rate = val

    def offset_changed(self, val):
        self.offset = val

    def progress_update(self):
        self.set_progress(self.progress_rate)
        self.rosbag_progress_proc.start("rostopic echo -n 1 /clock/clock/secs")

    def rosbag_progress_completed(self):
        time = self.rosbag_progress_proc.readAllStandardOutput().data()
        self.current_time = [int(s) for s in time.split() if s.isdigit()][0]
        self.progress_rate = 100*(self.current_time-self.start_time)/(self.end_time-self.start_time)

    def rosbag_info_completed(self):
        stdout = self.rosbag_info_proc.readAllStandardOutput().data().decode('utf-8')
        stderr = self.rosbag_info_proc.readAllStandardOutput().data().decode('utf-8')
        self.set_text(stdout + stderr)

    def rosbag_start_time_completed(self):
        self.start_time = int(self.rosbag_start_time_proc.readAllStandardOutput().data().split('.')[0])

    def rosbag_end_time_completed(self):
        self.end_time = int(self.rosbag_end_time_proc.readAllStandardOutput().data().split('.')[0])
# from python_qt_binding import QtCore
# from python_qt_binding import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

from autoware_launcher.core.launch import AwLaunchNode
import time


class Image(QtWidgets.QLabel):
    def __init__(self, img_path, width=200, height=200):
        super(Image, self).__init__()

        pixmap = QtGui.QPixmap(img_path)
        pixmap = pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio)

        self.setPixmap(pixmap)


class Images(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    def __init__(self, img_list, size=(100, 100), default=0):
        super(Images, self).__init__()

        self.cur_i = default
        self.pixamps = [QtGui.QPixmap(img) for img in img_list]

        self.update_img()
    
    # This function is called when mouse is pressed
    def mousePressEvent(self, event):
        self.clicked.emit()

    def update_img(self):
        self.setPixmap(self.pixamps[self.cur_i])

    def set_img(self, ind):
        if ind != self.cur_i and ind < len(self.pixamps):
            self.cur_i = ind
            self.update_img()


class HLine(QtWidgets.QFrame):
    def __init__(self):
        super(HLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class ToggleImage(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal()

    switchedOn = QtCore.pyqtSignal()
    switchedOff = QtCore.pyqtSignal()

    def __init__(self, off_img, on_img, size=(120, 120), default=0):
        super(ToggleImage, self).__init__()

        # size is (width, heigh)
        self._value = default
        self.off_img = QtGui.QPixmap(off_img).scaled(size[0], size[1], QtCore.Qt.KeepAspectRatio)
        self.on_img = QtGui.QPixmap(on_img).scaled(size[0], size[1], QtCore.Qt.KeepAspectRatio)

        self.update_img()
        self.clicked.connect(self.toggle)
    
    # This function is called when mouse is pressed
    def mousePressEvent(self, event):
        self.toggle()

    def update_img(self):
        if self._value == 0:
            self.setPixmap(self.off_img)
        else:
            self.setPixmap(self.on_img)

    def set_value(self, val):
        if val != self._value:
            self._value = val
            self.update_img()
            self.emit_switchedSignal()

    def emit_switchedSignal(self):
        if self._value == 1:
            self.switchedOn.emit()
        else:
            self.switchedOff.emit()

    def toggle(self):
        if self._value == 1:
            self.set_value(0)
        else:
            self.set_value(1)

    def isOn(self):
        if self._value == 1:
            return True
        else:
            return False


class LabeledLineEdit(QtWidgets.QWidget):
    textChanged = QtCore.pyqtSignal(str)

    def __init__(self, pre_text='', post_text=''):
        super(LabeledLineEdit, self).__init__()

        self.pre_label = QtWidgets.QLabel()
        self.set_pre_label_text(pre_text)

        self.post_label = QtWidgets.QLabel()
        self.set_post_label_text(post_text)

        self.lineedit = QtWidgets.QLineEdit()
        self.lineedit.textChanged.connect(self.emit_textChanged)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.pre_label)
        layout.addWidget(self.lineedit)
        layout.addWidget(self.post_label)

        self.setLayout(layout)

    def set_pre_label_text(self, txt):
        self.pre_label.setText(txt)

    def set_post_label_text(self, txt):
        self.post_label.setText(txt)

    def set_text(self, text):
        self.lineedit.setText(str(text))

    def emit_textChanged(self, text):
        self.textChanged.emit(text)


class NodeStatusList(QtWidgets.QWidget):

    def __init__(self, target, depth=0, ignore=None):
        # target is str or dict
        # str -> node_name (+ depth)
        # dict -> node_name: display_name
        super(NodeStatusList, self).__init__()
        self.nodes = []
        self.target = target
        self.depth = depth
        self.ignore = [] if ignore is None else ignore

        self.node_name_label = QtWidgets.QLabel()
        self.node_status_label = QtWidgets.QLabel()

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.node_name_label)
        layout.addWidget(self.node_status_label)

        self.setLayout(layout)
    
    def arrange_nodes(self):
        if isinstance(self.target, dict):
            arranged_nodes = []
            cur_nodes = {n['name']: n['status'] for n in self.nodes}
            for node_name, display_name in self.target.items():
                node = {'name': display_name, 'status': cur_nodes.get(node_name, 'Off')}
                arranged_nodes.append(node)
            self.nodes = arranged_nodes
        elif isinstance(self.target, str):
            self.nodes.sort(key=lambda x: x['name'])

    def update_node_list(self, nodes):
        self.nodes = nodes
        self.arrange_nodes()

        node_name_text = ''
        node_status_text = ''
        for n in self.nodes:
            node_name_text += n['name'] + '\n'
            node_status_text += n['status'] + '\n'
        self.node_name_label.setText(node_name_text)
        self.node_status_label.setText(node_status_text)
    
    def is_target_node(self, name):
        ns = name.split('/')
        
        if isinstance(self.target, dict):    # dict target
            return True if name in self.target else False
        elif isinstance(self.target, str):    # str target
            target = self.target.split('/')
            if len(ns) < len(target):
                return False
            
            if len(ns) > len(target) + self.depth:
                return False
            
            for i in range(len(target)):
                if ns[i] != target[i]:
                    return False
            return True

        return False
    
    # this function is called by NodeStatusWatcher
    def node_status_updated(self, nodes):
        target_nodes = []
        for name, val in nodes.items():
            if name in self.ignore:
                continue
            if not self.is_target_node(name):
                continue

            if val == AwLaunchNode.STOP:
                status = 'Off'
            elif val == AwLaunchNode.EXEC:
                status = 'On'
            elif val == AwLaunchNode.TERM:
                status = 'Terminating'
            else:
                status = 'N/A'
            
            target_nodes.append({'name': name, 'status': status})
                
        # print(target_nodes)
        self.update_node_list(target_nodes)


class Thread(QtCore.QThread):
    
    update = QtCore.pyqtSignal()
    
    def __init__(self, parent, period):
        super(Thread, self).__init__(parent)
        self.period = period
 
    def run(self):
        while True:
            time.sleep(self.period)
            self.update.emit()

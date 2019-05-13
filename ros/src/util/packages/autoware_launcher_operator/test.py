from autoware_launcher_operator.view.fieldoperator import AwFieldOperatorPanel
from PyQt5 import QtWidgets

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = AwFieldOperatorPanel('dummy')
    w.show()
    sys.exit(app.exec_())

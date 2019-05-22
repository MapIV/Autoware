from autoware_launcher_operator.view.fieldoperator import AwFieldOperatorPanel
from autoware_launcher_operator.context import Context
from PyQt5 import QtWidgets

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = AwFieldOperatorPanel(Context())
    w.show()
    sys.exit(app.exec_())

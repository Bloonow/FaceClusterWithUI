import sys

from PyQt5.QtWidgets import QApplication
from qt_material import apply_stylesheet

from ui.mainform import MainForm

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mf = MainForm()

    extra = {
        'font_family': 'Microsoft YaHei UI',
        'font_size': 16
    }
    apply_stylesheet(app, theme='dark_teal.xml', extra=extra)

    mf.show()
    sys.exit(app.exec_())

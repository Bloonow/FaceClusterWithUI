from PyQt5.QtWidgets import QWidget

from ui_autogen.logwindow import Ui_LogWindow


class LogForm(QWidget, Ui_LogWindow):
    def __init__(self, parent=None):
        super(LogForm, self).__init__(parent)
        self.setupUi(self)

    def append_log(self, content_str):
        self.textBrowser_train_log.append(content_str)

    def register_callback(self, callback_fn):
        self.close_callback = callback_fn

    def closeEvent(self, close_event):
        if self.close_callback is not None:
            self.close_callback(True)
        close_event.accept()

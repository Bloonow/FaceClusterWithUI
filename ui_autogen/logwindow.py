# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_autogen\logwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LogWindow(object):
    def setupUi(self, LogWindow):
        LogWindow.setObjectName("LogWindow")
        LogWindow.resize(800, 500)
        self.gridLayout = QtWidgets.QGridLayout(LogWindow)
        self.gridLayout.setObjectName("gridLayout")
        self.textBrowser_train_log = QtWidgets.QTextBrowser(LogWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser_train_log.sizePolicy().hasHeightForWidth())
        self.textBrowser_train_log.setSizePolicy(sizePolicy)
        self.textBrowser_train_log.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textBrowser_train_log.setObjectName("textBrowser_train_log")
        self.gridLayout.addWidget(self.textBrowser_train_log, 0, 0, 1, 1)

        self.retranslateUi(LogWindow)
        QtCore.QMetaObject.connectSlotsByName(LogWindow)

    def retranslateUi(self, LogWindow):
        _translate = QtCore.QCoreApplication.translate
        LogWindow.setWindowTitle(_translate("LogWindow", "详情"))

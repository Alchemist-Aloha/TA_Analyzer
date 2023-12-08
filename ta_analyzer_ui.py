# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ta_analyzer.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QHeaderView,
    QLabel, QLayout, QMainWindow, QPushButton,
    QSizePolicy, QStatusBar, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.NonModal)
        MainWindow.resize(1147, 741)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy1)
        self.centralwidget.setMinimumSize(QSize(0, 0))
        self.centralwidget.setMaximumSize(QSize(3820, 2160))
        self.centralwidget.setAutoFillBackground(False)
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout_main = QHBoxLayout()
        self.horizontalLayout_main.setSpacing(0)
        self.horizontalLayout_main.setObjectName(u"horizontalLayout_main")
        self.horizontalLayout_main.setSizeConstraint(QLayout.SetNoConstraint)
        self.verticalLayout_l = QVBoxLayout()
        self.verticalLayout_l.setSpacing(0)
        self.verticalLayout_l.setObjectName(u"verticalLayout_l")
        self.horizontalLayout_lt = QHBoxLayout()
        self.horizontalLayout_lt.setObjectName(u"horizontalLayout_lt")

        self.verticalLayout_l.addLayout(self.horizontalLayout_lt)

        self.horizontalLayout_lm = QHBoxLayout()
        self.horizontalLayout_lm.setObjectName(u"horizontalLayout_lm")

        self.verticalLayout_l.addLayout(self.horizontalLayout_lm)

        self.horizontalLayout_lb = QHBoxLayout()
        self.horizontalLayout_lb.setObjectName(u"horizontalLayout_lb")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout_lb.addWidget(self.label)


        self.verticalLayout_l.addLayout(self.horizontalLayout_lb)

        self.verticalLayout_l.setStretch(0, 8)
        self.verticalLayout_l.setStretch(1, 8)
        self.verticalLayout_l.setStretch(2, 1)

        self.horizontalLayout_main.addLayout(self.verticalLayout_l)

        self.verticalLayout_r = QVBoxLayout()
        self.verticalLayout_r.setSpacing(0)
        self.verticalLayout_r.setObjectName(u"verticalLayout_r")
        self.horizontalLayout_rt = QHBoxLayout()
        self.horizontalLayout_rt.setObjectName(u"horizontalLayout_rt")

        self.verticalLayout_r.addLayout(self.horizontalLayout_rt)

        self.horizontalLayout_table = QHBoxLayout()
        self.horizontalLayout_table.setObjectName(u"horizontalLayout_table")
        self.table_misc = QTableWidget(self.centralwidget)
        self.table_misc.setObjectName(u"table_misc")

        self.horizontalLayout_table.addWidget(self.table_misc)

        self.table_fit = QTableWidget(self.centralwidget)
        self.table_fit.setObjectName(u"table_fit")

        self.horizontalLayout_table.addWidget(self.table_fit)


        self.verticalLayout_r.addLayout(self.horizontalLayout_table)

        self.horizontalLayout_button1 = QHBoxLayout()
        self.horizontalLayout_button1.setObjectName(u"horizontalLayout_button1")
        self.button_load = QPushButton(self.centralwidget)
        self.button_load.setObjectName(u"button_load")

        self.horizontalLayout_button1.addWidget(self.button_load)

        self.button_bgcorr = QPushButton(self.centralwidget)
        self.button_bgcorr.setObjectName(u"button_bgcorr")

        self.horizontalLayout_button1.addWidget(self.button_bgcorr)

        self.button_tcorr = QPushButton(self.centralwidget)
        self.button_tcorr.setObjectName(u"button_tcorr")

        self.horizontalLayout_button1.addWidget(self.button_tcorr)

        self.mat_selector = QComboBox(self.centralwidget)
        self.mat_selector.setObjectName(u"mat_selector")

        self.horizontalLayout_button1.addWidget(self.mat_selector)

        self.button_glotaran = QPushButton(self.centralwidget)
        self.button_glotaran.setObjectName(u"button_glotaran")

        self.horizontalLayout_button1.addWidget(self.button_glotaran)


        self.verticalLayout_r.addLayout(self.horizontalLayout_button1)

        self.horizontalLayout_button2 = QHBoxLayout()
        self.horizontalLayout_button2.setObjectName(u"horizontalLayout_button2")
        self.button_spectra = QPushButton(self.centralwidget)
        self.button_spectra.setObjectName(u"button_spectra")

        self.horizontalLayout_button2.addWidget(self.button_spectra)

        self.button_traces = QPushButton(self.centralwidget)
        self.button_traces.setObjectName(u"button_traces")

        self.horizontalLayout_button2.addWidget(self.button_traces)

        self.button_fit = QPushButton(self.centralwidget)
        self.button_fit.setObjectName(u"button_fit")

        self.horizontalLayout_button2.addWidget(self.button_fit)

        self.trace_selector = QComboBox(self.centralwidget)
        self.trace_selector.setObjectName(u"trace_selector")

        self.horizontalLayout_button2.addWidget(self.trace_selector)


        self.verticalLayout_r.addLayout(self.horizontalLayout_button2)

        self.verticalLayout_r.setStretch(0, 8)
        self.verticalLayout_r.setStretch(1, 7)
        self.verticalLayout_r.setStretch(2, 1)
        self.verticalLayout_r.setStretch(3, 1)

        self.horizontalLayout_main.addLayout(self.verticalLayout_r)

        self.horizontalLayout_main.setStretch(0, 6)
        self.horizontalLayout_main.setStretch(1, 4)

        self.horizontalLayout.addLayout(self.horizontalLayout_main)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Transient Absorption Analyzer", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.button_load.setText(QCoreApplication.translate("MainWindow", u"load", None))
        self.button_bgcorr.setText(QCoreApplication.translate("MainWindow", u"bgcorr", None))
        self.button_tcorr.setText(QCoreApplication.translate("MainWindow", u"tcorr", None))
        self.button_glotaran.setText(QCoreApplication.translate("MainWindow", u"glotaran", None))
        self.button_spectra.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.button_traces.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
        self.button_fit.setText(QCoreApplication.translate("MainWindow", u"PushButton", None))
    # retranslateUi


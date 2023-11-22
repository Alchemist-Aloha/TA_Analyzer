import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
from ta_analyzer_ui import Ui_MainWindow  # Import the generated class from the generated file
from ta_analyzer_core import *

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the UI from the generated file
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Transient Absorption Analyzer")
        # Connect signals and slots or customize UI elements here
        
        # Initialize PlotWidgets
        self.canvasc = pg.GraphicsLayoutWidget()# contour
        self.canvass = pg.GraphicsLayoutWidget()# spectra
        self.canvask = pg.GraphicsLayoutWidget()# kinetic traces
        self.canvasc.setBackground('white')
        self.canvass.setBackground('white')
        self.canvask.setBackground('white')
        self.ui.horizontalLayout_rt.addWidget(self.canvasc)
        self.ui.horizontalLayout_lt.addWidget(self.canvass)   
        self.ui.horizontalLayout_lm.addWidget(self.canvask)  
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
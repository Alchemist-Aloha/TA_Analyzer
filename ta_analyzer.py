import sys
from PySide6.QtWidgets import QTableWidgetItem, QApplication, QMainWindow, QFileDialog,QMessageBox
import os
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
        self.init_plots()
        self.init_buttons()
        self.init_tables()
        
    def init_plots(self):
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
        
        # Add PlotWidget to the GraphicsLayoutWidget
        self.contour_plot = self.canvasc.addPlot()   
        # Add color bar
        color_map = pg.colormap.get('CET-I2')
        self.colorbar = pg.ColorBarItem(values = (-0.01,0.01))
        self.colorbar.setColorMap(color_map)
        self.canvasc.addItem(self.colorbar, colspan=1) 
    
    def init_buttons(self):  
        self.ui.button_load.clicked.connect(self.select_file) 
        self.ui.button_bgcorr.clicked.connect(self.bgcorr)
        
    def init_tables(self):  
        self.ui.table_misc.setRowCount(2)
        self.ui.table_misc.setColumnCount(10)
        self.ui.table_misc.setItem(0,0,QTableWidgetItem("bgcorr pts"))
        self.ui.table_misc.setItem(0,1,QTableWidgetItem("20"))
    
    def bgcorr(self):
        pts = int(self.ui.table_misc.item(0,1).text())
        self.obj_ta.auto_bgcorr(pts)
        self.plot_contour(mat = self.obj_ta.bgcorr)
    
    def select_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        self.file_path = QFileDialog.getOpenFileName(
        self, "Select File", os.getcwd(), options=options)

        if self.file_path:
            self.file_path = str(self.file_path[0])
            self.file_name = os.path.basename(self.file_path)
            self.file_name = self.file_name.split('.')[0]
            print("Selected file:", self.file_path)
            self.ui.label.setText("Current data file: "+self.file_path)
            self.dir = os.path.dirname(self.file_path)
            self.obj_ta = tamatrix_importer(self.file_path,0,1500)
            #self.sendto_table()
            self.plot_contour()
            #self.plot_data()
            
    def plot_contour(self,mat=None):
        if mat is None:
            mat = self.obj_ta.tamatrix
        y = np.insert(self.obj_ta.tatime,0,self.obj_ta.tatime[0])
        x = np.insert(self.obj_ta.tawavelength,0,self.obj_ta.tawavelength[0])
        Y, X = np.meshgrid(y, x)       
        self.contour_plot.clear()
        self.contour_plot_data = pg.PColorMeshItem(X,Y,mat,levels=5, autoLevels=False)
        self.contour_plot.addItem(self.contour_plot_data)
        
        '''# Create the contour plot using ImageItem()
        self.contour_plot_data = pg.ImageItem()
        self.contour_plot.addItem(self.contour_plot_data)
        self.contour_plot_data.setImage(self.obj_ta.tamatrix,autoLevels=True)
        #self.contour_plot_data.setRect(np.min(self.obj_ta.tawavelength), np.min(self.obj_ta.tatime), 
        #                                                np.max(self.obj_ta.tawavelength)-np.min(self.obj_ta.tawavelength), np.max(self.obj_ta.tatime)-np.min(self.obj_ta.tatime))
        self.contour_plot.setXRange(min(self.obj_ta.tawavelength), max(self.obj_ta.tawavelength))'''
        
        # Set labels and colormap
        self.contour_plot.setLabel('left', "Time (ps)")
        self.contour_plot.setLabel('bottom', "Wavelength (nm)")
        self.contour_plot.setAspectLocked(False)
        #self.contour_plot.showAxis('right')
        #self.contour_plot.showAxis('top')
        
        # Add color bar
        self.colorbar.setImageItem(self.contour_plot_data)
        
        # Show the GraphicsLayoutWidget
        self.canvasc.show()
        
def exception_hook(exctype, value, traceback):
    """
    Custom exception hook to handle uncaught exceptions.
    Display an error message box with the exception details.
    """
    msg = f"Unhandled exception: {exctype.__name__}\n{value}"
    QMessageBox.critical(None, "Unhandled Exception", msg)
    sys.__excepthook__(exctype, value, traceback)  # Call default exception hook

if __name__ == "__main__":
    app = QApplication(sys.argv)
    sys.excepthook = exception_hook
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
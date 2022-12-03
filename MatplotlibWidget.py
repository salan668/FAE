from PyQt5.QtWidgets import *
import matplotlib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MatplotlibWidget(QWidget):
    """·
    Implements a Matplotlib figure inside a QWidget.
    Use getFigure() and redraw() to interact with matplotlib.
    
    Example::
    
        mw = MatplotlibWidget()
        subplot = mw.getFigure().add_subplot(111)
        subplot.plot(x,y)
        mw.draw()
    """
    
    def __init__(self, parent=None, size=(5.0, 4.0), dpi=100):
        QWidget.__init__(self)
        super(MatplotlibWidget, self).__init__(parent)
        self.fig = Figure(size, dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)
        # self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.vbox = QVBoxLayout()
        # self.vbox.addWidget(self.toolbar)
        self.vbox.addWidget(self.canvas)
        
        self.setLayout(self.vbox)

    def getFigure(self):
        return self.fig
        
    def draw(self):
        self.canvas.draw()

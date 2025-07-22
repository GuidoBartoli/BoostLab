import math
import warnings

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHeaderView,
    QTableView,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")


def modify_font(obj, bold=False, italic=False, underline=False, mono=False):
    if mono:
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
    else:
        font = obj.font(0) if type(obj) is QTreeWidgetItem else obj.font()
    font.setBold(bold)
    font.setItalic(italic)
    font.setUnderline(underline)
    if type(obj) is QTreeWidgetItem:
        obj.setFont(0, font)
    elif type(obj) is QGroupBox and bold:
        obj.setStyleSheet("QGroupBox {font-weight: bold;}")
    else:
        obj.setFont(font)


class PandasView(QTableView):
    copied = Signal(str)

    class PandasModel(QAbstractTableModel):
        def __init__(self, data, index):
            QAbstractTableModel.__init__(self)
            self._data = data
            self._index = index

        def rowCount(self, parent=None):
            return self._data.shape[0]

        def columnCount(self, parent=None):
            return self._data.shape[1]

        def data(self, index, role):
            if role == Qt.DisplayRole:
                item = self._data.iloc[index.row(), index.column()]
                if isinstance(item, float):
                    # FIXME: Rounding is not performed on floating numbers (i.e. 0.3213124)
                    return str(int(item)) if item.is_integer() else str(round(item, 6))
                return str(item)

        def headerData(self, col, orientation, role):
            if role == Qt.DisplayRole:
                if orientation == Qt.Horizontal:
                    return str(self._data.columns[col])
                if orientation == Qt.Vertical and self._index:
                    return str(self._data.index[col])

    def __init__(
        self,
        parent=None,
        mono: bool = True,
        alternate: bool = False,
        index: bool = False,
    ):
        super().__init__(parent)
        self.setAlternatingRowColors(alternate)
        if mono:
            modify_font(self, mono=True)
        self._index = index
        self.doubleClicked.connect(self.copy_cell)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)

    def update_data(self, data: pd.DataFrame):
        self.setModel(self.PandasModel(data, self._index))
        for i in range(data.shape[0]):
            self.setRowHeight(i, 18)
        if math.sqrt(data.size) < 200:
            self.resizeColumnsToContents()
        else:
            for j in range(data.shape[1]):
                self.setColumnWidth(j, 85)

    def clear_data(self):
        self.setModel(self.PandasModel(pd.DataFrame(), self._index))

    def copy_cell(self, index):
        # FIXME: Floating values are copied with comma as decimal separator
        #        and unable to be pasted into line edits
        data = index.data()
        QApplication.clipboard().setText(data)
        self.copied.emit(data)


class MplWidget(QWidget):
    # TODO: Create separate axes for colorbar
    # TODO: Add parameter to specify subplots and access them by index

    class MplCanvas(FigureCanvas):
        def __init__(self, title, width, height, dpi):
            self.figure = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.figure.add_subplot(1, 1, 1)
            self.axes.set_title(title)
            self.figure.tight_layout()
            super().__init__(self.figure)

    def __init__(
        self, title="", width=8, height=6, dpi=100, toolbar=False, parent=None
    ):
        super().__init__(parent)
        self.canvas = self.MplCanvas(title, width, height, dpi)
        self.layout = QVBoxLayout()
        if toolbar:
            self.layout.addWidget(NavigationToolbar(self.canvas, self))
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.enable_ticks(False)

    def clear(self):
        self.canvas.axes.clear()
        self.refresh()

    def refresh(self):
        self.enable_ticks(True)
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def imshow(self, data, cmap, norm, interpolation):
        self.canvas.axes.imshow(
            data,
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
            aspect="auto",
        )

    def enable_ticks(self, enabled: bool):
        self.canvas.axes.tick_params(
            left=enabled,
            right=enabled,
            labelleft=enabled,
            labelbottom=enabled,
            bottom=enabled,
        )

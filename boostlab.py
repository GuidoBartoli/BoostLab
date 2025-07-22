import re
import sys

from PySide6.QtCore import QSettings, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QTabWidget
import resources

from modules.about import AboutWidget
from modules.dataset import DatasetWidget
from modules.evaluate import EvaluateWidget
from modules.model import ModelWidget

# VERSION HISTORY AND ROADMAP:
# -------------------------------------------------
#   0.1:  Training, tuning and deployment
#   0.2:  Model evaluation
#   0.3:  Dataset visualization
#   0.4:  Data preprocessing
#   0.5:  Parameter persistence
#   0.6:  Dataset statistics
#   0.7:  Heatmap customization
#   0.8:  Outlier detection
#   0.9:  Logitraw objective
#   0.10: Review metric usage
#   0.11: DET and Accuracy evaluation
# > 0.12: Rename AiLab to Boost-Lab
#   0.13: XGBoost 2.0 support
#   0.14: Tuning with K-Fold validation
#   0.15: Review feature importance
#   0.16: Multiclass booster support
#   0.17: Binclass package support
#   0.18: Manifold Learning support
#   0.19: Integrated documentation
# -------------------------------------------------
#   1.0:  Optimizations, bugfixes and refactoring
#   1.1:  Python 3.12 support
# -------------------------------------------------
#   2.x:  Add regressor support (classes = 0)
# -------------------------------------------------
#   3.x:  Alternative classifiers (LightGBM, ANN, SVM, ...)


class MainWindow(QTabWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        QApplication.setApplicationName("BoostLab")
        QApplication.setOrganizationName("ZCS Company")
        QApplication.setOrganizationDomain("https://www.zcscompany.com")
        QApplication.setApplicationVersion("0.12.1")
        QApplication.setWindowIcon(QIcon(":/boostlab"))
        self.setWindowTitle(
            f"{QApplication.applicationName()} {QApplication.applicationVersion()}"
        )

        dataset_widget = DatasetWidget(self)
        model_widget = ModelWidget(self)
        evaluate_widget = EvaluateWidget(self)
        about_widget = AboutWidget(self)

        self.addTab(dataset_widget, "Dataset")
        self.setTabIcon(0, QIcon(":/data"))
        self.addTab(model_widget, "Model")
        self.setTabIcon(1, QIcon(":/model"))
        self.addTab(evaluate_widget, "Evaluation")
        self.setTabIcon(2, QIcon(":/eval1"))
        self.addTab(about_widget, "About")
        self.setTabIcon(3, QIcon(":/about"))
        self.tabBar().setStyleSheet("font-weight: bold")

        dataset_widget.dataset_ready.connect(model_widget.enable_training)
        dataset_widget.dataset_ready.connect(evaluate_widget.set_dataset)
        dataset_widget.dataset_ready.connect(self.set_dataset)
        model_widget.model_ready.connect(evaluate_widget.set_model)
        model_widget.model_ready.connect(self.set_model)

        self.train = None
        self.test = None
        self.booster = None
        self.file_pattern = r"[a-zA-Z0-9\s\-\_\./]*"

        settings = QSettings()
        settings.beginGroup("main_window")
        if main_geometry := settings.value("geometry"):
            self.restoreGeometry(main_geometry)
        settings.endGroup()

    def closeEvent(self, event):
        settings = QSettings()
        settings.beginGroup("main_window")
        settings.setValue("geometry", self.saveGeometry())
        settings.endGroup()
        super(MainWindow, self).closeEvent(event)

    @Slot()
    def set_dataset(self, filename):
        # FIXME: Sometimes dataset name is incorrectly updated
        title = self.windowTitle()
        brackets = title.count("[")
        if brackets == 0:
            self.setWindowTitle(
                f"{QApplication.applicationName()} {QApplication.applicationVersion()} --- D[{filename}]"
            )
        else:
            pattern = re.compile(fr"\sD\[({self.file_pattern})\]")
            if match := pattern.search(title):
                title = title.replace(match[1], filename)
            else:
                title = title.replace("--- M[", f"--- D[{filename}] + M[")
            self.setWindowTitle(title)

    @Slot()
    def set_model(self, filename):
        title = self.windowTitle()
        brackets = title.count("[")
        if brackets == 0:
            self.setWindowTitle(
                f"{QApplication.applicationName()} {QApplication.applicationVersion()} --- M[{filename}]"
            )
        else:
            pattern = re.compile(fr"\sM\[({self.file_pattern})\]")
            if match := pattern.search(title):
                title = title.replace(match[1], filename)
            else:
                title += f" + M[{filename}]"
            self.setWindowTitle(title)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

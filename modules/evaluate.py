import contextlib
import itertools

from PySide6.QtCore import QSettings, Slot
from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lib import dataset, disk, evaluate, gui, model


class EvaluateWidget(QWidget):
    # TODO: Check consistency of dataset and model before evaluation (features should match)
    # TODO: Add option for logarithmic scale in probability distribution

    def __init__(self, parent):
        super().__init__(parent)

        self.file_types = {"csv": "CSV files (*.csv)"}

        self.parent = parent
        self.dataset_ready = False
        self.model_ready = False

        self.decimals = 6
        self.binthr_spin = QDoubleSpinBox()
        self.binthr_spin.setRange(-100, 100)
        self.binthr_spin.setSingleStep(0.01)
        self.binthr_spin.setDecimals(self.decimals)
        self.binthr_spin.setToolTip("Threshold used for binary classification")
        self.eval_button = QPushButton("Evaluate")
        self.eval_button.setIcon(QIcon(":/eval2"))
        self.eval_button.setShortcut(QKeySequence("Ctrl+E"))
        self.eval_button.clicked.connect(self.eval_model)
        self.eval_button.setToolTip("Evaluate the model on the validation set")
        gui.modify_font(self.eval_button, bold=True)
        self.select_combo = QComboBox()
        self.select_combo.addItems(["All samples", "Correct samples", "Wrong samples"])
        self.select_combo.setToolTip("Select the output samples to export")
        export_button = QPushButton("Export...")
        export_button.setIcon(QIcon(":/export"))
        export_button.clicked.connect(self.export_output)
        export_button.setToolTip("Export model predictions to a CSV file")

        eval_layout = QGridLayout()
        eval_layout.addWidget(self.eval_button, 0, 0)
        eval_layout.addWidget(QLabel("Threshold:"), 0, 1)
        eval_layout.addWidget(self.binthr_spin, 0, 2)
        eval_layout.addWidget(export_button, 1, 0)
        eval_layout.addWidget(QLabel("Selection:"), 1, 1)
        eval_layout.addWidget(self.select_combo, 1, 2)
        eval_group = QGroupBox()
        eval_group.setLayout(eval_layout)

        self.report_view = gui.PandasView()
        self.matrix_view = gui.PandasView()
        self.metrics_view = gui.PandasView()
        self.optthr_view = gui.PandasView()
        self.optthr_view.copied.connect(self.set_threshold)

        report_layout = QVBoxLayout()
        report_layout.addWidget(self.report_view)
        report_group = QGroupBox("CLASSIFICATION REPORT")
        report_group.setLayout(report_layout)
        report_group.setMaximumHeight(160)
        matrix_layout = QVBoxLayout()
        matrix_layout.addWidget(self.matrix_view)
        matrix_group = QGroupBox("CONFUSION MATRIX")
        matrix_group.setLayout(matrix_layout)
        matrix_group.setMaximumHeight(140)
        metrics_layout = QVBoxLayout()
        metrics_layout.addWidget(self.metrics_view)
        metrics_group = QGroupBox("PERFORMANCE METRICS")
        metrics_group.setLayout(metrics_layout)
        optthr_layout = QVBoxLayout()
        optthr_layout.addWidget(self.optthr_view)
        optthr_group = QGroupBox("OPTIMAL THRESHOLDS")
        optthr_group.setLayout(optthr_layout)
        optthr_group.setMaximumHeight(180)

        self.curves_widgets = [
            gui.MplWidget(),
            gui.MplWidget(),
            gui.MplWidget(),
            gui.MplWidget(),
            gui.MplWidget(),
            gui.MplWidget(),
        ]
        curves_layout = QGridLayout()
        for i, j in itertools.product(range(2), range(3)):
            curves_layout.addWidget(self.curves_widgets[i * 3 + j], i, j)
        curves_widget = QWidget()
        curves_widget.setLayout(curves_layout)

        self.hist_mpls = [gui.MplWidget(), gui.MplWidget()]
        hist_layout = QVBoxLayout()
        hist_layout.addWidget(self.hist_mpls[0])
        hist_layout.addWidget(self.hist_mpls[1])
        hist_widget = QWidget()
        hist_widget.setLayout(hist_layout)

        left_layout = QVBoxLayout()
        left_layout.addWidget(eval_group)
        left_layout.addWidget(report_group)
        left_layout.addWidget(matrix_group)
        left_layout.addWidget(metrics_group)
        left_layout.addWidget(optthr_group)
        self.left_widget = QWidget()
        self.left_widget.setLayout(left_layout)
        self.left_widget.setMinimumWidth(320)

        self.right_widget = QTabWidget()
        self.right_widget.addTab(curves_widget, "Metric curves")
        self.right_widget.setTabIcon(0, QIcon(":/curves"))
        self.right_widget.addTab(hist_widget, "Probability distribution")
        self.right_widget.setTabIcon(1, QIcon(":/hist"))
        self.right_widget.setMovable(True)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.left_widget)
        main_layout.addWidget(self.right_widget)
        self.setLayout(main_layout)

        # DEFAULT VALUES
        self.binthr_spin.setValue(0.5)
        self.select_combo.setCurrentIndex(0)
        self.left_widget.setEnabled(False)
        self.right_widget.setEnabled(False)

    def eval_model(self):
        if self.parent.booster is None:
            QMessageBox.warning(self, "Model evaluation", "No model loaded")
            return
        if self.parent.test is None:
            QMessageBox.warning(self, "Model evaluation", "No validation loaded")
            return
        self.eval_button.setEnabled(False)
        self.eval_button.setText("Evaluating...")
        gui.modify_font(self.eval_button, bold=False)
        QApplication.processEvents()

        success = True
        props = model.properties(self.parent.booster)
        try:
            prob_y = model.predict(
                self.parent.booster, self.parent.test, props["class_labels"]
            )
        except ValueError:
            QMessageBox.critical(
                self, "Model evaluation", "Dataset and model are incompatible!"
            )
            success = False
        self.eval_button.setText("Evaluate")
        gui.modify_font(self.eval_button, bold=True)
        self.eval_button.setEnabled(True)
        if not success:
            return

        binthr = self.binthr_spin.value()
        classes, labels = props["num_classes"], props["class_labels"]
        test_x, test_y = dataset.df2np(self.parent.test, labels)
        report, matrix, metrics = evaluate.performance(
            test_y, prob_y, binthr, classes, labels
        )
        if matrix is not None:
            self.matrix_view.update_data(matrix)
        self.report_view.update_data(report)
        self.metrics_view.update_data(metrics)
        if classes == 2:
            with contextlib.suppress(ValueError):
                optthr = evaluate.thresholds(
                    test_y, prob_y, binthr, self.curves_widgets
                )
                self.optthr_view.update_data(optthr)
            evaluate.histogram(prob_y, binthr, labels, self.hist_mpls)
            self.right_widget.setEnabled(True)
        else:
            for c in self.curves_widgets:
                c.clear()
            for c in self.hist_mpls:
                c.clear()
            self.right_widget.setEnabled(False)

    def export_output(self):
        if self.parent.booster is None:
            QMessageBox.warning(self, "Warning", "No model trained")
            return
        settings = QSettings()
        filename, extension = QFileDialog.getSaveFileName(
            self,
            "Save output",
            settings.value("save_output", "."),
            self.file_types["csv"],
        )
        if not filename:
            return
        settings.setValue("save_output", disk.get_folder(filename))
        props = model.properties(self.parent.booster)
        prob_y = model.predict(
            self.parent.booster, self.parent.test, props["class_labels"]
        )
        if prob_y is None:
            QMessageBox.critical(self, "Error", "Dataset and model are incompatible")
            return
        binthr = round(self.binthr_spin.value(), self.decimals)
        test_x, test_y = dataset.df2np(self.parent.test, props["class_labels"])
        props = dataset.properties(self.parent.test)
        mode = self.select_combo.currentText().lower().split()[0]
        try:
            evaluate.save_output(
                test_x,
                test_y,
                prob_y,
                binthr,
                props["labels"],
                props["columns"],
                filename,
                mode,
            )
            QMessageBox.information(self, "Save output", "Output saved successfully")
        except OSError:
            QMessageBox.critical(self, "Save output", "Could not save output!")

    @Slot()
    def set_dataset(self):
        self.dataset_ready = True
        if self.dataset_ready and self.model_ready:
            self.left_widget.setEnabled(True)

    @Slot()
    def set_model(self):
        self.model_ready = True
        if self.dataset_ready and self.model_ready:
            self.left_widget.setEnabled(True)

    @Slot()
    def set_threshold(self, threshold):
        with contextlib.suppress(ValueError):
            if threshold != "nan":
                self.binthr_spin.setValue(float(threshold))

import contextlib

import matplotlib

matplotlib.use("Qt5Agg")

from PySide6.QtCore import QSettings, Signal
from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from lib import dataset, disk, gui, misc

# TODO: Change colors of Andrews curves and Parallel Coordinates
# TODO: Add NearMiss algorithm version selection with a combo box
# TODO: Optimize filtered -> normalized -> ... pipeline using only one dataset


class DatasetWidget(QWidget):
    dataset_ready = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)

        self.setWindowTitle("Dataset")
        self.parent = parent

        self.original = None
        self.filtered = None
        self.normalized = None
        self.balanced = None
        self.ordered = None
        self.selection = None
        self.reduction = None
        self.stats = None
        self.max_values = int(5e7)

        self.file_types = {
            "dataset": "HDF5 files (*.h5 *.hdf5);;"
            "Pickle files (*.zip *.gz *.bz2 *.xz *.tar);;"
            "CSV files (*.csv);;Excel files (*.xls *.xlsx);;"
            "Dataset files (*.h5 *.hdf5 *.zip *.gz *.bz2 *.xz *.tar *.csv *.xls *.xlsx)",
            "csv": "CSV files (*.csv)",
        }

        # FILE
        load_button = QPushButton("Load...")
        load_button.setIcon(QIcon(":/load"))
        load_button.setShortcut(QKeySequence.Open)
        load_button.clicked.connect(self.load_dataset)
        load_button.setToolTip("Load a saved dataset from file")
        self.save_button = QPushButton("Save...")
        self.save_button.setIcon(QIcon(":/save"))
        self.save_button.setShortcut(QKeySequence.Save)
        self.save_button.clicked.connect(self.save_dataset)
        self.save_button.setToolTip("Save current dataset to file")
        import_button = QPushButton("Import...")
        import_button.setIcon(QIcon(":/import"))
        import_button.setShortcut(QKeySequence("Ctrl+I"))
        import_button.clicked.connect(self.import_folder)
        import_button.setToolTip(
            "Scan folder to import raw data (NOTE: filenames starting with '_' are ignored)"
        )

        self.suffix_radio = QRadioButton("Filename suffix")
        self.suffix_radio.setToolTip(
            "Use text after the last underscore as target label (i.e.: '00_data_class0.csv')"
        )
        self.column_radio = QRadioButton("Last column")
        self.column_radio.setToolTip("Use text in the last column as target label")
        self.csvsep_combo = QComboBox()
        self.csvsep_combo.addItems([",", ";", " "])
        self.csvsep_combo.setToolTip(
            "Character to treat as field delimiter for CSV files"
        )
        self.compress_spin = QSpinBox()
        self.compress_spin.setRange(0, 9)
        self.compress_spin.setSpecialValueText("OFF")
        self.compress_spin.setToolTip("HDF5 and Pickle data compression level")
        self.recursive_check = QCheckBox("Recursive folder scan")
        self.recursive_check.setToolTip("Scan subfolders recursively during import")
        self.header_check = QCheckBox("Use first row header")
        self.header_check.setToolTip("Use first row as column names during import")

        file1_layout = QGridLayout()
        file1_layout.addWidget(load_button, 0, 0)
        file1_layout.addWidget(import_button, 0, 1)
        file1_layout.addWidget(self.save_button, 0, 2)
        file1_layout.addWidget(QLabel("Targeting mode:"), 1, 0)
        file1_layout.addWidget(self.suffix_radio, 1, 1)
        file1_layout.addWidget(self.column_radio, 1, 2)
        file2_layout = QGridLayout()
        file2_layout.addWidget(QLabel("Compression:"), 0, 0)
        file2_layout.addWidget(self.compress_spin, 0, 1)
        file2_layout.addWidget(QLabel("CSV separator:"), 0, 2)
        file2_layout.addWidget(self.csvsep_combo, 0, 3)
        file2_layout.addWidget(self.recursive_check, 1, 0, 1, 2)
        file2_layout.addWidget(self.header_check, 1, 2, 1, 2)
        file_layout = QVBoxLayout()
        file_layout.addLayout(file1_layout)
        file_layout.addLayout(file2_layout)
        file_group = QGroupBox("FILE")
        file_group.setLayout(file_layout)

        # FILTER
        self.filter_check = QCheckBox("Enable")
        self.filter_check.stateChanged.connect(self.update_filtered)
        self.filter_check.setToolTip(
            "Enable/disable feature/class filtering and subsampling"
        )
        self.featsel_edit = QLineEdit()
        self.featsel_edit.editingFinished.connect(self.update_filtered)
        self.featsel_edit.setToolTip(
            "Feature selection (index based, comma separated, ranges allowed, i.e.: 0,3,5:10)"
        )
        self.featrem_check = QCheckBox("Remove")
        self.featrem_check.stateChanged.connect(self.update_filtered)
        self.featrem_check.setToolTip(
            "Feature removal mode (uncheck to keep only selected features)"
        )
        self.featren1_edit = QLineEdit()
        self.featren1_edit.editingFinished.connect(self.update_filtered)
        self.featren1_edit.setToolTip(
            "Old feature label(s) to be renamed (comma separated)"
        )
        self.featren2_edit = QLineEdit()
        self.featren2_edit.editingFinished.connect(self.update_filtered)
        self.featren2_edit.setToolTip("New feature label(s) (can be comma separated)")

        self.classsel_edit = QLineEdit()
        self.classsel_edit.editingFinished.connect(self.update_filtered)
        self.classsel_edit.setToolTip(
            "Class selection (can be comma separated, i.e.: 'cat, dog')"
        )
        self.classrem_check = QCheckBox("Remove")
        self.classrem_check.stateChanged.connect(self.update_filtered)
        self.classrem_check.setToolTip(
            "Class removal mode (uncheck to keep only selected classes)"
        )
        self.classren1_edit = QLineEdit()
        self.classren1_edit.editingFinished.connect(self.update_filtered)
        self.classren1_edit.setToolTip(
            "Old class label(s) to be renamed (can be comma separated)"
        )
        self.classren2_edit = QLineEdit()
        self.classren2_edit.editingFinished.connect(self.update_filtered)
        self.classren2_edit.setToolTip("New class label(s) (can be comma separated)")

        self.rowsub_spin = QSpinBox()
        self.rowsub_spin.setRange(1, 100)
        self.rowsub_spin.setSuffix("%")
        self.rowsub_spin.setSingleStep(5)
        self.rowsub_spin.setValue(100)
        self.rowsub_spin.valueChanged.connect(self.update_filtered)
        self.rowsub_spin.setToolTip("Percentage of samples (rows) to keep")
        self.colsub_spin = QSpinBox()
        self.colsub_spin.setRange(1, 100)
        self.colsub_spin.setSuffix("%")
        self.colsub_spin.setSingleStep(5)
        self.colsub_spin.setValue(100)
        self.colsub_spin.valueChanged.connect(self.update_filtered)
        self.colsub_spin.setToolTip("Percentage of features (columns) to keep")

        filter1_layout = QGridLayout()
        filter1_layout.addWidget(QLabel("Selection:"), 0, 0)
        filter1_layout.addWidget(self.featsel_edit, 0, 1)
        filter1_layout.addWidget(self.featrem_check, 0, 2)
        filter1_layout.addWidget(QLabel("Rename:"), 2, 0)
        filter1_layout.addWidget(self.featren1_edit, 2, 1)
        filter1_layout.addWidget(self.featren2_edit, 2, 2)
        filter1_widget = QWidget()
        filter1_widget.setLayout(filter1_layout)

        filter2_layout = QGridLayout()
        filter2_layout.addWidget(QLabel("Selection:"), 0, 0)
        filter2_layout.addWidget(self.classsel_edit, 0, 1)
        filter2_layout.addWidget(self.classrem_check, 0, 2)
        filter2_layout.addWidget(QLabel("Rename:"), 2, 0)
        filter2_layout.addWidget(self.classren1_edit, 2, 1)
        filter2_layout.addWidget(self.classren2_edit, 2, 2)
        filter2_widget = QWidget()
        filter2_widget.setLayout(filter2_layout)

        sub_layout = QHBoxLayout()
        sub_layout.addWidget(QLabel("Samples:"))
        sub_layout.addWidget(self.rowsub_spin)
        sub_layout.addWidget(QLabel("Features:"))
        sub_layout.addWidget(self.colsub_spin)
        filter3_layout = QVBoxLayout()
        filter3_layout.addStretch()
        filter3_layout.addLayout(sub_layout)
        filter3_layout.addStretch()
        filter3_widget = QWidget()
        filter3_widget.setLayout(filter3_layout)

        filter_tab = QTabWidget()
        filter_tab.addTab(filter1_widget, "Features")
        filter_tab.setTabIcon(0, QIcon(":/features"))
        filter_tab.addTab(filter2_widget, "Classes")
        filter_tab.setTabIcon(1, QIcon(":/classes"))
        filter_tab.addTab(filter3_widget, "Subsample")
        filter_tab.setTabIcon(2, QIcon(":/subsample"))
        filter_tab.tabBar().setExpanding(True)
        filter_tab.tabBar().setDocumentMode(True)

        filter_layout = QVBoxLayout()
        filter_layout.addWidget(self.filter_check)
        filter_layout.addWidget(filter_tab)
        filter_widget = QWidget()
        filter_widget.setLayout(filter_layout)

        # NORMALIZE
        self.normalize_check = QCheckBox("Enable")
        self.normalize_check.stateChanged.connect(self.update_normalized)
        self.normalize_check.setToolTip("Enable/disable feature normalization")
        self.sample_radio = QRadioButton("Sample")
        self.sample_radio.setChecked(True)
        self.sample_radio.toggled.connect(self.update_normalized)
        self.sample_radio.setToolTip("Normalize each sample (row) independently")
        self.feature_radio = QRadioButton("Feature")
        self.feature_radio.toggled.connect(self.update_normalized)
        self.feature_radio.setToolTip("Normalize each feature (column) independently")
        normalize_group = QButtonGroup()
        normalize_group.addButton(self.sample_radio)
        normalize_group.addButton(self.feature_radio)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(
            [
                "Standard",
                "MinMax",
                "MaxAbs",
                "Normalize",
                "Robust",
                "Quantile",
                "Minimum",
                "Maximum",
            ]
        )
        self.scaling_combo.setToolTip(
            "Algorithm for data normalization<br><br>"
            "<b>Standard</b>: Center to the mean and component wise scale to unit variance<br>"
            "<b>MinMax</b>: Scale values to the [0, 1] range<br>"
            "<b>MaxAbs</b>: Scale values to the [-1, +1] range (the maximal absolute value will be 1.0)<br>"
            "<b>Normalize</b>: Scale to unit norm (vector length)<br>"
            "<b>Robust</b>: Center to the median and component wise scale according to the interquartile range<br>"
            "<b>Quantile</b>: Transform values to follow a uniform or a normal distribution. "
            "This transformation tends to spread out the most frequent values and reduce the impact of outliers."
            "<b>Minimum</b>: Subtract the minimum value, so the new range is [0, max - min]<br>"
            "<b>Maximum</b>: Subtract the maximum value, se the new range is [min - max, 0]"
        )
        self.scaling_combo.currentIndexChanged.connect(self.update_normalized)

        normal_layout1 = QGridLayout()
        normal_layout1.addWidget(QLabel("Operation mode:"), 0, 0)
        normal_layout1.addWidget(self.sample_radio, 0, 1)
        normal_layout1.addWidget(self.feature_radio, 0, 2)
        normal_layout1.addWidget(QLabel("Scaling method:"), 1, 0)
        normal_layout1.addWidget(self.scaling_combo, 1, 1, 1, 2)
        normal_layout2 = QVBoxLayout()
        normal_layout2.addWidget(self.normalize_check)
        normal_layout2.addLayout(normal_layout1)
        normal_layout2.addStretch()
        normalize_widget = QWidget()
        normalize_widget.setLayout(normal_layout2)

        # BALANCE
        self.balance_check = QCheckBox("Enable")
        self.balance_check.stateChanged.connect(self.update_balanced)
        self.balance_check.setToolTip("Enable/disable automatic class balancing")

        self.over_radio = QRadioButton("Oversample")
        self.over_radio.setIcon(QIcon(":/over"))
        self.over_radio.setChecked(True)
        self.over_radio.toggled.connect(self.update_balanced)
        self.over_combo = QComboBox()
        self.over_combo.addItems(["RAND", "SMOTE", "ADASYN", "BORDER", "KMEANS", "SVM"])
        self.over_combo.currentIndexChanged.connect(self.update_balanced)
        self.over_combo.setToolTip(
            "Algorithm for data over-sampling<br><br>"
            "<b>RAND</b>: Over-sample the minority class(es) by picking samples at random with replacement<br>"
            "<b>SMOTE</b>: Implementation of SMOTE (Synthetic Minority Over-sampling Technique)<br>"
            "<b>ADASYN</b>: Oversample using Adaptive Synthetic (ADASYN) algorithm<br>"
            "This method is similar to SMOTE but it generates different number of samples<br>"
            "depending on an estimate of the local distribution of the class to be oversampled<br>"
            "<b>BORDER</b>: This algorithm is a variant of the original SMOTE algorithm<br>"
            "Borderline samples will be detected and used to generate new synthetic samples<br>"
            "<b>KMEANS</b>: Apply a KMeans clustering before to over-sample using SMOTE<br>"
            "<b>SVM</b>: Variant of SMOTE algorithm which use an SVM algorithm<br>"
            "to detect samples to use for generating new synthetic samples."
        )

        self.under_radio = QRadioButton("Undersample")
        self.under_radio.setIcon(QIcon(":/under"))
        self.under_radio.toggled.connect(self.update_balanced)
        self.under_combo = QComboBox()
        self.under_combo.addItems(
            [
                "RAND",
                "NMISS",
                "CC",
                "CNN",
                "ENN",
                "RENN",
                "AKNN",
                "IHT",
                "NCR",
                "OSS",
                "TOMEK",
            ]
        )
        self.under_combo.currentIndexChanged.connect(self.update_balanced)
        self.under_combo.setToolTip(
            "Algorithm for data under-sampling<br><br>"
            "<b>RAND</b>: Under-sample the majority class(es) by randomly picking samples<br>"
            "<b>NMISS</b>: kNN approach to unbalanced data distributions<br>"
            "<b>CC</b>: Method that under samples the majority class by replacing a cluster of majority samples<br>"
            "by the cluster centroid of a KMeans algorithm. This algorithm keeps N majority samples<br>"
            "by fitting the KMeans algorithm with N cluster to the majority class <br>"
            "and using the coordinates of the N cluster centroids as the new majority samples<br>"
            "<b>CNN</b>: Uses a 1 nearest neighbor rule to iteratively decide if a sample should be removed or not<br>"
            "<b>ENN</b>: This method will clean the database by removing samples close to the decision boundary<br>"
            "<b>RENN</b>: This method will repeat several time the ENN algorithm<br>"
            "<b>AKNN</b>: This method will apply ENN several time and will vary the number of nearest neighbours<br>"
            "<b>IHT</b>: A classifier is trained on the data and the samples with lower probabilities are removed<br>"
            "<b>NCR</b>: This class uses ENN and a k-NN to remove noisy samples from the datasets<br>"
            "<b>OSS</b>: Use TomekLinks and 1 nearest neighbor rule to remove noisy samples."
        )
        balance_group = QButtonGroup()
        balance_group.addButton(self.over_radio)
        balance_group.addButton(self.under_radio)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(3, 500)
        self.size_spin.setSuffix(" samples")
        self.size_spin.valueChanged.connect(self.update_balanced)
        self.size_spin.setToolTip(
            "Number of nearest neighbors to consider for balancing"
        )

        balance_layout1 = QGridLayout()
        balance_layout1.addWidget(self.balance_check, 0, 0, 1, 2)
        balance_layout1.addWidget(self.over_radio, 1, 0)
        balance_layout1.addWidget(self.over_combo, 1, 1)
        balance_layout1.addWidget(self.under_radio, 2, 0)
        balance_layout1.addWidget(self.under_combo, 2, 1)
        balance_layout1.addWidget(QLabel("Neighborhood size:"), 3, 0)
        balance_layout1.addWidget(self.size_spin, 3, 1)
        balance_layout2 = QVBoxLayout()
        balance_layout2.addLayout(balance_layout1)
        balance_layout2.addStretch()
        balance_widget = QWidget()
        balance_widget.setLayout(balance_layout2)

        # ORDERING
        self.order_check = QCheckBox("Enable")
        self.order_check.stateChanged.connect(self.update_ordered)
        self.order_check.setToolTip("Sort rows by target column value")
        self.ascend_radio = QRadioButton("Ascending")
        self.ascend_radio.toggled.connect(self.update_ordered)
        self.ascend_radio.setToolTip("Sort rows in ascending order")
        self.descend_radio = QRadioButton("Descending")
        self.descend_radio.toggled.connect(self.update_ordered)
        self.descend_radio.setToolTip("Sort rows in descending order")
        direction_group = QButtonGroup()
        direction_group.addButton(self.ascend_radio)
        direction_group.addButton(self.descend_radio)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("Sort direction:"))
        dir_layout.addWidget(self.ascend_radio)
        dir_layout.addWidget(self.descend_radio)
        order_layout = QVBoxLayout()
        order_layout.addWidget(self.order_check)
        order_layout.addLayout(dir_layout)
        order_layout.addStretch()
        order_widget = QWidget()
        order_widget.setLayout(order_layout)

        # TOOLBOX
        preproc_box = QToolBox()
        preproc_box.addItem(filter_widget, "Filter")
        preproc_box.setItemIcon(0, QIcon(":/filter"))
        preproc_box.addItem(normalize_widget, "Normalize")
        preproc_box.setItemIcon(1, QIcon(":/normalize"))
        preproc_box.addItem(balance_widget, "Balance")
        preproc_box.setItemIcon(2, QIcon(":/balance"))
        preproc_box.addItem(order_widget, "Ordering")
        preproc_box.setItemIcon(3, QIcon(":/order"))
        preproc_layout = QVBoxLayout()
        preproc_layout.addWidget(preproc_box)
        self.preproc_group = QGroupBox("PREPROCESSING")
        self.preproc_group.setLayout(preproc_layout)
        self.preproc_group.setEnabled(False)

        # SPLIT
        self.training_spin = QSpinBox()
        self.training_spin.setRange(0, 100)
        self.training_spin.setValue(80)
        self.training_spin.setSuffix("%")
        self.training_spin.setSingleStep(5)
        self.training_spin.valueChanged.connect(self.update_split)
        self.training_spin.setToolTip(
            "Percentage of samples to use for training (0% or 100% = no split)"
        )

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 10000)
        self.seed_spin.setValue(42)
        self.seed_spin.setSingleStep(1)
        self.seed_spin.valueChanged.connect(self.update_split)
        self.seed_spin.setToolTip("Seed for random shuffling of samples")

        exportsplit_button = QPushButton("Export...")
        exportsplit_button.setIcon(QIcon(":/export"))
        exportsplit_button.clicked.connect(self.export_split)
        exportsplit_button.setToolTip(
            "Export split data to CSV files ('train.csv' and 'test.csv')"
        )

        # SUMMARY
        self.classes_label = QLabel("Classes: 0")
        self.features_label = QLabel("Features: 0")
        self.values_label = QLabel("Values: 0")
        self.summary_view = gui.PandasView(alternate=False, index=False)
        self.count_radio = QRadioButton("Sample count")
        self.count_radio.setChecked(True)
        self.count_radio.toggled.connect(self.update_summary)
        self.ratio_radio = QRadioButton("Sample ratio")
        self.ratio_radio.toggled.connect(self.update_summary)

        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Training:"))
        split_layout.addWidget(self.training_spin)
        split_layout.addStretch()
        split_layout.addWidget(QLabel("Seed:"))
        split_layout.addWidget(self.seed_spin)
        split_layout.addWidget(exportsplit_button)
        self.split_group = QGroupBox("SPLIT")
        self.split_group.setLayout(split_layout)
        self.split_group.setEnabled(False)

        show_group = QButtonGroup()
        show_group.addButton(self.count_radio)
        show_group.addButton(self.ratio_radio)
        show_layout = QHBoxLayout()
        show_layout.addWidget(QLabel("Display mode:"))
        show_layout.addWidget(self.count_radio)
        show_layout.addWidget(self.ratio_radio)

        summary1_layout = QHBoxLayout()
        summary1_layout.addWidget(self.classes_label)
        summary1_layout.addStretch()
        summary1_layout.addWidget(self.features_label)
        summary1_layout.addStretch()
        summary1_layout.addWidget(self.values_label)
        summary2_layout = QVBoxLayout()
        summary2_layout.addLayout(summary1_layout)
        summary2_layout.addWidget(self.summary_view)
        summary2_layout.addLayout(show_layout)
        self.summary_group = QGroupBox("SUMMARY")
        self.summary_group.setLayout(summary2_layout)
        self.summary_group.setEnabled(False)
        self.summary_group.setMaximumHeight(220)

        # RAW DATA
        self.raw_view = gui.PandasView(alternate=True, index=True)

        # VALUE HEATMAP
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            [
                "Plasma",
                "Viridis",
                "Inferno",
                "Magma",
                "Cividis",
                "Gray",
                "Bone",
                "Pink",
                "Cool",
                "Hot",
                "Rainbow",
                "Jet",
                "Turbo",
            ]
        )
        self.colormap_combo.currentIndexChanged.connect(self.update_heatmap)
        self.colormap_combo.setToolTip(
            "Perceptually uniform colormaps for displaying heatmap"
        )
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(
            [
                "Nearest",
                "Antialiased",
                "Bilinear",
                "Bicubic",
                "Spline16",
                "Hanning",
                "Hamming",
                "Quadric",
                "Catrom",
                "Gaussian",
                "Bessel",
                "Mitchell",
                "Sinc",
                "Lanczos",
            ]
        )
        self.interp_combo.currentIndexChanged.connect(self.update_heatmap)
        self.interp_combo.setToolTip(
            "Value interpolation method for displaying heatmap"
        )
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(
            [
                "Linear",
                "Log",
                "Symlog",
                "Asinh",
            ]
        )
        self.norm_combo.currentIndexChanged.connect(self.update_heatmap)
        self.norm_combo.setToolTip(
            "Normalization method used to scale data before color mapping"
        )
        self.reversed_check = QCheckBox("Reversed")
        self.reversed_check.stateChanged.connect(self.update_heatmap)
        self.reversed_check.setToolTip("Reverse colormap before applying to data")
        self.heatmap_mpl = gui.MplWidget(toolbar=True)

        heatmap1_layout = QHBoxLayout()
        heatmap1_layout.addWidget(QLabel("Scaling:"))
        heatmap1_layout.addWidget(self.norm_combo)
        heatmap1_layout.addWidget(QLabel("Interpolation:"))
        heatmap1_layout.addWidget(self.interp_combo)
        heatmap1_layout.addWidget(QLabel("Colormap:"))
        heatmap1_layout.addWidget(self.colormap_combo)
        heatmap1_layout.addWidget(self.reversed_check)
        heatmap1_layout.addStretch()

        heatmap2_layout = QVBoxLayout()
        heatmap2_layout.addLayout(heatmap1_layout)
        heatmap2_layout.addWidget(self.heatmap_mpl)
        self.heatmap_widget = QWidget()
        self.heatmap_widget.setLayout(heatmap2_layout)

        # FEATURE ANALYSIS
        self.analyze_combo = QComboBox()
        self.analyze_combo.addItems(
            [
                "Descriptive",
                "Range",
                "Box",
                "Violin",
                "Scatter",
                "Histogram",
                "Density",
                "Parallel",
                "Andrews",
                "ScatterMtx",
                "Radial",
                "Correlation",
                "Bootstrap",
            ]
        )
        self.analyze_combo.setToolTip(
            "Feature analysis method<br><br>"
            "<b>Descriptive</b>: Compute descriptive statistics for each feature<br>"
            "<b>Range</b>: Plot minimum, average, maximum and deviation of each feature<br>"
            "<b>Box</b>: Make a box and whisker plot of the distribution of each feature<br>"
            "<b>Violin</b>: Make a violin plot of the distribution of each feature<br>"
            "<b>Scatter</b>: Make a scatter plot of selected features values (X/Y axis)<br>"
            "<b>Histogram</b>: Make a histogram of selected feature distributions (X/Y axis)<br>"
            "<b>Density</b>: Make a density plot of selected feature distributions (X/Y axis)<br>"
            "<b>Parallel</b>: Make a parallel coordinates plot of the whole dataset<br>"
            "<b>Andrews</b>: Make an Andrews curves plot of the whole dataset<br>"
            "<b>ScatterMtx</b>: Make a scatter matrix plot of the whole dataset<br>"
            "<b>Radial</b>: Make a radial plot of the whole dataset<br>"
            "<b>Correlation</b>: Make a correlation matrix plot of the whole dataset<br>"
            "<b>Bootstrap</b>: Make a bootstrap plot of the selected feature (X axis)"
        )
        self.multiple_combo = QComboBox()
        self.multiple_combo.addItems(["Layer", "Dodge", "Stack", "Fill"])
        self.multiple_combo.setToolTip("Method for drawing multiple elements")

        self.featx_combo = QComboBox()
        self.featx_combo.setToolTip("Selected feature for X axis")
        self.featy_combo = QComboBox()
        self.featy_combo.setToolTip("Selected feature for Y axis")
        self.notch_check = QCheckBox("Notches")
        self.notch_check.setToolTip("Draw notches around the median")
        self.caps_check = QCheckBox("Caps")
        self.caps_check.setToolTip("Draw whiskers as caps")
        self.fliers_check = QCheckBox("Fliers")
        self.fliers_check.setToolTip("Show outliers beyond caps")
        self.means_check = QCheckBox("Means")
        self.means_check.setToolTip("Show mean values")
        self.boxes_check = QCheckBox("Boxes")
        self.boxes_check.setToolTip("Draw box outlines")
        self.extrema_check = QCheckBox("Extrema")
        self.extrema_check.setToolTip("Show min/max values")
        self.filling_check = QCheckBox("Filling")
        self.filling_check.setToolTip("Fill boxes with color")
        self.logscale_check = QCheckBox("Log scale")
        self.logscale_check.setToolTip("Use logarithmic scale for Y axis")

        self.analyze_view = gui.PandasView(alternate=True)
        self.analyze_mpl = gui.MplWidget(toolbar=True)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setIcon(QIcon(":/process"))
        self.analyze_button.clicked.connect(self.analyze_features)
        self.analyze_button.setToolTip("Perform feature analysis")
        self.exportstats_button = QPushButton("Export...")
        self.exportstats_button.setIcon(QIcon(":/export"))
        self.exportstats_button.clicked.connect(self.export_stats)
        self.exportstats_button.setToolTip("Export analysis data to CSV file")

        analyze1_layout = QGridLayout()
        analyze1_layout.addWidget(QLabel("Analysis type:"), 0, 0)
        analyze1_layout.addWidget(self.analyze_combo, 0, 1)
        analyze1_layout.addWidget(QLabel("Multiple mode:"), 1, 0)
        analyze1_layout.addWidget(self.multiple_combo, 1, 1)
        analyze1_layout.addWidget(QLabel("X-axis feature:"), 0, 2)
        analyze1_layout.addWidget(self.featx_combo, 0, 3)
        analyze1_layout.addWidget(QLabel("Y-axis feature:"), 1, 2)
        analyze1_layout.addWidget(self.featy_combo, 1, 3)

        analyze1_layout.addWidget(self.means_check, 3, 0)
        analyze1_layout.addWidget(self.extrema_check, 3, 1)
        analyze1_layout.addWidget(self.caps_check, 3, 2)
        analyze1_layout.addWidget(self.filling_check, 3, 3)
        analyze1_layout.addWidget(self.notch_check, 4, 0)
        analyze1_layout.addWidget(self.boxes_check, 4, 1)
        analyze1_layout.addWidget(self.fliers_check, 4, 2)
        analyze1_layout.addWidget(self.logscale_check, 4, 3)

        analyze_width = 400
        analyze1_group = QGroupBox("OPTIONS")
        analyze1_group.setLayout(analyze1_layout)
        analyze1_group.setMaximumWidth(analyze_width)

        analyze2_layout = QHBoxLayout()
        analyze2_layout.addWidget(self.analyze_button)
        analyze2_layout.addWidget(self.exportstats_button)

        analyze3_layout = QVBoxLayout()
        analyze3_layout.addWidget(self.analyze_view)
        analyze3_group = QGroupBox("ANALYSIS")
        analyze3_group.setLayout(analyze3_layout)
        analyze3_group.setMaximumWidth(analyze_width)

        analyze_layout = QGridLayout()
        analyze_layout.addWidget(analyze1_group, 0, 0)
        analyze_layout.addLayout(analyze2_layout, 1, 0)
        analyze_layout.addWidget(analyze3_group, 2, 0)
        analyze_layout.addWidget(self.analyze_mpl, 0, 1, 3, 1)
        analyze_widget = QWidget()
        analyze_widget.setLayout(analyze_layout)

        # SELECTION
        self.select_combo = QComboBox()
        self.select_combo.addItems(
            [
                "VAR",
                "KBEST",
                "PERC",
                "FPR",
                "FDR",
                "FWE",
                "RFE",
                "SFM",
                "SFS",
            ]
        )
        self.select_combo.setToolTip(
            "Feature selection method<br><br>"
            "<b>VAR</b>: Feature selector that removes all low-variance features<br>"
            "<b>KBEST</b>: Select features according to the k highest scores<br>"
            "<b>PERC</b>: Select features according to a percentile of the highest scores<br>"
            "<b>FPR</b>: Select features based on False Positive Rate test (total of false detections)<br>"
            "<b>FDR</b>: Select features using an estimated false discovery rate (Benjamini-Hochberg procedure)<br>"
            "<b>FWE</b>: Select the p-values corresponding to Family-wise error rate<br>"
            "<b>RFE</b>: Feature ranking with recursive feature elimination<br>"
            "Recursively considers smaller and smaller sets of features<br>"
            "First, the estimator is trained on the initial set of features and the importance of each feature<br>"
            "is obtained. Then, the least important features are pruned from current set of features<br>"
            "That procedure is recursively repeated on the pruned set until the desired number of features is reached."
            "<b>SFM</b>: Meta-transformer for selecting features based on estimator importance weights<br>"
            "<b>SFS</b>: Sequential Feature Selection adds (forward selection) or removes (backward selection)<br>"
            "features to form a feature subset in a greedy fashion. At each stage, the estimator chooses<br>"
            "the best feature to add or remove based on the cross-validation score of an estimator"
        )
        self.score_combo = QComboBox()
        self.score_combo.addItems(["CHI2", "ANOVA", "MUTUAL"])
        self.score_combo.setToolTip(
            "Univariate feature scoring function for KBEST, PERC, FPR, FDR, FWE<br><br>"
            "<b>CHI2</b>: Chi-squared stats of non-negative features<br>"
            "<b>ANOVA</b>: F-value between label/feature<br>"
            "<b>MUTUAL</b>: Mutual information for discrete targets"
        )
        self.estim_combo = QComboBox()
        self.estim_combo.addItems(["ETC", "RFC", "SVC", "ADA"])
        self.estim_combo.setToolTip(
            "Model for RFE, SFM, SFS feature selection<br><br>"
            "<b>ETC</b>: Extremely randomized trees estimator<br>"
            "<b>RFC</b>: Random forest estimator<br>"
            "<b>SVC</b>: Support vector machine estimator<br>"
            "<b>ADA</b>: AdaBoost estimator"
        )
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0, 1e9)
        self.thr_spin.setSingleStep(0.1)
        self.thr_spin.setToolTip("Selection threshold for VAR, PERC, FPR, FDR, FWE")
        self.kbest_spin = QSpinBox()
        self.kbest_spin.setRange(1, 1000)
        self.kbest_spin.setToolTip("Number of features to select for KBEST")
        self.trees_spin = QSpinBox()
        self.trees_spin.setRange(10, 1000)
        self.trees_spin.setToolTip("Number of estimator trees for ETC, RFC, ADA")
        self.folds_spin = QSpinBox()
        self.folds_spin.setRange(2, 10)
        self.folds_spin.setToolTip("Number of cross-validation folds for RFE, SFS")
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.01, 1)
        self.rate_spin.setSingleStep(0.01)
        self.rate_spin.setDecimals(2)
        self.rate_spin.setToolTip("Learning rate for ADA estimator")
        self.creg_spin = QDoubleSpinBox()
        self.creg_spin.setRange(0.01, 1)
        self.creg_spin.setSingleStep(0.01)
        self.creg_spin.setToolTip("Regularization parameter for SVC estimator")
        self.tol_spin = QDoubleSpinBox()
        self.tol_spin.setRange(0.0001, 0.1)
        self.tol_spin.setToolTip("Tolerance for stopping criteria for SVC estimator")
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(10, 5000)
        self.iter_spin.setSingleStep(10)
        self.iter_spin.setToolTip("Maximum number of iterations for SVC estimator")
        self.select_button = QPushButton("Select")
        self.select_button.setIcon(QIcon(":/process"))
        self.select_button.clicked.connect(self.select_features)
        self.select_button.setToolTip("Perform feature selection")
        self.copy_button = QPushButton("Copy")
        self.copy_button.setIcon(QIcon(":/copy"))
        self.copy_button.clicked.connect(self.copy_selection)
        self.copy_button.setToolTip("Copy selected features to Preprocessing filter")
        self.select_view = gui.PandasView(alternate=True)
        self.select_mpl = gui.MplWidget()

        select_width = 280
        select1_layout = QGridLayout()
        select1_layout.addWidget(QLabel("Selection algorithm:"), 0, 0)
        select1_layout.addWidget(self.select_combo, 0, 1)
        select1_layout.addWidget(QLabel("Univariate scoring:"), 1, 0)
        select1_layout.addWidget(self.score_combo, 1, 1)
        select1_layout.addWidget(QLabel("RFE/SFM/SFS model:"), 2, 0)
        select1_layout.addWidget(self.estim_combo, 2, 1)
        select1_layout.addWidget(QLabel("Selection threshold:"), 3, 0)
        select1_layout.addWidget(self.thr_spin, 3, 1)
        select1_layout.addWidget(QLabel("K-Best subset size:"), 4, 0)
        select1_layout.addWidget(self.kbest_spin, 4, 1)
        select1_layout.addWidget(QLabel("ETC/RFC/ADA trees:"), 5, 0)
        select1_layout.addWidget(self.trees_spin, 5, 1)
        select1_layout.addWidget(QLabel("RFE/SFS validations"), 6, 0)
        select1_layout.addWidget(self.folds_spin, 6, 1)
        select1_layout.addWidget(QLabel("ADA learning rate:"), 7, 0)
        select1_layout.addWidget(self.rate_spin, 7, 1)
        select1_layout.addWidget(QLabel("SVC regularization:"), 8, 0)
        select1_layout.addWidget(self.creg_spin, 8, 1)
        select1_layout.addWidget(QLabel("SVC hard tolerance:"), 9, 0)
        select1_layout.addWidget(self.tol_spin, 9, 1)
        select1_layout.addWidget(QLabel("SVC max iterations:"), 10, 0)
        select1_layout.addWidget(self.iter_spin, 10, 1)
        select1_group = QGroupBox("OPTIONS")
        select1_group.setLayout(select1_layout)
        select1_group.setMaximumWidth(select_width)

        select2_layout = QHBoxLayout()
        select2_layout.addWidget(self.select_button)
        select2_layout.addWidget(self.copy_button)

        select3_layout = QVBoxLayout()
        select3_layout.addWidget(self.select_view)
        select3_group = QGroupBox("SELECTION")
        select3_group.setLayout(select3_layout)
        select3_group.setMaximumWidth(select_width)

        select_layout = QGridLayout()
        select_layout.addWidget(select1_group, 0, 0)
        select_layout.addLayout(select2_layout, 1, 0)
        select_layout.addWidget(select3_group, 2, 0)
        select_layout.addWidget(self.select_mpl, 0, 1, 3, 1)
        select_widget = QWidget()
        select_widget.setLayout(select_layout)

        # DIMENSIONALITY REDUCTION
        self.reduce_combo = QComboBox()
        self.reduce_combo.addItems(["PCA", "KPCA", "ICA", "LDA", "FA", "NMF"])
        self.reduce_combo.setToolTip(
            "Dimensionality reduction algorithm<br><br>"
            "<b>PCA</b>: Principal component analysis<br>"
            "<b>KPCA</b>: Kernel principal component analysis<br>"
            "<b>ICA</b>: Independent component analysis<br>"
            "<b>LDA</b>: Linear discriminant analysis<br>"
            "<b>FA</b>: Factor analysis<br>"
            "<b>NMF</b>: Non-negative matrix factorization"
        )
        self.whiten_combo = QComboBox()
        self.whiten_combo.addItems(["OFF", "Unit", "Arbitrary"])
        self.whiten_combo.setToolTip(
            "Specify the whitening strategy for ICA<br><br>"
            "<b>OFF</b>: No whitening is performed<br>"
            "<b>Unit</b>: Whitening matrix is rescaled to ensure that samples have unit variance<br>"
            "<b>Arbitrary</b>: A whitening with arbitrary variance is used"
        )
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(
            ["Linear", "Poly", "RBF", "Sigmoid", "Cosine", "Precomputed"]
        )
        self.kernel_combo.setToolTip(
            "Kernel function for KPCA<br><br>"
            "<b>Linear</b>: Linear kernel<br>"
            "<b>Poly</b>: Polynomial kernel<br>"
            "<b>RBF</b>: Radial basis function kernel<br>"
            "<b>Sigmoid</b>: Sigmoid kernel<br>"
            "<b>Cosine</b>: Cosine kernel<br>"
            "<b>Precomputed</b>: Precomputed kernel"
        )
        self.fititer_spin = QSpinBox()
        self.fititer_spin.setRange(1, 1000)
        self.fititer_spin.setSuffix(" rounds")
        self.fititer_spin.setToolTip(
            "Number of fitting iterations for ICA, LDA, FA, NMF"
        )
        self.fittol_spin = QDoubleSpinBox()
        self.fittol_spin.setRange(0.0001, 0.01)
        self.fittol_spin.setSingleStep(0.00001)
        self.fittol_spin.setDecimals(6)
        self.fittol_spin.setToolTip("Fitting tolerance for KPCA, ICA, FA, NMF")
        self.alpha_spin = QSpinBox()
        self.alpha_spin.setRange(0, 100)
        self.alpha_spin.setSuffix("%")
        self.alpha_spin.setToolTip("Alpha transparency for scatter plot")

        self.outlier_combo = QComboBox()
        self.outlier_combo.addItems(["OFF", "Local Factor", "Isolation Forest"])
        self.outlier_combo.setToolTip(
            "Outlier detection method<br><br>"
            "<b>OFF</b>: Disable outlier detection<br>"
            "<b>Local Factor</b>: Local outlier factor<br>"
            "<b>Isolation Forest</b>: Isolation forest"
        )
        self.distance_combo = QComboBox()
        self.distance_combo.addItems(
            [
                "Euclidean",
                "BrayCurtis",
                "Canberra",
                "Chebyshev",
                "CityBlock",
                "Correlation",
                "Cosine",
                "Manhattan",
                "SqEuclidean",
            ]
        )
        self.distance_combo.setToolTip("Distance metric for outlier detection")
        self.response_combo = QComboBox()
        self.response_combo.addItems(["Probability", "Decision", "Prediction"])
        self.response_combo.setToolTip(
            "Response type for outlier score<br><br>"
            "<b>Probability</b>: Probability of a sample to be outlier<br>"
            "<b>Decision</b>: Classifier decision function<br>"
            "<b>Prediction</b>: Output prediction function"
        )
        self.pmethod_combo = QComboBox()
        self.pmethod_combo.addItems(["Filled", "Contour", "Max", "Min"])
        self.pmethod_combo.setToolTip(
            "Plot method for Isolation Forest decision boundary"
        )
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(3, 100)
        self.neighbors_spin.setSuffix(" samples")
        self.neighbors_spin.setToolTip("Number of neighbors for outlier detection")
        self.leafsize_spin = QSpinBox()
        self.leafsize_spin.setRange(2, 100)
        self.leafsize_spin.setSuffix(" nodes")
        self.leafsize_spin.setToolTip("Leaf size for outlier detection")
        self.estimators_spin = QSpinBox()
        self.estimators_spin.setRange(10, 1000)
        self.estimators_spin.setSuffix(" estimators")
        self.estimators_spin.setToolTip("Number of estimators for Isolation Forest")
        self.bootstrap_check = QCheckBox("Sample with replacement for IF")
        self.bootstrap_check.setToolTip(
            "Enable sample with replacement (bootstrap) for Isolation Forest"
        )
        self.reduced_check = QCheckBox("Apply LFA on reduced components")
        self.reduced_check.setToolTip(
            "Apply Local Factor Analysis on reduced components"
        )
        self.reduce_button = QPushButton("Reduce")
        self.reduce_button.setIcon(QIcon(":/process"))
        self.reduce_button.clicked.connect(self.reduce_dimensions)
        self.reduce_button.setToolTip("Perform dimensionality reduction")
        self.exportreduce_button = QPushButton("Export...")
        self.exportreduce_button.setIcon(QIcon(":/export"))
        self.exportreduce_button.clicked.connect(self.export_reduction)
        self.exportreduce_button.setToolTip("Export reduced dataset to CSV file")
        self.reduce_view = gui.PandasView(alternate=True)
        self.reduce_mpl = gui.MplWidget(toolbar=True)

        reduce_width = 280
        reduce1_layout = QFormLayout()
        reduce1_layout.addRow("Reduction algorithm:", self.reduce_combo)
        reduce1_layout.addRow("Noise normalization:", self.whiten_combo)
        reduce1_layout.addRow("Approximation kernel:", self.kernel_combo)
        reduce1_layout.addRow("Fitting iterations:", self.fititer_spin)
        reduce1_layout.addRow("Fitting tolerance:", self.fittol_spin)
        reduce1_layout.addRow("Scatter plot alpha:", self.alpha_spin)
        reduce1_group = QGroupBox("OPTIONS")
        reduce1_group.setLayout(reduce1_layout)
        reduce1_group.setMaximumWidth(reduce_width)

        reduce2_layout = QFormLayout()
        reduce2_layout.addRow("Detection method:", self.outlier_combo)
        reduce2_layout.addRow("Distance metric:", self.distance_combo)
        reduce2_layout.addRow("Neighborhood:", self.neighbors_spin)
        reduce2_layout.addRow("Tree leaf size:", self.leafsize_spin)
        reduce2_layout.addRow("Base ensemble:", self.estimators_spin)
        reduce2_layout.addRow(self.bootstrap_check)
        reduce2_layout.addRow(self.reduced_check)
        reduce2_group = QGroupBox("OUTLIERS")
        reduce2_group.setLayout(reduce2_layout)
        reduce2_group.setMaximumWidth(reduce_width)

        reduce3_layout = QHBoxLayout()
        reduce3_layout.addWidget(self.reduce_button)
        reduce3_layout.addWidget(self.exportreduce_button)

        reduce4_layout = QVBoxLayout()
        reduce4_layout.addWidget(self.reduce_view)
        reduce4_group = QGroupBox("REDUCTION")
        reduce4_group.setLayout(reduce4_layout)
        reduce4_group.setMaximumWidth(reduce_width)

        reduce_layout = QGridLayout()
        reduce_layout.addWidget(reduce1_group, 0, 0)
        reduce_layout.addWidget(reduce2_group, 1, 0)
        reduce_layout.addLayout(reduce3_layout, 2, 0)
        reduce_layout.addWidget(reduce4_group, 3, 0)
        reduce_layout.addWidget(self.reduce_mpl, 0, 1, 4, 1)
        reduce_widget = QWidget()
        reduce_widget.setLayout(reduce_layout)

        # LEFT WIDGET
        left_width = 360
        left_layout = QVBoxLayout()
        left_layout.addWidget(file_group)
        left_layout.addWidget(self.preproc_group)
        left_layout.addStretch()
        left_layout.addWidget(self.split_group)
        left_layout.addWidget(self.summary_group)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMaximumWidth(left_width)

        # RIGHT WIDGET
        self.right_widget = QTabWidget()
        self.right_widget.addTab(self.raw_view, "Data matrix")
        self.right_widget.setTabIcon(0, QIcon(":/raw"))
        self.right_widget.addTab(self.heatmap_widget, "Value heatmap")
        self.right_widget.setTabIcon(1, QIcon(":/heat"))
        self.right_widget.addTab(analyze_widget, "Feature analysis")
        self.right_widget.setTabIcon(2, QIcon(":/analysis"))
        self.right_widget.addTab(select_widget, "Feature selection")
        self.right_widget.setTabIcon(3, QIcon(":/selection"))
        self.right_widget.addTab(reduce_widget, "Dimensionality reduction")
        self.right_widget.setTabIcon(4, QIcon(":/reduction"))

        # MAIN LAYOUT
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.right_widget)
        self.setLayout(main_layout)
        self.set_defaults()

    def set_defaults(self):
        self.right_widget.setEnabled(False)

        self.save_button.setEnabled(False)
        self.suffix_radio.setChecked(True)
        self.compress_spin.setValue(4)
        self.csvsep_combo.setCurrentIndex(0)
        self.recursive_check.setChecked(True)
        self.header_check.setChecked(False)

        self.filter_check.setChecked(False)
        self.featsel_edit.clear()
        self.featrem_check.setChecked(False)
        self.featren1_edit.clear()
        self.featren2_edit.clear()
        self.classsel_edit.clear()
        self.classrem_check.setChecked(False)
        self.classren1_edit.clear()
        self.classren2_edit.clear()
        self.rowsub_spin.setValue(100)
        self.colsub_spin.setValue(100)

        self.normalize_check.setChecked(False)
        self.sample_radio.setChecked(True)
        self.feature_radio.setChecked(False)
        self.scaling_combo.setCurrentIndex(0)

        self.balance_check.setChecked(False)
        self.over_radio.setChecked(True)
        self.over_combo.setCurrentIndex(1)
        self.under_combo.setCurrentIndex(1)
        self.size_spin.setValue(5)

        self.order_check.setChecked(False)
        self.ascend_radio.setChecked(True)

        self.colormap_combo.setCurrentIndex(0)
        self.interp_combo.setCurrentIndex(0)
        self.norm_combo.setCurrentIndex(0)
        self.reversed_check.setChecked(False)

        self.analyze_combo.setCurrentIndex(0)
        self.featx_combo.setCurrentIndex(0)
        self.featy_combo.setCurrentIndex(0)
        self.multiple_combo.setCurrentIndex(0)
        self.notch_check.setChecked(False)
        self.means_check.setChecked(True)
        self.extrema_check.setChecked(True)
        self.boxes_check.setChecked(True)
        self.caps_check.setChecked(True)
        self.fliers_check.setChecked(False)
        self.filling_check.setChecked(True)
        self.logscale_check.setChecked(False)

        self.reduce_combo.setCurrentIndex(0)
        self.whiten_combo.setCurrentIndex(0)
        self.kernel_combo.setCurrentIndex(0)
        self.fititer_spin.setValue(200)
        self.fittol_spin.setValue(0.0001)
        self.alpha_spin.setValue(75)
        self.outlier_combo.setCurrentIndex(0)
        self.distance_combo.setCurrentIndex(0)
        self.neighbors_spin.setValue(20)
        self.leafsize_spin.setValue(30)
        self.reduced_check.setChecked(False)

        self.select_combo.setCurrentIndex(0)
        self.score_combo.setCurrentIndex(0)
        self.estim_combo.setCurrentIndex(0)
        self.thr_spin.setValue(1)
        self.kbest_spin.setValue(10)
        self.trees_spin.setValue(100)
        self.folds_spin.setValue(5)
        self.rate_spin.setValue(1)
        self.creg_spin.setValue(1)
        self.tol_spin.setValue(0.0001)
        self.iter_spin.setValue(4000)

    def enable_controls(self):
        self.preproc_group.setEnabled(True)
        self.split_group.setEnabled(True)
        self.summary_group.setEnabled(True)
        self.right_widget.setEnabled(True)
        self.save_button.setEnabled(True)

    def import_folder(self):
        settings = QSettings()
        if folder := QFileDialog.getExistingDirectory(
            self, "Import folder", settings.value("import_folder", ".")
        ):
            settings.setValue("import_folder", folder)
            recursive = self.recursive_check.isChecked()
            target = "suffix" if self.suffix_radio.isChecked() else "column"
            header = self.header_check.isChecked()
            csvsep = self.csvsep_combo.currentText()
            original, imported = dataset.scan(folder, recursive, target, header, csvsep)
            if original is None:
                QMessageBox.critical(self, "Import folder", "No valid files found!")
                return
            if not dataset.check(original):
                QMessageBox.critical(self, "Import folder", "Invalid dataset!")
                return
            self.init_gui(original)
            self.dataset_ready.emit("IMPORTED")
            QMessageBox.information(
                self, "Import folder", f"Import completed ({imported} files processed)"
            )

    def load_dataset(self):
        settings = QSettings()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load dataset",
            settings.value("load_dataset", "."),
            self.file_types["dataset"],
        )
        if not filename:
            return
        settings.setValue("load_dataset", disk.get_folder(filename))
        try:
            original = dataset.load(filename)
        except ValueError as e:
            QMessageBox.critical(self, "Load dataset", repr(e))
            return
        self.init_gui(original)
        self.dataset_ready.emit(disk.get_basename(filename))

    def init_gui(self, df):
        self.original = df
        self.set_defaults()
        self.update_filtered()
        self.clear_graphs()
        self.enable_controls()
        self.check_size(self.original, warning=True)

    def check_size(self, df, warning) -> bool:
        result = dataset.properties(df)["values"] < self.max_values
        if not result and warning:
            QMessageBox.warning(
                self,
                "Load dataset",
                f"Dataset has >{misc.humanize(self.max_values)} values: data matrix and heatmap disabled",
            )
        return result

    def save_dataset(self):
        if self.balanced is None:
            QMessageBox.critical(
                self, "Save dataset", "Please load or import a dataset first"
            )
            return
        settings = QSettings()
        filename, extension = QFileDialog.getSaveFileName(
            self,
            "Save dataset",
            settings.value("save_dataset", "."),
            self.file_types["dataset"],
        )
        if not filename:
            return
        # FIXME: Add missing extension if needed (check 'extension' variable)
        settings.setValue("save_dataset", disk.get_folder(filename))
        compress = self.compress_spin.value()
        try:
            dataset.save(self.ordered, filename, compress)
            filesize = disk.human_size(disk.file_size(filename))
            self.dataset_ready.emit(disk.get_basename(filename))
            QMessageBox.information(
                self, "Save dataset", f"Dataset successfully saved ({filesize})"
            )
        except ValueError as e:
            QMessageBox.critical(self, "Save dataset", repr(e))

    def export_split(self):
        if self.parent.train is None or self.parent.test is None:
            QMessageBox.critical(self, "Export split", "Dataset not ready")
            return
        settings = QSettings()
        if folder := QFileDialog.getExistingDirectory(
            self, "Export split", settings.value("export_split", ".")
        ):
            settings.setValue("export_split", folder)
            train_file = disk.join_paths(folder, "train.csv")
            test_file = disk.join_paths(folder, "test.csv")
            try:
                dataset.save(self.parent.train, train_file)
                dataset.save(self.parent.test, test_file)
                QMessageBox.information(
                    self, "Export split", f"Split exported ('train.csv' and 'test.csv')"
                )
            except ValueError as e:
                QMessageBox.critical(self, "Export split", repr(e))

    def update_filtered(self):
        if self.original is None:
            return
        checked = self.filter_check.isChecked()
        gui.modify_font(self.filter_check, bold=checked)
        if checked:
            self.filtered = self.original
            for what, selection, check, rename1, rename2 in zip(
                ["feature", "class"],
                [self.featsel_edit, self.classsel_edit],
                [self.featrem_check, self.classrem_check],
                [self.featren1_edit, self.classren1_edit],
                [self.featren2_edit, self.classren2_edit],
            ):
                select = misc.rngs2vals(selection.text())
                if select:
                    mode = "exclude" if check.isChecked() else "include"
                    self.filtered = dataset.filter(self.filtered, what, mode, select)
                ren1 = misc.rngs2vals(rename1.text())
                ren2 = misc.rngs2vals(rename2.text())
                if ren1 and ren2:
                    self.filtered = dataset.rename(self.filtered, what, ren1, ren2)
            for spin, what in zip(
                [self.rowsub_spin, self.colsub_spin], ["rows", "cols"]
            ):
                factor = spin.value() / 100
                if factor < 1:
                    self.filtered = dataset.subsample(self.filtered, what, factor)
        else:
            self.filtered = self.original
        self.update_normalized()

    def update_normalized(self):
        if self.filtered is None:
            return
        checked = self.normalize_check.isChecked()
        gui.modify_font(self.normalize_check, bold=checked)
        if checked:
            mode = "sample" if self.sample_radio.isChecked() else "feature"
            method = self.scaling_combo.currentText()
            self.normalized = dataset.scale(self.filtered, mode, method)
        else:
            self.normalized = self.filtered
        self.update_balanced()

    def update_balanced(self):
        if self.normalized is None:
            return
        checked = self.balance_check.isChecked()
        gui.modify_font(self.balance_check, bold=checked)
        if checked:
            if self.over_radio.isChecked():
                mode = "oversample"
                method = self.over_combo.currentText()
            else:
                mode = "undersample"
                method = self.under_combo.currentText()
            size = self.size_spin.value()
            self.balanced = dataset.balance(self.normalized, mode, method, size)
        else:
            self.balanced = self.normalized
        self.update_ordered()

    def update_ordered(self):
        if self.balanced is None:
            return
        checked = self.order_check.isChecked()
        gui.modify_font(self.order_check, bold=checked)
        if checked:
            ascending = self.ascend_radio.isChecked()
            self.ordered = dataset.sort(self.balanced, ascending)
        else:
            self.ordered = self.balanced
        self.update_raw()

    def update_raw(self):
        if self.ordered is None:
            return
        if self.check_size(self.ordered, warning=False):
            self.raw_view.update_data(self.ordered)
            self.raw_view.setEnabled(True)
        else:
            self.raw_view.clear_data()
            self.raw_view.setEnabled(False)
        self.update_heatmap()

    def update_heatmap(self):
        if self.ordered is None:
            return
        if self.check_size(self.ordered, warning=False):
            cmap = self.colormap_combo.currentText().lower()
            interp = self.interp_combo.currentText().lower()
            norm = self.norm_combo.currentText().lower()
            reverse = self.reversed_check.isChecked()
            dataset.heatmap(self.ordered, cmap, norm, interp, reverse, self.heatmap_mpl)
            self.heatmap_widget.setEnabled(True)
        else:
            self.heatmap_mpl.clear()
            self.heatmap_widget.setEnabled(False)
        self.update_split()

    def update_split(self):
        if self.ordered is None:
            return
        training = self.training_spin.value() / 100
        seed = self.seed_spin.value()
        self.parent.train, self.parent.test = dataset.split(
            self.ordered, training, seed
        )
        self.update_summary()

    def update_summary(self):
        if self.ordered is None:
            return
        ratio = self.ratio_radio.isChecked()
        info = dataset.info(self.parent.train, self.parent.test, ratio)
        props = dataset.properties(self.ordered)
        self.summary_view.update_data(info)
        self.classes_label.setText(f"Classes: {props['classes']}")
        self.features_label.setText(f"Features: {props['features']}")
        self.values_label.setText(f"Values: {props['values']:,}")
        self.featx_combo.clear()
        self.featx_combo.addItems(["NONE"] + props["columns"])
        self.featy_combo.clear()
        self.featy_combo.addItems(["NONE"] + props["columns"])

    def clear_graphs(self):
        self.analyze_view.clear_data()
        self.stats = None
        self.analyze_mpl.clear()
        self.exportstats_button.setEnabled(False)
        self.select_view.clear_data()
        self.select_mpl.clear()
        self.copy_button.setEnabled(False)
        self.reduce_view.clear_data()
        self.reduce_mpl.clear()
        self.exportreduce_button.setEnabled(False)

    def get_options(self):
        featx = self.featx_combo.currentText()
        if featx == "NONE":
            featx = None
        featy = self.featy_combo.currentText()
        if featy == "NONE":
            featy = None
        return featx, featy

    def reduce_dimensions(self):
        if self.ordered is None:
            return
        self.reduce_button.setEnabled(False)
        self.reduce_button.setText("Reducing...")
        QApplication.processEvents()
        try:
            self.reduction, report = dataset.reduce(
                self.ordered,
                self.reduce_combo.currentText().lower(),
                self.whiten_combo.currentText().lower(),
                self.kernel_combo.currentText().lower(),
                self.fititer_spin.value(),
                self.fittol_spin.value(),
                self.alpha_spin.value(),
                self.outlier_combo.currentText().lower().split()[0],
                self.distance_combo.currentText().lower(),
                self.neighbors_spin.value(),
                self.leafsize_spin.value(),
                self.estimators_spin.value(),
                self.bootstrap_check.isChecked(),
                self.reduced_check.isChecked(),
                self.reduce_mpl,
            )
            self.reduce_view.update_data(report)
            self.exportreduce_button.setEnabled(True)
        except KeyboardInterrupt:
            QMessageBox.warning(
                self, "Dimensionality reduction", "Reduction interrupted"
            )
        self.reduce_button.setEnabled(True)
        self.reduce_button.setText("Reduce")

    def export_reduction(self):
        if self.reduction is None:
            return
        settings = QSettings()
        filename, extension = QFileDialog.getSaveFileName(
            self,
            "Export reduction",
            settings.value("export_reduction", "."),
            self.file_types["csv"],
        )
        if not filename:
            return
        settings.setValue("export_reduction", disk.get_folder(filename))
        try:
            headers = [f"Component {i+1}" for i in range(2)] + ["Target"]
            disk.save_csv(filename, self.reduction, headers=headers)
            filesize = disk.human_size(disk.file_size(filename))
            QMessageBox.information(
                self,
                "Export reduction",
                f"Reduction successfully exported ({filesize})",
            )
        except OSError:
            QMessageBox.critical(
                self, "Export reduction", "Unable to export reduction!"
            )

    def analyze_features(self):
        if self.ordered is None:
            return
        self.analyze_button.setEnabled(False)
        self.analyze_button.setText("Analyzing...")
        QApplication.processEvents()
        with contextlib.suppress(ValueError):
            featx, featy = self.get_options()
            df = self.ordered
            suffix = " [all classes]"
            if self.analyze_combo.currentIndex() == 0:
                self.stats = dataset.describe(df)
                self.analyze_view.update_data(self.stats)
                self.exportstats_button.setEnabled(True)
            else:
                if self.analyze_combo.currentIndex() == 1:
                    dataset.rangeplot(df, self.analyze_mpl)
                elif self.analyze_combo.currentIndex() == 2:
                    notch = self.notch_check.isChecked()
                    means = self.means_check.isChecked()
                    boxes = self.boxes_check.isChecked()
                    caps = self.caps_check.isChecked()
                    fliers = self.fliers_check.isChecked()
                    dataset.boxplot(
                        df, notch, means, boxes, caps, fliers, self.analyze_mpl
                    )
                elif self.analyze_combo.currentIndex() == 3:
                    means = self.means_check.isChecked()
                    extrema = self.extrema_check.isChecked()
                    dataset.violinplot(df, means, extrema, self.analyze_mpl)
                elif self.analyze_combo.currentIndex() == 7:
                    dataset.parallelplot(df, self.analyze_mpl)
                elif self.analyze_combo.currentIndex() == 8:
                    dataset.andrewsplot(df, self.analyze_mpl)
                elif self.analyze_combo.currentIndex() == 9:
                    dataset.scatmtxplot(df, self.analyze_mpl)
                elif self.analyze_combo.currentIndex() == 10:
                    dataset.radialplot(df, self.analyze_mpl)
                elif self.analyze_combo.currentIndex() == 11:
                    dataset.corrmtxplot(df, self.analyze_mpl)
                elif self.analyze_combo.currentIndex() == 12:
                    if featx is None:
                        QMessageBox.warning(
                            self,
                            "Analyze features",
                            "A feature X is needed for Bootstrap plot",
                        )
                        raise ValueError
                    dataset.bootstrapplot(df, featx, self.analyze_mpl)
                else:
                    if self.analyze_combo.currentIndex() == 4:
                        mode = "scatter"
                    elif self.analyze_combo.currentIndex() == 5:
                        mode = "hist"
                    elif self.analyze_combo.currentIndex() == 6:
                        mode = "density"
                    else:
                        raise ValueError
                    multiple = self.multiple_combo.currentText().lower()
                    filling = self.filling_check.isChecked()
                    logscale = self.logscale_check.isChecked()
                    dataset.statsplot(
                        df,
                        mode,
                        featx,
                        featy,
                        multiple,
                        filling,
                        logscale,
                        self.analyze_mpl,
                    )

                self.analyze_mpl.canvas.axes.set_title(
                    self.analyze_mpl.canvas.axes.get_title(), fontweight="bold"
                )
                self.analyze_mpl.refresh()
        self.analyze_button.setEnabled(True)
        self.analyze_button.setText("Analyze")

    def export_stats(self):
        if self.stats is None:
            return
        settings = QSettings()
        filename, extension = QFileDialog.getSaveFileName(
            self,
            "Export stats",
            settings.value("export_stats", "."),
            self.file_types["csv"],
        )
        if not filename:
            return
        settings.setValue("export_stats", disk.get_folder(filename))
        try:
            disk.save_csv(filename, self.stats.to_numpy(), headers=self.stats.columns)
            filesize = disk.human_size(disk.file_size(filename))
            QMessageBox.information(
                self, "Export stats", f"Statistics successfully exported ({filesize})"
            )
        except OSError:
            QMessageBox.critical(self, "Export stats", "Unable to export statistics!")

    def select_features(self):
        if self.ordered is None:
            return
        mode = self.select_combo.currentText().lower()
        score = self.score_combo.currentText().lower()
        estimator = self.estim_combo.currentText().lower()
        threshold = self.thr_spin.value()
        kbest = self.kbest_spin.value()
        trees = self.trees_spin.value()
        folds = self.folds_spin.value()
        rate = self.rate_spin.value()
        creg = self.creg_spin.value()
        tol = self.tol_spin.value()
        iters = self.iter_spin.value()
        self.select_button.setEnabled(False)
        self.select_button.setText("Selecting...")
        QApplication.processEvents()
        report = dataset.select(
            self.ordered,
            mode,
            score,
            estimator,
            threshold,
            kbest,
            trees,
            folds,
            rate,
            creg,
            tol,
            iters,
            self.select_mpl,
        )
        self.select_button.setEnabled(True)
        self.select_button.setText("Select")
        if report is not None:
            self.selection = report["Feature"].to_list()
            self.select_view.update_data(report)
            self.copy_button.setEnabled(True)
        else:
            self.selection = None

    def copy_selection(self):
        if self.selection is None:
            return
        self.featsel_edit.setText(",".join(self.selection))

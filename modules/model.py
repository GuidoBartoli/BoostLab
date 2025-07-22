import itertools
import json
from time import time

import xgboost as xgb
from PySide6.QtCore import QSettings, Signal, Slot
from PySide6.QtGui import QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lib import dataset, disk, gui, model


# TODO: Add partial dependence plots to feature importance tab (part of model inspection)
# (https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html
# TODO: Add sort "by importance" or "by index" option to feature importance tab
# TODO: Add binary threshold to error metric (f"error@{args.binthr}")
# TODO: Disable AUC and AUCPR if HistGPU is selected as method


class ModelWidget(QWidget):
    model_ready = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # FILE
        self.file_types = {
            "model": "UBJ files (*.ubj);;JSON files (*.json);;Model files (*.ubj *.json)",
            "export": "C/C++ files (*.c *.cpp);;Python files (*.py);;Text files (*.json *.txt);;Export files (*.c *.cpp *.py)",
            "params": "Params files (*.json)",
            "image": "Image files (*.png)",
            "csv": "CSV files (*.csv)",
        }

        load_button = QPushButton("Load...")
        load_button.setIcon(QIcon(":/load"))
        load_button.setShortcut(QKeySequence.Open)
        load_button.clicked.connect(self.load_model)
        load_button.setToolTip("Load a saved model from file")
        self.save_button = QPushButton("Save...")
        self.save_button.setIcon(QIcon(":/save"))
        self.save_button.setShortcut(QKeySequence.Save)
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setToolTip("Save current model to file")
        self.export_button = QPushButton("Export...")
        self.export_button.setIcon(QIcon(":/export"))
        self.export_button.setShortcut(QKeySequence("Ctrl+E"))
        self.export_button.clicked.connect(self.export_model)
        self.export_button.setToolTip("Convert current model to Python/C++ code")

        self.include_check = QCheckBox("Include header")
        self.include_check.setToolTip(
            "Include corresponding header file in exported C code"
        )
        self.integer_check = QCheckBox("Integer features")
        self.integer_check.setToolTip(
            "Use decimal input features instead of floating point numbers"
        )
        self.macro_check = QCheckBox("Use SRAM macro")
        self.macro_check.setToolTip(
            "Add C macro to compile predict() function in SRAM segment"
        )
        self.comments_check = QCheckBox("Prepend comments")
        self.comments_check.setToolTip(
            "Insert model details in the comments of exported C code"
        )
        self.double_check = QCheckBox("Double precision")
        self.double_check.setToolTip(
            "Use double precision floating point numbers for C code"
        )
        self.stats_check = QCheckBox("Add split statistics")
        self.stats_check.setToolTip(
            "Add split statistics to the model exported in text format"
        )
        self.inline_check = QCheckBox("Inline sigmoid()")
        self.inline_check.setToolTip(
            "Compile sigmoid function inline in exported C code"
        )
        self.memcpy_check = QCheckBox("Output memcpy()")
        self.memcpy_check.setToolTip(
            "Use memcpy() function to copy data in exported C code"
        )
        self.predict_label = QLabel("Prediction function:")
        self.predict_edit = QLineEdit()
        self.predict_edit.setToolTip(
            "Name generated for the predict() function in exported C code"
        )

        button_layout = QHBoxLayout()
        button_layout.addWidget(load_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.save_button)

        output_layout = QGridLayout()
        output_layout.addWidget(self.include_check, 0, 0)
        output_layout.addWidget(self.integer_check, 0, 1)
        output_layout.addWidget(self.inline_check, 1, 0)
        output_layout.addWidget(self.comments_check, 1, 1)
        output_layout.addWidget(self.double_check, 2, 0)
        output_layout.addWidget(self.stats_check, 2, 1)
        output_layout.addWidget(self.macro_check, 3, 0)
        output_layout.addWidget(self.memcpy_check, 3, 1)
        output_layout.addWidget(self.predict_label, 4, 0)
        output_layout.addWidget(self.predict_edit, 4, 1)

        file_layout = QVBoxLayout()
        file_layout.addLayout(button_layout)
        file_layout.addLayout(output_layout)
        file_group = QGroupBox("FILE")
        file_group.setLayout(file_layout)

        # INFORMATION
        self.info_view = gui.PandasView(alternate=False, index=False)
        info_layout = QVBoxLayout()
        info_layout.addWidget(self.info_view)
        self.info_group = QGroupBox("DETAILS")
        self.info_group.setLayout(info_layout)

        # TRAINING
        self.rounds_spin = QSpinBox()
        self.rounds_spin.setRange(10, 2000)
        self.rounds_spin.setSingleStep(5)
        self.rounds_spin.setToolTip("Number of boosting iterations")
        self.early_spin = QSpinBox()
        self.early_spin.setRange(0, 500)
        self.early_spin.setSingleStep(5)
        self.early_spin.setSpecialValueText("OFF")
        self.early_spin.setToolTip(
            "Validation metric needs to improve at least once in these rounds to continue training"
        )
        self.objective_combo = QComboBox()
        self.objective_combo.addItems(
            [
                "Logistic",
                "Logitraw",
            ]
        )
        self.objective_combo.setToolTip(
            "Binary objective function used for training<br><br>"
            "<b>Logistic</b>: Logistic regression for binary classification (probability output)<br>"
            "<b>Logitraw</b>: Logistic regression for binary classification (raw score output)<br>"
            "<b>Hinge</b>: Hinge loss for binary classification (prediction output is 0 or 1)<br>"
        )
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 100000)
        self.seed_spin.setToolTip("Random number seed for reproducibility")
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Logloss", "Error", "AUC", "AUCPR"])
        self.metric_combo.setToolTip(
            "Target metric used for early stopping and parameter tuning<br><br>"
            "<b>Error</b>: Classification error rate. It is calculated as <i>(wrong cases)/(all cases)</i><br>"
            "<b>AUC</b>: Receiver Operating Characteristic Area under the Curve<br>"
            "<b>AUCPR</b>: Area under the Precision-Recall Curve"
        )
        self.construct_combo = QComboBox()
        self.construct_combo.addItems(["Auto", "Exact", "Approx", "HistCPU", "HistGPU"])
        self.construct_combo.setToolTip(
            "Algorithm for constructing a new tree<br><br>"
            "<b>Auto</b>: Use heuristic to choose between <i>Exact</i> and <i>Approx</i><br>"
            "<b>Exact</b>: Exact greedy algorithm<br>"
            "<b>Approx</b>: Approximate greedy algorithm using quantile sketch and gradient histogram<br>"
            "<b>HistCPU</b>: Fast histogram optimized approximate greedy algorithm<br>"
            "<b>HistGPU</b>: GPU implementation of <i>HistCPU</i> algorithm"
        )
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["Uniform", "Gradient"])
        self.sampling_combo.setToolTip(
            "Algorithm for sampling the training data<br><br>"
            "<b>Uniform</b>: Each training instance has an equal probability of being selected<br>"
            "<b>Gradient</b>: The selection probability for each training instance is proportional<br>"
            "to the regularized absolute value of gradients (more specifically, <i>sqrt(g^2 + l*h^2)</i>)"
        )
        self.growing_combo = QComboBox()
        self.growing_combo.addItems(["Depthwise", "Lossguide"])
        self.growing_combo.setToolTip(
            "Algorithm for growing a new node in a tree<br><br>"
            "<b>Depthwise</b>: Split at nodes closest to the root<br>"
            "<b>Lossguide</b>: Split at nodes with highest loss change"
        )
        self.train_button = QPushButton("Train")
        self.train_button.setIcon(QIcon(":/train"))
        self.train_button.setShortcut(QKeySequence("Ctrl+T"))
        self.train_button.setToolTip("Train the model using the provided parameters")
        self.train_button.clicked.connect(self.train_model)
        gui.modify_font(self.train_button, bold=True)

        train_layout = QGridLayout()
        train_layout.addWidget(QLabel("Iterations:"), 0, 0)
        train_layout.addWidget(self.rounds_spin, 0, 1)
        train_layout.addWidget(QLabel("Early stop:"), 1, 0)
        train_layout.addWidget(self.early_spin, 1, 1)
        train_layout.addWidget(QLabel("Rand seed:"), 2, 0)
        train_layout.addWidget(self.seed_spin, 2, 1)
        train_layout.addWidget(QLabel("Objective:"), 3, 0)
        train_layout.addWidget(self.objective_combo, 3, 1)
        train_layout.addWidget(QLabel("Metric:"), 0, 2)
        train_layout.addWidget(self.metric_combo, 0, 3)
        train_layout.addWidget(QLabel("Method:"), 1, 2)
        train_layout.addWidget(self.construct_combo, 1, 3)
        train_layout.addWidget(QLabel("Sampling:"), 2, 2)
        train_layout.addWidget(self.sampling_combo, 2, 3)
        train_layout.addWidget(QLabel("Growing:"), 3, 2)
        train_layout.addWidget(self.growing_combo, 3, 3)
        train_layout.addWidget(self.train_button, 4, 0, 1, 4)
        self.train_group = QGroupBox("TRAINING")
        self.train_group.setLayout(train_layout)

        # PARAMETERS
        self.model_params = {
            "max_depth": {
                "label": "Max tree depth:",
                "min": 1,
                "max": 12,
                "def": 6,
                "step": 1,
                "double": False,
                "tooltip": "Maximum depth of a tree.\n"
                "Increasing this value will make the model more complex and more likely to overfit.\n"
                "Beware that XGBoost aggressively consumes memory when training a deep tree.\n"
                "Range: [0, ∞], Default: 6",
            },
            "learning_rate": {
                "label": "Learning rate:",
                "min": 0,
                "max": 1,
                "def": 0.3,
                "step": 0.01,
                "double": True,
                "tooltip": "Step size shrinkage used in update to prevents overfitting.\n"
                "After each boosting step, we can directly get the weights of new features,\n"
                "and eta shrinks the feature weights to make the boosting process more conservative.\n"
                "Range: [0, 1], Default: 0.3",
            },
            "subsample": {
                "label": "Row subsample:",
                "min": 0.01,
                "max": 1,
                "def": 1,
                "step": 0.01,
                "double": True,
                "tooltip": "Subsample ratio of the training instances.\n"
                "Setting it to 0.5 means that XGBoost would randomly sample\n"
                "half of the training data prior to growing trees\n"
                "and this will prevent overfitting. "
                "Subsampling will occur once in every boosting iteration.\n"
                "Range: (0, 1], Default: 1",
            },
            "colsample_bytree": {
                "label": "Col subsample:",
                "min": 0.01,
                "max": 1,
                "def": 1,
                "step": 0.01,
                "double": True,
                "tooltip": "Subsample ratio of columns when constructing each tree.\n"
                "Subsampling occurs once for every tree constructed.\n"
                "Range: (0, 1], Default: 1",
            },
            "min_child_weight": {
                "label": "Min child weight:",
                "min": 0,
                "max": 100,
                "def": 1,
                "step": 0.1,
                "double": True,
                "tooltip": "Minimum sum of instance weight (hessian) needed in a child.\n"
                "If the tree partition step results in a leaf node with the sum of instance weight\n"
                "less than min_child_weight, then the building process will give up further partitioning.\n"
                "The larger min_child_weight is, the more conservative the algorithm will be.\n"
                "Range: [0, ∞], Default: 1",
            },
            "min_split_loss": {
                "label": "Min split loss:",
                "min": 0,
                "max": 100,
                "def": 0,
                "step": 0.01,
                "double": True,
                "tooltip": "Minimum loss reduction required to make a further partition on a leaf node of the tree.\n"
                "The larger gamma is, the more conservative the algorithm will be.\n"
                "Range: [0, ∞], Default: 0",
            },
            "reg_alpha": {
                "label": "L1-norm alpha:",
                "min": 0,
                "max": 100,
                "def": 0,
                "step": 0.01,
                "double": True,
                "tooltip": "L1 regularization term on weights.\n"
                "Increasing this value will make model more conservative.\n"
                "Range: [0, ∞], Default: 0",
            },
            "reg_lambda": {
                "label": "L2-norm lambda:",
                "min": 0,
                "max": 100,
                "def": 1,
                "step": 0.01,
                "double": True,
                "tooltip": "L2 regularization term on weights.\n"
                "Increasing this value will make model more conservative.\n"
                "Range: [0, ∞], Default: 1",
            },
            "max_delta_step": {
                "label": "Max delta step:",
                "min": 0,
                "max": 100,
                "def": 0,
                "step": 0.01,
                "double": True,
                "tooltip": "Maximum delta step we allow each leaf output to be.\n"
                "If the value is set to 0, it means there is no constraint.\n"
                "If it is set to a positive value, it can help making the update step more conservative.\n"
                "Usually this parameter is not needed, but it might help in logistic regression\n"
                "when class is extremely imbalanced.\n"
                "Range: [0, ∞], Default: 0",
            },
            "scale_pos_weight": {
                "label": "Weight scale:",
                "min": 0.01,
                "max": 10,
                "def": 1,
                "step": 0.01,
                "double": True,
                "tooltip": "Control the balance of positive and negative weights, useful for unbalanced classes.\n"
                "A typical value to consider: sum(negative instances) / sum(positive instances)\n"
                "Default: 1",
            },
        }

        params_layout = QGridLayout()
        min_label = QLabel("Minimum")
        min_label.setToolTip("Lower bound value during parameter tuning")
        gui.modify_font(min_label, italic=True)
        params_layout.addWidget(min_label, 0, 1)
        max_label = QLabel("Maximum")
        max_label.setToolTip("Upper bound value during parameter tuning")
        gui.modify_font(max_label, italic=True)
        params_layout.addWidget(max_label, 0, 2)
        offset = 1
        self.decimals = 6
        for row, (n, p) in enumerate(self.model_params.items()):
            spin1 = QDoubleSpinBox() if p["double"] else QSpinBox()
            spin1.setRange(p["min"], p["max"])
            spin1.setSingleStep(p["step"])
            if p["double"]:
                spin1.setDecimals(self.decimals)
            spin2 = QDoubleSpinBox() if p["double"] else QSpinBox()
            spin2.setRange(p["min"], p["max"])
            spin2.setSingleStep(p["step"])
            spin2.setSpecialValueText("OFF")
            if p["double"]:
                spin2.setDecimals(self.decimals)
            label = QLabel(p["label"])
            label.setToolTip(p["tooltip"])
            params_layout.addWidget(label, row + offset, 0)
            params_layout.addWidget(spin1, row + offset, 1)
            params_layout.addWidget(spin2, row + offset, 2)
            self.model_params[n]["spin1"] = spin1
            self.model_params[n]["spin2"] = spin2

        total_params = len(self.model_params) + offset
        loadparam_button = QPushButton("Load...")
        loadparam_button.setIcon(QIcon(":/load"))
        loadparam_button.clicked.connect(self.load_params)
        params_layout.addWidget(loadparam_button, total_params, 0)
        saveparam_button = QPushButton("Save...")
        saveparam_button.setIcon(QIcon(":/save"))
        saveparam_button.clicked.connect(self.save_params)
        params_layout.addWidget(saveparam_button, total_params, 1)
        defaultparam_button = QPushButton("Default")
        defaultparam_button.setIcon(QIcon(":/reset"))
        defaultparam_button.clicked.connect(self.reset_params)
        params_layout.addWidget(defaultparam_button, total_params, 2)
        self.params_group = QGroupBox("PARAMETERS")
        self.params_group.setLayout(params_layout)

        # TUNING
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(8, 8192)
        self.trials_spin.setSingleStep(8)
        self.trials_spin.setToolTip("Total number of trials used for optimization")
        self.report_check = QCheckBox("Show final report")
        self.report_check.setToolTip(
            "Show graphical report of tuning results after optimization"
        )

        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(
            [
                "Random sampling",
                "Parzen estimator",
                "Covariance matrix",
                "Quasi Monte-Carlo",
                "Brute force search",
            ]
        )
        self.sampler_combo.setToolTip(
            "Algorithm for sampling the next hyperparameter values to evaluate<br><br>"
            "<b>Random sampling</b>: This sampler uses random samples for suggesting values<br>"
            "<b>Parzen estimator</b>: Sampler using TPE (Tree-structured Parzen Estimator) algorithm<br>"
            "On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) l(x)<br>"
            "to the set of parameter values associated with the best objective values,<br>"
            "and another GMM g(x) to the remaining parameter values<br>"
            "It chooses the parameter value x that maximizes the ratio l(x)/g(x)<br>"
            "<b>Covariance matrix</b>: Sampler with Covariance Matrix Adaptation Evolution Strategy<br>"
            "CMA-ES is a stochastic derivative-free numerical optimization method<br>"
            "<b>Quasi Monte-Carlo</b>: A Quasi Monte Carlo Sampler that generates low-discrepancy<br>"
            "QMC sequences have lower discrepancies than standard random sequences<br>"
            "They are known to perform better than the standard random sequences<br>"
            "<b>Brute Force</b>: This sampler performs exhaustive search on the defined search space."
        )

        self.restart_combo = QComboBox()
        self.restart_combo.addItems(
            ["Disabled", "Increasing population", "Bi-population strategy"]
        )
        self.restart_combo.setToolTip(
            "Strategy for restarting CMA-ES optimization when objective converges to a local minimum<br><br>"
            "<b>Disabled</b>: CMA-ES will not restart<br>"
            "<b>Increasing population</b>: CMA-ES will restart with increasing population size<br>"
            "<b>Bi-population strategy</b>: CMA-ES will restart with the population size increased or decreased<br>"
        )

        self.multivar_check = QCheckBox("Multivariate TPE")
        self.multivar_check.setToolTip(
            "This algorithm uses multivariate kernel density estimation\n"
            "to estimate the joint density of the parameters.\n"
            "This algorithm outperforms TPE on non-separable functions."
        )

        self.group_check = QCheckBox("Group search TPE")
        self.group_check.setToolTip(
            "Use group decomposed search space is used when suggesting parameters.\n"
            "The sampling algorithm decomposes the search space based on past trials and samples\n"
            "from the joint distribution in each decomposed subspace.\n"
            "The decomposed subspaces are a partition of the whole search space.\n"
            "Each subspace is a maximal subset of the whole search space, which satisfies the following:\n"
            "for a completed trial, the intersection of the subspace and the search space of the trial\n"
            "becomes subspace itself or an empty set.\n"
            "Sampling from the joint distribution on the subspace is realized by multivariate TPE."
        )

        self.separable_check = QCheckBox("Separable covariance")
        self.separable_check.setToolTip(
            "The covariance matrix is constrained to be diagonal\n"
            "and learning rate is increased to reduce complexity.\n"
            "This algorithm outperforms CMA-ES on separable functions."
        )

        self.margin_check = QCheckBox("CMA-ES with margin")
        self.margin_check.setToolTip(
            "This algorithm prevents samples in each discrete distribution from being fixed to a single point.\n"
            "The margin is the minimum distance between the best objective value and the current population."
        )

        self.pruner_combo = QComboBox()
        self.pruner_combo.addItems(
            ["Disabled", "Median", "Percentile", "Hyperband", "Halving"]
        )
        self.pruner_combo.setToolTip(
            "Algorithm for pruning unpromising trials<br><br>"
            "<b>Disabled</b>: No trials are pruned.<br>"
            "<b>Median</b>: Prune if the trial’s best intermediate result is worse than median\n"
            "of intermediate results of previous trials at the same step.<br>"
            "<b>Percentile</b>: Prune if the best intermediate value is in the bottom percentile\n"
            "among trials at the same step.<br>"
            "<b>Hyperband</b>: As Halving requires the number of configurations <i>n</i> as its hyperparameter.<br>"
            "For a given budget <i>B</i>, all the configurations have the resources of <i>B/n</i> on average.<br>"
            "As you can see, there will be a trade-off of <i>B</i> and <i>B/n</i>.<br>"
            "Hyperband attacks this trade-off by trying different <i>n</i> values for a fixed budget.<br>"
            "<b>Halving</b>: Successive Halving is a bandit-based algorithm to identify the best configuration."
        )

        self.startup_spin = QSpinBox()
        self.startup_spin.setRange(0, 100)
        self.startup_spin.setSuffix("%")
        self.startup_spin.setToolTip(
            "Pruning is disabled until the given ratio of trials finish in the same study"
        )

        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 100)
        self.warmup_spin.setSuffix("%")
        self.warmup_spin.setToolTip(
            "Pruning is disabled until the trial exceeds the given ratio of steps"
        )

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 10)
        self.interval_spin.setSuffix(" step(s)")
        self.interval_spin.setToolTip(
            "Interval in number of steps between the pruning checks, offset by the warmup steps"
        )

        self.percent_spin = QSpinBox()
        self.percent_spin.setRange(1, 100)
        self.percent_spin.setSuffix("%")
        self.percent_spin.setToolTip("Percentile threshold used for pruning trials")

        self.reduction_spin = QSpinBox()
        self.reduction_spin.setRange(1, 10)
        self.reduction_spin.setSuffix(" x")
        self.reduction_spin.setToolTip(
            "A parameter <i>r</i> for specifying reduction factor of promotable trials.<br>"
            "At the completion point of each rung, about <i>1/r</i> trials will be promoted."
        )

        self.resource_spin = QSpinBox()
        self.resource_spin.setRange(0, 10)
        self.resource_spin.setSuffix(" step(s)")
        self.resource_spin.setSpecialValueText("auto")
        self.resource_spin.setToolTip(
            "A parameter for specifying the minimum resource allocated to a trial.<br>"
            "A smaller value will give a result faster, but a larger one will give a better guarantee<br>"
            "of successful judging between configurations."
        )

        self.tune_button = QPushButton("Tune")
        self.tune_button.setIcon(QIcon(":/tune"))
        self.tune_button.setShortcut(QKeySequence("Ctrl+U"))
        self.tune_button.setToolTip(
            "Optimal tune of hyperparameters inside the specified ranges"
        )
        self.tune_button.clicked.connect(self.tune_model)
        gui.modify_font(self.tune_button, bold=True)

        trials_layout = QHBoxLayout()
        trials_layout.addWidget(QLabel("Optimization trials:"))
        trials_layout.addWidget(self.trials_spin)
        trials_layout.addWidget(self.report_check)

        sampler_layout = QGridLayout()
        sampler_layout.addWidget(QLabel("Sampling algorithm:"), 0, 0)
        sampler_layout.addWidget(self.sampler_combo, 0, 1)
        sampler_layout.addWidget(QLabel("CMA restart strategy:"), 1, 0)
        sampler_layout.addWidget(self.restart_combo, 1, 1)
        sampler_layout.addWidget(self.multivar_check, 2, 0)
        sampler_layout.addWidget(self.separable_check, 2, 1)
        sampler_layout.addWidget(self.group_check, 3, 0)
        sampler_layout.addWidget(self.margin_check, 3, 1)
        sampler_widget = QWidget()
        sampler_widget.setLayout(sampler_layout)

        pruner_layout = QGridLayout()
        pruner_layout.addWidget(QLabel("Pruning algorithm:"), 0, 0, 1, 2)
        pruner_layout.addWidget(self.pruner_combo, 0, 2, 1, 2)
        pruner_layout.addWidget(QLabel("Reduction:"), 1, 0)
        pruner_layout.addWidget(self.reduction_spin, 1, 1)
        pruner_layout.addWidget(QLabel("Resource:"), 1, 2)
        pruner_layout.addWidget(self.resource_spin, 1, 3)
        pruner_layout.addWidget(QLabel("Startup:"), 2, 0)
        pruner_layout.addWidget(self.startup_spin, 2, 1)
        pruner_layout.addWidget(QLabel("Interval:"), 2, 2)
        pruner_layout.addWidget(self.interval_spin, 2, 3)
        pruner_layout.addWidget(QLabel("Warmup:"), 3, 0)
        pruner_layout.addWidget(self.warmup_spin, 3, 1)
        pruner_layout.addWidget(QLabel("Percentile:"), 3, 2)
        pruner_layout.addWidget(self.percent_spin, 3, 3)
        pruner_widget = QWidget()
        pruner_widget.setLayout(pruner_layout)

        tuning_tab = QTabWidget()
        tuning_tab.addTab(sampler_widget, "Sampler")
        tuning_tab.setTabIcon(0, QIcon(":/sampler"))
        tuning_tab.addTab(pruner_widget, "Pruner")
        tuning_tab.setTabIcon(1, QIcon(":/pruner"))
        # tuning_tab.tabBar().setDocumentMode(True)
        # tuning_tab.tabBar().setExpanding(True)

        tuning_layout = QVBoxLayout()
        tuning_layout.addLayout(trials_layout)
        tuning_layout.addWidget(tuning_tab)
        tuning_layout.addWidget(self.tune_button)
        self.tuning_group = QGroupBox("TUNING")
        self.tuning_group.setLayout(tuning_layout)

        # RESULTS
        self.results_widgets = [
            gui.MplWidget(),
            gui.MplWidget(),
            gui.MplWidget(),
            gui.MplWidget(),
        ]
        results_layout = QGridLayout()
        for i, j in itertools.product(range(2), range(2)):
            results_layout.addWidget(self.results_widgets[i * 2 + j], i, j)
        results_widget = QWidget()
        results_widget.setLayout(results_layout)

        # VISUALIZATION
        showtree_button = QPushButton("Show")
        showtree_button.setIcon(QIcon(":/show"))
        showtree_button.clicked.connect(self.show_tree)
        showtree_button.setToolTip("Show graph of selected tree")
        self.tree_spin = QSpinBox()
        self.tree_spin.setToolTip("Index of tree to show")
        self.tree_mpl = gui.MplWidget(toolbar=True)
        tree_layout = QHBoxLayout()
        tree_layout.addWidget(showtree_button)
        tree_layout.addWidget(QLabel("Tree index:"))
        tree_layout.addWidget(self.tree_spin)
        tree_layout.addStretch()
        visual_layout = QVBoxLayout()
        visual_layout.addLayout(tree_layout)
        visual_layout.addWidget(self.tree_mpl)
        visual_widget = QWidget()
        visual_widget.setLayout(visual_layout)

        # SHAP
        explain_button = QPushButton("Explain")
        explain_button.setIcon(QIcon(":/explain"))
        explain_button.clicked.connect(self.show_shap)
        explain_button.setToolTip(
            "Explain predictions using SHAP (SHapley Additive exPlanations)"
        )
        self.shap_combo = QComboBox()
        self.shap_combo.addItems(
            ["Average", "Scatter", "Force", "Beeswarm", "Waterfall"]
        )
        self.shap_combo.setToolTip(
            "SHAP plot type<br><br>"
            "<b>Average</b>: Plot the mean absolute value for each feature column as a bar chart.<br>"
            "<b>Scatter</b>: SHAP dependence scatter plot, colored by an interaction feature.<br>"
            "Plots the value of the feature on the x-axis and the SHAP value of the same feature on the y-axis.<br>"
            "This shows how the model depends on the given feature, thus a richer extension of classical partial<br>"
            "dependence plots. Vertical dispersion of the data points represents interaction effects.<br>"
            "Grey ticks along the y-axis are data points where the feature’s value was NaN.<br>"
            "<b>Force</b>: Visualize the given SHAP values with an additive force layout.<br>"
            "<b>Beeswarm</b>: Create a SHAP beeswarm plot, colored by feature values when they are provided.<br>"
            "<b>Waterfall</b>: Plots an explanation of a single prediction as a waterfall plot.<br>"
            "The SHAP value of a feature is the impact provided by that feature on the model’s output.<br>"
            "The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature<br>"
            "move the model output from our prior expectation under the background data distribution,<br>"
            "to the final model prediction given the evidence of all the features."
        )
        self.row_spin = QSpinBox()
        self.row_spin.setToolTip("Index of sample to explain (force/waterfall plot)")
        self.col_spin = QSpinBox()
        self.col_spin.setToolTip("Index of feature to explain (scatter plot)")
        saveshap_button = QPushButton("Export...")
        saveshap_button.clicked.connect(self.save_shap)
        saveshap_button.setToolTip("Export computed SHAP values to CSV file")

        shap_layout = QHBoxLayout()
        shap_layout.addWidget(explain_button)
        shap_layout.addWidget(self.shap_combo)
        shap_layout.addWidget(QLabel("Sample:"))
        shap_layout.addWidget(self.row_spin)
        shap_layout.addWidget(QLabel("Feature:"))
        shap_layout.addWidget(self.col_spin)
        shap_layout.addWidget(saveshap_button)
        shap_group = QGroupBox("SHAP")
        shap_group.setLayout(shap_layout)

        # PERMUTATION
        permute_button = QPushButton("Permute")
        permute_button.setIcon(QIcon(":/permute"))
        permute_button.clicked.connect(self.show_perm)
        permute_button.setToolTip(
            "First, a baseline metric, defined by scoring, is evaluated on a (potentially different) dataset.<br>"
            "Next, a feature column from the validation set is permuted and the metric is evaluated again.<br>"
            "The permutation importance is defined to be the difference between the baseline metric<br>"
            "and metric from permutating the feature column."
        )
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 50)
        self.repeats_spin.setValue(5)
        self.repeats_spin.setToolTip("Number of times to permute each feature column")
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1, 100)
        self.samples_spin.setValue(100)
        self.samples_spin.setSuffix("%")
        self.samples_spin.setToolTip(
            "Percentage of samples to use for permutation (randomly selected)"
        )
        saveperm_button = QPushButton("Export...")
        saveperm_button.clicked.connect(self.save_perm)
        saveperm_button.setToolTip("Export computed permutation importance to CSV file")

        permute_layout = QHBoxLayout()
        permute_layout.addWidget(permute_button)
        permute_layout.addWidget(QLabel("Repeats:"))
        permute_layout.addWidget(self.repeats_spin)
        permute_layout.addWidget(QLabel("Samples:"))
        permute_layout.addWidget(self.samples_spin)
        permute_layout.addWidget(saveperm_button)
        permute_group = QGroupBox("Permutation")
        permute_group.setLayout(permute_layout)

        # IMPORTANCE
        self.gain_mpl = gui.MplWidget(toolbar=True)
        top_layout = QHBoxLayout()
        top_layout.addWidget(shap_group)
        top_layout.addWidget(permute_group)
        top_layout.addStretch()
        top_widget = QWidget()
        top_widget.setLayout(top_layout)
        top_widget.setMaximumHeight(90)
        importance_layout = QVBoxLayout()
        importance_layout.addWidget(top_widget)
        importance_layout.addWidget(self.gain_mpl)
        importance_widget = QWidget()
        importance_widget.setLayout(importance_layout)

        # MAIN
        col_width = 420
        col1_layout = QVBoxLayout()
        col1_layout.addWidget(file_group)
        col1_layout.addWidget(self.info_group)
        col1_widget = QWidget()
        col1_widget.setLayout(col1_layout)
        col1_widget.setMaximumWidth(col_width)

        col2_layout = QVBoxLayout()
        col2_layout.addWidget(self.train_group)
        col2_layout.addWidget(self.params_group)
        col2_layout.addWidget(self.tuning_group)
        col2_layout.addStretch()
        col2_widget = QWidget()
        col2_widget.setLayout(col2_layout)
        col2_widget.setMaximumWidth(col_width)

        self.right_widget = QTabWidget()
        self.right_widget.addTab(results_widget, "Training results")
        self.right_widget.setTabIcon(0, QIcon(":/results"))
        self.right_widget.addTab(visual_widget, "Tree visualization")
        self.right_widget.setTabIcon(1, QIcon(":/tree"))
        self.right_widget.addTab(importance_widget, "Feature importance")
        self.right_widget.setTabIcon(2, QIcon(":/importance"))

        main_layout = QHBoxLayout()
        main_layout.addWidget(col1_widget)
        main_layout.addWidget(col2_widget)
        main_layout.addWidget(self.right_widget)
        self.setLayout(main_layout)
        self.set_defaults()

    def set_defaults(self) -> None:
        self.include_check.setChecked(True)
        self.integer_check.setChecked(False)
        self.double_check.setChecked(False)
        self.comments_check.setChecked(True)
        self.macro_check.setChecked(False)
        self.stats_check.setChecked(False)
        self.inline_check.setChecked(True)
        self.memcpy_check.setChecked(False)
        self.predict_edit.setText("predict")

        self.tree_spin.setEnabled(False)
        self.train_group.setEnabled(False)
        self.info_group.setEnabled(False)
        self.params_group.setEnabled(False)
        self.tuning_group.setEnabled(False)
        self.right_widget.setEnabled(False)

        self.export_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.include_check.setEnabled(False)
        self.integer_check.setEnabled(False)
        self.inline_check.setEnabled(False)
        self.comments_check.setEnabled(False)
        self.double_check.setEnabled(False)
        self.stats_check.setEnabled(False)
        self.macro_check.setEnabled(False)
        self.memcpy_check.setEnabled(False)
        self.predict_label.setEnabled(False)
        self.predict_edit.setEnabled(False)
        self.reset_params()

    def get_training(self) -> dict:
        params = {
            n: round(p["spin1"].value(), self.decimals)
            for n, p in self.model_params.items()
        }
        params["boosting_rounds"] = self.rounds_spin.value()
        early_stopping = self.early_spin.value()
        params["early_stopping"] = None if early_stopping == 0 else early_stopping
        props = dataset.properties(self.parent.train)
        params["num_classes"] = props["classes"]
        params["class_labels"] = props["labels"]
        params["num_samples"] = props["samples"]
        params["num_features"] = props["features"]
        params["feature_names"] = props["columns"]
        objective = self.objective_combo.currentText().lower()
        metric = self.metric_combo.currentText().lower()
        # TODO: Add AUC and AUCPR metrics laeving logloss as last item for early stopping
        if props["classes"] == 2:
            params["objective"] = f"binary:{objective}"
            metrics = [
                self.metric_combo.itemText(i).lower()
                for i in range(self.metric_combo.count())
            ]
            metrics.remove(metric)
            params["eval_metric"] = metrics + [metric]
            # FIXME: Add "error@t" binary metric with custom threshold
            # TODO: https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
        else:
            params["objective"] = "multi:softprob"
            if metric == "error":
                metric = "merror"
            params["eval_metric"] = [metric, "mlogloss"]
            params["num_class"] = props["classes"]  # compatibility with scikit-learn
        params["seed"] = self.seed_spin.value()
        construct = self.construct_combo.currentText()
        if construct in ["Auto", "Exact", "Approx"]:
            params["tree_method"] = construct.lower()
        else:
            if construct == "HistCPU":
                params["tree_method"] = "hist"
            elif construct == "HistGPU":
                params["tree_method"] = "gpu_hist"
        sampling = self.sampling_combo.currentText()
        if sampling == "Uniform":
            params["sampling_method"] = "uniform"
        elif sampling == "Gradient":
            params["sampling_method"] = "gradient_based"
        params["grow_policy"] = self.growing_combo.currentText().lower()
        return params

    def set_training(self, params: dict) -> None:
        for n, p in self.model_params.items():
            p["spin1"].setValue(params[n])
            p["spin2"].setValue(0)
        self.rounds_spin.setValue(params["boosting_rounds"])
        if params["early_stopping"] is None:
            self.early_spin.setValue(0)
        else:
            self.early_spin.setValue(params["early_stopping"])
        self.seed_spin.setValue(params["seed"])
        objective = params["objective"].split(":")[1]
        # FIXME: Does not work with multiclass classification (check for 'multi' prefix in objective?)
        self.objective_combo.setCurrentText(objective.capitalize())
        metric = params["eval_metric"][-1]
        if any(m in metric for m in ["error", "loss"]):
            self.metric_combo.setCurrentText(metric.capitalize())
        else:
            self.metric_combo.setCurrentText(metric.upper())
        construct = params["tree_method"]
        if construct in ["auto", "exact", "approx"]:
            self.construct_combo.setCurrentText(construct.capitalize())
        else:
            if construct == "hist":
                self.construct_combo.setCurrentText("HistCPU")
            elif construct == "gpu_hist":
                self.construct_combo.setCurrentText("HistGPU")
        sampling = params["sampling_method"]
        if sampling == "uniform":
            self.sampling_combo.setCurrentText("Uniform")
        elif sampling == "gradient_based":
            self.sampling_combo.setCurrentText("Gradient")
        self.growing_combo.setCurrentText(params["grow_policy"].capitalize())

    def reset_params(self) -> None:
        for p in self.model_params.values():
            p["spin1"].setValue(p["def"])
            p["spin2"].setValue(0)

        self.rounds_spin.setValue(100)
        self.early_spin.setValue(10)
        self.seed_spin.setValue(0)
        self.objective_combo.setCurrentIndex(0)
        self.metric_combo.setCurrentIndex(0)
        self.construct_combo.setCurrentIndex(0)
        self.sampling_combo.setCurrentIndex(0)
        self.growing_combo.setCurrentIndex(0)

        self.trials_spin.setValue(1024)
        self.sampler_combo.setCurrentIndex(1)
        self.restart_combo.setCurrentIndex(0)
        self.pruner_combo.setCurrentIndex(3)
        self.startup_spin.setValue(25)
        self.warmup_spin.setValue(50)
        self.interval_spin.setValue(1)
        self.percent_spin.setValue(25)
        self.reduction_spin.setValue(3)
        self.resource_spin.setValue(1)

    def get_tuning(self) -> tuple[dict, dict]:
        tuning_params = {
            "trials": self.trials_spin.value(),
            "report": self.report_check.isChecked(),
            "sampler": self.sampler_combo.currentText(),
            "restart": self.restart_combo.currentText(),
            "multivar": self.multivar_check.isChecked(),
            "group": self.group_check.isChecked(),
            "separable": self.separable_check.isChecked(),
            "margin": self.margin_check.isChecked(),
            "pruner": self.pruner_combo.currentText(),
            "startup": self.startup_spin.value(),
            "warmup": self.warmup_spin.value(),
            "interval": self.interval_spin.value(),
            "percent": self.percent_spin.value(),
            "reduction": self.reduction_spin.value(),
            "resource": self.resource_spin.value(),
        }
        tuning_ranges = {
            n: [p["spin1"].value(), p["spin2"].value()]
            for n, p in self.model_params.items()
        }
        for k, v in tuning_ranges.items():
            if v[1] < v[0]:
                tuning_ranges[k][1] = tuning_ranges[k][0]
        return tuning_params, tuning_ranges

    def set_tuning(self, tuning_params: dict, tuning_ranges: dict) -> None:
        self.trials_spin.setValue(tuning_params["trials"])
        self.report_check.setChecked(tuning_params["report"])
        self.sampler_combo.setCurrentText(tuning_params["sampler"])
        self.restart_combo.setCurrentText(tuning_params["restart"])
        self.multivar_check.setChecked(tuning_params["multivar"])
        self.group_check.setChecked(tuning_params["group"])
        self.separable_check.setChecked(tuning_params["separable"])
        self.margin_check.setChecked(tuning_params["margin"])
        self.pruner_combo.setCurrentText(tuning_params["pruner"])
        self.startup_spin.setValue(tuning_params["startup"])
        self.warmup_spin.setValue(tuning_params["warmup"])
        self.interval_spin.setValue(tuning_params["interval"])
        self.percent_spin.setValue(tuning_params["percent"])
        self.reduction_spin.setValue(tuning_params["reduction"])
        self.resource_spin.setValue(tuning_params["resource"])
        for k, v in tuning_ranges.items():
            self.model_params[k]["spin1"].setValue(v[0])
            if v[1] > v[0]:
                self.model_params[k]["spin2"].setValue(v[1])
            else:
                self.model_params[k]["spin2"].setValue(0)

    def load_params(self) -> None:
        settings = QSettings()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load parameters",
            settings.value("load_params", "."),
            self.file_types["params"],
        )
        if not filename:
            return
        settings.setValue("load_params", disk.get_folder(filename))
        try:
            with open(filename, "r") as file:
                params = json.load(file)
                self.set_training(params["training"])
                self.set_tuning(params["tuning"], params["ranges"])
        except OSError:
            QMessageBox.critical(self, "Load parameters", "Could not load parameters!")
            return

    def save_params(self) -> None:
        settings = QSettings()
        filename, extension = QFileDialog.getSaveFileName(
            self,
            "Save parameters",
            settings.value("save_params", "."),
            self.file_types["params"],
        )
        if not filename:
            return
        settings.setValue("save_params", disk.get_folder(filename))
        try:
            tuning_params, tuning_ranges = self.get_tuning()
            train_params = self.get_training()
            params = {
                "training": train_params,
                "tuning": tuning_params,
                "ranges": tuning_ranges,
            }
            with open(filename, "w") as file:
                json.dump(params, file, indent=4, sort_keys=True)
        except OSError:
            QMessageBox.critical(self, "Save parameters", "Could not save parameters!")

    def load_model(self):
        settings = QSettings()
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load model",
            settings.value("load_model", "."),
            self.file_types["model"],
        )
        if not filename:
            return
        settings.setValue("load_model", disk.get_folder(filename))
        try:
            self.parent.booster = model.load(filename)
        except OSError:
            QMessageBox.critical(self, "Load model", "Could not load model!")
            return
        self.set_training(model.properties(self.parent.booster))
        self.update_info()
        self.enable_controls()
        self.model_ready.emit(disk.get_basename(filename))

    def save_model(self):
        if self.parent.booster is None:
            QMessageBox.warning(
                self, "Save model", "Please load or train a model first"
            )
            return
        settings = QSettings()
        filename, extension = QFileDialog.getSaveFileName(
            self,
            "Save model",
            settings.value("save_model", "."),
            self.file_types["model"],
        )
        if not filename:
            return
        settings.setValue("save_model", disk.get_folder(filename))
        try:
            model.save(self.parent.booster, filename)
            self.model_ready.emit(disk.get_basename(filename))
        except OSError:
            QMessageBox.critical(self, "Save model", "Could not save model!")

    def train_model(self):
        if self.parent.train is None or self.parent.test is None:
            QMessageBox.critical(self, "Model training", "Please load a dataset first")
            return
        if dataset.properties(self.parent.train)["classes"] < 2:
            QMessageBox.critical(
                self, "Model training", "Dataset must have at least two classes!"
            )
            return
        params = self.get_training()
        if params["tree_method"] == "gpu_hist":
            QMessageBox.warning(
                self,
                "Model training",
                "AUC and AUCPR metrics are disabled with HistGPU method",
            )
        self.train_button.setEnabled(False)
        self.train_button.setText("Training...")
        QApplication.processEvents()
        try:
            booster, results = model.train(params, self.parent.train, self.parent.test)
            self.parent.booster = booster
            model.plot_results(
                self.parent.booster, params, results, self.results_widgets
            )
            self.update_info()
            self.model_ready.emit("TRAINED")
            self.enable_controls()
        except xgb.core.XGBoostError as e:
            QMessageBox.critical(self, "Model training", repr(e).split("\\n")[0])
        except KeyboardInterrupt:
            QMessageBox.warning(self, "Model training", "Training interrupted")
        self.train_button.setEnabled(True)
        self.train_button.setText("Train")

    def tune_model(self):
        if self.parent.train is None or self.parent.test is None:
            QMessageBox.warning(self, "Model tuning", "Please load a dataset first")
            return
        training_params = self.get_training()
        tuning_params, tuning_ranges = self.get_tuning()
        self.tune_button.setEnabled(False)
        self.tune_button.setText("Tuning...")
        QApplication.processEvents()
        start = time()
        success = True
        try:
            best_params = model.tune(
                self.parent.train,
                self.parent.test,
                tuning_params,
                tuning_ranges,
                training_params,
            )
            if best_params is None:
                raise RuntimeError
            self.set_training(best_params)
        except xgb.core.XGBoostError as e:
            QMessageBox.critical(self, "Model tuning", repr(e).split("\\n")[0])
        except RuntimeError:
            QMessageBox.critical(
                self, "Model tuning", "Please specify at least a tuning interval"
            )
            success = False
        except KeyboardInterrupt:
            QMessageBox.warning(self, "Model tuning", "Tuning interrupted")
            success = False
        self.tune_button.setEnabled(True)
        self.tune_button.setText("Tune")
        if not success:
            return
        elapsed = time() - start
        QMessageBox.information(
            self, "Model tuning", f"Tuning completed ({elapsed:.2f}s)"
        )

    def update_info(self):
        props = model.properties(self.parent.booster)
        self.info_view.update_data(model.info(self.parent.booster))
        self.tree_mpl.clear()
        self.tree_spin.setEnabled(True)
        self.tree_spin.setMaximum(props["best_ntree_limit"] - 1)
        self.tree_spin.setValue(0)
        self.row_spin.setMaximum(props["num_samples"] - 1)
        self.row_spin.setValue(0)
        self.col_spin.setMaximum(props["num_features"] - 1)
        self.col_spin.setValue(0)
        self.info_group.setEnabled(True)
        self.right_widget.setEnabled(True)
        model.plot_gain(self.parent.booster, self.gain_mpl)

    def export_model(self):
        if self.parent.booster is None:
            QMessageBox.warning(
                self, "Model export", "Please train or load a model first"
            )
            return
        settings = QSettings()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Model export",
            settings.value("export_model", "."),
            self.file_types["export"],
        )
        if not filename:
            return
        settings.setValue("export_model", disk.get_folder(filename))
        include = self.include_check.isChecked()
        integer = self.integer_check.isChecked()
        inline = self.inline_check.isChecked()
        comments = self.comments_check.isChecked()
        double = self.double_check.isChecked()
        stats = self.stats_check.isChecked()
        macro = self.macro_check.isChecked()
        memcpy = self.memcpy_check.isChecked()
        predict = self.predict_edit.text()
        try:
            model.export(
                self.parent.booster,
                filename,
                include,
                integer,
                inline,
                comments,
                double,
                stats,
                macro,
                memcpy,
                predict,
            )
            lines = disk.count_lines(filename)
            QMessageBox.information(
                self, "Model export", f"Model successfully exported ({lines} lines)"
            )
        except (ValueError, OSError):
            QMessageBox.critical(self, "Export model", "Could not export model!")

    def show_tree(self):
        if self.parent.booster is None:
            return
        index = self.tree_spin.value()
        model.plot_tree(self.parent.booster, index, self.tree_mpl)

    def show_shap(self):
        if self.parent.booster is None:
            QMessageBox.warning(
                self, "SHAP explainer", "Please load or train a model first"
            )
            return
        if self.parent.test is None:
            QMessageBox.warning(
                self, "SHAP explainer", "Dataset needed for SHAP evaluation"
            )
            return
        mode = self.shap_combo.currentText()
        row = self.row_spin.value()
        props = dataset.properties(self.parent.test)
        col = props["columns"][self.col_spin.value()]
        model.plot_shap(self.parent.booster, self.parent.test, mode, row, col)

    def save_shap(self):
        if self.parent.booster is None:
            QMessageBox.warning(self, "Save SHAP", "Please load or train a model first")
            return
        if self.parent.test is None:
            QMessageBox.warning(self, "Save SHAP", "SHAP values not evaluated yet")
            return
        settings = QSettings()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save SHAP", settings.value("save_shap", "."), self.file_types["csv"]
        )
        if not filename:
            return
        settings.setValue("save_shap", disk.get_folder(filename))
        try:
            model.save_shap(self.parent.booster, self.parent.test, filename)
        except (ValueError, OSError):
            QMessageBox.critical(self, "Save SHAP", "Could not save SHAP values!")

    def show_perm(self):
        if self.parent.booster is None:
            QMessageBox.warning(
                self, "Show permutation", "Please load or train a model first"
            )
            return
        if self.parent.test is None:
            QMessageBox.warning(
                self, "Show permutation", "Dataset needed for feature permutation"
            )
            return
        repeats = self.repeats_spin.value()
        samples = self.samples_spin.value() / 100
        model.plot_permute(self.parent.booster, self.parent.test, repeats, samples)

    def save_perm(self):
        if self.parent.booster is None:
            QMessageBox.warning(
                self, "Save permutation", "Please load or train a model first"
            )
            return
        if self.parent.test is None:
            QMessageBox.warning(
                self, "Save permutation", "Feature permutation not computed yet"
            )
            return
        settings = QSettings()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save permutation",
            settings.value("save_perm", "."),
            self.file_types["csv"],
        )
        if not filename:
            return
        settings.setValue("save_perm", disk.get_folder(filename))
        repeats = self.repeats_spin.value()
        samples = self.samples_spin.value() / 100
        try:
            model.save_permute(
                self.parent.booster, self.parent.test, repeats, samples, filename
            )
        except (ValueError, OSError):
            QMessageBox.critical(
                self, "Save permutation", "Could not save feature permutation!"
            )

    def enable_controls(self):
        self.save_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.include_check.setEnabled(True)
        self.integer_check.setEnabled(True)
        self.inline_check.setEnabled(True)
        self.comments_check.setEnabled(True)
        self.double_check.setEnabled(True)
        self.stats_check.setEnabled(True)
        self.macro_check.setEnabled(True)
        self.memcpy_check.setEnabled(True)
        self.predict_label.setEnabled(True)
        self.predict_edit.setEnabled(True)

    @Slot()
    def enable_training(self):
        self.train_group.setEnabled(True)
        self.params_group.setEnabled(True)
        self.tuning_group.setEnabled(True)

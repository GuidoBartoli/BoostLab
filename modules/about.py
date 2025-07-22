from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QLabel,
)


class AboutWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        logo_size = 256
        logo_label = QLabel()
        logo_label.setPixmap(
            QPixmap(":/boostlab").scaled(
                logo_size, logo_size, mode=Qt.TransformationMode.SmoothTransformation
            )
        )
        logo_label.setMinimumSize(logo_size, logo_size)

        info_label = QLabel(
            f"<h1>{QApplication.applicationName()} {QApplication.applicationVersion()}</h1>"
            "<h2>Supervised Learning with Gradient Boosting</h2>"
            f'<h3>Author: <a href="{QApplication.organizationDomain()}">{QApplication.organizationName()}</a></h3>'
            '<h3>Libraries: <a href="https://doc.qt.io/qtforpython-6">pyside6</a>, '
            '<a href="https://matplotlib.org">matplotlib</a>, '
            '<a href="https://pandas.pydata.org">pandas</a>, '
            '<a href="https://seaborn.pydata.org">seaborn</a>, '
            '<a href="https://scikit-learn.org">scikit-learn</a>, <br>'
            '<a href="https://optuna.org">optuna</a>, '
            '<a href="https://shap.readthedocs.io/en/latest">shap</a>, '
            '<a href="https://xgboop.readthedocs.io/en/latest">xgboost</a>, '
            '<a href="https://imbalanced-learn.org">imbalanced-learn</a>, '
            '<a href="https://github.com/BayesWitnesses/m2cgen">m2cgen</a>, '
            '<a href="https://plotly.com/">plotly</a></h3>'
            '<h3>License: <a href="https://www.gnu.org/licenses/lgpl-3.0.en.html">GNU LGPL v3</a></h3>'
        )
        info_label.setMinimumWidth(600)

        about_layout = QHBoxLayout()
        about_layout.addStretch(0)
        about_layout.addWidget(logo_label)
        about_layout.addWidget(info_label)
        about_layout.addStretch(0)
        self.setLayout(about_layout)

        palette = QPalette()
        palette.setColor(QPalette.Window, "#edfaed")
        self.setAutoFillBackground(True)
        self.setPalette(palette)

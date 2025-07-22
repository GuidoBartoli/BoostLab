import contextlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    hinge_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    zero_one_loss,
    DetCurveDisplay,
)

from lib import disk


# TODO: Embed binclass-tools plots (https://github.com/lucazav/binclass-tools)


def performance(
    test_y: np.ndarray, prob_y: np.ndarray, binthr: float, classes: int, labels: list
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate model performance computing performance metrics.

    :param test_y: ground truth target values
    :param prob_y: predicted probabilities
    :param binthr: binary classification threshold
    :param classes: total number of classes
    :param labels:  labels
    :return: classification report, confusion matrix, performance metrics
    """

    # TODO: Add metrics for imbalanced datasets:
    #       (https://imbalanced-learn.org/stable/references/generated/imblearn.metrics.classification_report_imbalanced.html)

    if classes == 2:
        pred_y = (prob_y >= binthr).astype(int)
    else:
        pred_y = np.argmax(prob_y, axis=1)

    # TODO: Use sklearn.metrics.ConfusionMatrixDisplay?
    matrix = np.zeros((classes, classes), dtype=int)
    for t, p in zip(test_y, pred_y):
        matrix[int(t), int(p)] += 1
    matrix = np.append(matrix, np.sum(matrix, axis=0, keepdims=True), axis=0)
    matrix = np.append(matrix, np.sum(matrix, axis=1, keepdims=True), axis=1)
    matrix = [list(row) for row in matrix]
    names = labels + ["TOTAL"]
    matrix = [[names[i]] + row for i, row in enumerate(matrix)]
    matrix = pd.DataFrame(matrix, columns=[""] + names)

    report = (
        pd.DataFrame(
            classification_report(
                test_y,
                pred_y,
                target_names=labels,
                labels=np.arange(classes),
                digits=6,
                zero_division=0,
                output_dict=True,
            )
        )
        .transpose()
        .reset_index()
        .round(6)
        .rename(
            columns={
                "index": "",
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-Score",
            }
        )
        .drop(columns="support")
        .drop(classes, axis=0)
    )
    report.iloc[-2, 0] = "Macro"
    report.iloc[-1, 0] = "Weighted"

    metrics = [
        ["[GENERIC]", ""],
        ["Average ACC", accuracy_score(test_y, pred_y)],
        ["Balanced ACC", balanced_accuracy_score(test_y, pred_y)],
        [
            "F1-measure",
            f1_score(
                test_y,
                pred_y,
                average="binary" if classes == 2 else "weighted",
                zero_division=0,
            ),
        ],
        ["Kappa factor", cohen_kappa_score(test_y, pred_y)],
        ["Matthews CC", matthews_corrcoef(test_y, pred_y)],
        [
            "Jaccard score",
            jaccard_score(
                test_y,
                pred_y,
                average="binary" if classes == 2 else "weighted",
                zero_division=0,
            ),
        ],
        ["Hamming loss", hamming_loss(test_y, pred_y)],
    ]
    if classes > 2:
        metrics.append(["Zero-one loss", zero_one_loss(test_y, pred_y)])
    else:
        metrics.append(["[BINARY]", ""])
        with contextlib.suppress(ValueError):
            metrics.extend(
                (
                    ["AUC-ROC score", roc_auc_score(test_y, prob_y)],
                    [
                        "Average P/R",
                        average_precision_score(test_y, prob_y),
                    ],
                )
            )
            metrics.extend(
                (
                    [
                        "Logistic loss",
                        log_loss(test_y, pred_y.astype(np.float64)),
                    ],
                    ["Hinge loss", hinge_loss(test_y, pred_y)],
                )
            )
            if ((prob_y >= 0) & (prob_y <= 1)).all():
                metrics.append(["Brier score", brier_score_loss(test_y, prob_y)])
    for metric in metrics:
        with contextlib.suppress(ValueError):
            metric[1] = round(float(metric[1]), 6)
    metrics = pd.DataFrame(metrics, columns=["Metric", "Value"])
    if metrics.iloc[-1, 0] == "[BINARY]":
        metrics.drop(metrics.tail(1).index, inplace=True)
    return report, matrix, metrics


def histogram(prob_y: np.ndarray, binthr: float, labels: list, widgets) -> None:
    """Plot probability distribution histogram.

    :param prob_y: predicted probabilities
    :param binthr: binary classification threshold
    :param labels: class labels
    :param widgets: list of Matplotlib widgets
    """
    if len(labels) != 2:
        return
    prob = {labels[0]: prob_y[prob_y < binthr], labels[1]: prob_y[prob_y >= binthr]}
    for i in range(2):
        widgets[i].clear()
        ax = widgets[i].canvas.axes
        # FIXME: Compute histogram only once and plot with linear and log scales
        sns.kdeplot(
            prob, legend=True, fill=True, log_scale=[False, i == 1], ax=ax, cut=0
        )
        ax.set(ylabel="density" if i == 0 else "log(density)")
        ax.axvline(x=binthr, linestyle=":", color="k", label="Threshold")
        ax.grid()
        if i == 1:
            ax.set_xlabel("threshold")
        widgets[i].refresh()


def thresholds(test_y: np.ndarray, prob_y: np.ndarray, binthr: float, widgets: list):
    """Plot ROC and PR curves and compute optimal thresholds.

    :param test_y: ground truth target values
    :param prob_y: predicted probabilities
    :param binthr: binary classification threshold
    :param widgets: list of Matplotlib widgets
    :return: table of optimal thresholds
    """

    # TODO: Implement automatic thresholds to get same precision and recall for both classes (remove AVERAGE threshold)

    alpha = 0.1
    red = "tab:red"
    green = "tab:green"
    blue = "tab:blue"
    orange = "tab:orange"
    gray = "tab:gray"
    brown = "tab:brown"

    def format_axes(
        axes: plt.Axes,
        title: str,
        xlabel: str | None,
        ylabel: str | None,
        square: bool,
        legend: bool,
    ):
        axes.set_title(title)
        if xlabel is not None:
            axes.set_xlabel(xlabel)
        if ylabel is not None:
            axes.set_ylabel(ylabel)
        axes.grid(visible=True, which="both")
        if legend:
            axes.legend(loc="lower center")
        if square:
            axes.set_aspect("equal")

    if np.count_nonzero(test_y == 0) == 0 or np.count_nonzero(test_y == 1) == 0:
        raise ValueError("Both classes must be non-empty for binary thresholds")

    # ROC curve --> maximize Youden’s J statistic
    roc_fpr, roc_tpr, roc_thr = roc_curve(test_y, prob_y)
    roc_thr[0] = 1
    roc_idx = np.argmax(roc_tpr - roc_fpr)
    roc_opt = roc_thr[roc_idx]
    roc_fpr0 = roc_fpr[roc_idx]
    roc_tpr0 = roc_tpr[roc_idx]
    roc_val = np.sqrt(roc_tpr0 * (1 - roc_fpr0))

    # Best balance between precision and recall
    prc_prec, prc_rcll, prc_thr = precision_recall_curve(test_y, prob_y)
    prc_idx = np.argmax(prc_prec >= prc_rcll)
    prc_opt = prc_thr[prc_idx]
    prc_val = prc_prec[prc_idx]

    # F1-score --> harmonic mean of precision and recall
    f1_scr = (2 * prc_prec * prc_rcll) / (prc_prec + prc_rcll)
    f1_idx = np.argmax(f1_scr)
    f1_opt = prc_thr[f1_idx]
    f1_val = f1_scr[f1_idx]

    # Find the threshold that maximizes accuracy
    thrs = np.linspace(start=0, stop=1, num=200)
    accs = np.array([accuracy_score(test_y, prob_y >= thr) for thr in thrs])
    acc_idx = np.argmax(accs)
    acc_opt = thrs[acc_idx]
    acc_val = accs[acc_idx]

    # Find the threshold that balances both recalls
    rcl0 = np.array([recall_score(test_y, prob_y >= thr) for thr in thrs])
    rcl1 = np.array([recall_score(1 - test_y, prob_y < thr) for thr in thrs])
    rcl_idx = np.argmax(rcl1 >= rcl0)
    rcl_opt = thrs[rcl_idx]
    rcl_val = rcl0[rcl_idx]

    # Summary table
    table = [
        ["ROC-AUC", roc_opt, roc_val],
        ["PR-AUC", prc_opt, prc_val],
        ["F1-score", f1_opt, f1_val],
        ["Accuracy", acc_opt, acc_val],
        ["Recall", rcl_opt, rcl_val],
    ]
    table = pd.DataFrame(table, columns=["Metric", "Threshold", "Value"]).round(6)

    # Plot curves
    for i in range(len(widgets)):
        widgets[i].clear()

    ax = widgets[0].canvas.axes
    ax.plot(roc_fpr, roc_tpr, color=blue)
    ax.fill_between(roc_fpr, roc_fpr, roc_tpr, alpha=alpha, color=blue)
    ax.plot([0, 1], [0, 1], linestyle=":", color=gray)
    format_axes(
        ax,
        title="Receiver Operating Characteristic (curve)",
        xlabel="false positive rate",
        ylabel="true positive rate",
        square=True,
        legend=False,
    )

    ax = widgets[1].canvas.axes
    ax.plot(roc_thr, roc_tpr, label="TPR", color=blue)
    ax.plot(roc_thr, roc_fpr, label="FPR", color=orange)
    ax.axvline(x=roc_opt, color=green, label="ROC", linestyle="--")
    ax.axvline(x=acc_opt, color=red, label="ACC", linestyle="--")
    ax.axvline(x=rcl_opt, color=brown, label="RCL", linestyle="--")
    ax.axvline(x=binthr, color=gray, label="THR", linestyle="--")
    format_axes(
        ax,
        title="False/True Positive Rate (thresholds)",
        xlabel="threshold",
        ylabel="false/true positive rate",
        square=False,
        legend=True,
    )

    ax = widgets[2].canvas.axes
    DetCurveDisplay.from_predictions(test_y, prob_y, ax=ax)
    format_axes(
        ax,
        title="Detection Error Tradeoff (curve)",
        xlabel=None,
        ylabel=None,
        square=True,
        legend=False,
    )
    ax.get_legend().remove()

    ax = widgets[3].canvas.axes
    ax.plot(prc_rcll, prc_prec, color=orange)
    ax.fill_between(prc_rcll, 1 - prc_rcll, prc_prec, alpha=alpha, color=orange)
    ax.plot([0, 1], [1, 0], linestyle=":", color=gray)
    format_axes(
        ax,
        title="Precision-Recall (curve)",
        xlabel="recall",
        ylabel="precision",
        square=True,
        legend=False,
    )

    ax = widgets[4].canvas.axes
    ax.plot(prc_thr, prc_rcll[:-1], label="RCL", color=blue)
    ax.plot(prc_thr, prc_prec[:-1], label="PRC", color=orange)
    ax.axvline(x=prc_opt, color=green, label="PR", linestyle="--")
    ax.axvline(x=f1_opt, color=red, label="F1", linestyle="--")
    ax.axvline(x=binthr, color=gray, label="THR", linestyle="--")
    format_axes(
        ax,
        title="Precision-Recall (thresholds)",
        xlabel="threshold",
        ylabel="precision/recall value",
        square=False,
        legend=True,
    )

    ax = widgets[5].canvas.axes
    CalibrationDisplay.from_predictions(test_y, prob_y, n_bins=10, ax=ax)
    format_axes(
        ax,
        title="Probability calibration (curve)",
        xlabel=None,
        ylabel=None,
        square=True,
        legend=False,
    )

    for i in range(len(widgets)):
        widgets[i].refresh()
    return table


def save_output(
    test_x: np.ndarray,
    test_y: np.ndarray,
    prob_y: np.ndarray,
    binthr: float,
    labels: list,
    columns: list,
    filename: str,
    mode: str,
) -> None:
    """Save classification output to CSV file.

    :param test_x: test sample values
    :param test_y: test target values
    :param prob_y: predicted probabilities
    :param binthr: binary classification threshold
    :param labels: class labels
    :param columns: feature names
    :param filename: output filename
    :param mode: output mode (all, correct, wrong)
    """
    classes = len(labels)
    if classes == 2:
        pred_y = (prob_y >= binthr).astype(int)
    else:
        pred_y = np.argmax(prob_y, axis=1)
    correct = test_y == pred_y
    wrong = np.logical_not(correct)
    headers = (
        columns
        + [f"P({label}) [{i}]" for i, label in enumerate(labels)]
        + ["Prediction", "Target", "Correct"]
    )
    if len(prob_y.shape) == 1:
        prob_y = np.column_stack((1 - prob_y, prob_y))
    if mode == "all":
        output = np.concatenate(
            (test_x, prob_y, pred_y[:, None], test_y[:, None], correct[:, None]), axis=1
        )
    elif mode == "correct":
        output = np.concatenate(
            (
                test_x[correct],
                prob_y[correct],
                pred_y[correct][:, None],
                test_y[correct][:, None],
                np.ones_like(test_y[correct])[:, None],
            ),
            axis=1,
        )
    elif mode == "wrong":
        output = np.concatenate(
            (
                test_x[wrong],
                prob_y[wrong],
                pred_y[wrong][:, None],
                test_y[wrong][:, None],
                np.zeros_like(test_y[wrong])[:, None],
            ),
            axis=1,
        )
    else:
        return
    disk.save_csv(filename, output, headers=headers)

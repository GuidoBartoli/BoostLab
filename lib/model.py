import contextlib
import os
import re
import warnings
from datetime import date, datetime

import m2cgen as m2c
import matplotlib.pyplot as plt
import numpy as np
import optuna as opt
import pandas as pd
import shap
import xgboost as xgb
from sklearn.inspection import permutation_importance

from lib import dataset, disk

xgb.set_config(verbosity=0)
warnings.filterwarnings(
    action="ignore",
    message=".*Loading a native XGBoost model with Scikit-Learn interface.*",
)

# TODO: Use exceptions instead of returning None inside library functions


def train(
    params: dict, training: pd.DataFrame, validation: pd.DataFrame, callbacks=None
) -> tuple[xgb.Booster, dict]:
    """Train a Gradient Boosting model with provided parameters.

    :param params: training parameters
    :param training: training dataset
    :param validation: validation dataset
    :return: trained XGBoost model and evaluation results
    """
    results = {}
    props = dataset.properties(training)
    dtrain = dataset.df2dm(training, props["labels"])
    dtest = dataset.df2dm(validation, props["labels"])
    evals = [(dtrain, "training"), (dtest, "validation")]
    if params["tree_method"] == "gpu_hist":
        if params["eval_metric"][-1] == "logloss":
            params["eval_metric"] = ["error", "logloss"]
        else:
            params["eval_metric"] = ["logloss", "error"]
        reference = params["eval_metric"][-1]
    else:
        reference = None
    booster = xgb.train(
        params,
        dtrain=dtrain,
        num_boost_round=params["boosting_rounds"],
        evals=evals,
        early_stopping_rounds=params["early_stopping"],
        evals_result=results,
        verbose_eval=False,
        callbacks=callbacks,
    )
    if (
        params["early_stopping"] is not None
        and params["boosting_rounds"] != booster.best_ntree_limit
    ):
        early = booster.best_iteration
        booster = booster[: booster.best_ntree_limit]
        booster.best_iteration = early
        booster.best_ntree_limit = early + 1
    for k, v in zip(params.keys(), params.values()):
        eval(f"booster.set_attr({k}=str(v))")
    if reference is not None:
        empty = [0] * len(results["training"]["logloss"])
        results["training"]["auc"] = empty
        results["training"]["aucpr"] = empty
        results["validation"]["auc"] = empty
        results["validation"]["aucpr"] = empty
    return booster, results


def tune(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    tuning_params: dict,
    tuning_ranges: dict,
    training_params: dict,
) -> dict | None:
    """Automatic hyperparameter tuning.

    :param tuning_params: tuning parameters
    :param tuning_ranges: tuning parameter ranges
    :param training_params: training parameters
    :param training: training dataset
    :param validation: validation dataset
    :return: best training parameters
    """
    boosting_params = [
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "min_split_loss",
        "reg_alpha",
        "reg_lambda",
        "max_delta_step",
        "scale_pos_weight",
    ]

    def check(params, key) -> bool:
        return params[key][1] > params[key][0]

    if not any(check(tuning_ranges, k) for k in tuning_ranges):
        return None
    if tuning_params["sampler"] == "Random sampling":
        sampler = opt.samplers.RandomSampler()
    elif tuning_params["sampler"] == "Parzen estimator":
        sampler = opt.samplers.TPESampler(
            multivariate=tuning_params["multivar"] or tuning_params["group"],
            group=tuning_params["group"],
        )
    elif tuning_params["sampler"] == "Covariance matrix":
        if tuning_params["restart"] == "Increasing population":
            restart = "ipop"
        elif tuning_params["restart"] == "Bi-population strategy":
            restart = "bipop"
        else:
            restart = None
        sampler = opt.samplers.CmaEsSampler(
            restart_strategy=restart,
            use_separable_cma=tuning_params["separable"],
            with_margin=tuning_params["margin"],
            consider_pruned_trials=tuning_params["pruner"] == "hyperband",
        )
    elif tuning_params["sampler"] == "Quasi Monte-Carlo":
        sampler = opt.samplers.QMCSampler()
    elif tuning_params["sampler"] == "Brute force search":
        sampler = opt.samplers.BruteForceSampler()
    else:
        sampler = opt.samplers.BaseSampler()

    startup = int(tuning_params["startup"] * tuning_params["trials"] / 100)
    warmup = int(tuning_params["warmup"] * training_params["boosting_rounds"] / 100)
    interval = tuning_params["interval"]
    percent = tuning_params["percent"]
    reduction = tuning_params["reduction"]
    if tuning_params["pruner"] == "Median":
        pruner = opt.pruners.MedianPruner(
            n_startup_trials=startup, n_warmup_steps=warmup, interval_steps=interval
        )
    elif tuning_params["pruner"] == "Percentile":
        pruner = opt.pruners.PercentilePruner(
            percentile=percent,
            n_startup_trials=startup,
            n_warmup_steps=warmup,
            interval_steps=interval,
        )
    elif tuning_params["pruner"] == "Hyperband":
        pruner = opt.pruners.HyperbandPruner(
            min_resource=tuning_params["resource"], reduction_factor=reduction
        )
    elif tuning_params["pruner"] == "Halving":
        pruner = opt.pruners.SuccessiveHalvingPruner(
            min_resource=tuning_params["resource"], reduction_factor=reduction
        )
    else:
        pruner = opt.pruners.NopPruner()

    def objfun(trial: opt.trial.Trial) -> float:
        tuning = {}
        for p in boosting_params:
            if check(tuning_ranges, p):
                f = trial.suggest_int if p == "max_depth" else trial.suggest_float
                tuning[p] = f(p, tuning_ranges[p][0], tuning_ranges[p][1])
        params = training_params.copy()
        params.update(tuning)
        last = params["eval_metric"][-1]

        callbacks = [
            opt.integration.XGBoostPruningCallback(trial, f"validation-{last}")
        ]
        booster, results = train(params, training, validation, callbacks)
        score = results["validation"][last][booster.best_iteration]
        return score

    last = training_params["eval_metric"][-1]
    if last in ["auc", "aucpr"]:
        direction = "maximize"
    elif last in ["error", "logloss"]:
        direction = "minimize"
    else:
        direction = None

    study = opt.create_study(
        storage=None,
        sampler=sampler,
        pruner=pruner,
        study_name=f"boostlab_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        direction=direction,
    )
    opt.logging.set_verbosity(opt.logging.WARNING)
    study.optimize(
        objfun,
        n_trials=tuning_params["trials"],
        show_progress_bar=True,
    )

    best_params = training_params.copy()
    for param in boosting_params:
        if check(tuning_ranges, param):
            best_params[param] = study.best_params[param]

    if tuning_params["report"]:
        opt.visualization.plot_optimization_history(study).show()
        opt.visualization.plot_intermediate_values(study).show()
        opt.visualization.plot_edf(study).show()
        opt.visualization.plot_contour(study).show()
        opt.visualization.plot_slice(study).show()
        opt.visualization.plot_param_importances(study).show()
        opt.visualization.plot_parallel_coordinate(study).show()

    return best_params


def predict(
    booster: xgb.Booster, df: pd.DataFrame, labels: list[str]
) -> np.ndarray | None:
    """Predict class probabilities for input dataset.

    :param booster: trained model
    :param df: input dataset
    :return: predicted class probabilities
    """

    # FIXME: Check for dataset and model compatibility does not work
    classes1 = int(booster.attr("num_classes"))
    props = dataset.properties(df)
    if 1 < props["classes"] != classes1:
        raise ValueError("Dataset and model are incompatible!")
    try:
        return booster.predict(dataset.df2dm(df, labels=None))
    except xgb.core.XGBoostError:
        raise ValueError("Error while predicting probabilities!")


def export(
    booster: xgb.Booster,
    filename: str,
    include: bool,
    integer: bool,
    inline: bool,
    comments: bool,
    double: bool,
    stats: bool,
    macro: bool,
    memcpy: bool,
    predict: str,
) -> None:
    """Export trained model to C/C++/Python code.

    :param booster: trained XGBoost model
    :param filename: output filename
    :param include: include header file
    :param integer: use integers for input data
    :param inline: use inline sigmoid function
    :param comments: add comments to code
    :param double: use doubles for output data
    :param stats: include statistics in code
    :param macro: add macro for using SRAM memory
    :param memcpy: use memcpy function for output data
    :param predict: prediction function name
    :return: True if code was exported successfully
    """

    # TODO: Add option to change function and header name, option to disable lint [-e91, #if !defined(_lint)]
    # TODO: Add export in '.gz' format (pickled dataframe) using booster.trees_to_dataframe()

    xgbclf = bst2clf(booster)
    extension = disk.get_ext(filename)
    code = None
    if extension in ["c", "cpp"]:
        code = (
            m2c.export_to_c(xgbclf, function_name=predict)
            .replace("double *x", "double* x", 1)
            .replace("double *result", "double* result", 1)
            .replace("double sigmoid", "\nstatic double sigmoid", 1)
            .replace(f"void {predict}", f"\nvoid {predict}", 1)
            .replace("(double * input, double * output)", "(double* x, double* y)", 1)
            .replace("var", "v")
            .replace("input", "x")
            .replace("output", "y")
            .replace("#include <string.h>\n", "\n", 1)
        )
        if include:
            code = (
                f'#include "{disk.remove_ext(disk.get_basename(filename))}.h"\n\n'
                + code
            )
        if inline:
            code = code.replace(
                "static double sigmoid", "\nstatic inline double sigmoid", 1
            )
        if integer:
            code = code.replace("double* x", "uint32_t* x")
            code = code.replace(".0)", ")")
        if not double:
            code = code.replace("double", "float")
            code = code.replace("exp", "expf")
            code = re.sub(r"(-?\d+\.\d+)", r"\1f", code)
        if macro:
            code = code.replace(
                f"void {predict}(",
                f"AT_SRAM_ITC_DATA_SECTION void {predict}(",
            )
            code = code.replace(
                "#include <math.h>", "#include <math.h>\n#include <mimxrt1060/hw.h>"
            )
        if comments:
            props = properties(booster)
            comment = (
                "/*\n"
                "GRADIENT BOOSTING MODEL\n"
                "- Source dataset:\n"
                f"- Input features: {props['num_features']}\n"
                "- Test logloss:\n"
                "- Test error:\n"
                "- Test accuracy:\n"
                f"- Build date:     {date.today().strftime('%d/%m/%Y')}\n"
                f"- Maximum depth:  {props['max_depth']}\n"
                f"- Total rounds:   {props['best_ntree_limit']}\n"
                f"- Total branches: {props['active_branches']}\n"
                f"- Avg. branches:  {round(props['average_branches'], 2)}\n"
                "*/\n\n"
            )
            code = comment + code
        if not memcpy:
            code = re.sub(
                r"memcpy\(.+\{(.+),\s(.+)\}.+\);", r"y[0] = \1;\n    y[1] = \2;", code
            )
        # FIXME: If model uses "logitraw" objective, remove sigmoid function
    elif extension == "py":
        code = (
            m2c.export_to_python(xgbclf, function_name=predict)
            .replace("import math", "import math\n", 1)
            .replace(f"def {predict}", f"\ndef {predict}", 1)
            .replace("var", "v")
            .replace("input", "x")
            .replace("output", "y")
            .replace(".0:", ":")
        )
    elif extension == "json":
        booster.dump_model(filename, with_stats=stats, dump_format="json")
    elif extension == "txt":
        booster.dump_model(filename, with_stats=stats, dump_format="text")
    if code is None:
        raise ValueError("Invalid generated code!")
    with open(filename, "w") as out:
        out.write(code)


def save(booster: xgb.Booster, filename: str) -> None:
    """Save model to disk.

    :param booster: trained XGBoost model
    :param filename: output filename
    """
    booster.save_model(filename)


def load(filename: str) -> xgb.Booster:
    """Load model from disk.

    :param filename: input filename
    :return: loaded model and parameters
    """
    booster = xgb.Booster()
    booster.load_model(filename)
    # TODO: Fallback code to import old format models
    return booster


def properties(booster: xgb.Booster) -> dict:
    """Get trained model properties.

    :param booster: trained model
    :return: model properties
    """
    attr = booster.attributes()
    for k, v in zip(attr.keys(), attr.values()):
        try:
            if k in [
                "best_ntree_limit",
                "boosting_rounds",
                "early_stopping",
                "max_depth",
                "num_classes",
                "num_features",
                "num_samples",
                "seed",
            ]:
                attr[k] = int(v)  # decimal values
            else:
                attr[k] = float(v)  # numeric values
        except ValueError:
            with contextlib.suppress(NameError, SyntaxError):
                attr[k] = eval(v)  # strings or lists of values
    depth = attr["max_depth"]
    rounds = attr["best_ntree_limit"]
    max_branches = (2 ** (depth + 1) - 1) * rounds
    active_branches = tree_leaves = average_depth = 0
    dump = booster.get_dump()
    for estimator in dump:
        active_branches += estimator.count(":[")
        for node in estimator.split("\n"):
            if "leaf=" in node:
                tree_leaves += 1
                average_depth += node.count("\t")
    ratio = len(dump) // rounds
    active_branches //= ratio
    tree_leaves //= ratio
    average_depth /= ratio
    pruning_ratio = 1 - active_branches / max_branches
    average_depth = average_depth / tree_leaves
    average_branches = int((2 ** (average_depth + 1) - 1))
    attr["max_branches"] = max_branches
    attr["active_branches"] = active_branches
    attr["pruning_ratio"] = pruning_ratio
    attr["average_depth"] = average_depth
    attr["average_branches"] = average_branches
    attr["tree_leaves"] = tree_leaves
    return attr


def info(booster: xgb.Booster) -> pd.DataFrame:
    """Get model information.

    :param booster: trained model
    :return: training parameters, boosting parameters, structure information
    """
    props = properties(booster)
    train_df = pd.DataFrame(
        [
            ["[TRAINING]", ""],
            ["Input features", props["num_features"]],
            ["Output classes", props["num_classes"]],
            ["Learned samples", props["num_samples"]],
        ],
    )
    params_df = pd.DataFrame(
        [
            ["[PARAMETERS]", ""],
            ["Max tree depth", props["max_depth"]],
            ["Learning rate", props["learning_rate"]],
            ["Row subsample", props["subsample"]],
            ["Column subsample", props["colsample_bytree"]],
            ["Min child weight", props["min_child_weight"]],
            ["Min split loss", props["min_split_loss"]],
            ["L1-norm alpha", props["reg_alpha"]],
            ["L2-norm lambda", props["reg_lambda"]],
            ["Max delta step", props["max_delta_step"]],
            ["Weight scale", props["scale_pos_weight"]],
        ],
    )
    boost_df = pd.DataFrame(
        [
            ["[BOOSTING]", ""],
            ["Boosting rounds", props["boosting_rounds"]],
            ["Early stopping", props["early_stopping"]],
            ["Objective function", props["objective"].split(":")[1]],
            ["Evaluation metric", props["eval_metric"][0]],
            ["Construction method", props["tree_method"]],
            ["Sampling method", props["sampling_method"]],
            ["Growing policy", props["grow_policy"]],
            ["Random seed", props["seed"]],
        ],
    )
    struct_df = pd.DataFrame(
        [
            ["[STRUCTURE]", ""],
            ["Boosting tree limit", props["best_ntree_limit"]],
            ["Max tree branches", props["max_branches"]],
            ["Active tree branches", props["active_branches"]],
            ["Tree pruning ratio", f"{props['pruning_ratio']:.2%}"],
            ["Average tree depth", round(props["average_depth"], 2)],
            ["Average tree branches", round(props["average_branches"], 2)],
            ["Tree estimator leaves", props["tree_leaves"]],
        ],
    )
    table = pd.concat([train_df, params_df, boost_df, struct_df], ignore_index=True)
    table = table.rename(columns={0: "Property", 1: "Value"})
    return table


def bst2clf(booster: xgb.Booster) -> xgb.XGBClassifier:
    """Convert trained model to Scikit-Learn classifier.

    :param booster: trained model
    :return: Scikit-Learn classifier
    """
    temp_file = "temp.ubj"
    booster.save_model(temp_file)
    skclf = xgb.XGBClassifier()
    skclf.load_model(temp_file)
    os.remove(temp_file)
    skclf.n_classes_ = skclf.classes_ = int(booster.attr("num_classes"))
    skclf.base_score = 0
    skclf.num_parallel_tree = 1
    return skclf


def plot_results(
    booster: xgb.Booster, params: dict, results: dict, widgets: list
) -> None:
    """Plot training and validation results.

    :param booster: trained model
    :param params: training parameters
    :param results: evaluation results
    :param widgets: graphical widgets
    """

    metrics = ["logloss", "error", "auc", "aucpr"]
    colors = dict(
        zip(
            metrics,
            [
                ["tab:blue", "tab:orange"],
                ["tab:green", "tab:red"],
                ["tab:purple", "tab:olive"],
                ["tab:brown", "tab:pink"],
            ],
        )
    )
    total_rounds = len(results["training"][metrics[0]])
    best_round = booster.best_iteration
    x = range(total_rounds)

    for metric, widget in zip(metrics, widgets):
        training, validation = (
            results["training"][metric],
            results["validation"][metric],
        )
        train_color, valid_color = colors[metric]
        train_val, valid_val = training[best_round], validation[best_round]

        widget.clear()
        ax = widget.canvas.axes
        ax.plot(x, training, color=train_color, label=f"training ({train_val:.6f})")
        ax.plot(x, validation, color=valid_color, label=f"validation ({valid_val:.6f})")
        if best_round != total_rounds - 1:
            ax.axvline(x=best_round, linestyle=":", color="k")
        ax.set_xlabel("rounds")
        ax.set_ylabel(metric.upper())
        ax.legend(loc="best")
        ax.grid(True, which="both")
        widget.refresh()


def plot_tree(booster, index, widget):
    """Plot a single tree from the model.

    :param booster: trained model
    :param index: tree index
    :param widget: Matplotlib widget
    """
    widget.clear()
    xgb.plot_tree(booster, num_trees=index, rankdir="TB", ax=widget.canvas.axes)
    widget.refresh()


def get_shap(booster: xgb.Booster, df: pd.DataFrame) -> shap.Explainer:
    """Get SHAP values for trained model on input dataset.

    :param booster: trained model
    :param df: input dataset
    """
    explainer = shap.Explainer(booster)
    return explainer(df.drop(columns=["target"]))


def plot_shap(
    booster: xgb.Booster, df: pd.DataFrame, mode: str, row: int, col: str
) -> None:
    """Plot SHAP values for trained model on input dataset.

    :param booster: trained model
    :param df: input dataset
    :param mode: plot mode (average, scatter, force, beeswarm, waterfall)
    :param row: row index
    :param col: column name
    """
    shap_values = get_shap(booster, df)
    mode = mode.lower()
    if mode == "average":
        shap.plots.bar(shap_values)
    elif mode == "scatter":
        shap.plots.scatter(shap_values[:, col], color=shap_values)
    elif mode == "force":
        shap.plots.force(shap_values[row])
    elif mode == "beeswarm":
        shap.plots.beeswarm(shap_values, alpha=0.9)
    elif mode == "waterfall":
        shap.plots.waterfall(shap_values[row])
    plt.tight_layout()


def save_shap(booster: xgb.Booster, df: pd.DataFrame, filename: str) -> None:
    """Save SHAP values for trained model on input dataset.

    :param booster: trained model
    :param df: input dataset
    :param filename: output filename
    """
    shap_values = get_shap(booster, df)
    props = dataset.properties(df)
    values = np.vstack((shap_values, np.mean(np.absolute(shap_values), 0)))
    disk.save_csv(filename, values, props["columns"])


def get_permute(
    booster, df, repeats, samples
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get permutation importance for trained model on input dataset.

    :param booster: trained model
    :param df: input dataset
    :param repeats: number of repetitions
    :param samples: number of samples
    :return: mean, low and high values
    """
    props = dataset.properties(df)
    test_x, test_y = dataset.df2np(df, props["labels"])
    skclf = bst2clf(booster)
    perm_values = permutation_importance(
        skclf, test_x, test_y, n_repeats=repeats, max_samples=samples, n_jobs=-1
    )
    mean = np.array(perm_values["importances_mean"])
    low = np.array(mean - 3 * perm_values["importances_std"])
    high = np.array(mean + 3 * perm_values["importances_std"])
    return mean, low, high


def plot_permute(booster, df, repeats, samples) -> None:
    """Plot permutation importance for trained model on input dataset.

    :param booster: trained model
    :param df: input dataset
    :param repeats: number of repetitions
    :param samples: number of samples
    """
    # FIXME: Avoid recomputation, use saved results from get_permute()
    mean, low, high = get_permute(booster, df, repeats, samples)
    props = dataset.properties(df)
    plt.figure(figsize=[12, 8])
    plt.title("Feature Permutation Importance", fontweight="bold")
    plt.bar(props["columns"], mean, label="average")
    plt.fill_between(props["columns"], low, high, alpha=0.20, label="deviation")
    plt.xlabel("feature")
    plt.ylabel("importance")
    plt.grid(True, which="both")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def save_permute(booster, df, repeats, samples, filename) -> None:
    """Save permutation importance for trained model on input dataset.

    :param booster: trained model
    :param df: input dataset
    :param repeats: number of repetitions
    :param samples: number of samples
    :param filename: output filename
    """
    mean, low, high = get_permute(booster, df, repeats, samples)
    props = dataset.properties(df)
    table = np.column_stack((props["columns"], low, mean, high))
    headers = ["feature", "low", "mean", "high"]
    disk.save_csv(filename, table, headers)


def plot_gain(booster, widget) -> None:
    """Plot feature gain importance for trained model.

    :param booster: trained model
    :param widget: Matplotlib widget
    """
    widget.clear()
    xgb.plot_importance(
        booster,
        title="Gain",
        importance_type="gain",
        grid=False,
        height=0.75,
        ax=widget.canvas.axes,
        show_values=False,
    )
    widget.refresh()

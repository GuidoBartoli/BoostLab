import contextlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from h5py import File
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    ClusterCentroids,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from sklearn.decomposition import (
    FactorAnalysis as FA,
    FastICA as ICA,
    KernelPCA as KPCA,
    LatentDirichletAllocation as LDA,
    NMF,
    PCA,
)
from sklearn.ensemble import (
    ExtraTreesClassifier as ETC,
    RandomForestClassifier as RFC,
    AdaBoostClassifier as ADA,
    IsolationForest as IF,
)
from sklearn.feature_selection import (
    RFECV,
    SelectFdr,
    SelectFpr,
    SelectFromModel,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    SequentialFeatureSelector,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.preprocessing import (
    LabelEncoder,
    maxabs_scale,
    minmax_scale,
    normalize as normal_scale,
    quantile_transform,
    robust_scale,
    scale as standard_scale,
)
from sklearn.svm import LinearSVC as SVC

from lib import disk

DF_EXT = {
    "csv": ["csv"],
    "excel": ["xls", "xlsx"],
    "hdf5": ["h5", "hdf5"],
    "pickle": ["zip", "gz", "bz2", "xz", "tar"],
}
DF_TGT = "target"
H5_KEY = "data"


def load(
    filename: str, header: int = 0, csvsep: str = ",", valid: bool = True
) -> pd.DataFrame:
    """Load a dataset from disk (supports CSV, Excel, HDF5 and Pickle).

    :param filename: input filename
    :param header: header row index
    :param csvsep: CSV separator
    :param valid: validity check
    :return: imported dataset
    """
    ext = disk.get_ext(filename)
    if ext in DF_EXT["csv"]:
        df = pd.read_csv(filename, sep=csvsep, header=header, on_bad_lines="skip")
    elif ext in DF_EXT["excel"]:
        df = pd.read_excel(filename, header=header)
    elif ext in DF_EXT["hdf5"]:
        try:
            df = pd.DataFrame(pd.read_hdf(filename))
        except ValueError:
            db = File(filename)
            if "data" not in db or "target" not in db or "labels" not in db:
                raise ValueError("Data, target or labels not present")
            data = db["data"][:]
            target = db["target"][:]
            if target.shape[0] != data.shape[0]:
                raise ValueError("Data and target have different number of rows")
            labels = [label.decode("utf-8") for label in list(db["labels"][:])]
            if len(labels) < 2 or np.any(
                np.logical_not(np.in1d(target, range(len(labels))))
            ):
                raise ValueError("Invalid labels")
            features = data.shape[1]
            if "columns" in db:
                columns = [column.decode("utf-8") for column in list(db["columns"][:])]
                if len(columns) != features:
                    raise ValueError("Invalid columns")
            else:
                columns = [str(c) for c in range(features)]
            if data.size == 0 or target.size == 0:
                raise ValueError("Empty data or target")
            df = pd.DataFrame(data, columns=columns)
            tgt_col = "target"
            df = df.rename(columns={tgt_col: f"{tgt_col}0"})
            df[tgt_col] = [labels[int(t)] for t in target]
            df.columns = df.columns.astype(str)
    elif ext in DF_EXT["pickle"]:
        compression = {"method": "zstd"} if ext == "xz" else "infer"
        df = pd.read_pickle(filename, compression)
    else:
        raise ValueError("Unsupported file format")
    if valid and not check(df):
        raise ValueError("Invalid dataset")
    return df


def save(
    df: pd.DataFrame, filename: str, compression: int = 4, valid: bool = True
) -> None:
    """Save dataset to disk (supports CSV, Excel, HDF5 and Pickle).

    :param df: input dataset
    :param filename: output filename
    :param compression: compression level [0:9]
    :param valid: validity check
    """
    if valid and not check(df):
        raise ValueError("Invalid dataset")
    ext = disk.get_ext(filename)
    if ext in DF_EXT["csv"]:
        df.to_csv(filename, index=False)
    elif ext in DF_EXT["excel"]:
        df.to_excel(filename, index=False)
    elif ext in DF_EXT["hdf5"]:
        df.to_hdf(filename, key=H5_KEY, mode="w", index=False, complevel=compression)
    elif ext in DF_EXT["pickle"]:
        if compression == 0:
            compress = None
        elif ext == "zip":
            compress = {"method": "zip", "compresslevel": compression}
        elif ext == "gz":
            compress = {"method": "gzip", "compresslevel": compression}
        elif ext == "bz2":
            compress = {"method": "bz2", "compresslevel": compression}
        elif ext == "xz":
            compress = {"method": "zstd", "level": compression}
        else:
            raise ValueError("Unsupported compression method")
        df.to_pickle(filename, compress)
    else:
        raise ValueError("Unsupported file format")


def scan(
    folder: str, recursive: bool, target: str, header: bool, csvsep: str
) -> tuple[pd.DataFrame | None, int | None]:
    """Import and convert raw files to a structured dataset.

    :param folder: input folder
    :param recursive: recursive search
    :param target: target column type
    :param header: header row index
    :param csvsep: CSV separator
    :return: dataset and number of imported files
    """
    files = disk.get_files(folder, sort=True, fullpath=True, recursive=recursive)
    df = None
    imported = 0
    for f in files:
        try:
            basename = disk.get_basename(f)
            if basename.startswith("_"):
                continue
            d = load(f, header=0 if header else None, csvsep=csvsep, valid=False)
            if d is None:
                continue
            if disk.get_ext(f) in DF_EXT["csv"]:
                d = d.rename(columns={DF_TGT: f"{DF_TGT}0"})
                if target == "suffix":
                    t = disk.remove_ext(basename).split("_")[-1]
                    d[DF_TGT] = [t] * len(d.index)
                elif target == "column":
                    d.rename(columns={d.columns[-1]: DF_TGT}, inplace=True)
                    # FIXME: Apply round() if column contains floats or raise exception
                else:
                    continue
                d.columns = d.columns.astype(str)
            df = append(df, d)
            imported += 1
        except ValueError:
            continue
    if df is None:
        return None, None
    return unique(df), imported


def check(df: pd.DataFrame) -> bool:
    """Check dataset integrity and validity.

    :param df: input dataset
    :return: True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    if DF_TGT not in df.columns:
        return False
    if df[DF_TGT].dtype != "object":
        return False
    if not all(isinstance(col, str) for col in df.columns):
        return False
    if not any(df[c].dtype not in ["float64", "int64"] for c in df.columns):
        return False
    return True


def properties(df: pd.DataFrame) -> dict[str, int | list[str]]:
    """Get dataset properties.

    :param df: input dataset
    :return: classes, labels, samples, features, columns
    """
    encoder = LabelEncoder()
    encoder.fit(df[DF_TGT])
    labels = list(encoder.classes_)
    classes = len(labels)
    samples, features = df.shape
    features -= 1
    columns = df.columns.to_list()[:-1]
    values = samples * features
    return {
        "classes": classes,
        "labels": labels,
        "samples": samples,
        "features": features,
        "columns": columns,
        "values": values,
    }


def sort(df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    """Sort dataset by target column values.

    :param df: input dataset
    :param ascending: sort ascending
    :return: sorted dataset
    """
    return df.sort_values(by=DF_TGT, ascending=ascending, ignore_index=True)


def unique(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows.

    :param df: input dataset
    :return: unique dataset and removed rows
    """
    return df.drop_duplicates(ignore_index=True)


# FIXME: Change function name to avoid conflict with Python built-in function
def filter(
    df: pd.DataFrame, selection: str, mode: str, items: list[str]
) -> pd.DataFrame:
    """Filter columns or rows by feature or class labels respectively.

    :param df: input dataset
    :param selection: 'feature' or 'class'
    :param mode: 'include' or 'exclude'
    :param items: feature or class labels
    :return: filtered dataset
    """
    if not items:
        return df
    result = pd.DataFrame()
    if selection == "class":
        if mode == "exclude":
            result = df[~df[DF_TGT].isin(items)]
        elif mode == "include":
            result = df[df[DF_TGT].isin(items)]
    elif selection == "feature":
        if mode == "exclude":
            if DF_TGT in items:
                items.remove(DF_TGT)
            result = df.drop(columns=items)
        elif mode == "include":
            if DF_TGT not in items:
                items.append(DF_TGT)
            with contextlib.suppress(KeyError):
                # FIXME: Does not work with already filtered datasets
                result = df[items]
    return df if result.empty else result


def rename(df: pd.DataFrame, what: str, old: list, new: list) -> pd.DataFrame:
    """Replace target labels.

    :param df: input dataset
    :param what: "feature" or "class"
    :param old: old labels
    :param new: new labels
    :return: updated dataset
    """
    if len(old) != len(new):
        return df
    result = df.copy()
    for x, y in zip(old, new):
        if what == "feature":
            result.rename(columns={x: y}, inplace=True)
        elif what == "class":
            result.replace(to_replace={DF_TGT: x}, value=y, inplace=True)
    return result


def subsample(df: pd.DataFrame, what: str, factor: float) -> pd.DataFrame:
    """Resample dataset rows.

    :param df: input dataset
    :param what: "rows" or "cols"
    :param factor: sampling factor
    :return: resampled dataset
    """
    if what == "rows":
        rows = len(df.index)
        num = max(int(rows * factor), 1)
        indices = np.linspace(start=0, stop=rows - 1, num=num).astype(int)
        return df.iloc[indices]
    if what == "cols":
        target = df[DF_TGT]
        data = df.drop(DF_TGT, axis=1)
        cols = len(data.columns)
        num = max(int(cols * factor), 1)
        indices = np.linspace(start=0, stop=cols - 1, num=num).astype(int)
        result = data.iloc[:, indices].copy()
        result[DF_TGT] = target
        return result
    return df


def limit(df: pd.DataFrame, samples: int) -> pd.DataFrame:
    """Limit dataset rows.

    :param df: input dataset
    :param samples: number of rows to keep
    """
    return df.head(samples)


def append(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    if df1 is None:
        return df2
    if df2 is None:
        return df1
    if set(df1.columns) != set(df2.columns):
        return df1
    return pd.concat([df1, df2], ignore_index=True)


def idx2feat(df: pd.DataFrame, index: list) -> list[str]:
    """Return feature labels by index.

    :param df: input dataset
    :param index: index list
    :return: feature names
    """
    return list(df.columns[index])


def df2np(
    df: pd.DataFrame, labels: list[str] | None
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Separate data and target parts.

    :param df: input dataset
    :param labels: input labels
    :return: data and target
    """
    data = df.iloc[:, :-1].to_numpy()
    if labels is None:
        return data
    # FIXME: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version.
    pd.set_option('future.no_silent_downcasting', True)  
    target = df[DF_TGT].replace(labels, range(len(labels))).to_numpy(dtype=np.int32)
    return data, target


def np2df(
    data: np.ndarray, target: np.ndarray, labels: list[str], columns: list[str]
) -> pd.DataFrame:
    """Combine data and target parts into a dataset.

    :param data: data part
    :param target: target values
    :param labels: target labels
    :param columns: feature labels
    :return: combined dataset
    """
    df = pd.DataFrame(data, columns=columns)
    df[DF_TGT] = [labels[i] for i in target]
    return df


def df2dm(df: pd.DataFrame, labels: list[str] | None) -> xgb.DMatrix:
    """Convert a dataframe to a XGBoost DMatrix.

    :param df: input dataset
    :param labels: input labels
    :return: XGBoost DMatrix
    """
    if labels is not None:
        x, y = df2np(df, labels)
        return xgb.DMatrix(x, label=y)
    x = df2np(df, labels=None)
    return xgb.DMatrix(x)


def split(
    df: pd.DataFrame, ratio: float, seed: int = 0
) -> list[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and test subsets.

    :param df: input dataset
    :param ratio: training subset ratio
    :param seed: random seed
    :return: training and test subsets
    """
    if ratio in {0, 1}:
        return [df, df]
    return train_test_split(df, train_size=ratio, random_state=seed)


def info(train: pd.DataFrame, test: pd.DataFrame, ratio: bool) -> pd.DataFrame:
    """Get dataset information.

    :param train: training subset
    :param test: test subset
    :param ratio: show ratio instead of absolute numbers
    :return: summary table
    """
    df = pd.concat([train, test])
    props = properties(df)
    table = []
    totals = [0] * 3
    for l in props["labels"]:
        support_num = len(df[df[DF_TGT] == l])
        support_prc = (
            f"{support_num / props['samples']:.2%}" if ratio else f"{support_num}"
        )
        training_num = len(train[train[DF_TGT] == l])
        training_prc = (
            f"{training_num / props['samples']:.2%}" if ratio else f"{training_num}"
        )
        validation_num = len(test[test[DF_TGT] == l])
        validation_prc = (
            f"{validation_num / props['samples']:.2%}" if ratio else f"{validation_num}"
        )
        totals[0] += support_num
        totals[1] += training_num
        totals[2] += validation_num
        table.append([l, support_prc, training_prc, validation_prc])
    if ratio:
        total_support = f"{totals[0] / props['samples']:.0%}"
        total_training = f"{totals[1] / props['samples']:.0%}"
        total_validation = f"{totals[2] / props['samples']:.0%}"
    else:
        total_support = f"{totals[0]}"
        total_training = f"{totals[1]}"
        total_validation = f"{totals[2]}"
    table.append(["TOTAL"] + [total_support, total_training, total_validation])
    return pd.DataFrame(table, columns=["Class", "Support", "Training", "Test"])


def heatmap(
    df: pd.DataFrame, colormap, normalization, interpolation, reversed, widget
) -> None:
    """Show graphical heatmap.

    :param df: input dataset
    :param colormap: plasma, viridis, inferno, magma, cividis, gray, bone, pink, cool, hot,
                     rainbow, jet, turbo
    :param normalization: linear, log, symlog, asinh
    :param interpolation: nearest, antialiased, bilinear, bicubic, spline16, hanning,
                          hamming, quadric, catrom, gaussian, bessel, mitchell, sinc, lanczos
    :param reversed: reverse colormap
    :param widget: Matplotlib canvas
    """
    x = df2np(df, labels=None)
    cmap = colormap
    if reversed:
        cmap += "_r"
    widget.clear()
    widget.imshow(x, cmap=cmap, norm=normalization, interpolation=interpolation)
    widget.refresh()


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics.

    :param df: input dataset
    :return: statistics table
    """
    columns = properties(df)["columns"]
    table = df.describe().transpose().drop(columns="count")
    table.insert(loc=0, column="#", value=columns)
    return table.round(4)


def rangeplot(df: pd.DataFrame, widget) -> None:
    """Show Range plot (minimum, average, maximum and deviation of feature values).

    :param df: input dataset
    :param widget: Matplotlib canvas
    """
    x = df2np(df, labels=None)
    data_min, data_avg, data_max, data_std = (
        np.min(x, axis=0),
        np.mean(x, axis=0),
        np.max(x, axis=0),
        np.std(x, axis=0),
    )
    dev = 3
    data_low, data_high = data_avg - dev * data_std, data_avg + dev * data_std
    props = properties(df)
    try:
        x0 = [int(c) for c in props["columns"]]
    except ValueError:
        x0 = props["columns"]

    widget.clear()
    ax = widget.canvas.axes
    ax.set_title("Range plot", fontweight="bold")
    ax.plot(x0, data_min, label="minimum", linewidth=1, color="tab:green")
    ax.plot(x0, data_avg, label="average", linewidth=2, color="tab:blue")
    ax.plot(x0, data_max, label="maximum", linewidth=1, color="tab:red")
    ax.fill_between(
        x0, data_low, data_high, label=f"{dev}*sigma", color="tab:orange", alpha=0.25
    )
    if type(x0[0]) != str:
        ax.set_xlabel("feature")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid()
    widget.refresh()


def boxplot(
    df: pd.DataFrame,
    notch: bool,
    means: bool,
    boxes: bool,
    caps: bool,
    fliers: bool,
    widget,
) -> None:
    """Box plot.

    :param df: input dataset
    :param notch: show notches
    :param means: show means
    :param boxes: show boxes
    :param caps: show caps
    :param fliers: show fliers
    :param widget: Matplotlib canvas
    """
    widget.clear()
    ax = widget.canvas.axes
    ax.set_title("Box plot", fontweight="bold")
    params = {
        "x": df2np(df, labels=None),
        "notch": notch,
        "showmeans": means,
        "showbox": boxes,
        "showcaps": caps,
        "showfliers": fliers,
        "labels": properties(df)["columns"],
    }
    ax.boxplot(**params)
    ax.yaxis.grid(True)
    widget.refresh()


def violinplot(df: pd.DataFrame, means: bool, extrema: bool, widget) -> None:
    """Violin plot.

    :param df: input dataset
    :param means: show distribution means
    :param extrema: show distribution extrema
    :param widget: Matplotlib canvas
    """
    widget.clear()
    ax = widget.canvas.axes
    ax.set_title("Violin plot", fontweight="bold")
    params = {
        "dataset": df2np(df, labels=None),
        "showmeans": means,
        "showextrema": extrema,
    }
    ax.violinplot(**params)
    columns = properties(df)["columns"]
    ax.set_xticks(range(1, len(columns) + 1), labels=columns)
    ax.yaxis.grid(True)
    widget.refresh()


def parallelplot(df: pd.DataFrame, widget) -> None:
    """Parallel coordinates plotting.

    :param df: input dataset
    :param widget: Matplotlib canvas
    """
    widget.clear()
    ax = widget.canvas.axes
    ax.set_title("Parallel coordinates plot", fontweight="bold")
    props = properties(df)
    pd.plotting.parallel_coordinates(
        df, DF_TGT, cols=props["columns"], axvlines=False, ax=ax
    )
    widget.refresh()


def statsplot(
    df: pd.DataFrame,
    mode: str,
    featx: str | None,
    featy: str | None,
    multiple: str,
    fill: bool,
    logscale: bool,
    widget,
) -> None:
    """Various statistical plots.

    :param df: input dataset
    :param mode: plot mode (scatter, hist, density)
    :param featx: selected feature on the x-axis (None for all features)
    :param featy: selected feature on the y-axis (None for all features)
    :param multiple: multiple elements (dodge, stack, fill)
    :param fill: fill areas under curves
    :param logscale: use logarithmic scale for y-axis
    :param widget: Matplotlib canvas
    """
    hue = DF_TGT
    wide = featx is None and featy is None

    sns.set()
    widget.clear()
    ax = widget.canvas.axes
    if mode == "scatter":
        title = "Scatter plot"
        if wide:
            sns.scatterplot(data=df, ax=ax)
        else:
            sns.scatterplot(data=df, x=featx, y=featy, hue=hue, ax=ax)
            sns.rugplot(data=df, x=featx, y=featy, hue=hue, ax=ax)
    else:
        if mode == "hist":
            title = "Histogram plot"
            func = sns.histplot
            params = {
                "data": df,
                "ax": ax,
                "multiple": multiple,
                "fill": fill,
                "log_scale": [False, logscale],
            }
        elif mode == "density":
            title = "Density plot"
            func = sns.kdeplot
            mult = "layer" if multiple == "dodge" else multiple
            params = {
                "data": df,
                "ax": ax,
                "multiple": mult,
                "fill": fill,
                "log_scale": [False, logscale],
            }
        else:
            return
        if not wide:
            params.update({"x": featx, "y": featy, "hue": hue})
        func(**params)

    ax.set_title(title, fontweight="bold")
    ax.grid(visible=False, which="both")
    if featx is not None:
        ax.set_xlabel(featx)
    if featy is not None:
        ax.set_ylabel(featy)
    widget.refresh()
    sns.reset_orig()


def andrewsplot(df: pd.DataFrame, widget) -> None:
    """Visualize clusters of multivariate data using Andrews curves.

    :param df: input dataset
    :param widget: Matplotlib canvas
    """
    widget.clear()
    ax = widget.canvas.axes
    ax.set_title("Andrews curves plot", fontweight="bold")
    pd.plotting.andrews_curves(df, DF_TGT, ax=ax)
    widget.refresh()


def scatmtxplot(df: pd.DataFrame, widget) -> None:
    """Draw a matrix of scatter plots.

    :param df: input dataset
    :param widget: Matplotlib canvas
    """
    widget.clear()
    ax = widget.canvas.axes
    pd.plotting.scatter_matrix(df, alpha=0.2)
    plt.tight_layout()
    plt.show()
    widget.refresh()


def radialplot(df: pd.DataFrame, widget) -> None:
    """Plot a multidimensional dataset in 2D.

    :param df: input dataset
    :param widget: Matplotlib canvas
    """
    widget.clear()
    ax = widget.canvas.axes
    ax.set_title("Radial plot", fontweight="bold")
    pd.plotting.radviz(df, DF_TGT, ax=ax)
    widget.refresh()


def corrmtxplot(df: pd.DataFrame, widget) -> None:
    """Plot correlation matrix.

    :param df: input dataset
    :param widget: Matplotlib canvas
    """

    # TODO: Add correlation matrix text display in Analysis widget

    sns.set()
    widget.clear()
    data = df2np(df, labels=None)
    corr = pd.DataFrame(data).corr().to_numpy()
    mask = np.triu_indices_from(corr)
    corr[mask] = np.abs(corr)[mask]
    np.fill_diagonal(corr, val=1)
    ax = widget.canvas.axes
    ax.set_title(
        "Correlation matrix (low = signed, high = absolute)", fontweight="bold"
    )
    sns.heatmap(corr, vmin=-1, vmax=+1, center=0, square=True, cbar=False, ax=ax)
    ax.set(ylabel="feature index")
    widget.refresh()
    sns.reset_orig()


def bootstrapplot(df: pd.DataFrame, feature: str, widget) -> None:
    widget.clear()
    ax = widget.canvas.axes
    ax.set_title(f"Bootstrap plot (feature: {feature})", fontweight="bold")
    pd.plotting.bootstrap_plot(df[feature])
    plt.show()
    widget.refresh()


def reduce(
    df: pd.DataFrame,
    mode: str,
    whiten: str,
    kernel: str,
    iters: int,
    tol: float,
    alpha: int,
    outlier: str,
    distance: str,
    neighbors: int,
    leafsize: int,
    estimators: int,
    bootstrap: bool,
    reduced: bool,
    widget,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Perform dimensionality reduction.

    :param df: input dataset
    :param mode: reduction mode (PCA, KPCA, ICA, LDA, FA, NMF)
    :param whiten: whiten data (unit, arbitrary)
    :param kernel: kernel type (linear, poly, rbf, sigmoid, cosine)
    :param iters: maximum number of iterations
    :param tol: optimization tolerance
    :param alpha: scatter plot transparency
    :param outlier: outlier detection (local, isolation)
    :param distance: distance metric (euclidean, manhattan, chebyshev, minkowski, ...)
    :param neighbors: number of neighbors
    :param leafsize: tree leaf size
    :param estimators: number of estimators
    :param bootstrap: sample with replacement
    :param reduced: compute outliers on reduced components
    :param widget: Matplotlib widget
    """
    n_components = 2
    if mode == "pca":
        reducer = PCA(n_components=n_components)
    elif mode == "kpca":
        reducer = KPCA(n_components=n_components, kernel=kernel, tol=tol)
    elif mode == "ica":
        reducer = ICA(
            n_components=n_components,
            max_iter=iters,
            tol=tol,
            whiten=whiten + "-variance" if whiten != "OFF" else False,
        )
    elif mode == "lda":
        reducer = LDA(
            n_components=n_components,
            max_iter=iters,
        )
    elif mode == "fa":
        reducer = FA(n_components=n_components, max_iter=iters, tol=tol)
    elif mode == "nmf":
        reducer = NMF(n_components=n_components, max_iter=iters, tol=tol)
    else:
        return np.ndarray([]), pd.DataFrame(data=[[]], columns=["Property", "Value"])

    props = properties(df)
    data, target = df2np(df, props["labels"])
    reduction = reducer.fit_transform(data, target)

    if mode == "pca":
        results = [
            ["Explained variance", reducer.explained_variance_],
            ["Variance ratio", reducer.explained_variance_ratio_],
            ["Singular values", reducer.singular_values_],
            ["Class centroids", reducer.mean_],
            ["Noise variance", reducer.noise_variance_],
        ]
    elif mode == "kpca":
        results = [
            ["Eigenvalues", reducer.eigenvalues_],
            ["Eigenvectors", reducer.eigenvectors_],
            ["Inverse matrix", reducer.dual_coef_],
        ]
    elif mode == "ica":
        results = [
            ["Total iterations", reducer.n_iter_],
            ["Linear operator", reducer.components_],
        ]
        if whiten is not False:
            results.append(["Feature centroids", reducer.mean_])
            results.append(["Pre-whitening matrix", reducer.whitening_])
    elif mode == "lda":
        results = [
            ["EM step total iterations", reducer.n_batch_iter_],
            ["Number of dataset passes", reducer.n_iter_],
            ["Final perplexity score", reducer.bound_],
            ["Variational parameters", reducer.components_],
            ["Exponential expectation", reducer.exp_dirichlet_component_],
        ]
    elif mode == "fa":
        results = [
            ["Total iterations", reducer.n_iter_],
            ["Empirical mean", reducer.mean_],
            ["Maximum variance", reducer.components_],
            ["Noise variance", reducer.noise_variance_],
        ]
    elif mode == "nmf":
        results = [
            ["Total iterations", reducer.n_iter_],
            ["Factorization matrix", reducer.components_],
            ["Reconstruction error", reducer.reconstruction_err_],
        ]
    else:
        return np.ndarray([]), pd.DataFrame(data=[[]], columns=["Property", "Value"])

    widget.clear()
    ax = widget.canvas.axes
    ax.set_xlabel("first component")
    ax.set_ylabel("second component")

    if outlier == "local":
        lof = LOF(n_neighbors=neighbors, leaf_size=leafsize, metric=distance, n_jobs=-1)
        source = reduction if reduced else data
        outliers = np.where(lof.fit_predict(source) == -1)[0].tolist()
        scores = lof.negative_outlier_factor_
        radius = (scores.max() - scores) / (scores.max() - scores.min())
        ax.scatter(
            reduction[outliers, 0],
            reduction[outliers, 1],
            label="outliers",
            s=1000 * radius[outliers],
            edgecolors="k",
            facecolors="none",
            alpha=0.5,
        )
    elif outlier == "isolation":
        forest = IF(n_estimators=estimators, bootstrap=bootstrap, n_jobs=-1)
        forest.fit(reduction)
        DecisionBoundaryDisplay.from_estimator(
            forest,
            reduction,
            ax=ax,
            alpha=0.5,
            response_method="predict",
            plot_method="pcolormesh",
        )

    for c in range(props["classes"]):
        mask = np.where(target == c)[0].tolist()
        x = reduction[mask, 0]
        y = reduction[mask, 1]
        ax.scatter(x, y, label=props["labels"][c] + " (data)", s=5, alpha=alpha / 100)
        ax.scatter(
            x.mean(),
            y.mean(),
            label=props["labels"][c] + " (mean)",
            marker="x",
            zorder=props["classes"] + 1,
        )

    ax.grid(True, which="both")
    ax.legend(loc="best")
    widget.refresh()

    reduction = np.hstack((reduction, target.reshape(-1, 1)))
    return reduction, pd.DataFrame(results, columns=["Property", "Value"]).round(4)


def balance(df: pd.DataFrame, mode: str, method: str, size: int) -> pd.DataFrame:
    """Balance dataset classes.

    :param df: input dataset
    :param mode: balancing mode (oversample, undersample)
    :param method: balancing method
                   (oversample: rand, smote, adasyn, border, kmeans, svm)
                   (undersample: rand, nmiss, cc, cnn, enn, renn, aknn, iht, ncr, oss, tomek)
    :param size: number of nearest neighbors
    :return: balanced dataset
    """
    mode = mode.lower()
    method = method.lower()
    result = df.copy()
    if mode == "oversample":
        if method == "rand":
            sampler = RandomOverSampler()
        elif method == "smote":
            sampler = SMOTE(k_neighbors=size)
        elif method == "adasyn":
            sampler = ADASYN(n_neighbors=size)
        elif method == "border":
            sampler = BorderlineSMOTE(k_neighbors=size, m_neighbors=size * 2)
        elif method == "kmeans":
            sampler = KMeansSMOTE(k_neighbors=size)
        elif method == "svm":
            sampler = SVMSMOTE(k_neighbors=size, m_neighbors=size * 2)
        else:
            return result
    elif mode == "undersample":
        if method == "rand":
            sampler = RandomUnderSampler()
        elif method == "nmiss":
            sampler = NearMiss(version=3, n_neighbors=size, n_neighbors_ver3=size)
        elif method == "cc":
            sampler = ClusterCentroids()
        elif method == "cnn":
            sampler = CondensedNearestNeighbour(n_neighbors=size)
        elif method == "enn":
            sampler = EditedNearestNeighbours(n_neighbors=size)
        elif method == "renn":
            sampler = RepeatedEditedNearestNeighbours(n_neighbors=size)
        elif method == "aknn":
            sampler = AllKNN(n_neighbors=size)
        elif method == "iht":
            sampler = InstanceHardnessThreshold(estimator="gradient-boosting")
        elif method == "ncr":
            sampler = NeighbourhoodCleaningRule(n_neighbors=size)
        elif method == "oss":
            sampler = OneSidedSelection(n_neighbors=size)
        elif method == "tomek":
            sampler = TomekLinks()
        else:
            return result
    else:
        return result
    props = properties(result)
    data1, target1 = df2np(result, props["labels"])
    data2, target2 = sampler.fit_resample(data1, target1)
    return np2df(data2, target2, props["labels"], props["columns"])


def scale(df: pd.DataFrame, mode: str, method: str) -> pd.DataFrame:
    """Scale dataset features using various algorithms.

    :param df: input dataset
    :param mode: scaling mode (feature, sample)
    :param method: scaling method (standard, minmax, maxabs, normalize, robust, quantile)
    :return scaled dataset
    """
    mode2 = mode.lower()
    if mode2 == "feature":
        axis = 0
    elif mode2 == "sample":
        axis = 1
    else:
        return df
    props = properties(df)
    data, target = df2np(df, props["labels"])
    params = {"X": data, "axis": axis, "copy": False}
    method2 = method.lower()
    if method2 == "standard":
        standard_scale(**params)
    elif method2 == "minmax":
        minmax_scale(**params)
    elif method2 == "maxabs":
        maxabs_scale(**params)
    elif method2 == "normalize":
        normal_scale(**params)
    elif method2 == "robust":
        robust_scale(**params)
    elif method2 == "quantile":
        quantile_transform(
            **params
        )  # FIXME: Apply to train/test separately to prevent data leakage
    elif method2 == "minimum":
        data = np.apply_along_axis(lambda x: x - x.min(), axis, data)
    elif method2 == "maximum":
        data = np.apply_along_axis(lambda x: x - x.max(), axis, data)
    return np2df(data, target, props["labels"], props["columns"])


def select(
    df: pd.DataFrame,
    mode: str,
    score: str,
    estim: str,
    thr: float,
    kbest: int,
    trees: int,
    folds: int,
    rate: float,
    creg: float,
    tol: float,
    iters: int,
    widget,
) -> pd.DataFrame:
    """Automatic feature selection using various algorithms.

    :param df: input dataset
    :param mode: algorithm mode (var, kbest, perc, fpr, fdr, fwe, rfe, sfm, sfs)
    :param score: scoring function (chi2, anova, mutual)
    :param estim: estimator model (etc, rfc, ada, svc)
    :param thr: selection threshold
    :param kbest: number of best features to select
    :param trees: number of trees in the forest
    :param folds: number of folds in cross-validation
    :param rate: learning rate
    :param creg: regularization parameter
    :param tol: tolerance for stopping criteria
    :param iters: maximum number of iterations
    :param widget: Matplotlib canvas
    :return: selected features
    """
    props = properties(df)
    if mode == "var":
        selector = VarianceThreshold(threshold=thr)
    elif mode in ("rfe", "sfm", "sfs"):
        if estim == "etc":
            estimator = ETC(n_estimators=trees)
        elif estim == "rfc":
            estimator = RFC(n_estimators=trees)
        elif estim == "ada":
            estimator = ADA(n_estimators=trees, learning_rate=rate)
        elif estim == "svc":
            estimator = SVC(max_iter=iters, tol=tol, C=creg)
        else:
            return df
        if mode == "rfe":
            selector = RFECV(estimator, cv=folds)
        elif mode == "sfs":
            selector = SequentialFeatureSelector(
                estimator, n_features_to_select="auto", cv=folds
            )
        else:
            selector = SelectFromModel(estimator, max_features=int(thr))
    else:
        if score == "chi2":
            scorer = chi2
        elif score == "anova":
            scorer = f_classif
        elif score == "mutual":
            scorer = mutual_info_classif
        else:
            return df
        if mode == "kbest":
            if kbest > props["features"]:
                kbest = props["features"]
            selector = SelectKBest(score_func=scorer, k=kbest)
        elif mode == "perc":
            selector = SelectPercentile(score_func=scorer, percentile=thr)
        elif mode == "fpr":
            selector = SelectFpr(score_func=scorer, alpha=thr)
        elif mode == "fdr":
            selector = SelectFdr(score_func=scorer, alpha=thr)
        elif mode == "fwe":
            selector = SelectFwe(score_func=scorer, alpha=thr)
        else:
            return df

    data, target = df2np(df, props["labels"])
    try:
        selector.fit(data, target)
    except ValueError:
        return df
    if mode == "var":
        scores = selector.variances_
        pvalues = scores * 0
    elif mode == "rfe":
        scores = selector.cv_results_["mean_test_score"]
        pvalues = selector.cv_results_["std_test_score"]
    elif mode == "sfm":
        scores = selector.estimator_.feature_importances_
        pvalues = scores * 0
    elif mode == "sfs":
        scores = pvalues = np.zeros_like(props["columns"])  # , dtype=float)
    else:
        scores = selector.scores_
        pvalues = selector.pvalues_
    df = pd.DataFrame(
        np.column_stack((scores, pvalues)),
        columns=["Score", "P-value"],
    )
    df.insert(
        loc=0, column="Feature" if mode != "rfe" else "Subset", value=props["columns"]
    )
    selection = selector.get_support(indices=True)
    df = df.iloc[selection].sort_values(by="Score", ascending=False, ignore_index=True)

    if mode != "sfs":
        widget.clear()
        ax = widget.canvas.axes
        if mode != "rfe":
            ax.set_ylabel("variance" if mode == "var" else "score")
            x = range(props["features"])
            ax.bar(x, scores, label="raw score", color="tab:blue")
            ax.scatter(
                selection,
                np.ones_like(selection) * max(scores) / 100,
                label="selection",
                color="tab:orange",
            )
            if mode == "var":
                ax.axhline(thr, label="threshold", color="tab:red")
                ax.set_xlabel("feature")
            ax.legend(loc="upper right")
        else:
            x = np.arange(1, props["features"] + 1)
            ax.errorbar(
                x,
                scores,
                yerr=pvalues,
                uplims=True,
                lolims=True,
                label="average + deviation",
            )
            ax.set_xlabel("subset size")
            ax.set_ylabel("validation score")
            ax.legend(loc="best")
        widget.refresh()
    return df.round(4)

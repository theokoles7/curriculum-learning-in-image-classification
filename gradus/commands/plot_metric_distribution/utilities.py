"""# gradus.commands.plot_metric_distribution.utilities

Auxiliary methods for facilitating metric distribution plots.
"""

__all__ =   [
                "load_parquet",
                "plot_metric",
                "shapiro_label",
            ]

from typing             import Union

from matplotlib.pyplot  import Axes
from numpy              import isnan, linspace
from numpy.random       import default_rng
from numpy.typing       import NDArray
from pandas             import DataFrame, read_parquet
from scipy              import stats

def load_parquet(
    dataset_id: str,
    seed:       int
) -> DataFrame:
    """# Load Parquet File into DataFrame.

    ## Args:
        * dataset_id    (str):  Identifier of dataset whose metrics are being loaded.
        * seed          (int):  RNG seed used when metrics were computed.

    ## Returns:
        * DataFrame:    Metrics dataframe loaded from parquet file.
    """
    # Resolve path to parquet file.
    path:   str =   f".cache/scores/{dataset_id}/metric-scores_seed-{seed}.parquet"

    # Load file & drop non-metric labels.
    try: return read_parquet(path = path).drop(labels = ["index", "class"], axis = 1)

    # Indicate that file was not found.
    except FileNotFoundError: exit(code = 1)

def plot_metric(
    axes:       Axes,
    values:     NDArray,
    column:     str,
    bins:       Union[int, str] =   "auto",
    alpha:      float =             0.05,
    normalize:  bool =              False
) -> None:
    """# Plot Metric Histogram.

    ## Args:
        * axes      (Axes): Plot axes position.
        * values    (NDArray): Metric values.
        * column    (str): Name of metric.
        * bins      (int | str): Number of histogram bins.
        * alpha     (float): Normality test threshold.
        * normalize (bool, optional): Normalize values. Defaults to False.
    """
    # Drop NaNs.
    values: NDArray =   values[~isnan(values)]

    # If no values are left, no plot should be shown.
    if len(values) == 0: axes.set_visible(False); return

    # Calculate mean and location.
    mu, sigma =         values.mean(), values.std(ddof = 1)

    # Normalize values if specified.
    if normalize: values = (values - mu) / sigma if sigma > 0 else values - mu

    # Calculate histogram.
    axes.hist(
        values,
        bins = bins,
        density = True,
        alpha = 0.35,
        color = "#4C72B0",
        label = "Empirical"
    )

    # Get linear space values.
    x:  NDArray =   linspace(values.min(), values.max(), 500)

    # Calculate KDE.
    axes.plot(
        x,
        stats.gaussian_kde(values)(x),
        color = "#4C72B0",
        lw = 2,
        label = "KDE"
    )

    # Create normalized overlay.
    if normalize: normal_pdf = stats.norm.pdf(x, 0, 1); normal_label = "N(0, 1)"
    else: normal_pdf = stats.norm.pdf(x, mu, sigma); normal_label = f"N(μ={mu:.3g}, σ={sigma:.3g})"

    # Plot normal overlay.
    axes.plot(x, normal_pdf, color = "#DD8452", lw = 2, linestyle = "--", label = normal_label)

    # Handle labels and legends.
    sw: str =   shapiro_label(values, alpha)
    axes.set_title(column, fontsize = 9, fontweight = "bold")
    axes.set_xlabel(sw, fontsize = 6.5, color = "gray")
    axes.set_ylabel("Density", fontsize = 8)
    axes.legend(fontsize = 10, loc = "upper right")
    axes.spines[["top", "right"]].set_visible(False)

def shapiro_label(
    values: NDArray,
    alpha:  float = 0.05
) -> str:
    """# Construct Shapiro-Wilk Label.

    ## Args:
        * values    (NDArray):  Metric values.
        * alpha     (float):    Normality test significance level. Defaults to 0.05.

    ## Returns:
        * str:  Human-readable Shapiro-Wilk result string.
    """
    # Note quatity of values.
    n:          int =       len(values)

    # Skip if there are less than 3 values.
    if n < 3: return "Shapiro-Wilk: n < 3 (Skipped)"

    # Shapiro-Wilk is most reliable for n <= 5,000. Sample down if needed.
    sample:     NDArray =   values if n <= 5000 else default_rng(0).choice(
                                values,
                                5000,
                                replace = False
                            )
    
    # Test normality.
    stat, p =               stats.shapiro(sample)

    # Decide normality.
    verdict:    str =       "Normal" if p >= alpha else "Non-Normal"

    # Indicate if sampled from values or not.
    suffix:     str =       " (Sampled n=5000)" if n > 5000 else ""

    # Provide label.
    return f"Shapiro-Wilk: W={stat:.4f}, p={p:.4g} -> {verdict}{suffix}"
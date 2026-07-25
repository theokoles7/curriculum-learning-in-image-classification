"""# gradus.commands.plot_metric_distribution.main

Main process entry point for `plot-metric-distribution` command.
"""

from gradus.commands.plot_metric_distribution.__args__  import MetricDistributionConfig
from gradus.registration                                import register_command

@register_command(
    id =        "plot-metric-distribution",
    config =    MetricDistributionConfig
)
def plot_metric_distribution_entry_point(
    dataset_id: str,
    seed:       int =   1,
    bins:       int =   0,
    alpha:      float = 0.05,
    normalize:  bool =  False,
    *args,
    **kwargs
) -> str:
    """# Plot Dataset Metric Distributions.

    ## Args:
        * dataset_id    (str):      Identifier of dataset whose metric distributions are being 
                                    plotted.
        * seed          (int):      RNG seed used when metrics were computed. Defaults to 1.
        * bins          (int):      Number of histogram bins. Defaults to auto.
        * alpha         (float):    Normality test significance level. Defaults to 0.05.
        * normalize     (bool):     Z-score each metric before plotting. Defaults to False.

    ## Returns:
        * str:  Path at which metric distribution plots were saved.
    """
    from matplotlib.pyplot                                  import subplots
    from numpy                                              import array, nan
    from numpy.typing                                       import NDArray
    from pandas                                             import DataFrame

    from gradus.commands.plot_metric_distribution.utilities import load_parquet, plot_metric

    # Resolve number of bins.
    bins =                  "auto" if bins == 0 else bins

    # Load parquet file into data frame.
    metrics:    DataFrame = load_parquet(dataset_id = dataset_id, seed = seed)

    # Count metrics & calculate rows needed.
    n:          int =       len(metrics.columns)
    n_rows:     int =       (n + 2) // 3

    # Calculate height & width.
    fig_h:      float =     3.5 * n_rows
    fig_w:      float =     4.5 * 3

    # Construct subplots.
    fig, axes =             subplots(n_rows, 3, figsize = (fig_w, fig_h))

    # Flatten for easy indexing.
    axes:       NDArray =   array(axes).reshape(-1)

    # For each metric...
    for m, metric in enumerate(metrics.columns):

        # Plot the histogram.
        plot_metric(
            axes =      axes[m],
            values =    metrics[metric].to_numpy(dtype = float, na_value = nan),
            column =    metric,
            bins =      bins,
            alpha =     alpha,
            normalize = normalize
        )

    # Hide unused plots.
    for p in range(n, len(axes)): axes[p].set_visible(False)

    # Set title and configurations.
    title:  str =   f"""{dataset_id.upper()} Metric Distributions{f" (Normalized)" if normalize else ""}"""
    fig.suptitle(title, fontsize = 13, fontweight = "bold", y = 0.99)
    fig.tight_layout()
    fig.savefig(f".cache/scores/{dataset_id}/metrics-scores_seed-{seed}_distributions.png")

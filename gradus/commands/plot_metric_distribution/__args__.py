"""# gradus.commands.plot_metric_distribution.args

Argument definitions & parsing for plot-metric-distribution command.
"""

__all__ = ["MetricDistributionConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import CommandConfig

class MetricDistributionConfig(CommandConfig):
    """# Metric Distribution Plotting Configuration"""

    def __init__(self):
        """# Instantiate Metric Distribution Plotting Configuration."""
        # Initialize configuration.
        super(MetricDistributionConfig, self).__init__(
            name =              "plot-metric-distribution",
            help =              """Plot distribution of metrics across dataset samples.""",
            subparser_title =   "dataset-id",
            subparser_help =    """Identifier of dataset whose metric distribution is being 
                                plotted."""
        )

    # HELPERS ======================================================================================

    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Parser Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        from gradus.registration    import DATASET_REGISTRY

        parser.add_argument(
            "dataset_id",
            type =      str,
            choices =   DATASET_REGISTRY.list_entries(),
            help =      """Dataset whose metric distributions are being plotted."""
        )

        parser.add_argument(
            "--seed",
            dest =      "seed",
            type =      int,
            default =   1,
            help =      """RNG seed used when metrics were computed. Defaults to 1."""
        )

        parser.add_argument(
            "--bins",
            dest =      "bins",
            type =      int,
            default =   0,
            help =      """Number of histogram bins. Defaults to auto."""
        )

        parser.add_argument(
            "--alpha",
            dest =      "alpha",
            type =      float,
            default =   0.05,
            help =      """Normality test significance level. Defaults to 0.05."""
        )

        parser.add_argument(
            "--normalize",
            dest =      "normalize",
            action =    "store_true",
            default =   False,
            help =      """Z-score each metric before plotting (μ=0, σ=1)."""
        )
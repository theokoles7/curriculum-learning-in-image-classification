"""# gradus.curricula

Curriculum construction/components module.
"""

__all__ =   [
                # Protocol
                "Curriculum",

                # METRICS ==========================================================================

                # Protocol
                "Metric",

                # Complexity
                "ColorVariance",
                "CompressionRatio",
                "EdgeDensity",
                "ObjectCount",
                "SpatialFrequency",
                "WaveletEnergy",
                "WaveletEntropy",

                # Model-informed
                "TimeToConvergence",
                "TimeToSaturation",

                # RANKS ============================================================================

                # Protocol
                "Rank",

                # Concrete
                "Ascending",
                "Descending",
                "DistanceFromMean",
                "Lexicographic",
                "NormalizedMean",
                "PairwiseCorrelation",
                "Weighted",

                # SCHEDULES ========================================================================

                # Protocol
                "Schedule",

                # Concrete
                "AdaptiveSchedule",
                "BanditSchedule",
                "GradientSchedule",
                "LinearSchedule",
                "UncertaintySchedule",
            ]

from gradus.curricula.protocol  import Curriculum
from gradus.curricula.metrics   import *
from gradus.curricula.ranks     import *
from gradus.curricula.schedules import *

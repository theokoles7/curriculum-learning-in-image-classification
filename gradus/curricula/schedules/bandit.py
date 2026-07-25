"""# gradus.curricula.schedules.bandit

Nonstationary multi-armed bandit curriculum pacing schedule.

Treats contiguous difficulty strata of the (metric-ranked) curriculum as the arms
of an EXP3.S bandit. Each epoch, the bandit commits a distribution over strata and
observes a per-stratum learning-progress reward derived from the change in mean
softmax entropy of that stratum's batches. Because stratum value changes as the
model learns (a stratum that is highly informative early may be exhausted later),
the bandit is nonstationary: EXP3.S redistributes a small share of weight uniformly
each round so that no arm collapses permanently and previously-exhausted strata can
recover.

This is the "policy" rung of the project: unlike static curricula, which fix an
ordering against a frozen difficulty reference, the bandit conditions its selection
on a signal recomputed from the model's current state each epoch.

## Assumptions (verify against your plumbing):
    * The Curriculum's batches are difficulty-ordered: batch index 0 holds the
      easiest samples, batch index N-1 the hardest. This holds when the Curriculum
      is constructed with a metric + ascending rank, since holistic ranking sorts
      all samples by the metric before chunking. Strata are contiguous bands over
      these indices.
    * `dataset.step(...)` forwards `entropy_df` into this schedule's `_order_`, and
      the value returned by `_order_` is applied via `Curriculum.set_order`.
    * `entropy_df` is indexed by curriculum batch index (actual_idx) with a column
      'mean_entropy'. (This is the case in the current train loop.)
    * `Curriculum.set_order` tolerates repeated and omitted indices (it does, in the
      current implementation - it filters to valid range without deduplicating).
      Only relevant in 'allocate' mode.
"""

__all__ = ["BanditSchedule"]

from typing                                 import Any, Dict, List, Literal, override

from gradus.curricula.schedules.protocol    import Schedule
from gradus.registration                    import register_schedule

@register_schedule(
    id =    "bandit",
    tags =  ["adaptive", "bandit", "pacing"]
)
class BanditSchedule(Schedule):
    """# EXP3.S Bandit Curriculum Pacing Schedule"""

    def __init__(self,
        total_samples:      int,
        total_epochs:       int,
        batch_size:         int,
        num_strata:         int =                               10,
        # mode:               Literal["reorder", "allocate"] =    "reorder",
        cold_start_epochs:  int =                               1,
        gamma:              float =                             None,
        start_fraction:     float =                             0.3
    ):
        """# Instantiate Bandit Curriculum Pacing Schedule.

        ## Args:
            * total_samples     (int):      Total number of training samples.
            * total_epochs      (int):      Total number of training epochs.
            * batch_size        (int):      Number of samples per batch.
            * num_strata        (int):      Number of difficulty strata (bandit arms).
                                            Defaults to 10.
            * mode              (str):      'reorder' exposes every batch once per epoch,
                                            ordered by descending arm probability (DSI=0,
                                            isolates ordering); 'allocate' draws batch
                                            slots in proportion to arm probability with
                                            replacement (DSI>0, adaptive coverage).
                                            Defaults to 'reorder'.
            * cold_start_epochs (int):      Epochs of natural curriculum order before the
                                            bandit activates. Defaults to 1.
            * gamma             (float):    EXP3.S exploration rate in (0, 1]. Defaults to
                                            the Auer et al. (2002) recommendation when None.
            * start_fraction    (float):    Unused; kept for CLI consistency with other
                                            schedules. Defaults to 0.3.
        """
        from math   import e, log, sqrt

        # Initialize protocol.
        super(BanditSchedule, self).__init__(
            schedule_id =   f"bandit-reorder",
            total_samples = total_samples,
            total_epochs =  total_epochs,
            batch_size =    batch_size
        )

        # Validate parameters.
        if num_strata < 2:
            raise ValueError(f"num_strata must be >= 2; got {num_strata}")
        # if mode not in ("reoder", "allocate"):
        #     raise ValueError(f"mode must be reoder or allocate; got {mode}")

        # Clamp strata to the number of available batches.
        self._num_strata_:      int =           min(num_strata, self._total_batches_)
        self._mode_:            str =           "reorder"
        self._cold_start_:      int =           cold_start_epochs

        # EXP3.S exploration & nonstationarity parameters.
        # gamma: exploration; alpha: per-round share term (set to 1/T per Auer et al.).
        T:                      int =           max(2, self._total_epochs_)
        self._gamma_:           float =         gamma if gamma is not None else min(
                                                    1.0,
                                                    sqrt(
                                                        self._num_strata_ * log(self._num_strata_ * T)
                                                        / ((e - 1.0) * T)
                                                    )
                                                )
        self._alpha_:           float =         1.0 / T

        # EXP3.S arm weights (init uniform).
        self._weights_:         List[float] =   [1.0] * self._num_strata_

        # Bookkeeping for the deferred reward update: the distribution used last epoch
        # and the per-stratum mean entropy observed last epoch.
        self._last_probs_:      List[float] =   None
        self._prev_entropy_:    List[float] =   None

        # Precompute stratum -> batch-index ranges (contiguous difficulty bands).
        self._strata_:          List[List[int]] =   self._build_strata_()

    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Bandit Schedule Dictionary Representation"""
        return  {
                    **super().dict,
                    "num_strata":           self._num_strata_,
                    "mode":                 self._mode_,
                    "cold_start_epochs":    self._cold_start_,
                    "gamma":                round(self._gamma_, 6),
                    "alpha":                round(self._alpha_, 6),
                }

    # HELPERS ======================================================================================

    def _build_strata_(self) -> List[List[int]]:
        """# Partition Batch Indices into Contiguous Difficulty Strata.

        ## Returns:
            * List[List[int]]:  Stratum k holds a contiguous band of batch indices,
                                easiest (k=0) to hardest (k=K-1).
        """
        from numpy  import array_split

        # array_split handles non-even divisions gracefully.
        return [list(map(int, band)) for band in array_split(range(self._total_batches_), self._num_strata_)]

    def _probabilities_(self) -> List[float]:
        """# Current EXP3.S Sampling Distribution over Strata.

        ## Returns:
            * List[float]:  Probability per stratum, mixing exploitation (weights) with
                            uniform exploration (gamma).
        """
        total:  float =         sum(self._weights_)
        K:      int =           self._num_strata_
        return  [
                    (1.0 - self._gamma_) * (w / total) + self._gamma_ / K
                    for w in self._weights_
                ]

    def _update_(self,
        entropy_df,
    ) -> None:
        """# EXP3.S Weight Update from Last Epoch's Per-Stratum Learning Progress.

        Reward for a stratum is the normalized drop in its mean softmax entropy since
        the previous epoch - a stratum whose uncertainty is falling fastest is where
        the model is currently making the most progress. Rewards are min-max normalized
        to [0, 1] across strata each epoch, then fed through the importance-weighted
        EXP3.S update.

        ## Args:
            * entropy_df    (DataFrame):    Per-batch mean entropy for the just-finished
                                            epoch, indexed by curriculum batch index.
        """
        from math   import e

        # Compute current per-stratum mean entropy from this epoch's readings.
        cur_entropy:    List[float] =   self._stratum_entropy_(entropy_df)

        # First activation: nothing to compare against yet; just store and return.
        if self._prev_entropy_ is None or self._last_probs_ is None:
            self._prev_entropy_ = cur_entropy
            return

        # Per-stratum progress = entropy drop (prev - cur). Missing readings -> 0 progress.
        progress:   List[float] =   [
                                        (self._prev_entropy_[k] - cur_entropy[k])
                                        if (cur_entropy[k] is not None and self._prev_entropy_[k] is not None)
                                        else 0.0
                                        for k in range(self._num_strata_)
                                    ]

        # Min-max normalize progress to [0, 1] (guard against a flat vector).
        lo:     float =     min(progress)
        hi:     float =     max(progress)
        spread: float =     hi - lo
        rewards: List[float] =  [
                                    (p - lo) / spread if spread > 1e-12 else 0.5
                                    for p in progress
                                ]

        # EXP3.S update: importance-weighted multiplicative weights + uniform share term.
        K:          int =       self._num_strata_
        weight_sum: float =     sum(self._weights_)
        for k in range(K):

            # Only strata that were reachable last epoch (prob > 0) yield an estimate.
            p_k:        float = self._last_probs_[k]
            observed:   bool =  (cur_entropy[k] is not None) and (p_k > 0.0)
            x_hat:      float = (rewards[k] / p_k) if observed else 0.0

            # Multiplicative update.
            self._weights_[k] *= pow(e, self._gamma_ * x_hat / K)

        # Share term (the ".S"): redistribute a little weight uniformly for nonstationarity.
        weight_sum = sum(self._weights_)
        share:      float =     (e * self._alpha_ / K) * weight_sum
        self._weights_ =        [w + share for w in self._weights_]

        # Carry forward this epoch's entropy as the next baseline.
        self._prev_entropy_ =   cur_entropy

    def _stratum_entropy_(self,
        entropy_df,
    ) -> List[float]:
        """# Mean Entropy per Stratum from a Per-Batch Entropy DataFrame."""
        # Handle absent/empty signal.
        if entropy_df is None or getattr(entropy_df, "empty", True) \
                or "mean_entropy" not in getattr(entropy_df, "columns", []):
            return [None] * self._num_strata_

        # Collapse duplicate batch indices (a batch may be presented more than once
        # per epoch under reordering/oversampling) to a single mean per batch.
        per_batch = entropy_df.groupby(level = 0)["mean_entropy"].mean()

        result: List[float] =   []
        for band in self._strata_:
            vals:   List[float] =   [
                                        float(per_batch.loc[b])
                                        for b in band
                                        if b in per_batch.index
                                    ]
            result.append(sum(vals) / len(vals) if vals else None)

        return result

    @override
    def _order_(self,
        epoch:          int,
        entropy_df =    None,
        **kwargs:       Any
    ) -> List[int]:
        """# Compute Bandit Batch Ordering for Current Epoch.

        ## Args:
            * epoch         (int):              Current epoch (1-indexed).
            * entropy_df    (DataFrame | None): Per-batch mean entropy from the previous
                                                epoch, indexed by curriculum batch index.
            * **kwargs      (Any):              Other signals (loss, val_acc, grad_norm_df,
                                                std_df); ignored by this schedule.

        ## Returns:
            * List[int]:    Batch indices to expose this epoch.
        """
        from numpy.random    import default_rng

        # Fold in last epoch's observed reward before deciding this epoch's distribution.
        self._update_(entropy_df)

        # Cold start: natural difficulty order, every batch once.
        if epoch <= self._cold_start_:
            self._last_probs_ = self._probabilities_()
            return list(range(self._total_batches_))

        # Current sampling distribution over strata.
        probs:  List[float] =   self._probabilities_()
        self._last_probs_ =     probs

        # --- REORDER MODE: every batch once, strata ordered by descending probability ---
        if self._mode_ == "reorder":
            stratum_order:  List[int] = sorted(
                                            range(self._num_strata_),
                                            key =       lambda k: probs[k],
                                            reverse =   True
                                        )
            order:          List[int] = [b for k in stratum_order for b in self._strata_[k]]

            self.__logger__.debug(
                f"Epoch {epoch}: reorder; top stratum = {stratum_order[0]}, "
                f"probs = {[round(p, 3) for p in probs]}"
            )
            return order

        # --- ALLOCATE MODE: draw N batch-slots ~ Multinomial(probs), batches w/ replacement ---
        rng = default_rng(seed = epoch)

        # Number of batch-slots to draw from each stratum this epoch (total = total_batches).
        counts: List[int] =     rng.multinomial(self._total_batches_, probs).tolist()

        order:  List[int] =     []
        for k, n in enumerate(counts):
            if n <= 0 or not self._strata_[k]:
                continue
            # Draw n batches from stratum k with replacement.
            order.extend(int(b) for b in rng.choice(self._strata_[k], size = n, replace = True))

        # Guarantee non-empty order (protocol requires it).
        if not order:
            order = list(range(self._total_batches_))

        self.__logger__.debug(
            f"Epoch {epoch}: allocate; stratum counts = {counts}, "
            f"probs = {[round(p, 3) for p in probs]}"
        )
        return order

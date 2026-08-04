"""
Probability calibration for Tm predictions.

The model emits a point estimate Tm in degrees C. Two different probabilities can be
attached to it, and they answer different questions:

**local**       P(Tm_true >= T | Tm_pred ~ x)
                "Is *this* protein above T?" -- the per-sequence annotation. A protein
                predicted exactly at the threshold sits near 0.5.

**cumulative**  P(Tm_true >= T | Tm_pred >= x)
                "I screen everything scoring >= x -- what fraction are hits?" -- a
                set-level enrichment precision. Always >= the local value at the same x,
                because it averages the local curve over the whole upper tail. It is a
                property of the *screened library* as much as of the model.

The local curve is modelled as a heteroscedastic location-scale residual fit

    Tm_true = m(x) + s(x) * z,    m(x) = a + b*x,    log s(x) = c + d*x
    P(Tm_true >= T | Tm_pred = x) = 1 - F_z((T - m(x)) / s(x))

so a single fit serves *every* threshold. That matters because the threshold is
user-adjustable at inference time. A per-threshold logistic would instead bake the
fitting set's base rate into its intercept, which transports badly across datasets with
different fractions of thermostable proteins; it is kept here only as a cross-check
(`fit_per_cutoff_logistic`).

The cumulative curve is purely empirical -- `cumulative_precision` is the logic that
previously lived inline in `ranking.py:plot_tm_cutoff_curve_number`.
"""
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

# Plot styling, matched to src/eval/ranking.py
FIGSIZE = (16, 6)
LABEL_SIZE = 16
TICK_SIZE = 13
LEGEND_SIZE = 12
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
# Okabe-Ito. COLORS keeps ranking.py's look, but its orange/green pair is only
# dE 4.5 apart under protanopia, so plots that rely on hue to separate series use
# this set instead (worst all-pairs dE 7.6, cleared with direct labels).
CVD_COLORS = ["#0072B2", "#E69F00", "#CC79A7", "#009E73"]
LOW_SUPPORT_FILL = "#d9d9d9"

# Cells thinner than this are too noisy to contribute to aggregate calibration error.
MIN_BIN_FOR_ECE = 5
MIN_BIN_FOR_MCE = 20


# ---------------------------------------------------------------------------
# Binomial intervals
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the Wald interval throughout this module: with the per-bin counts
    seen on the small independent evaluation sets, Wald intervals routinely leave
    [0, 1] and collapse to zero width when k is 0 or n.

    Args:
        k (int): Successes.
        n (int): Trials.
        alpha (float): Significance level.

    Returns:
        Tuple[float, float]: (lower, upper), or (nan, nan) when n == 0.
    """
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# Cumulative (set-level) enrichment
# ---------------------------------------------------------------------------

def cumulative_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cutoffs: Sequence[float],
    threshold: float,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Set-level hit rate P(Tm_true >= T | Tm_pred >= x) for each cutoff x.

    Args:
        y_true (np.ndarray): Actual Tm values.
        y_pred (np.ndarray): Predicted Tm values.
        cutoffs (Sequence[float]): Predicted-Tm cutoffs to evaluate.
        threshold (float): T, the actual-Tm threshold defining a hit.
        strict (bool): Use strict '>' on both comparisons instead of '>='. This
            reproduces the legacy behaviour of the inline comprehension in
            `ranking.py` exactly; new callers should leave it False.

    Returns:
        pd.DataFrame: cutoff, n_selected, n_hits, rate, ci_low, ci_high. Cutoffs that
            select nothing yield NaN rather than a divide-by-zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows = []
    for cutoff in cutoffs:
        selected = y_pred > cutoff if strict else y_pred >= cutoff
        n_selected = int(selected.sum())
        if n_selected == 0:
            rows.append((cutoff, 0, 0, np.nan, np.nan, np.nan))
            continue
        hits = y_true[selected] > threshold if strict else y_true[selected] >= threshold
        n_hits = int(hits.sum())
        low, high = wilson_interval(n_hits, n_selected)
        rows.append((cutoff, n_selected, n_hits, n_hits / n_selected, low, high))

    return pd.DataFrame(
        rows, columns=["cutoff", "n_selected", "n_hits", "rate", "ci_low", "ci_high"]
    )


def cumulative_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cutoffs: Sequence[float],
    thresholds: Sequence[float],
) -> Dict[str, List[List[float]]]:
    """
    Cumulative precision over a full (threshold x cutoff) grid.

    Returns:
        dict: Row-major arrays keyed 'p', 'n_selected', 'n_hits', 'ci_low', 'ci_high',
            each indexed [threshold_index][cutoff_index].
    """
    out: Dict[str, List[List[float]]] = {
        k: [] for k in ("p", "n_selected", "n_hits", "ci_low", "ci_high")
    }
    for threshold in thresholds:
        df = cumulative_precision(y_true, y_pred, cutoffs, threshold)
        out["p"].append(df["rate"].tolist())
        out["n_selected"].append(df["n_selected"].tolist())
        out["n_hits"].append(df["n_hits"].tolist())
        out["ci_low"].append(df["ci_low"].tolist())
        out["ci_high"].append(df["ci_high"].tolist())
    return out


# ---------------------------------------------------------------------------
# Local (per-sequence) calibrator
# ---------------------------------------------------------------------------

@dataclass
class LocationScaleCalibrator:
    """
    Heteroscedastic location-scale model of Tm_true given Tm_pred.

    Attributes:
        a, b: Location coefficients, m(x) = a + b*x.
        c, d: Log-scale coefficients, log s(x) = c + d*x.
        scale_floor: Lower bound on s(x), degrees C, so the tails cannot collapse.
        z_grid, z_cdf: Tabulated CDF of the standardized residual.
        t_df, tail_lo, tail_hi: Student-t tail parameters and the splice points.
        fit_meta: Provenance and fit diagnostics.
    """

    a: float
    b: float
    c: float
    d: float
    scale_floor: float
    z_grid: np.ndarray
    z_cdf: np.ndarray
    t_df: float
    tail_lo: float
    tail_hi: float
    fit_meta: dict = field(default_factory=dict)

    def location(self, x: np.ndarray) -> np.ndarray:
        """Conditional mean m(x)."""
        return self.a + self.b * np.asarray(x, dtype=float)

    def scale(self, x: np.ndarray) -> np.ndarray:
        """Conditional standard deviation s(x), floored."""
        return np.maximum(np.exp(self.c + self.d * np.asarray(x, dtype=float)), self.scale_floor)

    def cdf_z(self, z: np.ndarray) -> np.ndarray:
        """Tabulated CDF of the standardized residual, clamped outside the grid."""
        return np.interp(np.asarray(z, dtype=float), self.z_grid, self.z_cdf, left=0.0, right=1.0)

    def local_prob(self, tm_pred: np.ndarray, threshold: float) -> np.ndarray:
        """
        P(Tm_true >= threshold | Tm_pred = tm_pred).

        Args:
            tm_pred (np.ndarray): Predicted Tm values.
            threshold (float): Actual-Tm threshold in degrees C.

        Returns:
            np.ndarray: Probabilities in [0, 1].
        """
        x = np.asarray(tm_pred, dtype=float)
        z = (threshold - self.location(x)) / self.scale(x)
        return np.clip(1.0 - self.cdf_z(z), 0.0, 1.0)

    def local_grid(self, pred_grid: np.ndarray, threshold_grid: np.ndarray) -> np.ndarray:
        """Probability table of shape (len(threshold_grid), len(pred_grid))."""
        return np.vstack([self.local_prob(pred_grid, float(t)) for t in threshold_grid])

    def crossing_point(self, threshold: float, lo: float = 0.0, hi: float = 150.0) -> float:
        """
        Predicted Tm at which local_prob crosses 0.5, found by bisection.

        This is (threshold - a) / b only when the standardized residual has zero median;
        it is solved numerically because s(x) varies with x.
        """
        f = lambda x: self.local_prob(np.array([x]), threshold)[0] - 0.5
        if f(lo) > 0 or f(hi) < 0:
            return float("nan")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) < 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def to_dict(self) -> dict:
        """Serializable parametric block for the artifact."""
        return {
            "type": "location_scale_residual",
            "location": {"form": "linear", "a": self.a, "b": self.b},
            "scale": {"form": "loglinear", "c": self.c, "d": self.d, "floor_c": self.scale_floor},
            "z_cdf": {
                "form": "empirical_grid_with_t_tails",
                "z_grid": np.round(self.z_grid, 4).tolist(),
                "F": np.round(self.z_cdf, 6).tolist(),
                "tail": {
                    "form": "student_t",
                    "df": self.t_df,
                    "lower_z": self.tail_lo,
                    "upper_z": self.tail_hi,
                },
            },
            "fit_meta": self.fit_meta,
        }


def fit_location_scale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale_floor: float = 1.0,
    z_grid_step: float = 0.05,
    z_grid_limit: float = 6.0,
    tail_q: float = 0.01,
) -> LocationScaleCalibrator:
    """
    Fit the conditional distribution of Tm_true given Tm_pred.

    Three stages: OLS for the location, Gaussian MLE (location held fixed) for the
    log-linear scale, then a tabulated empirical CDF of the standardized residual with
    Student-t tails spliced on beyond the `tail_q` quantiles, where empirical counts run
    out. The splice is continuity-preserving and the result is forced monotone.

    Args:
        y_true (np.ndarray): Actual Tm.
        y_pred (np.ndarray): Predicted Tm.
        scale_floor (float): Minimum conditional SD in degrees C.
        z_grid_step (float): Resolution of the tabulated CDF.
        z_grid_limit (float): Grid extent in standardized units.
        tail_q (float): Quantile beyond which the Student-t tail takes over.

    Returns:
        LocationScaleCalibrator: The fitted calibrator.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n = len(x)

    # Stage 1 -- location by OLS. Keeping this un-weighted preserves (a, b) as an
    # interpretable bias diagnostic: b > 1 means the point predictor is under-dispersed.
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    s2 = float(resid @ resid) / (n - 2)
    sxx = float(((x - x.mean()) ** 2).sum())
    b_se = math.sqrt(s2 / sxx)
    a_se = math.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx))

    # Stage 2 -- log-linear scale. Seed from a regression of log|r| on x, corrected for
    # E[log|Z|] = -(gamma + log 2)/2, then refine by MLE with the location held fixed.
    log_abs = np.log(np.abs(resid) + 1e-6)
    d0, c0 = np.polyfit(x, log_abs, 1)
    c0 += 0.5 * (math.log(2.0) + np.euler_gamma)

    def nll(params: np.ndarray) -> float:
        c, d = params
        s = np.maximum(np.exp(c + d * x), scale_floor)
        return float(np.sum(np.log(s) + 0.5 * (resid / s) ** 2))

    opt = minimize(nll, x0=np.array([c0, d0]), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    c, d = (float(opt.x[0]), float(opt.x[1])) if opt.success else (float(c0), float(d0))

    scale = np.maximum(np.exp(c + d * x), scale_floor)
    z = resid / scale

    # Stage 3 -- tabulated CDF of z. Hazen plotting positions on the sorted sample.
    z_sorted = np.sort(z)
    f_hazen = (np.arange(n) + 0.5) / n
    z_grid = np.arange(-z_grid_limit, z_grid_limit + z_grid_step, z_grid_step)
    f_grid = np.interp(z_grid, z_sorted, f_hazen, left=0.0, right=1.0)

    # Student-t tails beyond the empirical quantiles, spliced so the CDF stays
    # continuous at the join and still approaches 0 and 1.
    t_df, t_loc, t_scale = stats.t.fit(z)
    z_lo = float(np.quantile(z, tail_q))
    z_hi = float(np.quantile(z, 1 - tail_q))
    f_lo = float(np.interp(z_lo, z_sorted, f_hazen))
    f_hi = float(np.interp(z_hi, z_sorted, f_hazen))
    t_cdf = stats.t.cdf(z_grid, t_df, loc=t_loc, scale=t_scale)
    t_at_lo = stats.t.cdf(z_lo, t_df, loc=t_loc, scale=t_scale)
    t_at_hi = stats.t.cdf(z_hi, t_df, loc=t_loc, scale=t_scale)

    lower = z_grid < z_lo
    if t_at_lo > 0:
        f_grid[lower] = f_lo * t_cdf[lower] / t_at_lo
    upper = z_grid > z_hi
    if t_at_hi < 1:
        f_grid[upper] = 1.0 - (1.0 - f_hi) * (1.0 - t_cdf[upper]) / (1.0 - t_at_hi)

    f_grid = np.clip(np.maximum.accumulate(f_grid), 0.0, 1.0)

    return LocationScaleCalibrator(
        a=float(a), b=float(b), c=c, d=d,
        scale_floor=scale_floor,
        z_grid=z_grid, z_cdf=f_grid,
        t_df=float(t_df), tail_lo=z_lo, tail_hi=z_hi,
        fit_meta={
            "n": int(n),
            "a_se": a_se,
            "b_se": b_se,
            "scale_mle_converged": bool(opt.success),
            "resid_sd": float(resid.std(ddof=2)),
            "z_median": float(np.median(z)),
            "t_loc": float(t_loc),
            "t_scale": float(t_scale),
        },
    )


def _bilinear(table: np.ndarray, xs: np.ndarray, ts: np.ndarray, x: float, t: float) -> float:
    """Bilinear lookup into a (len(ts), len(xs)) table, clamping outside the grid."""
    x = min(max(x, xs[0]), xs[-1])
    t = min(max(t, ts[0]), ts[-1])
    j = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    wx = (x - xs[j]) / (xs[j + 1] - xs[j])
    wt = (t - ts[i]) / (ts[i + 1] - ts[i])
    top = table[i, j] * (1 - wx) + table[i, j + 1] * wx
    bot = table[i + 1, j] * (1 - wx) + table[i + 1, j + 1] * wx
    return float(np.clip(top * (1 - wt) + bot * wt, 0.0, 1.0))


@dataclass
class KernelConditionalCalibrator:
    """
    Non-parametric estimate of P(Tm_true >= T | Tm_pred = x).

    Where the location-scale model assumes the conditional distribution is a single
    shape shifted and stretched along x, this estimates the conditional survival
    function directly by kernel-weighted averaging of the indicator 1[y >= T]. That
    makes no assumption about the shape, which matters when the predictor is close to
    bimodal -- a straight location line then splits the difference between the two modes
    and is badly wrong in between.

    Monotonicity holds in both directions by construction: in T exactly, because
    1[y >= T] is pointwise non-increasing in T and the kernel weights are non-negative;
    in x by a weighted pool-adjacent-violators projection, which preserves the ordering
    across T because the same weights are used at every threshold.

    Attributes:
        x_grid, t_grid: Prediction and threshold grids.
        p: Probability table, shape (len(t_grid), len(x_grid)).
        bandwidth: Gaussian kernel bandwidth in degrees C.
        x_data_range: Observed prediction range; outside it the table is extrapolation.
        fit_meta: Provenance and selection diagnostics.
    """

    x_grid: np.ndarray
    t_grid: np.ndarray
    p: np.ndarray
    bandwidth: float
    x_data_range: Tuple[float, float]
    fit_meta: dict = field(default_factory=dict)

    def local_prob(self, tm_pred: np.ndarray, threshold: float) -> np.ndarray:
        """P(Tm_true >= threshold | Tm_pred), by bilinear lookup."""
        arr = np.atleast_1d(np.asarray(tm_pred, dtype=float))
        return np.array([_bilinear(self.p, self.x_grid, self.t_grid, float(v), float(threshold))
                         for v in arr])

    def local_grid(self, pred_grid: np.ndarray, threshold_grid: np.ndarray) -> np.ndarray:
        """Probability table on an arbitrary grid, shape (len(threshold_grid), len(pred_grid))."""
        return np.vstack([self.local_prob(pred_grid, float(t)) for t in threshold_grid])

    def crossing_point(self, threshold: float) -> float:
        """Predicted Tm at which the probability crosses 0.5, or NaN if never."""
        probs = self.local_prob(self.x_grid, threshold)
        above = np.nonzero(probs >= 0.5)[0]
        if len(above) == 0 or above[0] == 0:
            return float("nan")
        i = above[0]
        p0, p1 = probs[i - 1], probs[i]
        if p1 == p0:
            return float(self.x_grid[i])
        return float(self.x_grid[i - 1] + (0.5 - p0) / (p1 - p0) * (self.x_grid[i] - self.x_grid[i - 1]))

    def to_dict(self) -> dict:
        return {
            "type": "kernel_conditional_survival",
            "bandwidth_c": self.bandwidth,
            "kernel": "gaussian",
            "monotone": {"in_threshold": "exact", "in_prediction": "weighted PAV projection"},
            "x_data_range": list(self.x_data_range),
            "extrapolation": "clamped to the observed prediction range",
            "fit_meta": self.fit_meta,
        }


def _kernel_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Kernel-weighted conditional survival table, then monotonised along x."""
    from sklearn.isotonic import IsotonicRegression

    w = np.exp(-0.5 * ((x_grid[:, None] - y_pred[None, :]) / bandwidth) ** 2)
    denom = w.sum(axis=1)
    denom[denom <= 0] = 1.0

    table = np.vstack([(w @ (y_true >= t).astype(float)) / denom for t in t_grid])

    # Weight each grid point by its local data mass so sparsely-supported regions do not
    # dominate the projection.
    mass = w.sum(axis=1)
    for i in range(table.shape[0]):
        row = table[i]
        if np.all(np.diff(row) >= -1e-12):
            continue
        table[i] = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
            x_grid, row, sample_weight=mass
        )
    return np.clip(table, 0.0, 1.0)


def select_bandwidth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_grid: np.ndarray,
    thresholds: Sequence[float],
    candidates: Sequence[float] = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    n_folds: int = 5,
    seed: int = 0,
) -> Tuple[float, pd.DataFrame]:
    """
    Choose the kernel bandwidth by k-fold cross-validated log loss.

    Selection uses only the fitting set, so the held-out evaluation set stays clean.

    Returns:
        Tuple[float, pd.DataFrame]: Best bandwidth and the full CV table.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    t_grid = np.asarray(thresholds, dtype=float)

    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(y_true)) % n_folds

    rows = []
    for h in candidates:
        losses = []
        for k in range(n_folds):
            tr, te = folds != k, folds == k
            table = _kernel_table(y_true[tr], y_pred[tr], x_grid, t_grid, h)
            for i, t in enumerate(t_grid):
                labels = (y_true[te] >= t).astype(int)
                if labels.sum() in (0, len(labels)):
                    continue
                p = np.clip(np.interp(y_pred[te], x_grid, table[i]), 1e-6, 1 - 1e-6)
                losses.append(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
        rows.append((h, float(np.mean(losses))))

    cv = pd.DataFrame(rows, columns=["bandwidth", "cv_log_loss"])
    return float(cv.loc[cv["cv_log_loss"].idxmin(), "bandwidth"]), cv


def fit_kernel_conditional(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    bandwidth: Optional[float] = None,
    cv_thresholds: Sequence[float] = (50, 55, 60, 65, 70),
) -> KernelConditionalCalibrator:
    """
    Fit the non-parametric conditional-survival calibrator.

    Args:
        y_true (np.ndarray): Actual Tm.
        y_pred (np.ndarray): Predicted Tm.
        x_grid (np.ndarray): Prediction grid for the stored table.
        t_grid (np.ndarray): Threshold grid for the stored table.
        bandwidth (float, optional): Gaussian bandwidth in degrees C. Cross-validated
            on the fitting set when omitted.
        cv_thresholds (Sequence[float]): Thresholds used for bandwidth selection.

    Returns:
        KernelConditionalCalibrator: The fitted calibrator.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    cv = None
    if bandwidth is None:
        bandwidth, cv = select_bandwidth(y_true, y_pred, x_grid, cv_thresholds)

    table = _kernel_table(y_true, y_pred, x_grid, t_grid, bandwidth)
    return KernelConditionalCalibrator(
        x_grid=np.asarray(x_grid, dtype=float),
        t_grid=np.asarray(t_grid, dtype=float),
        p=table,
        bandwidth=float(bandwidth),
        x_data_range=(float(y_pred.min()), float(y_pred.max())),
        fit_meta={
            "n": int(len(y_true)),
            "bandwidth_selected_by": "5-fold CV log loss" if cv is not None else "caller",
            "cv_table": cv.to_dict(orient="records") if cv is not None else None,
        },
    )


def compare_calibrators(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    thresholds: Sequence[float] = (50, 55, 60, 65, 70),
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Cross-validate the candidate calibrator families on the fitting set.

    Selection happens here, on the fitting set only, so the held-out evaluation set is
    never used to choose the model and its reported calibration error stays honest.

    Args:
        y_true (np.ndarray): Actual Tm.
        y_pred (np.ndarray): Predicted Tm.
        x_grid (np.ndarray): Prediction grid for the kernel calibrator.
        t_grid (np.ndarray): Threshold grid for the kernel calibrator.
        thresholds (Sequence[float]): Thresholds scored.
        n_folds (int): Cross-validation folds.
        seed (int): Fold assignment seed.

    Returns:
        pd.DataFrame: method, cv_log_loss, cv_brier, cv_ece.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(y_true)) % n_folds

    scores: Dict[str, Dict[str, List[float]]] = {
        m: {"log_loss": [], "brier": [], "ece": []}
        for m in ("kernel_conditional", "location_scale")
    }

    for k in range(n_folds):
        tr, te = folds != k, folds == k
        fitted = {
            "kernel_conditional": fit_kernel_conditional(
                y_true[tr], y_pred[tr], x_grid, t_grid, bandwidth=None, cv_thresholds=thresholds
            ),
            "location_scale": fit_location_scale(y_true[tr], y_pred[tr]),
        }
        for name, calib in fitted.items():
            for t in thresholds:
                labels = (y_true[te] >= t).astype(int)
                if labels.sum() in (0, len(labels)):
                    continue
                p = calib.local_prob(y_pred[te], float(t))
                clipped = np.clip(p, 1e-6, 1 - 1e-6)
                scores[name]["log_loss"].append(
                    float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
                )
                scores[name]["brier"].append(float(np.mean((p - labels) ** 2)))
                scores[name]["ece"].append(calibration_report(p, labels)["ece"])

    return pd.DataFrame([
        {
            "method": name,
            "cv_log_loss": float(np.mean(s["log_loss"])),
            "cv_brier": float(np.mean(s["brier"])),
            "cv_ece": float(np.nanmean(s["ece"])),
        }
        for name, s in scores.items()
    ]).sort_values("cv_log_loss").reset_index(drop=True)


def fit_per_cutoff_logistic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Sequence[float],
    min_positives: int = 30,
) -> pd.DataFrame:
    """
    Per-threshold logistic (Platt) fit of P(Tm_true >= T | Tm_pred), as a cross-check.

    Not shipped: the intercept absorbs the fitting set's base rate, so these curves do
    not transport to datasets with a different fraction of thermostable proteins. Used
    only to confirm the location-scale model agrees where data are plentiful.

    Args:
        y_true (np.ndarray): Actual Tm.
        y_pred (np.ndarray): Predicted Tm.
        thresholds (Sequence[float]): Thresholds to fit.
        min_positives (int): Skip thresholds with fewer positives (or negatives).

    Returns:
        pd.DataFrame: threshold, n_pos, intercept, slope. NaN where skipped.
    """
    x = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=float)

    rows = []
    for t in thresholds:
        labels = (y >= t).astype(int)
        n_pos = int(labels.sum())
        if n_pos < min_positives or (len(labels) - n_pos) < min_positives:
            rows.append((t, n_pos, np.nan, np.nan))
            continue
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(x, labels)
        rows.append((t, n_pos, float(lr.intercept_[0]), float(lr.coef_[0][0])))
    return pd.DataFrame(rows, columns=["threshold", "n_pos", "intercept", "slope"])


def logistic_prob(row: pd.Series, tm_pred: np.ndarray) -> np.ndarray:
    """Evaluate a `fit_per_cutoff_logistic` row at the given predicted Tm values."""
    if not np.isfinite(row["slope"]):
        return np.full(np.shape(tm_pred), np.nan)
    return 1.0 / (1.0 + np.exp(-(row["intercept"] + row["slope"] * np.asarray(tm_pred, float))))


# ---------------------------------------------------------------------------
# Calibration quality
# ---------------------------------------------------------------------------

def reliability_bins(
    p_pred: np.ndarray,
    y_bin: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """
    Bin predicted probabilities and measure the observed frequency in each bin.

    Args:
        p_pred (np.ndarray): Predicted probabilities.
        y_bin (np.ndarray): Binary outcomes.
        n_bins (int): Number of bins.
        strategy (str): 'quantile' for equal-count bins (default -- probabilities pile
            up at the low end, so equal-width bins would leave the top bins nearly
            empty) or 'uniform' for equal-width.

    Returns:
        pd.DataFrame: bin, p_lo, p_hi, n, mean_pred, frac_pos, ci_low, ci_high.
    """
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(y_bin, dtype=int)

    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    if len(edges) < 2:
        edges = np.array([p.min(), p.min() + 1e-9])
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    idx = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for i in range(len(edges) - 1):
        mask = idx == i
        n = int(mask.sum())
        if n == 0:
            rows.append((i, edges[i], edges[i + 1], 0, np.nan, np.nan, np.nan, np.nan))
            continue
        k = int(y[mask].sum())
        low, high = wilson_interval(k, n)
        rows.append((i, edges[i], edges[i + 1], n, float(p[mask].mean()), k / n, low, high))

    return pd.DataFrame(
        rows, columns=["bin", "p_lo", "p_hi", "n", "mean_pred", "frac_pos", "ci_low", "ci_high"]
    )


def calibration_report(p_pred: np.ndarray, y_bin: np.ndarray, n_bins: int = 10) -> dict:
    """
    Summarize how well predicted probabilities match observed frequencies.

    `brier_baserate` is reported alongside `brier` so skill is visible rather than just
    magnitude: at a 12% base rate a constant predictor already scores about 0.104.
    `cal_in_large` (mean predicted minus base rate) is the single number that exposes a
    base-rate shift between the fitting set and the evaluation set.

    Args:
        p_pred (np.ndarray): Predicted probabilities.
        y_bin (np.ndarray): Binary outcomes.
        n_bins (int): Reliability bins.

    Returns:
        dict: ece, mce, brier, brier_baserate, log_loss, n, n_pos, base_rate,
            mean_pred, cal_in_large, frac_dropped, bins.
    """
    p = np.asarray(p_pred, dtype=float)
    y = np.asarray(y_bin, dtype=int)
    n = len(y)
    base_rate = float(y.mean()) if n else float("nan")

    bins = reliability_bins(p, y, n_bins=n_bins)
    usable = bins[bins["n"] >= MIN_BIN_FOR_ECE]
    total = int(usable["n"].sum())
    if total > 0:
        gaps = (usable["frac_pos"] - usable["mean_pred"]).abs()
        ece = float((usable["n"] / total * gaps).sum())
    else:
        ece = float("nan")

    strong = bins[bins["n"] >= MIN_BIN_FOR_MCE]
    mce = float((strong["frac_pos"] - strong["mean_pred"]).abs().max()) if len(strong) else float("nan")

    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": n,
        "n_pos": int(y.sum()),
        "base_rate": base_rate,
        "mean_pred": float(p.mean()) if n else float("nan"),
        "cal_in_large": float(p.mean() - base_rate) if n else float("nan"),
        "ece": ece,
        "mce": mce,
        "brier": float(np.mean((p - y) ** 2)) if n else float("nan"),
        "brier_baserate": float(base_rate * (1 - base_rate)) if n else float("nan"),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))) if n else float("nan"),
        "frac_dropped": float(1 - total / n) if n else float("nan"),
        "bins": bins.to_dict(orient="records"),
    }


def residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calib: LocationScaleCalibrator,
    n_bins: int = 20,
) -> dict:
    """
    Check the location-scale model's assumptions on a given set.

    Every assumption the model makes is tested here and the result stored in the
    artifact: linearity of the location, whether the scale really varies with the
    prediction, and whether the standardized residual is as heavy-tailed as assumed.
    Running this on a set other than the fitting set tests whether the residual
    distribution transports -- the property the whole approach relies on.

    Returns:
        dict: location / scale / shape sub-dicts.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    resid = y - calib.location(x)
    z = resid / calib.scale(x)

    # Location linearity: compare a binned conditional mean against the fitted line.
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    bin_rows, deviations = [], []
    for i in range(len(edges) - 1):
        mask = idx == i
        if mask.sum() < 5:
            continue
        xm, ym = float(x[mask].mean()), float(y[mask].mean())
        dev = ym - float(calib.location(np.array([xm]))[0])
        deviations.append(abs(dev))
        bin_rows.append({
            "x_mean": xm, "n": int(mask.sum()), "y_mean": ym,
            "deviation_c": dev, "resid_sd": float(resid[mask].std(ddof=1)),
        })

    # Heteroscedasticity: Breusch-Pagan LM statistic plus a rank check on |r|.
    r2 = resid**2
    slope, intercept = np.polyfit(x, r2, 1)
    fitted = intercept + slope * x
    ss_tot = float(((r2 - r2.mean()) ** 2).sum())
    r_squared = float(((fitted - r2.mean()) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    bp_stat = len(x) * r_squared
    rho, rho_p = stats.spearmanr(np.abs(resid), x)

    # Shape: Shapiro-Wilk is unreliable above ~5000 points, so subsample for it.
    rng = np.random.default_rng(0)
    sub = z if len(z) <= 2000 else rng.choice(z, 2000, replace=False)
    return {
        "location": {
            "a": calib.a, "b": calib.b,
            "a_se": calib.fit_meta.get("a_se"), "b_se": calib.fit_meta.get("b_se"),
            "max_bin_deviation_c": float(max(deviations)) if deviations else float("nan"),
            "bins": bin_rows,
        },
        "scale": {
            "c": calib.c, "d": calib.d,
            "bp_stat": float(bp_stat),
            "bp_pvalue": float(stats.chi2.sf(bp_stat, 1)),
            "spearman_abs_resid": float(rho),
            "spearman_pvalue": float(rho_p),
            "resid_sd_overall": float(resid.std(ddof=1)),
        },
        "shape": {
            "z_mean": float(z.mean()),
            "z_sd": float(z.std(ddof=1)),
            "skew": float(stats.skew(z)),
            "excess_kurtosis": float(stats.kurtosis(z)),
            "shapiro_p": float(stats.shapiro(sub).pvalue),
            "anderson_stat": float(stats.anderson(z, dist="norm").statistic),
            "t_df": calib.t_df,
        },
    }


# ---------------------------------------------------------------------------
# Plots (styling matched to src/eval/ranking.py)
# ---------------------------------------------------------------------------

def _finish(fig, path_save: Optional[str], footnotes: Optional[List[str]] = None) -> None:
    if footnotes:
        fig.text(0.01, 0.01, "\n".join(footnotes), fontsize=9, color="gray", va="bottom")
        fig.tight_layout(rect=(0, 0.04 + 0.018 * len(footnotes), 1, 1))
    else:
        fig.tight_layout()
    if path_save:
        fig.savefig(path_save, dpi=200)
        print(f"[calibration] wrote {path_save}")
    plt.close(fig)


def plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calib: LocationScaleCalibrator,
    path_save: str,
    footnotes: Optional[List[str]] = None,
) -> None:
    """Residual vs predicted Tm with the fitted scale band, plus a QQ plot of z."""
    x = np.asarray(y_pred, dtype=float)
    resid = np.asarray(y_true, dtype=float) - calib.location(x)
    z = resid / calib.scale(x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    ax1.scatter(x, resid, s=6, alpha=0.15, color=COLORS[0], edgecolors="none")
    grid = np.linspace(x.min(), x.max(), 200)
    band = calib.scale(grid)
    ax1.plot(grid, band, color=COLORS[1], linewidth=2.5, label="Fitted $\\pm$s(x)")
    ax1.plot(grid, -band, color=COLORS[1], linewidth=2.5)
    ax1.axhline(0, color="black", linewidth=1)

    edges = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    centres, sds = [], []
    for i in range(len(edges) - 1):
        mask = idx == i
        if mask.sum() >= 5:
            centres.append(float(x[mask].mean()))
            sds.append(float(resid[mask].std(ddof=1)))
    ax1.plot(centres, sds, marker="o", linestyle="--", color=COLORS[2], linewidth=2, label="Binned SD")
    ax1.plot(centres, [-s for s in sds], marker="o", linestyle="--", color=COLORS[2], linewidth=2)

    ax1.set_xlabel("Predicted T$_m$ [°C]", fontsize=LABEL_SIZE)
    ax1.set_ylabel("Residual (actual $-$ fitted) [°C]", fontsize=LABEL_SIZE)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=LEGEND_SIZE)

    stats.probplot(z, dist="norm", plot=ax2)
    ax2.get_lines()[0].set(marker="o", markersize=3, alpha=0.4, color=COLORS[0])
    ax2.get_lines()[1].set(color=COLORS[1], linewidth=2.5)
    ax2.set_title("")
    ax2.set_xlabel("Normal theoretical quantiles", fontsize=LABEL_SIZE)
    ax2.set_ylabel("Standardized residual z", fontsize=LABEL_SIZE)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.grid(True, linestyle="--", alpha=0.7)

    _finish(fig, path_save, footnotes)


def plot_reliability(
    reports: Dict[str, dict],
    path_save: str,
    threshold: float,
    footnotes: Optional[List[str]] = None,
) -> None:
    """Reliability diagrams with Wilson bars, one panel per evaluation set."""
    names = list(reports.keys())
    fig, axes = plt.subplots(1, max(len(names), 2), figsize=FIGSIZE, squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, names):
        rep = reports[name]
        bins = pd.DataFrame(rep["bins"])
        bins = bins[bins["n"] > 0]
        ax.plot([0, 1], [0, 1], linestyle="--", color="c", linewidth=2, label="Perfect calibration")
        if len(bins):
            yerr = np.vstack([
                (bins["frac_pos"] - bins["ci_low"]).to_numpy(),
                (bins["ci_high"] - bins["frac_pos"]).to_numpy(),
            ])
            ax.errorbar(bins["mean_pred"], bins["frac_pos"], yerr=yerr, marker="o",
                        color=COLORS[0], linewidth=2.5, capsize=3, label="Observed")
            for _, r in bins.iterrows():
                ax.annotate(f"n={int(r['n'])}", (r["mean_pred"], r["frac_pos"]),
                            textcoords="offset points", xytext=(0, 8), fontsize=8, color="gray")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"Predicted P(T$_m$ $\\geq$ {threshold:g}°C)", fontsize=LABEL_SIZE)
        ax.set_ylabel("Observed frequency", fontsize=LABEL_SIZE)
        ax.set_title(
            f"{name}  (n={rep['n']}, base rate {rep['base_rate']:.3f}, ECE {rep['ece']:.3f})",
            fontsize=LABEL_SIZE - 3,
        )
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=LEGEND_SIZE, loc="upper left")

    for ax in axes[len(names):]:
        ax.axis("off")
    _finish(fig, path_save, footnotes)


def plot_local_curves(
    calib: LocationScaleCalibrator,
    thresholds: Sequence[float],
    pred_grid: np.ndarray,
    empirical: Optional[pd.DataFrame] = None,
    focus_threshold: float = 60.0,
    path_save: Optional[str] = None,
    footnotes: Optional[List[str]] = None,
    support: Optional[np.ndarray] = None,
    min_support: int = 30,
) -> None:
    """
    Local probability curves, and the focus threshold overlaid with binned empirical
    frequencies. The right panel is the definitive visual check: model curve should
    thread the Wilson bars of the well-populated bins.

    Args:
        empirical: Output of `local_empirical_bins` for `focus_threshold`.
        support: Per-grid-point fit counts from `local_support`. Where these fall
            below `min_support` the curve is drawn faded over a shaded band, so a
            probability resting on a handful of proteins cannot be read off as
            confidently as one resting on hundreds.
        min_support: Count below which a location is treated as unsupported.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    spans = ([] if support is None
             else low_support_spans(pred_grid, support, min_support))
    solid = None if support is None else np.asarray(support, dtype=float) >= min_support
    for lo, hi in spans:
        ax1.axvspan(lo, hi, color=LOW_SUPPORT_FILL, zorder=0)
        ax2.axvspan(lo, hi, color=LOW_SUPPORT_FILL, zorder=0)

    palette = CVD_COLORS if support is not None else COLORS
    for i, t in enumerate(thresholds):
        colour = palette[i % len(palette)]
        p = calib.local_prob(pred_grid, float(t))
        if solid is None:
            ax1.plot(pred_grid, p, color=colour, linewidth=2.5, label=f"T = {t:g}°C")
        else:
            # Faded full curve underneath, opaque only where the data supports it.
            ax1.plot(pred_grid, p, color=colour, linewidth=2.5, alpha=0.25)
            ax1.plot(pred_grid, np.where(solid, p, np.nan), color=colour,
                     linewidth=2.5, label=f"T = {t:g}°C")
        cross = calib.crossing_point(float(t))
        if np.isfinite(cross):
            supported = (solid is None
                         or bool(solid[int(np.argmin(np.abs(np.asarray(pred_grid) - cross)))]))
            ax1.plot([cross], [0.5], marker="o" if supported else "x", color=colour,
                     markersize=7 if supported else 9,
                     markeredgewidth=2 if not supported else None)
    ax1.axhline(0.5, color="c", linestyle="--", linewidth=2, label="P = 0.5")
    ax1.set_xlabel("Predicted T$_m$ [°C]", fontsize=LABEL_SIZE)
    ax1.set_ylabel("P(actual T$_m$ $\\geq$ T | predicted)", fontsize=LABEL_SIZE)
    ax1.set_ylim(0, 1)
    ax1.tick_params(labelsize=TICK_SIZE)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=LEGEND_SIZE, loc="upper left")

    p_focus = calib.local_prob(pred_grid, focus_threshold)
    if solid is None:
        ax2.plot(pred_grid, p_focus, color=palette[0], linewidth=2.5, label="Calibrated model")
    else:
        ax2.plot(pred_grid, p_focus, color=palette[0], linewidth=2.5, alpha=0.25)
        ax2.plot(pred_grid, np.where(solid, p_focus, np.nan), color=palette[0],
                 linewidth=2.5, label=f"Calibrated model (n $\\geq$ {min_support})")
    if empirical is not None and len(empirical):
        e = empirical[empirical["n"] > 0]
        yerr = np.vstack([
            (e["frac_pos"] - e["ci_low"]).to_numpy(),
            (e["ci_high"] - e["frac_pos"]).to_numpy(),
        ])
        ax2.errorbar(e["x_mean"], e["frac_pos"], yerr=yerr, marker="o", linestyle="none",
                     color=palette[1], capsize=3, linewidth=2, label="Observed (Wilson 95%)")
    ax2.axhline(0.5, color="c", linestyle="--", linewidth=2)
    ax2.axvline(focus_threshold, color="gray", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Predicted T$_m$ [°C]", fontsize=LABEL_SIZE)
    ax2.set_ylabel(f"P(actual T$_m$ $\\geq$ {focus_threshold:g}°C)", fontsize=LABEL_SIZE)
    ax2.set_ylim(0, 1)
    ax2.tick_params(labelsize=TICK_SIZE)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=LEGEND_SIZE, loc="upper left")

    if spans:
        covered = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in spans)
        footnotes = (footnotes or []) + [
            f"Shaded / faded: fewer than {min_support} fit proteins within one bandwidth "
            f"({covered}°C). Crossings there are marked x, not o, and are not evidence."]
    _finish(fig, path_save, footnotes)


def plot_prediction_density(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_grid: np.ndarray,
    bandwidth: float,
    thresholds: Sequence[float],
    path_save: Optional[str] = None,
    min_support: int = 30,
    footnotes: Optional[List[str]] = None,
) -> None:
    """
    Where the model actually puts its predictions, and how much evidence backs each
    threshold.

    The left panel contrasts the predicted and actual Tm distributions: the model
    sorts proteins into a mesophile and a thermophile mode and rarely predicts
    anything between, a corridor the actual distribution fills perfectly well. The
    right panel counts fit-set proteins within one bandwidth of each prediction
    value, which is the evidence any local probability at that value rests on.
    Thresholds falling in the thin region cannot be calibrated from this fit set.

    Args:
        y_true: Actual Tm of the fit set.
        y_pred: Predicted Tm of the fit set.
        pred_grid: Grid the calibrator is evaluated on.
        bandwidth: Kernel bandwidth in degrees C.
        thresholds: Thresholds being calibrated, drawn as vertical rules.
        min_support: Count below which a location is treated as unsupported.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    support = local_support(x, pred_grid, bandwidth)
    spans = low_support_spans(pred_grid, support, min_support)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True)
    bins = np.arange(min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 2, 1.0)

    for lo, hi in spans:
        ax1.axvspan(lo, hi, color=LOW_SUPPORT_FILL, zorder=0)
        ax2.axvspan(lo, hi, color=LOW_SUPPORT_FILL, zorder=0)

    ax1.hist(x, bins=bins, color=CVD_COLORS[0], alpha=0.75, label="Predicted T$_m$")
    ax1.hist(y, bins=bins, histtype="step", linewidth=2.5, color=CVD_COLORS[1],
             label="Actual T$_m$")
    ax1.set_ylabel(f"Proteins per 1°C bin (n = {len(x)})", fontsize=LABEL_SIZE)
    ax1.legend(fontsize=LEGEND_SIZE, loc="upper right")

    ax2.plot(pred_grid, np.maximum(support, 0.1), color=CVD_COLORS[3], linewidth=2.5)
    ax2.axhline(min_support, color=CVD_COLORS[2], linestyle="--", linewidth=2,
                label=f"n = {min_support}")
    ax2.set_yscale("log")
    ax2.set_ylabel(f"Proteins within $\\pm${bandwidth:g}°C of x", fontsize=LABEL_SIZE)
    ax2.legend(fontsize=LEGEND_SIZE, loc="upper right")

    for ax in (ax1, ax2):
        for t in thresholds:
            ax.axvline(float(t), color="black", linestyle=":", linewidth=1.5)
            ax.annotate(f"{float(t):g}", xy=(float(t), 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=LEGEND_SIZE, color="black")
        ax.set_xlabel("T$_m$ [°C]", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--", alpha=0.7)

    if spans:
        covered = [f"{lo:g}-{hi:g}" for lo, hi in spans]
        note = (f"Shaded: fewer than {min_support} proteins within one bandwidth "
                f"({', '.join(covered)}°C).")
        footnotes = (footnotes or []) + [note]
    _finish(fig, path_save, footnotes)


def plot_calibration_table(
    table: np.ndarray,
    pred_grid: np.ndarray,
    thr_grid: np.ndarray,
    support: Optional[np.ndarray] = None,
    thresholds: Optional[Sequence[float]] = None,
    min_support: int = 30,
    path_save: Optional[str] = None,
    footnotes: Optional[List[str]] = None,
) -> None:
    """
    The whole artifact in one image: P(Tm_true >= T | Tm_pred = x) over both grids.

    Every other plot shows a slice of this table -- one threshold, one cutoff. Here all
    of it is visible at once, which is the only view in which the unsupported corridor
    reads as what it is: a vertical band the calibrator fills by carrying values sideways
    rather than by observing anything. The right panel is the evidence behind the left.

    Args:
        table: Local probability table, shape (len(thr_grid), len(pred_grid)).
        support: Per-prediction-value fit counts from `local_support`.
        thresholds: Reported thresholds, drawn as horizontal rules.
        min_support: Count below which a column counts as unsupported.
    """
    P = np.asarray(table, dtype=float)
    x, t = np.asarray(pred_grid, float), np.asarray(thr_grid, float)
    extent = (x[0], x[-1], t[-1], t[0])

    ncol = 1 if support is None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(FIGSIZE[0] if ncol == 2 else FIGSIZE[0] / 2,
                                               FIGSIZE[1]))
    axes = np.atleast_1d(axes)
    ax = axes[0]

    im = ax.imshow(P, aspect="auto", extent=extent, origin="upper",
                   cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("P(actual T$_m$ $\\geq$ T | predicted)", fontsize=LABEL_SIZE)
    cb.ax.tick_params(labelsize=TICK_SIZE)

    spans = [] if support is None else low_support_spans(x, support, min_support)
    for lo, hi in spans:
        # Hatch rather than fill: the probabilities underneath must stay legible,
        # the hatching says only that nothing measured them.
        ax.add_patch(plt.Rectangle((lo, t[0]), hi - lo, t[-1] - t[0], fill=False,
                                   hatch="///", edgecolor="white", linewidth=0, alpha=0.55))
    for thr in (thresholds or []):
        ax.axhline(float(thr), color="white", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Predicted T$_m$ [°C]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Threshold T [°C]", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    if support is not None:
        ax2 = axes[1]
        s = np.maximum(np.asarray(support, float), 0.1)
        ax2.fill_between(x, 0.1, s, color=CVD_COLORS[0], alpha=0.35)
        ax2.plot(x, s, color=CVD_COLORS[0], linewidth=2.5)
        ax2.axhline(min_support, color=CVD_COLORS[1], linestyle="--", linewidth=2,
                    label=f"n = {min_support}")
        for lo, hi in spans:
            ax2.axvspan(lo, hi, color=LOW_SUPPORT_FILL, zorder=0)
        ax2.set_yscale("log")
        ax2.set_xlim(x[0], x[-1])
        ax2.set_xlabel("Predicted T$_m$ [°C]", fontsize=LABEL_SIZE)
        ax2.set_ylabel("Fit proteins backing each column", fontsize=LABEL_SIZE)
        ax2.tick_params(labelsize=TICK_SIZE)
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(fontsize=LEGEND_SIZE, loc="upper right")

    if spans:
        covered = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in spans)
        footnotes = (footnotes or []) + [
            f"Hatched: fewer than {min_support} fit proteins back the column "
            f"({covered}°C). Values there are carried sideways, not measured."]
    _finish(fig, path_save, footnotes)


def per_set_summary(
    per_set: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reports: Dict[str, Dict[float, dict]],
    threshold: float,
) -> pd.DataFrame:
    """
    One row per evaluation set: does it rank, and are its probabilities honest?

    Splits the two questions deliberately. A set can score a fine ECE purely by having
    no positives to get wrong -- ERED-WT is calibrated at 0.037 while ranking at
    PCC -0.015 -- so the verdict column names that case rather than leaving a reader to
    infer it from a base rate of zero.

    Args:
        per_set: name -> (actual Tm, predicted Tm).
        reports: name -> output of the per-set calibration evaluation.
        threshold: Threshold the calibration columns are quoted at.
    """
    rows = []
    for name, (y, x) in per_set.items():
        rep = reports.get(name, {}).get(threshold, {})
        pcc = float(stats.pearsonr(x, y)[0]) if len(y) > 2 and y.std() > 0 else float("nan")
        scc = float(stats.spearmanr(x, y)[0]) if len(y) > 2 and y.std() > 0 else float("nan")
        ss_res = float(((y - x) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        brier, brier_base = rep.get("brier", float("nan")), rep.get("brier_baserate", float("nan"))

        # NaN is a distinct answer from a good score here: calibration_report returns
        # ece=NaN when no reliability bin is populated enough to measure, which must
        # never fall through to a "calibrated" verdict.
        ece = rep.get("ece", float("nan"))
        if not np.isfinite(pcc) or abs(pcc) < 0.2:
            verdict = "no rank signal"
        elif rep.get("undefined") or not np.isfinite(brier):
            verdict = "ranks; calibration undefined (no positives)"
        elif brier >= brier_base:
            verdict = "ranks; probabilities no better than base rate"
        elif not np.isfinite(ece):
            verdict = "ranks; too few per bin to assess calibration"
        elif ece > 0.1:
            verdict = "ranks; poorly calibrated"
        else:
            verdict = "ranks and calibrated"

        rows.append({
            "set": name,
            "n": int(len(y)),
            "tm_max": float(y.max()),
            "base_rate": rep.get("base_rate", float((y >= threshold).mean())),
            "PCC": pcc,
            "SCC": scc,
            "R2": r2,
            "RMSE": float(np.sqrt(((y - x) ** 2).mean())),
            "ECE": rep.get("ece", float("nan")),
            "Brier": brier,
            "Brier_baserate": brier_base,
            "verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def write_summary_table(df: pd.DataFrame, path_csv: str, path_md: Optional[str] = None) -> None:
    """
    Write the per-set summary as CSV, and optionally as markdown for pasting.

    The markdown is formatted here rather than via `DataFrame.to_markdown`, which needs
    `tabulate` -- an extra dependency for one table, in a project whose runtime is
    deliberately kept to torch/transformers/peft.
    """
    df.to_csv(path_csv, index=False)
    print(f"[calibration] wrote {path_csv}")
    if not path_md:
        return

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)

    cols = list(df.columns)
    rows = [[cell(v) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    line = lambda vals: "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"

    with open(path_md, "w") as f:
        f.write(line(cols) + "\n")
        f.write("|" + "|".join("-" * (w + 2) for w in widths) + "|\n")
        for r in rows:
            f.write(line(r) + "\n")
    print(f"[calibration] wrote {path_md}")


def plot_calibration_definition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    example_pred: float,
    threshold: float,
    path_save: Optional[str] = None,
    window: float = 0.75,
    footnotes: Optional[List[str]] = None,
) -> None:
    """
    What a calibrated probability *is*, unpacked for one cell of the table.

    Takes every fit protein the model predicted near `example_pred`, shows the spread
    of what they actually turned out to be, and shades the part above `threshold`.
    That shaded fraction is the calibrated probability -- no formula needed. This is
    the methods figure: it explains the whole artifact in one panel.

    Args:
        example_pred: The predicted Tm to unpack, degrees C.
        threshold: The threshold whose probability is being explained.
        window: Half-width of the prediction window, degrees C.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    sel = y[(x >= example_pred - window) & (x <= example_pred + window)]
    if not len(sel):
        raise ValueError(f"no fit proteins within {window}C of {example_pred}C")

    k = int((sel >= threshold).sum())
    frac = k / len(sel)
    lo, hi = wilson_interval(k, len(sel))

    fig, ax = plt.subplots(figsize=(FIGSIZE[0] / 2, FIGSIZE[1]))
    bins = np.arange(np.floor(sel.min()) - 1, np.ceil(sel.max()) + 2, 2.0)
    counts, edges, patches = ax.hist(sel, bins=bins, color=CVD_COLORS[0], edgecolor="white")
    for patch, left in zip(patches, edges[:-1]):
        if left >= threshold - 1e-9:
            patch.set_facecolor(CVD_COLORS[1])

    ax.axvline(example_pred, color="black", linewidth=2.5,
               label=f"Model predicted {example_pred:g}°C")
    ax.axvline(threshold, color=CVD_COLORS[2], linestyle="--", linewidth=2.5,
               label=f"Threshold {threshold:g}°C")
    # Anchored in axes coordinates: a data-space offset from the threshold runs off the
    # right edge whenever the threshold sits near the top of the observed range.
    ax.text(0.97, 0.62, f"{k} of {len(sel)} above {threshold:g}°C\n"
                        f"= {frac:.3f}\n(95% CI {lo:.3f}–{hi:.3f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=LABEL_SIZE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdbdbd"))
    ax.set_xlabel("Actual T$_m$ [°C]", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"Proteins predicted {example_pred:g}$\\pm${window:g}°C", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")

    note = (f"All {len(sel)} fit proteins predicted at {example_pred:g}$\\pm${window:g}°C,\n"
            f"by what they actually turned out to be. The shaded fraction\n"
            f"above the threshold is the calibrated probability.")
    _finish(fig, path_save, (footnotes or []) + [note])


def plot_prediction_interval_fan(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path_save: Optional[str] = None,
    n_bins: int = 14,
    min_bin: int = 20,
    footnotes: Optional[List[str]] = None,
) -> None:
    """
    How wrong the model typically is, as a function of what it predicted.

    Bins by predicted Tm and draws the 50% and 90% intervals of the actual Tm in each
    bin against the identity line. Answers "if it says 45, what is it really?" without
    reference to any calibrator -- this is measured, not modelled.

    A readability-first alternative to `plot_residual_diagnostics`; showing both is
    redundant, since they carry the same information.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

    c, med, q25, q75, q05, q95 = [], [], [], [], [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < min_bin:
            continue
        c.append(float(x[m].mean()))
        med.append(float(np.median(y[m])))
        for store, q in ((q25, 25), (q75, 75), (q05, 5), (q95, 95)):
            store.append(float(np.percentile(y[m], q)))

    fig, ax = plt.subplots(figsize=(FIGSIZE[0] / 2, FIGSIZE[1]))
    lims = [min(x.min(), y.min()) - 2, max(x.max(), y.max()) + 2]
    ax.plot(lims, lims, color="gray", linestyle="--", linewidth=2, label="Perfect prediction")
    ax.fill_between(c, q05, q95, color=CVD_COLORS[0], alpha=0.20, label="90% of proteins")
    ax.fill_between(c, q25, q75, color=CVD_COLORS[0], alpha=0.45, label="50% of proteins")
    ax.plot(c, med, color=CVD_COLORS[1], linewidth=2.5, marker="o", markersize=6,
            label="Median actual T$_m$")

    ax.set_xlabel("Predicted T$_m$ [°C]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Actual T$_m$ [°C]", fontsize=LABEL_SIZE)
    ax.set_xlim(lims)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left")
    _finish(fig, path_save, footnotes)


def plot_icon_array(
    cases: Sequence[Tuple[float, float, float, int]],
    threshold: float,
    path_save: Optional[str] = None,
    min_support: int = 30,
    footnotes: Optional[List[str]] = None,
) -> None:
    """
    "Of 100 proteins predicted at x, this many really were above T."

    A product asset rather than an analysis output -- intended for the web server or
    CLI docs, where the audience has no reason to read a reliability diagram. Cases
    whose support is below `min_support` are drawn hollow and labelled as unknown,
    so an unsupported cell cannot be mistaken for a confident one.

    Args:
        cases: (predicted_tm, probability, _unused, support) per panel.
        threshold: The threshold all panels are about.
    """
    fig, axes = plt.subplots(1, len(cases), figsize=(FIGSIZE[0], FIGSIZE[1] * 0.9))
    axes = np.atleast_1d(axes)
    for ax, (pred, prob, _, support) in zip(axes, cases):
        supported = support >= min_support
        filled = int(round(prob * 100))
        for i in range(100):
            row, col = divmod(i, 10)
            on = i < filled
            ax.scatter(col, -row, s=170,
                       facecolors=(CVD_COLORS[0] if on else "white") if supported
                       else ("#bdbdbd" if on else "white"),
                       edgecolors="#9e9e9e" if not supported else CVD_COLORS[0],
                       linewidths=1.2, zorder=2)
        headline = (f"{filled} in 100" if supported else "not enough data")
        ax.set_title(f"Predicted {pred:g}°C\n{headline}", fontsize=LABEL_SIZE)
        ax.set_xlim(-1, 10)
        ax.set_ylim(-10, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        if not supported:
            ax.text(4.5, -4.5, f"only {support}\nsimilar proteins", ha="center", va="center",
                    fontsize=LABEL_SIZE, color="#616161", zorder=3)

    fig.suptitle(f"Chance the protein is really above {threshold:g}°C", fontsize=LABEL_SIZE + 2)
    _finish(fig, path_save, footnotes)


def local_support(y_pred: np.ndarray, pred_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Fit-set points within one bandwidth of each grid location.

    This is the evidence behind each point of a local probability curve. The kernel
    calibrator will happily return a probability wherever the grid reaches, including
    regions holding a handful of proteins, so the curves are only interpretable
    alongside this count.

    Args:
        y_pred: Predicted Tm of the fit set.
        pred_grid: Grid the curves are evaluated on.
        bandwidth: Kernel bandwidth in degrees C.

    Returns:
        np.ndarray: Count per grid point, same length as pred_grid.
    """
    x = np.sort(np.asarray(y_pred, dtype=float))
    g = np.asarray(pred_grid, dtype=float)
    return (np.searchsorted(x, g + bandwidth, side="right")
            - np.searchsorted(x, g - bandwidth, side="left")).astype(float)


def low_support_spans(
    pred_grid: np.ndarray, support: np.ndarray, min_support: float
) -> List[Tuple[float, float]]:
    """Contiguous [lo, hi] stretches of pred_grid where support < min_support."""
    g = np.asarray(pred_grid, dtype=float)
    bad = np.asarray(support, dtype=float) < min_support
    spans, start = [], None
    for i, flag in enumerate(bad):
        if flag and start is None:
            start = g[i]
        elif not flag and start is not None:
            spans.append((start, g[i]))
            start = None
    if start is not None:
        spans.append((start, g[-1]))
    return spans


def local_empirical_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    n_bins: int = 12,
) -> pd.DataFrame:
    """
    Model-free local frequencies: bin by predicted Tm, measure the fraction above T.

    This is the ground truth the local calibrator is fitted to approximate, and is what
    the right-hand panel of `plot_local_curves` compares against.

    Returns:
        pd.DataFrame: x_lo, x_hi, x_mean, n, n_pos, frac_pos, ci_low, ci_high.
    """
    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

    rows = []
    for i in range(len(edges) - 1):
        mask = idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        k = int((y[mask] >= threshold).sum())
        low, high = wilson_interval(k, n)
        rows.append((edges[i], edges[i + 1], float(x[mask].mean()), n, k, k / n, low, high))
    return pd.DataFrame(
        rows, columns=["x_lo", "x_hi", "x_mean", "n", "n_pos", "frac_pos", "ci_low", "ci_high"]
    )


def plot_local_vs_cumulative(
    calib: LocationScaleCalibrator,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: float,
    cutoffs: Sequence[float],
    path_save: str,
    footnotes: Optional[List[str]] = None,
) -> None:
    """
    The two probability definitions on shared axes, with the selected count alongside.

    Cumulative sits above local by construction: it averages the local curve over the
    whole upper tail. Seeing the gap is the point of this figure -- it is exactly how
    much a set-level enrichment number overstates any individual protein's odds.
    """
    names = list(datasets.keys())
    fig, axes = plt.subplots(1, max(len(names), 2), figsize=FIGSIZE, squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, names):
        y_true, y_pred = datasets[name]
        cum = cumulative_precision(y_true, y_pred, cutoffs, threshold)
        grid = np.asarray(cutoffs, dtype=float)

        ax.plot(grid, calib.local_prob(grid, threshold), color=COLORS[0], linewidth=2.5,
                marker="o", label="Local  P(T$_m$ $\\geq$ T | pred $=$ x)")
        ax.plot(cum["cutoff"], cum["rate"], color=COLORS[1], linewidth=2.5, marker="o",
                label="Cumulative  P(T$_m$ $\\geq$ T | pred $\\geq$ x)")
        ax.fill_between(cum["cutoff"], cum["ci_low"], cum["ci_high"], color=COLORS[1], alpha=0.15)
        ax.axhline(0.5, color="c", linestyle="--", linewidth=2)
        ax.set_xlabel("Cut-off for predicted T$_m$ [°C]", fontsize=LABEL_SIZE)
        ax.set_ylabel(f"P(actual T$_m$ $\\geq$ {threshold:g}°C)", fontsize=LABEL_SIZE)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name} (n={len(y_true)})", fontsize=LABEL_SIZE - 3)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=LEGEND_SIZE, loc="upper left")

        ax_n = ax.twinx()
        ax_n.plot(cum["cutoff"], cum["n_selected"], color="gray", linestyle=":", linewidth=2)
        ax_n.set_ylabel("Proteins selected", fontsize=LABEL_SIZE - 3, color="gray")
        ax_n.tick_params(axis="y", labelsize=TICK_SIZE, labelcolor="gray")

    for ax in axes[len(names):]:
        ax.axis("off")
    _finish(fig, path_save, footnotes)


def plot_ece_vs_threshold(
    reports: Dict[str, Dict[float, dict]],
    path_save: str,
    footnotes: Optional[List[str]] = None,
) -> None:
    """ECE and calibration-in-the-large against threshold, one line per evaluation set."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    for i, (name, per_t) in enumerate(reports.items()):
        ts = sorted(per_t.keys())
        ax1.plot(ts, [per_t[t]["ece"] for t in ts], marker="o",
                 color=COLORS[i % len(COLORS)], linewidth=2.5, label=name)
        ax2.plot(ts, [per_t[t]["cal_in_large"] for t in ts], marker="o",
                 color=COLORS[i % len(COLORS)], linewidth=2.5, label=name)

    ax1.set_ylabel("Expected calibration error", fontsize=LABEL_SIZE)
    ax2.axhline(0, color="c", linestyle="--", linewidth=2)
    ax2.set_ylabel("Mean predicted $-$ base rate", fontsize=LABEL_SIZE)
    for ax in (ax1, ax2):
        ax.set_xlabel("Threshold T [°C]", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(fontsize=LEGEND_SIZE)

    _finish(fig, path_save, footnotes)


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

def _clean(obj):
    """Recursively replace non-finite floats with None so the JSON stays strict."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def build_grid(start: float, stop: float, step: float) -> np.ndarray:
    """Inclusive arithmetic grid, the layout the artifact records under `grids`."""
    return np.round(np.arange(start, stop + step / 2, step), 6)


def save_artifact(artifact: dict, path: str) -> None:
    """Write the artifact as strict JSON (non-finite values become null)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(artifact), f, indent=2, allow_nan=False)
    print(f"[calibration] wrote {path}")


def load_artifact(path: str) -> dict:
    """Read a calibration artifact."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def interp_local(artifact: dict, tm_pred: float, threshold: float) -> float:
    """
    Reference consumer: bilinear lookup of the local probability table.

    This is deliberately dependency-free (no scipy, no sklearn) because it is the exact
    algorithm the shipped `tmprot` package will reimplement -- that package depends only
    on torch/transformers/peft/click and should stay that way. Out-of-range queries
    clamp to the grid edge.

    Args:
        artifact (dict): Loaded calibration artifact.
        tm_pred (float): Predicted Tm in degrees C.
        threshold (float): Threshold T in degrees C.

    Returns:
        float: P(Tm_true >= threshold | Tm_pred = tm_pred).
    """
    g = artifact["grids"]
    xs = build_grid(**g["tm_pred_grid"])
    ts = build_grid(**g["threshold_grid"])
    table = np.asarray(artifact["local"]["p"], dtype=float)

    x = min(max(tm_pred, xs[0]), xs[-1])
    t = min(max(threshold, ts[0]), ts[-1])
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    j = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))

    wt = (t - ts[i]) / (ts[i + 1] - ts[i])
    wx = (x - xs[j]) / (xs[j + 1] - xs[j])
    top = table[i, j] * (1 - wx) + table[i, j + 1] * wx
    bot = table[i + 1, j] * (1 - wx) + table[i + 1, j + 1] * wx
    return float(np.clip(top * (1 - wt) + bot * wt, 0.0, 1.0))

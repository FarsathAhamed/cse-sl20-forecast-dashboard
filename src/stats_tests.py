"""
Phase 7: Statistical Evaluation & Multiple-Testing Utilities.

Provides:
- wilson_ci: Wilson score 95% confidence interval for classification accuracy
- binomial_test_gt: Exact one-sided binomial test (H1: acc > benchmark)
- oos_r2: Out-of-sample R^2 relative to a benchmark forecast
- dm_test_hac: One-sided Diebold-Mariano test with Newey-West (HAC) robust errors
- holm_adjust: Holm-Bonferroni multiple-testing adjustment across stocks
"""

import numpy as np
import statsmodels.api as sm
from scipy.stats import binomtest, norm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    low, high = proportion_confint(count=k, nobs=n, alpha=alpha, method="wilson")
    return float(low), float(high)


def binomial_test_gt(k: int, n: int, p0: float) -> float:
    res = binomtest(k=k, n=n, p=p0, alternative="greater")
    return float(res.pvalue)


def oos_r2(y_true: np.ndarray, y_pred_model: np.ndarray, y_pred_bench: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred_model = np.asarray(y_pred_model, dtype=float)
    y_pred_bench = np.asarray(y_pred_bench, dtype=float)
    sse_m = float(np.sum((y_true - y_pred_model) ** 2))
    sse_b = float(np.sum((y_true - y_pred_bench) ** 2))
    if sse_b == 0.0:
        return float("nan")
    return 1.0 - sse_m / sse_b


def dm_test_hac(
    y_true: np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_bench: np.ndarray,
    maxlags: int | None = None,
) -> dict:
    """
    One-sided Diebold-Mariano test under squared-error loss:
        d_t = (y_t - y_bench_t)^2 - (y_t - y_model_t)^2
    Positive DM statistic means `model` has lower squared error than `bench`.
    Uses Newey-West (HAC) autocorrelation-robust standard errors.
    """
    y_t = np.asarray(y_true, dtype=float)
    e_m = y_t - np.asarray(y_pred_model, dtype=float)
    e_b = y_t - np.asarray(y_pred_bench, dtype=float)
    d = e_b**2 - e_m**2
    n = len(d)

    if maxlags is None:
        maxlags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))

    ols = sm.OLS(d, np.ones((n, 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )
    t_stat = float(ols.tvalues[0])
    p_one_sided = float(1.0 - norm.cdf(t_stat))
    return {
        "dm_stat": round(t_stat, 4),
        "p_raw": round(p_one_sided, 6),
        "maxlags": int(maxlags),
    }


def holm_adjust(p_values: list[float] | np.ndarray, alpha: float = 0.05) -> list[float]:
    _, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return [round(float(p), 6) for p in p_adj]

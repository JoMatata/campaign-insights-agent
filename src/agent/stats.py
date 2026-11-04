"""Campaign metrics and a two-proportion z-test."""
import math

from scipy.stats import norm


def _div(a, b):
    return a / b if b else None


def campaign_metrics(impressions, clicks, conversions, spend, revenue):
    return {
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "spend": round(spend, 2),
        "revenue": round(revenue, 2),
        "ctr": _div(clicks, impressions),
        "cvr": _div(conversions, clicks),
        "cpa": _div(spend, conversions),
        "roas": _div(revenue, spend),
    }


def two_proportion_ztest(x1, n1, x2, n2, alpha=0.05):
    """Pooled z-test for p1 vs p2, plus an unpooled Wald CI for the difference."""
    if min(n1, n2) <= 0:
        raise ValueError("Both groups need n > 0")
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zcrit = norm.ppf(1 - alpha / 2)
    diff = p1 - p2
    return {
        "p1": p1,
        "p2": p2,
        "diff": diff,
        "relative_lift": _div(diff, p2),
        "z": z,
        "p_value": p_value,
        "ci_low": diff - zcrit * se_diff,
        "ci_high": diff + zcrit * se_diff,
        "significant": p_value < alpha,
    }

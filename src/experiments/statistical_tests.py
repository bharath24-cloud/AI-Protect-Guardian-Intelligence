from typing import Dict, Any, List, Tuple, Callable
import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_absolute_error

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculates Cohen's d effect size between two distributions."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def calculate_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable = r2_score,
    n_bootstraps: int = 500,
    ci_level: float = 0.95,
    random_seed: int = 42
) -> List[float]:
    """Computes non-parametric empirical bootstrap confidence intervals."""
    rng = np.random.RandomState(random_seed)
    n_samples = len(y_true)
    boot_scores = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        try:
            score = metric_fn(y_true[idx], y_pred[idx])
            boot_scores.append(score)
        except Exception:
            continue
            
    if not boot_scores:
        return [0.0, 0.0]
        
    alpha = (1.0 - ci_level) / 2.0
    low = float(np.percentile(boot_scores, 100 * alpha))
    high = float(np.percentile(boot_scores, 100 * (1.0 - alpha)))
    return [round(low, 4), round(high, 4)]


def perform_statistical_hypothesis_test(
    model_residuals: np.ndarray,
    baseline_residuals: np.ndarray,
    significance_alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs paired Student's t-test and Wilcoxon signed-rank test between proposed model 
    errors and baseline errors to prove statistical superiority.
    """
    # Paired t-test
    t_stat, p_val_t = stats.ttest_rel(model_residuals, baseline_residuals)
    
    # Wilcoxon signed rank test (non-parametric)
    try:
        w_stat, p_val_w = stats.wilcoxon(model_residuals, baseline_residuals)
    except Exception:
        p_val_w = p_val_t
        
    effect_d = calculate_cohens_d(baseline_residuals, model_residuals) # positive d means model errors are lower
    
    is_significant = (p_val_t < significance_alpha) and (effect_d > 0.2)
    
    return {
        "p_value": float(p_val_t) if not np.isnan(p_val_t) else 1.0,
        "p_value_wilcoxon": float(p_val_w) if not np.isnan(p_val_w) else 1.0,
        "effect_size_cohens_d": round(float(effect_d), 4),
        "is_statistically_significant": bool(is_significant)
    }

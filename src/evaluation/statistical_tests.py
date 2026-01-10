"""
Statistical testing utilities for comparing models.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


def bootstrap_confidence_interval(
    values: List[float],
    num_samples: int = 1000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        values: List of metric values
        num_samples: Number of bootstrap resamples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        mean, lower_bound, upper_bound
    """
    values = np.array(values)
    n = len(values)
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(num_samples):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean = np.mean(values)
    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)
    
    return mean, lower, upper


def paired_t_test(
    values1: List[float],
    values2: List[float]
) -> Tuple[float, float]:
    """
    Perform paired t-test between two sets of values.
    
    Args:
        values1: Metric values from model 1
        values2: Metric values from model 2
    
    Returns:
        t_statistic, p_value
    """
    t_stat, p_value = stats.ttest_rel(values1, values2)
    return t_stat, p_value


def wilcoxon_test(
    values1: List[float],
    values2: List[float]
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        values1: Metric values from model 1
        values2: Metric values from model 2
    
    Returns:
        statistic, p_value
    """
    stat, p_value = stats.wilcoxon(values1, values2)
    return stat, p_value


def compute_statistical_significance(
    model_results: Dict[str, List[float]],
    baseline_name: str = 'random',
    alpha: float = 0.05
) -> Dict[str, Dict[str, any]]:
    """
    Compute statistical significance for all models vs baseline.
    
    Args:
        model_results: Dictionary mapping model_name -> list of per-user metric values
        baseline_name: Name of baseline model
        alpha: Significance level
    
    Returns:
        Dictionary with significance test results
    """
    if baseline_name not in model_results:
        raise ValueError(f"Baseline '{baseline_name}' not found in results")
    
    baseline_values = model_results[baseline_name]
    
    significance_results = {}
    
    for model_name, values in model_results.items():
        if model_name == baseline_name:
            continue
        
        # Ensure same number of samples
        if len(values) != len(baseline_values):
            print(f"Warning: {model_name} has different number of samples than baseline")
            continue
        
        # Paired t-test
        t_stat, p_value_t = paired_t_test(values, baseline_values)
        
        # Wilcoxon test
        w_stat, p_value_w = wilcoxon_test(values, baseline_values)
        
        # Bootstrap CI
        mean, ci_lower, ci_upper = bootstrap_confidence_interval(values)
        
        significance_results[model_name] = {
            'mean': mean,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            't_statistic': t_stat,
            'p_value_ttest': p_value_t,
            'significant_ttest': p_value_t < alpha,
            'wilcoxon_statistic': w_stat,
            'p_value_wilcoxon': p_value_w,
            'significant_wilcoxon': p_value_w < alpha
        }
    
    return significance_results


def format_significance_table(
    significance_results: Dict[str, Dict],
    metric_name: str = 'NDCG@10'
) -> str:
    """
    Format significance results as a table.
    
    Args:
        significance_results: Output from compute_statistical_significance
        metric_name: Name of the metric being compared
    
    Returns:
        Formatted table string
    """
    table = []
    table.append("=" * 80)
    table.append(f"Statistical Significance Tests - {metric_name}")
    table.append("=" * 80)
    table.append(f"{'Model':<20} {'Mean':<10} {'95% CI':<25} {'p-value':<10} {'Significant':<12}")
    table.append("-" * 80)
    
    for model_name, results in sorted(significance_results.items(), key=lambda x: x[1]['mean'], reverse=True):
        mean = results['mean']
        ci_lower = results['ci_95_lower']
        ci_upper = results['ci_95_upper']
        p_value = results['p_value_ttest']
        significant = '✓' if results['significant_ttest'] else '✗'
        
        ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        table.append(f"{model_name:<20} {mean:.4f}    {ci_str:<25} {p_value:.4f}    {significant:<12}")
    
    table.append("=" * 80)
    table.append("Note: Significance tested using paired t-test (α=0.05)")
    table.append("=" * 80)
    
    return "\n".join(table)

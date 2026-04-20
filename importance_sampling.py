# Functions for importance sampling and related calculations.

from typing import Literal
import numpy as np
from sklearn.metrics import fbeta_score
from scipy.special import expit

def score_to_classification(
        score: float,
        threshold: float = 0.5) -> int:
    """
    Converts a continuous score to a binary classification based on the given threshold."""
    if not(0 < score < 1):
        raise ValueError("Score must be between 0 and 1.")
    return int(score >= threshold)

def classification_to_score(
        classification: int,
        precision_prior: float,
        false_inclusion_rate_prior: float) -> float:
    """
    Converts a binary classification to a continuous score by returning the 
    probability of being in the positive class based given priors for a classifier's 
    precision and false inclusion rate priors."""
    if classification not in (0, 1):
        raise ValueError("Classification must be 0 or 1.")
    if not (0 < precision_prior < 1):
        raise ValueError("Precision prior must be between 0 and 1 (exclusive).")
    if not (0 < false_inclusion_rate_prior < 1):
        raise ValueError("Negative precision prior must be between 0 and 1 (exclusive).")

    posterior_probability = (
    precision_prior * (classification == 1) +
    (false_inclusion_rate_prior) * (classification == 0)
    )

    return posterior_probability

def precision_to_recall(precision: float,
                        false_inclusion_rate: float,
                        classifier_positive_rate) -> float:
    """
    Determines a classifier's recall given its precision, false inclusion rate,
    and positive rate."""
    if not (0 < precision < 1):
        raise ValueError("Precision must be between 0 and 1 (exclusive).")
    if not (0 < false_inclusion_rate < 1):
        raise ValueError("False inclusion rate must be between 0 and 1 (exclusive).")
    if not (0 < classifier_positive_rate < 1):
        raise ValueError("Classifier positive rate must be between 0 and 1 (exclusive).")
    recall = (
        precision / (precision + false_inclusion_rate *
        (1 - classifier_positive_rate) / classifier_positive_rate)
    )

    return recall

def recall_to_precision(recall: float,
                        false_exclusion_rate: float,
                        prevalence) -> float:
    """
    Determines a classifier's precision given its recall, false exclusion rate,
    and prevalence."""
    if not (0 < recall < 1):
        raise ValueError("Recall must be between 0 and 1 (exclusive).")
    if not (0 < false_exclusion_rate < 1):
        raise ValueError("False exclusion rate must be between 0 and 1 (exclusive).")
    if not (0 < prevalence < 1):
        raise ValueError("Prevalence must be between 0 and 1 (exclusive).")
    precision = (
        recall / (recall + false_exclusion_rate * prevalence / (1 - prevalence))
    )

    return precision
                                
def effective_sample_size(weights: np.ndarray) -> float:
    """
    Computes the effective sample size given an array of sampling weights."""
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    sum_weights = np.sum(weights)
    sum_weights_squared = np.sum(weights ** 2)
    if sum_weights_squared == 0:
        return 0.0
    return (sum_weights ** 2) / sum_weights_squared

def _sigmoid(score: float,
             threshold: float = 0.5,
             temperature: float = 0.1) -> float:
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    scaled_score = (score - threshold) / temperature
    return expit(scaled_score)

def _prevalence_importance_score(score: float) -> float:
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    return score

def _prevalence_importance_score_anchored(score: float) -> float:
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    return score * (1 - score)

def _prevalence_importance_score_informed(score: float, prevalence_prior: float) -> float:
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    if not (0 < prevalence_prior < 1):
        raise ValueError("Prevalence prior must be between 0 and 1 (exclusive).")
    return (1 - prevalence_prior) ** 2 * score + prevalence_prior ** 2 * (1 - score)

def _precision_importance_function_informed(
        score: float,
        precision_prior: float,
        method: Literal["heaviside", "sigmoid"] = "heaviside") -> float:
    """
    Computes the sampling weight based on the given precision score and prior."""
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    if not (0 < precision_prior < 1):
        raise ValueError("Precision prior must be between 0 and 1 (exclusive).")
    
    draw_weight = (
        (score_to_classification(score) == 1) * 
        (precision_prior ** 2 * (1 - score) +
        (1 - precision_prior) ** 2 * score)
    )

    return draw_weight

def _recall_importance_function_informed(
        score: float,
        recall_prior: float) -> float:
    """
    Computes the sampling weight based on the given recall score and prior."""
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    if not (0 < recall_prior < 1):
        raise ValueError("Recall prior must be between 0 and 1 (exclusive).")
    return (score_to_classification(score) - recall_prior) ** 2 * score

def _false_inclusion_rate_importance_function_informed(
        score: float,
        false_inclusion_rate_prior: float) -> float:
    """
    Computes the sampling weight based on the given false inclusion rate score and prior."""
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    if not (0 < false_inclusion_rate_prior < 1):
        raise ValueError("False inclusion rate prior must be between 0 and 1 (exclusive).")
    
    draw_weight = (
        (score_to_classification(score) == 0) * 
        (false_inclusion_rate_prior ** 2 * (1 - score) +
        (1 - false_inclusion_rate_prior) ** 2 * score)
    )

    return draw_weight

def _false_exclusion_rate_importance_function_informed(
        score: float,
        false_exclusion_rate_prior: float) -> float:
    """
    Computes the sampling weight based on the given false exclusion rate score and prior."""
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    if not (0 < false_exclusion_rate_prior < 1):
        raise ValueError("False exclusion rate prior must be between 0 and 1 (exclusive).")
    
    draw_weight = (
        (score_to_classification(score) - false_exclusion_rate_prior) ** 2
        * (1 - score)
    )

    return draw_weight

def _fbeta_importance_function_informed(
        score: float,
        fbeta_prior: float,
        beta: float = 1) -> float:
    """
    Computes the sampling weight based on the given F-beta score and prior."""
    if not (0 <= score <= 1):
        raise ValueError("Score must be between 0 and 1.")
    if beta <= 0:
        raise ValueError("Beta must be positive.")
    if not (0 < fbeta_prior < 1):
        raise ValueError("F-beta prior must be between 0 and 1 (exclusive).")
    
    draw_weight = (
        (score * (1 - fbeta_prior) + (1 - score) * fbeta_prior ** 2) *
        (score_to_classification(score) == 1) +
        ((1 - fbeta_prior) ** 2 * score * fbeta_prior ** 2) *
        (score_to_classification(score) == 0)
        )
    
    return draw_weight

def _classification_importance_function(
        score: float,
        target_weight: float = 1,
        mixture_weight: float = 1,
        defensive_score: float = 1) -> float:
    """
    Computes the sampling weight based on the given score and target weight.

    Args:
        score(float): The classifier score.
        target_weight(float): The desired weighting scheme for the population.
        mixture_weight(float): The mixture weight for defensive sampling.
    Returns:
        float: The computed sampling weight.
    """
    if score < 0 or score > 1:
        raise ValueError("Score must be between 0 and 1.")
    if target_weight < 0:
        raise ValueError("Target weight must be non-negative.")
    if not(0 <= mixture_weight <= 1):
        raise ValueError("Mixture weight must be between 0 and 1.")
    
    draw_weight = (
        target_weight * (mixture_weight * np.sqrt(score) 
                         + (1 - mixture_weight) * np.sqrt(defensive_score))
    )

    return draw_weight

def recall_stratified_ratio(
        precision: float,
        false_inclusion_rate: float,
        classifier_positive_rate: float) -> float:
    """
    Computes the ratio of positive to negative classifications needed to construct
    confidence intervals for precision and recall of equal length."""

    ratio_neg_pos = (1 - classifier_positive_rate) / classifier_positive_rate

    partial_dervivative_precision = (
        (- ratio_neg_pos * precision) /
        (precision + ratio_neg_pos * false_inclusion_rate) ** 2
    )
    
    partial_derivative_false_inclusion_rate = (
        (ratio_neg_pos * false_inclusion_rate) /
        (precision + ratio_neg_pos * false_inclusion_rate) ** 2
    )

    ratio = (
        (1. - partial_dervivative_precision ** 2) / 
        (partial_derivative_false_inclusion_rate ** 2 ** false_inclusion_rate *
         (1. - false_inclusion_rate) / precision / (1. - precision))
    )

    return ratio

def _rectifier(metric_value_benchmark: float,
               metric_value_proxy: float) -> float:
    """
    Calculates a rectifier value based on benchmark and proxy metric values."""
    if not (0 <= metric_value_benchmark <= 1):
        raise ValueError("Benchmark metric value must be between 0 and 1.")
    if not (0 <= metric_value_proxy <= 1):
        raise ValueError("Proxy metric value must be between 0 and 1.")
    return metric_value_proxy - metric_value_benchmark

def ratio_covariance_variance(
        covariance: float,
        variance: float,
        small_sample_size: int,
        large_sample_size: int) -> float:
    """
    Computes the ratio of covariance to variance."""
    if variance == 0:
        raise ValueError("Variance must be non-zero.")
    if small_sample_size <= 1:
        raise ValueError("Small sample size must be greater than 1.")
    if large_sample_size <= 1:
        raise ValueError("Large sample size must be greater than 1.")
    
    omega = (
        covariance / (1 + (small_sample_size / large_sample_size) * variance)
    )

    return omega

def prediction_powered_inference_estimator(
        metric_value_benchmark: float,
        rectifier_value: float,
        ratio_covariance_variance: float
        ) -> float:
    """
    Estimates the population performance improvement (PPI) based on benchmark
    and rectifier values."""
    if not (0 <= metric_value_benchmark <= 1):
        raise ValueError("Benchmark metric value must be between 0 and 1.")
    
    return metric_value_benchmark + ratio_covariance_variance * rectifier_value

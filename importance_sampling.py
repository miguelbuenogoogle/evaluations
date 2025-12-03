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
                                

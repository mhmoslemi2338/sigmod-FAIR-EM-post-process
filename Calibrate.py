

## Defining necessary functions:

# This script defines several functions and classes for evaluating fairness metrics and calibrating predictions for machine learning models. The steps are as follows:

# 1. **Quantile Function Class (EQF)**:
#    - The `EQF` class is used to calculate the empirical quantile function from sample data, allowing for interpolation.

# 2. **Fairness Objects Calculation (`get_fairness_objects`)**:
#    - This function calculates empirical cumulative distribution functions (ECDF) and quantile functions (EQF) for sensitive and non-sensitive groups to estimate fairness-related metrics.

# 3. **Data Reading Function (`read_data`)**:
#    - Reads and combines training, validation, and test datasets for a given task and model.

# 4. **Fairness Estimation Function (`get_fair_estimation`)**:
#    - Estimates fair probabilities for sensitive and non-sensitive groups by combining ECDF and EQF results to ensure fairness.

# 5. **Fairness Metrics Calculation (`_stats_`)**:
#    - Calculates fairness metrics including Demographic Parity, Equal Opportunity, and Equal Opportunity Difference based on the prediction probabilities for both sensitive and non-sensitive groups.

# The goal of these calculations is to measure fairness disparities between different demographic groups and adjust the model predictions accordingly to reduce biases.





import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from utils import *
import time
from sklearn.metrics import roc_curve, auc



# ---- Fairness Metrics Calculation ----
def _stats_calibrate(sens_array, prob_array, y_array):
    """
    Calculate fairness metrics including Demographic Parity, Equal Opportunity, and Equal Opportunity Difference.
    
    Parameters:
    - sens_array: Sensitive attribute array.
    - prob_array: Probability predictions array.
    - y_array: Ground truth labels array.
    
    Returns:
    - DSP_EOD: Average Equal Opportunity Difference.
    - DSP_EO: Average Equal Opportunity.
    - DSP_DP: Average Demographic Parity.
    - auc_all: AUC score for all data.
    - Frac: Fraction of thresholds.
    """
    prob_minor = prob_array[sens_array == 1]
    prob_major = prob_array[sens_array == 0]

    y_minor = y_array[sens_array == 1]
    y_major = y_array[sens_array == 0]

    DP, EO, EOD = [], [], []
    PR_minor_all, TPR_minor_all = [], []
    PR_major_all, TPR_major_all = [], []

    # Calculate overall AUC
    fpr, tpr, _ = roc_curve(y_array, prob_array)
    auc_all = 100 * auc(fpr, tpr)

    N = 500
    for theta in np.linspace(0, 1, N):
        # Calculate metrics for minor group
        y_pred = np.array([1 if score > theta else 0 for score in prob_minor])
        tn, fp, fn, tp = confusion_matrix(y_minor, y_pred).ravel()
        tpr = tp / (tp + fn)
        pr = (tp + fp) / len(y_minor)
        fpr = fp / (fp + tn)
        PR_minor_all.append(pr)
        TPR_minor_all.append(tpr)

        # Calculate metrics for major group
        y_pred = np.array([1 if score > theta else 0 for score in prob_major])
        tn, fp, fn, tp = confusion_matrix(y_major, y_pred).ravel()
        tpr = tp / (tp + fn)
        pr = (tp + fp) / len(y_major)
        fpr = fp / (fp + tn)
        PR_major_all.append(pr)
        TPR_major_all.append(tpr)

        # Calculate fairness disparities
        EOD.append((np.abs(TPR_major_all[-1] - TPR_minor_all[-1]) + np.abs(fpr - fpr)))
        EO.append(np.abs(TPR_major_all[-1] - TPR_minor_all[-1]))
        DP.append(np.abs(PR_major_all[-1] - PR_minor_all[-1]))

    DSP_EOD = 100 * np.average(EOD)
    DSP_EO = 100 * np.average(EO)
    DSP_DP = 100 * np.average(DP)
    Frac = np.linspace(0, 1, N)

    return DSP_EOD, DSP_EO, DSP_DP, auc_all, Frac





# ---- Quantile Function Class ----
class EQF:
    def __init__(self, sample_data):
        self._calculate_eqf(sample_data)

    def _calculate_eqf(self, sample_data):
        """ Calculate the quantile function for the given sample data """
        sorted_data = np.sort(sample_data)
        linspace = np.linspace(0, 1, num=len(sample_data))
        self.interpolater = interp1d(linspace, sorted_data)
        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_):
        """ Interpolate a value based on the quantile function """
        try:
            return self.interpolater(value_)
        except ValueError:
            if value_ < self.min_val:
                return 0.0
            elif value_ > self.max_val:
                return 1.0
            else:
                raise ValueError('Error with input value')

# ---- Fairness Objects Calculation ----
def get_fairness_objects(sensitive_vector, predictions_s_0, predictions_s_1):
    """
    Calculate ECDF (Empirical CDF) and EQF (Empirical Quantile Function) for sensitive and non-sensitive groups.
    
    Parameters:
    - sensitive_vector: Array indicating sensitive attributes.
    - predictions_s_0: Predictions for non-sensitive group.
    - predictions_s_1: Predictions for sensitive group.
    
    Returns:
    - pi_dict: Proportions of sensitive and non-sensitive groups.
    - ecdf_dict: ECDF objects for both groups.
    - eqf_dict: EQF objects for both groups.
    """
    # Calculate CDF and quantile function for both groups
    ecdf_dict = {
        'p_non_sensitive': ECDF(predictions_s_0.reshape(-1,)),
        'p_sensitive': ECDF(predictions_s_1.reshape(-1,))
    }

    eqf_dict = {
        'p_non_sensitive': EQF(predictions_s_0.reshape(-1,)),
        'p_sensitive': EQF(predictions_s_1.reshape(-1,))
    }

    # Calculate group proportions
    pi_dict = {
        'p_non_sensitive': sensitive_vector[sensitive_vector == 0.0].shape[0] / sensitive_vector.shape[0],
        'p_sensitive': 1 - sensitive_vector[sensitive_vector == 0.0].shape[0] / sensitive_vector.shape[0]
    }

    return pi_dict, ecdf_dict, eqf_dict

# ---- Data Reading Function ----
def read_data(path_base, model, task, sens_dict):
    """
    Read and combine train, validation, and test data for a specific task and model.
    
    Parameters:
    - path_base: Base path for the data.
    - model: Model name.
    - task: Task name.
    - sens_dict: Sensitive attribute dictionary.
    
    Returns:
    - TRAIN: List containing training data, training scores, and training sensitive attributes.
    - TEST: List containing test data, test scores, and test sensitive attributes.
    """
    # Load train and validation datasets
    df_train = pd.read_csv(f'{path_base}/DATA_VLDB/{task}/train.csv')
    scores_train = pd.read_csv(f'{path_base}/VLDB_RES/{task}_{model}/score_train.csv')
    sens_train = make_sens_vector(df_train, task, sens_dict)

    df_valid = pd.read_csv(f'{path_base}/DATA_VLDB/{task}/valid.csv')
    scores_valid = pd.read_csv(f'{path_base}/VLDB_RES/{task}_{model}/score_valid.csv')
    sens_valid = make_sens_vector(df_valid, task, sens_dict)

    # Combine train and validation datasets
    df_train = pd.concat([df_train, df_valid])
    scores_train = pd.concat([scores_train, scores_valid])
    sens_train = np.concatenate((sens_train, sens_valid))

    # Load test dataset
    df_test = pd.read_csv(f'{path_base}/DATA_VLDB/{task}/test.csv')
    scores_test = pd.read_csv(f'{path_base}/VLDB_RES/{task}_{model}/score_test.csv')
    sens_test = make_sens_vector(df_test, task, sens_dict)

    TRAIN = [df_train, scores_train, sens_train]
    TEST = [df_test, scores_test, sens_test]

    return TRAIN, TEST

# ---- Fairness Estimation Function ----
def get_fair_estimation(p_dict, ecdf_dict, eqf_dict, predictions_nonsensitive, predictions_sensitive, jitter=0.0001):
    """
    Estimate fair probabilities for both sensitive and non-sensitive groups.
    
    Parameters:
    - p_dict: Proportions of sensitive and non-sensitive groups.
    - ecdf_dict: ECDF objects for both groups.
    - eqf_dict: EQF objects for both groups.
    - predictions_nonsensitive: Predictions for non-sensitive group.
    - predictions_sensitive: Predictions for sensitive group.
    - jitter: Small random noise added to avoid numerical issues.
    
    Returns:
    - vals_1: Calibrated values for non-sensitive group.
    - vals_2: Calibrated values for sensitive group.
    """
    # Sample jitters
    np.random.seed(int(time.time()))
    jitter_matrix = np.random.uniform(-jitter, jitter, (predictions_sensitive.shape[0] + predictions_nonsensitive.shape[0]))

    # ECDF-ified values
    f_preds_nonsensitive = ecdf_dict['p_non_sensitive'](predictions_nonsensitive)
    f_preds_sensitive = ecdf_dict['p_sensitive'](predictions_sensitive)

    # Calculate calibrated values for non-sensitive group
    vals_1 = np.zeros_like(predictions_nonsensitive)
    vals_1 += p_dict['p_non_sensitive'] * eqf_dict['p_non_sensitive'](f_preds_nonsensitive)
    vals_1 += p_dict['p_sensitive'] * eqf_dict['p_sensitive'](f_preds_nonsensitive)

    # Calculate calibrated values for sensitive group
    vals_2 = np.zeros_like(predictions_sensitive)
    vals_2 += p_dict['p_non_sensitive'] * eqf_dict['p_non_sensitive'](f_preds_sensitive)
    vals_2 += p_dict['p_sensitive'] * eqf_dict['p_sensitive'](f_preds_sensitive)

    return vals_1, vals_2

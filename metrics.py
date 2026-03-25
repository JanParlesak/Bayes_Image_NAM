import torch
import sklearn
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    balanced_accuracy_score,
    roc_curve,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score, # Import the rest of the metrics
)


def var_exp_score(predictions, targets):
  mean_sum_of_squares = torch.mean((predictions - targets)**2)
  variance_targets = torch.var(targets)
  var_exp = 1- mean_sum_of_squares/variance_targets

  return var_exp

def coef_det(x, y):
  sum_x = torch.sum(x)
  sum_y = torch.sum(y)
  n = len(x)

  numerator = n * torch.sum(x * y) - sum_x*sum_x
  denominator = (n * torch.sum(x**2) - sum_x**2)**0.5 * (n * torch.sum(y**2) - sum_y**2)**0.5

  return numerator/denominator


def mad_explained(predictions, targets):
  mean_sum_of_ad = torch.mean(torch.abs(predictions - targets))
  deviation_median = torch.mean(torch.abs(targets - torch.median(targets)))

  mad_exp = 1 - mean_sum_of_ad/deviation_median

  return mad_exp


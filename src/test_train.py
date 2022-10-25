"""
Tests to validate training procedure

Date: Oct 2022
Author: joesider
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))

import numpy as np
import pandas as pd

from ml.model import compute_model_metrics


def test_output(eval_data):
    y_pred = eval_data['y_pred']
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_metrics(eval_data):
    y_pred = eval_data['y_pred']
    y_test = eval_data['y_test']

    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    assert precision > 0.5 and recall > 0.5 and f1 > 0.5


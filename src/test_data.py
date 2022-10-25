"""
Tests to validate input data

Date: Oct 2022
Author: joesider
"""
import numpy as np
import pandas as pd


def test_columns(data):
    expected_columns = ['age',
                        'workclass',
                        'fnlgt',
                        'education',
                        'education_num',
                        'marital_status',
                        'occupation',
                        'relationship',
                        'race',
                        'sex',
                        'capital_gain',
                        'capital_loss',
                        'hours_per_week',
                        'native_country',
                        'salary']
    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_columns) == list(these_columns)

def test_categorical_feats(data):
    categorical_expected = [
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country'
    ]
    assert all([True for col in categorical_expected if col in data.columns])
    assert all([data[col].dtype == 'object' for col in categorical_expected])


def test_numerical_feats(data):
    numerical_expected = [
        'age',
        'fnlgt',
        'education_num',
        'capital_gain',
        'capital_loss',
        'hours_per_week'
    ]
    assert all([True for col in numerical_expected if col in data.columns])
    assert all([data[col].dtype == 'int64' for col in numerical_expected])


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper ages [0, 110]
    """
    idx = data['age'].between(0, -110)

    assert np.sum(idx) == 0


def test_row_count(data):
    """
    Checks that the size of the dataset is reasonable (not too small, not too large)
    :param data: Dataframe
    """
    assert 1500 < data.shape[0] < 100000


def test_salary(data):
    assert data.salary.isin([0, 1]).all()

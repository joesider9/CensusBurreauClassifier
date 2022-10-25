"""
Script with functions that reads data and make the data preprocessing pipeline

Date: October 2022
Author: joesider
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline



def get_data():
    """
    This function reads the data, fix the column names, replace the missing values '?' with nans and convert the native
    country column to two groups

    Output:
            clean_data, pd.Dataframe, the new dataframe with clean data
            y: pd.Series, the target values
            categorical_feats, index column, the list of categorical features
            numerical_feats, index column, the list of numerical features
    """
    data = pd.read_csv('./data/census.csv')

    data.columns = [col.strip().replace('-', '_') for col in data.columns]

    data = data.drop_duplicates()

    y = data.pop('salary')

    categorical_feats = data.select_dtypes(include=['object']).columns
    numerical_feats = data.select_dtypes(include=['int64']).columns

    for feature in categorical_feats:
        data[feature] = data[feature].str.strip()
        data[feature].iloc[np.where(data.workclass == '?')] = np.nan
    data.native_country[data.native_country != 'United-States'] = 'Other_value'
    y = y.str.strip()
    y = y.replace(to_replace=['<=50K', '>50K'], value=[0, 1])

    return data, y, categorical_feats, numerical_feats


def get_data_pipeline(categorical_features, numerical_feats):
    """
    Make the pipeline to process the data and create a Random Forest instance.

    Processes the data using ordinary encoding for the categorical features and standard scales the numerical features.
    This can be used in either training or inference/validation.


    Inputs
    ------
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    numerical_feats: list[str]
        List containing the names of the numerical features (default=[])

    Returns
    -------
    X_pipe: returns the inference pipeline
    """
    categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder()
    )

    numerical_scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('Categorical_processing', categorical_preproc, categorical_features),
            ('Numerical_preprocessing', numerical_scaler, numerical_feats)],
        remainder="passthrough",
    )

    return preprocessor

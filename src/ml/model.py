"""
This script contains the functions that train and inference the machine learning pipeline

Date: Oct 2022
Author: joesider

"""
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from data import *

params = {
    'random_forest__max_depth': np.unique(np.linspace(1, 15, num=5).astype('int')).tolist() + [None],
    'random_forest__max_features': ['sqrt', 'log2'],
}


def get_model_pipeline(categorical_features, numerical_feats):
    preprocessor = get_data_pipeline(categorical_features, numerical_feats)
    random_Forest = RandomForestClassifier(random_state=42)

    inference_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("random_forest", random_Forest),
    ])

    return inference_pipe


def train_model(X_train, y_train, categorical_features, numerical_feats):
    """
    Trains the machine learning pipeline and return the best estimator.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    ------
    the best estimator

    """
    pipe = get_model_pipeline(categorical_features, numerical_feats)
    search = GridSearchCV(pipe, params, cv=3, scoring='roc_auc', n_jobs=-1)
    search.fit(X_train, y_train)

    return search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(pipe, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    pipe : ???
        Trained machine learning pipeline.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    if isinstance(X, dict):
        X = pd.DataFrame().from_dict({0: X}).T
    preds = pipe.predict(X)

    return preds

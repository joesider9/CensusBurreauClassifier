import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))
import pytest
import pandas as pd

from ml.data import get_data
from ml.model import get_model_pipeline
from ml.model import inference

from sklearn.model_selection import train_test_split


@pytest.fixture(scope='session')
def data():
    """
    This checks loading the data
    Returns
    -------
    data: pd.Dataframe
    """
    file = '../data/census.csv'

    if not os.path.exists(file):
        pytest.fail('File does not exists')
    try:
        data = pd.read_csv(file)
    except:
        pytest.fail(f'Cannot open {file}')
    data.columns = [col.strip().replace('-', '_') for col in data.columns]

    categorical_feats = data.select_dtypes(include=['object']).columns
    for col in categorical_feats:
        data[col] = data[col].str.strip()

    data.salary = data.salary.replace(to_replace=['<=50K', '>50K'], value=[0, 1])

    return data


@pytest.fixture(scope='session')
def eval_data():
    """
    On this test a Random forest is trained by the default parameters and its output shape is cheked to be the same
    with the input and the target y

    Returns
    -------
    eval_data: dict with 'y_pred' as np.array, the model output with X_test as input
                        and y_test as np.array
    """
    data, y, categorical_feats, numerical_feats = get_data()

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20, stratify=y)

    pipe = get_model_pipeline(categorical_feats, numerical_feats)

    pipe.fit(X_train, y_train)

    y_pred = inference(pipe, X_test)

    if y_pred.shape[0] != X_test.shape[0]:
        pytest.fail(f'The shape of output {y_pred.shape[0]} does not match with input shape {X_test.shape[0]}')

    if y_pred.shape[0] != y_test.shape[0]:
        pytest.fail(f'The shape of output {y_pred.shape[0]} does not match with target shape {y_test.shape[0]}')

    eval_data = {'y_pred': y_pred, 'y_test': y_test}
    return eval_data

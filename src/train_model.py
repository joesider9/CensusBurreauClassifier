# Script to train machine learning model.
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))

import joblib
import logging

from ml.data import get_data
from ml.model import train_model
from ml.model import inference
from ml.model import compute_model_metrics
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def evaluate_slice(ml_pipe, X_test, y_test, feature):
    output = []
    for category in X_test[feature].unique():
        if isinstance(category, str):
            indices = X_test.index[X_test[feature] == category]
            y_pred = inference(ml_pipe, X_test.loc[indices])
            precision, recall, f1 = compute_model_metrics(y_test.loc[indices], y_pred)
            output.append(f'Evaluation on {category} of feature {feature}-> Precision: {precision:.3f}, '
                          f'Recall: {recall:.3f}, F1 {f1:.3f}\n')
    return output


def fit():
    logger.info('Census bureau classification problem starts...')

    data, y, categorical_feats, numerical_feats = get_data()
    logger.info('Data loaded successfully')

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20, random_state=42, stratify=y)

    logger.info('Training process starts...')
    ml_pipe = train_model(X_train, y_train, categorical_feats, numerical_feats)
    logger.info('Training process ends successfully')

    joblib.dump(ml_pipe, '../models/model_pipe.pkl')
    logger.info('Model saved on /models path')

    y_pred = inference(ml_pipe, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    logger.info(f'Evaluation on test data -> Precision: {precision:.3f}, Recall: {recall:.3f}, F1 {f1:.3f}')

    outputs = []
    for feature in categorical_feats:
        outputs += evaluate_slice(ml_pipe, X_test, y_test, feature)

    with open('../output_files/slice_output.txt', 'w') as fp:
        for output in outputs:
            fp.write(output)

    logger.info('Evaluation on data slices are performed successfully')


if __name__ == '__main__':
    fit()

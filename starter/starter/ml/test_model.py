
from .model import train_model, inference, compute_model_metrics
from sklearn.model_selection import train_test_split
import pytest
import pandas as pd
import os
import sklearn
import numpy as np
from .data import process_data


@pytest.fixture
def data():
    dir_path = os.path.dirname(__file__).split('/starter')[0]
    path_to_data = dir_path + '/starter/data/census.csv'
    data = pd.read_csv(filepath_or_buffer=path_to_data)

    data.columns = data.columns.str.replace(' ', '')

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function

    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, label="salary", training=False,
                                        encoder=encoder, lb=lb)

    # Train and save a model.
    return X_train, y_train, X_test, y_test

def test_train_model(data):
    X_train, y_train, X_test, y_test = data
    model = train_model(X_train, y_train)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[0] == X_train.shape[0]
    assert isinstance(model, sklearn.ensemble.RandomForestClassifier)

def test_inference(data):
    X_train, y_train, X_test, y_test = data
    model = train_model(X_train, y_train)
    assert np.any(inference(model, X_test))


def test_compute_model_metrics(data):
    X_train, y_train, X_test, y_test = data
    model = train_model(X_train, y_train)
    pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=pred)

    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64






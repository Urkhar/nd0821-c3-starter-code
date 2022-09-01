from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rforest = RandomForestClassifier(random_state=10)

    grid = {
               'bootstrap': [True],
               'max_depth': [2, 5, 7, None],
               'min_samples_leaf': [250, 500, 1000],
               'min_samples_split': [500, 1000, 2500],
               'n_estimators': [25],
               'criterion': ['gini', 'entropy', 'log_loss']

           },

    r_forest = GridSearchCV(estimator=rforest, param_grid=grid, cv=5,
                            verbose=2, refit=True)

    model = r_forest.fit(X_train, y_train)

    return model.best_estimator_



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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions

def performance_on_slices(df,cat_features,encoder,lb, model,write_to_file):
    """
    Run performance of the model on categorical slices of data

    Parameters
    ----------
    df: Pandas data frame with test data
    cat_features: ist containing the names of the categorical features
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
    lb: sklearn.preprocessing._label.LabelBinarizer
    model: Trained machine learning model.
    write_to_file: txt file where the results of data slicing will be saved



    """
    for i in cat_features:
        for j in df[i].unique():
            tmp_df = df[df[i] ==j]
            X_test_tmp, y_test_tmp, _, _ = process_data(tmp_df, categorical_features=cat_features, label="salary", training=False,
                                                encoder=encoder, lb=lb)
            tmp_predictions = inference(model=model, X=X_test_tmp)
            precision, recall, fbeta = compute_model_metrics(y=y_test_tmp, preds=tmp_predictions)

            print("Performance of the model on data slice: ", j, "That belongs to category: ", i, file=write_to_file)
            print("Precision:", round(precision, 4), "Recall: ", round(recall, 4), "Fbeta: ", round(fbeta, 4),
                  file=write_to_file)
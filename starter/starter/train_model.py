# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle
# Add the necessary imports for the starter code.
import pandas as pd
import os
from starter.starter.ml.data import process_data
from starter.starter.ml.model import train_model, inference, compute_model_metrics, performance_on_slices

# Add code to load in the data.
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
model = train_model(X_train=X_train, y_train=y_train)

save_path = dir_path + '/starter/model/saved_model_pickle.sav'
pickle.dump(model, open(save_path, 'wb'))

predictions = inference(model=model, X=X_test)

print(predictions)

precision, recall, fbeta = compute_model_metrics(y=y_test, preds=predictions)

print()
#

write_to_file = open(dir_path + '/starter/model/slice_output.txt', 'w')

print("Performance of the model on whole test data set:", file=write_to_file)
print("Precision:", round(precision,4), "Recall: ", round(recall,4), "Fbeta: ", round(fbeta,4), file=write_to_file)


performance_on_slices(df = test, cat_features = cat_features, encoder=encoder,lb=lb,model=model, write_to_file=write_to_file)
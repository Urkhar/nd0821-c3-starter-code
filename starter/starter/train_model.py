# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import os
# Add code to load in the data.

dir_path = os.path.dirname(__file__).split('/starter')[0]
path_to_data = dir_path + '/starter/data/census.csv'
data = pd.read_csv(filepath_or_buffer = path_to_data)

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

# Proces the test data with the process_data function.

# Train and save a model.

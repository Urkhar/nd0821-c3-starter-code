# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Karol Urban created the model. It is Random Forest Classifier using the hyperparameters tuning.

## Intended Use
Prediction task is to determine whether a person makes over 50K a year.


## Training Data
Data was trained on the 80% of the original dataset

## Evaluation Data
Data was tested on 20% of the original dataset

## Metrics
_Please include the metrics used and your model's performance on those metrics._ \
Precision: 0.7882 \
Recall:  0.4397 \
Fbeta:  0.5645


## Ethical Considerations
In some countries it is illegal to use potentially discriminatory variables like sex or race in any kind of modelling examples

## Caveats and Recommendations
Increase number of estimators in hyperparameter tuning stage. Currently we have only 25 which is not much however it makes the whole code run quickly. If this model was really going for production it would be recommended to use higher number of estimators

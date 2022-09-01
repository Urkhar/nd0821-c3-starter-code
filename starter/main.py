# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os
import pandas as pd
import numpy as np

dir_path = os.path.dirname(__file__).split('/starter')[0]
model = pickle.load(open(dir_path + '/starter/model/saved_model_pickle.sav', 'rb'))
encoder = pickle.load(open(dir_path + '/starter/model/encoder.sav', 'rb'))

app = FastAPI()

class Census(BaseModel):
   age: int
   workclass: str
   fnglt: int
   education: str
   education_num: int
   marital_status: str
   occupation: str
   relationship: str
   race: str
   sex: str
   capital_gain: int
   capital_loss: int
   hours_per_week: int
   native_country: str

   class Config:
       schema_extra = {
           "example": {
                  "age": 33,
                  "workclass": "Private",
                  "fnglt": 45781,
                  "education": "Masters",
                  "education_num": 14,
                  "marital_status": "Never-married",
                  "occupation": "Tech-support",
                  "relationship": "Unmarried",
                  "race": "White",
                  "sex": "Male",
                  "capital_gain": 14084,
                  "capital_loss": 0,
                  "hours_per_week": 40,
                  "native_country": "Mexico"
           }
       }

@app.get("/")
async def say_hello():
    return {"greeting": "Welcome!!"}



@app.post('/predict', tags=["predictions"])
async def get_prediction(input: Census):

    columns = ['age', 'workclass', 'fnlgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
               'hours-per-week', 'native-country']

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

    example = [[input.age, input.workclass, input.fnglt, input.education,
                input.education_num, input.marital_status, input.occupation,
                input.relationship, input.race, input.sex, input.capital_gain,
                input.capital_loss, input.hours_per_week, input.native_country]]


    df = pd.DataFrame(example, columns=columns)
    X_categorical = df[cat_features].values
    X_continuous = df.drop(*[cat_features], axis=1)
    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    prediction = model.predict(X)
    proba = model.predict_proba(X).tolist()
    proba = np.around(proba,3)


    return {'prediction': str(prediction[0]),
            'proba': str(proba)
            }

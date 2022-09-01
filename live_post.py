import requests
input_data = {
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

response = requests.post('https://udacity-fastapi-project-urbank.herokuapp.com/predict', json=input_data)
print("status code: ", response.status_code)
print("response: ", response.json())
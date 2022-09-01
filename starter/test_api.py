from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome!!"}

def test_post_example():
    example= {
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

    r = client.post('/predict', json=example)

    assert r.status_code == 200
    assert r.json() == {'prediction': "0",
            'proba': "[[0.644 0.356]]"
            }

def test_invalid_input():
    invalid_input = {
        "wrong_variable": "wrong_value"
    }

    r = client.post('/predict', json=invalid_input)

    assert r.status_code == 422
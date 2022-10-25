import requests
from pydantic import BaseModel, Field

data = {'age': 52,
            'workclass': ' Self-emp-inc',
            'fnlgt': 287927,
            'education': ' HS-grad',
            'education_num': 9,
            'marital_status': ' Married-civ-spouse',
            'occupation': ' Exec-managerial',
            'relationship': ' Wife',
            'race': ' White',
            'sex': ' Female',
            'capital_gain': 15024,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country': 'Other value'
            }

if __name__=='__main__':
    r = requests.post('http://127.0.0.1:8000/predict', json=data)
    print(f'status code is: {r.status_code}')
    print(r.json())
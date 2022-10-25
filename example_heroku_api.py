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
            'native_country': 'Other value',
            'salary': ' >50K'}

if __name__=='__main__':
    r = requests.post('https://censusbureau.herokuapp.com//predict', json=data, auth=('joesider9@gmail.com', '^)"qfC.^7$(J7Zh'))
    print(f'status code is: {r.status_code}')
    print(r.json())
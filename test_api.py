"""
This script contains test functions for testing the api

Date: Oct 2022
Author: joesider
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_greetings():
    get = client.get('/')
    greeting = get.json()
    assert get.status_code == 200
    assert 'Hi' in greeting.keys()
    assert greeting['Hi'] == 'This a Census Bureau classifier'


def test_output_less_than_50K():
    data = {'age': 60,
            'workclass': ' Private',
            'fnlgt': 132529,
            'education': ' HS-grad',
            'education_num': 9,
            'marital_status': ' Married-civ-spouse',
            'occupation': ' Craft-repair',
            'relationship': ' Husband',
            'race': ' White',
            'sex': ' Male',
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country': 'Other value',
            'salary': ' <=50K'}
    post = client.post('/predict', json=data)
    pred = post.json()
    assert post.status_code == 200
    assert 'Answer' in pred.keys()
    assert pred['Answer'] == '<=50K'


def test_output_greater_than_50K():
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
    post = client.post('/predict', json=data)
    pred = post.json()
    assert post.status_code == 200
    assert 'Answer' in pred.keys()
    assert pred['Answer'] == '>50K'

def test_mal_output():
    data = {'age': 52,
            'workclass': ' Self-emp-inc',
            'fnlgt': 287927,
            'education': ' HS-grad',
            'relationship': ' Wife',
            'race': ' White',
            'sex': ' Female',
            'capital_gain': 15024,
            'capital_loss': 0,
            'hours_per_week': 40,
            'native_country': 'Other value',
            'salary': ' >50K'}
    post = client.post('/predict', json=data)
    pred = post.json()
    assert post.status_code != 200

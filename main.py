# Put the code for your API here.
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.ml.model import *

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class InputData(BaseModel):
    age: int = Field(example=60)
    workclass: str = Field(example=' Private')
    fnlgt: int = Field(example=132529)
    education: str = Field(example=' HS-grad')
    education_num: int = Field(example=9)
    marital_status: str = Field(example=' Married-civ-spouse')
    occupation: str = Field(example=' Craft-repair')
    relationship: str = Field(example=' Husband')
    race: str = Field(example=' White')
    sex: str = Field(example=' Male')
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example='Other value')
    salary: str = Field(example='<=50K')


app = FastAPI()


@app.get("/")
async def greetings():
    return {'Hi': 'This a Census Bureau classifier'}


@app.post('/predict')
async def predict(data: InputData):
    pipe = joblib.load('./models/model_pipe.pkl')
    x = dict()
    for key, value in data.dict().items():
        if key != 'salary':
            x[key] = [value]
    x = pd.DataFrame(x)

    x.columns = [col.strip().replace('-', '_') for col in x.columns]
    categorical_feats = x.select_dtypes(include=['object']).columns

    for feature in categorical_feats:
        x[feature] = x[feature].str.strip()
        x[feature].iloc[np.where(x.workclass == '?')] = np.nan
    x.native_country[x.native_country != 'United-States'] = 'Other_value'

    data.salary = data.salary.strip()

    y_pred = inference(pipe, x)[0]

    tag = '<=50K' if y_pred == 0 else '>50K'

    return {'Answer': tag}

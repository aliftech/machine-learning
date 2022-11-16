from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel


app = FastAPI()

class request_body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

iris = load_iris()

x = iris.data
y = iris.target

clf = GaussianNB()
clf.fit(x, y)

@app.post('/')
def main(data: request_body):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    index = clf.predict(test_data)[0]
    return {'class': iris.target_names[index]}
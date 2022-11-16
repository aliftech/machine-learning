from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

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
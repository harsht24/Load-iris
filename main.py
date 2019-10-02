import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset['data']
y = dataset['target']

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y= train_test_split(x, y)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(train_X,train_y)
prediction = model.predict(test_X)

accuracy = accuracy_score(prediction,test_y)


print("prediction = ",prediction)
print("accuracy = ",accuracy)


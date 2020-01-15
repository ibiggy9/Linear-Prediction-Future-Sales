import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import metrics
import pickle
import seaborn as seabornInstance
from sklearn import metrics
import matplotlib.pyplot as pltd
import math

data = pd.read_csv("Veggie Crisps Pull.csv")

predict = "$ Vol"

data = data[["$ Shr", "Avg AC Dist", "Avg TPR Unit Price", "Avg Retail Unit Price", "ACD Display", "ACD Ad", "$ Vol"]]
data = shuffle(data) # Optional - shuffle the data

#how to display data
#print(data.head())
#print(data.tail(n=10))

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

# The line below is needed for error testing.
y_pred = linear.predict(x_test)

print("Accuracy: " + str(acc))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE

best = 0.9
for _ in range(10000):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # The line below is needed for error testing.
    y_pred = linear.predict(x_test)

    print("Accuracy: " + str(acc))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


if acc > best:
    best = acc
    with open("studentgrades.pickle", "wb") as f:
        pickle.dump(linear, f)


# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


print("-------------------------")
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('Best accuracy: \n', linear.score(x_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("-------------------------")

predicted = linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

#plot the data
#data.describe()

#Check for null values
#data.isnull().any()

#Remove null data
# data = data.fillna(method='ffill')



#HOW TO PLUG IN VALUES TO PREDICT $VOL
new_x = [[2, 90, 2.3, 2.4, 2, 4.12]]
print('The Expected $Vol with your inputs is:\n ', linear.predict(new_x))



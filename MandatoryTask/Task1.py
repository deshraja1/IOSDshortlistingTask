import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

trainingDataset = pd.read_csv(r"MandatoryTask\train.csv")
testDataset = pd.read_csv(r"MandatoryTask\test.csv")

x_train = np.array(trainingDataset.iloc[:, 0:2])
y_train = np.array(trainingDataset.iloc[:, 2])

x_test = np.array(testDataset.iloc[:,0:2])
y_test = np.array(testDataset.iloc[:, 2])

knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print('Accuracy is : ', accuracy_score(y_test, y_pred))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

irisDataset = pd.read_csv(r"OptionalTask\iris.csv")

x = np.array(irisDataset.iloc[:, 0:4])
y = np.array(irisDataset.iloc[:, 4])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

svcpoly = SVC(kernel='poly', random_state=0)
svcpoly.fit(x_train, y_train)

y_poly_pred = svcpoly.predict(x_test)
print(y_poly_pred, y_test)
poly_pr_sc = (precision_score(y_test, y_poly_pred, average=None))
poly_f1_sc = (f1_score(y_test, y_poly_pred, average=None))

svcsigmoid = SVC(kernel='sigmoid', random_state=0)
svcsigmoid.fit(x_train, y_train)

y_sigmoid_pred = svcsigmoid.predict(x_test)

sigmoid_pr_sc = (precision_score(y_test, y_sigmoid_pred, average=None))
sigmoid_f1_sc = (f1_score(y_test, y_sigmoid_pred, average=None))

svcguas_rbf = SVC(kernel='rbf', random_state=0)
svcguas_rbf.fit(x_train, y_train)

y_guas_rbf_pred = svcguas_rbf.predict(x_test)

g_rbf_pr_sc = (precision_score(y_test, y_guas_rbf_pred, average=None))
g_rbf_f1_sc = (f1_score(y_test, y_guas_rbf_pred, average=None))

width = 0.2
iris = ['','               Setosa', '','               Versicolor', '','               Virginica']

fig, axs = plt.subplots(1, 2)

bar1 = np.arange(3)
bar2 = [i+width for i in bar1]
bar3 = [i+width for i in bar2]

axs[0].bar(bar1, poly_pr_sc, width, label='Polynomial')
axs[0].bar(bar2, sigmoid_pr_sc, width, label='Sigmoid')
axs[0].bar(bar3, g_rbf_pr_sc, width, label='Guassian')

axs[1].bar(bar1, poly_f1_sc, width, label='Polynomial')
axs[1].bar(bar2, sigmoid_f1_sc, width, label='Sigmoid')
axs[1].bar(bar3, g_rbf_f1_sc, width, label='Guassian')

axs[0].set_title('Precision Scores of different irises for different kernels')
axs[0].set_xlabel('Iris Type', color='red', fontsize=13)
axs[0].set_ylabel('Precision Score', color='red', fontsize=13)
axs[0].set_xticklabels(iris)
axs[0].tick_params(axis='x', bottom=False)
axs[0].legend(bbox_to_anchor=(-0.1, 1), fontsize='small')

axs[1].set_title('F1-scores of different irises for different kernels')
axs[1].set_xlabel('Iris Type', color='red', fontsize=13)
axs[1].set_ylabel('F1-score', color='red', fontsize=13)
axs[1].set_xticklabels(iris)
axs[1].tick_params(axis='x', bottom=False)
axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.get_current_fig_manager().window.state('zoomed')
plt.show()
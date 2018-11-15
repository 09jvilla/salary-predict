import csv
import pdb
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.metrics import accuracy_score

use_full_data = False
#Use full data
if use_full_data:
	df = pd.read_csv('output/cleaned_data.csv')
else:
	df = pd.read_csv('output/low_salaries_removed.csv')	


df_XforSVM = df.filter(['is_acquired', 'is_public', 'remote_ok', 'NYC', \
	'LA', 'SF', 'SEA', 'senior', 'back_end', 'full_stack', 'front_end'], axis=1)
df_minYforSVM = df.filter(['min_salary'], axis=1 )
df_maxYforSVM = df.filter(['max_salary'], axis=1 )

X_train_min, X_test_min, Y_train_min, Y_test_min = train_test_split( df_XforSVM.values, df_minYforSVM.values.ravel() , test_size=0.05)
X_train_max, X_test_max, Y_train_max, Y_test_max = train_test_split( df_XforSVM.values, df_maxYforSVM.values.ravel() , test_size=0.05)

#currently episilon is at default, 0.1
min_rbf_SVR = SVR(gamma='scale')
min_rbf_SVR.fit(X_train_min, Y_train_min)

#test for max salary
max_rbf_SVR = SVR(gamma='scale')
max_rbf_SVR.fit(X_train_max, Y_train_max)

##look at performance
Y_train_pred_min = min_rbf_SVR.predict(X_train_min)
Y_train_pred_max = max_rbf_SVR.predict(X_train_max)

#plot
plt.scatter(x=Y_train_min, y=Y_train_pred_min)
plt.show()

plt.clf()
plt.scatter(x=Y_train_max, y=Y_train_pred_max)
plt.show()

##Well this currently sucks! Let's try on a smaller subset, where we only look at salaries above 40k
##You'd make at least 40k on minimum wage, so lets only predict things in that range


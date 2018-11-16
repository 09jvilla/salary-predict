import csv
##bread and butter packages
import pdb
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#for tree:
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

#for data visualization & result analysis
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydot, pydotplus
import graphviz

from sklearn import preprocessing
data  = pd.read_csv('output/cleaned_data_better.csv')

# data.shapemax
#build the feature matrix
want = 'max_salary'
# want = 'min_salary'
y = data[want]
# pdb.set_trace()
# y = data[['min_salary', 'max_salary']]
labels = ['NYC', 'LA', 'SF', 'SEA',
'senior', 'back_end', 'full_stack', 'front_end',
'remote_ok', 'avg_size',
'is_acquired', 'is_public']

X = data[labels]

print('=============your data statistics=============')
print(data.describe())
# pdb.set_trace()

# preprocess string labels as decision tree can't take
# string features. Literally NO STRING VALUES ON DATASET
# (later goest through dtype.int32 conversion for comparison; current DT can't
# process categorical variables -- known issue.)

# for label in labels:
# 	print(label, ' is ', X[label].dtype , 'dtype')

le = preprocessing.LabelEncoder()
for label in labels:
	# pdb.set_trace()
	# assuming all dtypes are in dtype == bool.
	# obviously this assumption needs to change as we use complicated features

	# X.loc[:, label] turned out to be a little bitch that returns
	# a copy of the specific column not the original data itself
	# Can't update directly
	# ===>Need Pandas to Pandas copy
	# X.loc[:, label] = le.fit_transform(X[label])
	encoded = le.fit_transform(X[label].astype(bool)) #returns numpy array
	processed_val = pd.DataFrame({'Column1': encoded}) # Numpy -> Pandas
	X.loc[:,label] = processed_val.values #Pandas to Pandas copy by values
	# pdb.set_trace()

	# else:
		# pass
print('um')
# pdb.set_trace()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)

y_test_np = pd.Series.as_matrix(y_test)
# pdb.set_trace()

result = pd.DataFrame({'Actual': y_test_np, 'Predicted': y_pred_reg})

print('')
print('============================== RESULT ===============================')
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred_reg))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred_reg))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)))

## dope graphics
# iris = load_iris()
# reg = reg.fit(iris.data, iris.target)

# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("graph.pdf")

dot_data = tree.export_graphviz(reg, feature_names = labels,
									filled=True,
									special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
# graph = pydotplus.graph_from_dot_data(dot_data)
graph.render("./output/salary-predict", view=True)

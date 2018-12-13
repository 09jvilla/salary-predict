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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# import pydot, pydotplus
import graphviz
from sklearn import preprocessing

#import external functions
from plot_confusion_matrix import plot_confusion_matrix
from perform_random_forest import random_forest_reg, random_forest_clf
from tree_evaluation import reg_eval, clf_eval

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
'remote_ok', 'is_acquired', 'is_public']

X = data[labels]

print('=============your data statistics=============')
print(data.describe())
print('===============================================')
# pdb.set_trace()

##################### uncomment the section below if
################## wish to use label encoder (includes NaN) ##############
#
# # preprocess string labels as decision tree can't take
# # string features. Literally NO STRING VALUES ON DATASET
# # (later goest through dtype.int32 conversion for comparison; current DT can't
# # process categorical variables -- known issue.)
#
# # for label in labels:
# # 	print(label, ' is ', X[label].dtype , 'dtype')
#
# le = preprocessing.LabelEncoder()
# for label in labels:
# 	# pdb.set_trace()
# 	# assuming all dtypes are in dtype == bool.
# 	# obviously this assumption needs to change as we use complicated features
#
# 	# X.loc[:, label] turned out to be a little bitch that returns
# 	# a copy of the specific column not the original data itself
# 	# Can't update directly
# 	# ===>Need Pandas to Pandas copy
# 	# X.loc[:, label] = le.fit_transform(X[label])
# 	encoded = le.fit_transform(X[label].astype(bool)) #returns numpy array
# 	processed_val = pd.DataFrame({'Column1': encoded}) # Numpy -> Pandas
# 	X.loc[:,label] = processed_val.values #Pandas to Pandas copy by values

############Alternative to label encoding #############################
def transform(input):
	# pdb.set_trace()
	# print(input)
	if str(input) == 'True' or '1':
		return 1.0
	elif str(input) == 'False' or '0':
		return 0

for label in labels:
	X[label].apply(transform)

############Alternative to label encoding #############################
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500, random_state=1991)

# pdb.set_trace()

###############################################################################
# Regression Tree #############################################################
print('Single regression tree')
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)

y_pred_reg_train = reg.predict(X_train)
y_pred_reg = reg.predict(X_test)

print('on train set:')
reg_eval(reg, y_train, y_pred_reg_train)

print('on test set:')
reg_eval(reg, y_test.values.ravel(),y_pred_reg )

# # # dope graphics
# # dot_data = StringIO()t_file=dot_data)
# # graph = pydot.graph_from_dot_data(dot_data.getvalue())
# # graph.write_pdf("graph.pdf")

# dot_data = tree.export_graphviz(reg, feature_names = labels,
# 									filled=True,
# 									special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.format = 'png'
# # graph = pydotplus.graph_from_dot_data(dot_data)
# # graph.render("./output/salary-predict", view=True) #plt.show

###############################################################################
# Random Forest Regressor ####################################################
#
print("Baseline case - Regressor: ")
random_forest_reg(X_train, y_train, X_test, y_test, tune=False)
print("Optimized case - Regressor: ")
random_forest_reg(X_train, y_train, X_test, y_test, num_estimator=250, min_samples_split =0.01,
 					max_depth = 20, max_features = 4, random=1,tune=False)


# def random_forest_reg(x_train, y_train, x_test, y_test,estimator,state)
# pdb.set_trace()
#
###############################################################################
#Classification Tree #########################################################
# data bucketing
# min_salary = y.min()
# max_salary = y.max()
# num_buckets =200
# increment = (max_salary - min_salary ) * 1.0 /num_buckets
# quick_bucket_cutoff_lines = [min_salary + i * increment for i in range (0,num_buckets)]
#
# #dumb way of bucketing but whatever it works
# new_y = np.zeros(shape=(np.shape(y)))
# for i in range(0,len(y)):
# 	for j in range(0,len(quick_bucket_cutoff_lines)):
# 		if y[i] > quick_bucket_cutoff_lines[j] :
# 			new_y[i] = j
# 			continue
#
# # pdb.set_trace()
# y_clf = pd.DataFrame({'bucket_y': new_y})
# X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size = 0.1, random_state = 0)
#
# clf = DecisionTreeClassifier()
# clf.fit(X_train, y_train)
# y_pred_clf = clf.predict(X_test)
# y_pred_train_clf = clf.predict(X_train)
# # y_test_np = pd.Series.as_matrix(y_test)
# y_test_np = y_test.values.ravel()
#
# # # pdb.set_trace()
# # result = pd.DataFrame({'Actual': y_test_np, 'Predicted': y_pred_reg})
# result = pd.DataFrame({'Actual (clf)': y_test_np, 'Predicted (clf)': y_pred_clf})
#
# print('====================SINGLE CLASSIFIER===================')
# print("on train set::")
# clf_eval(clf, y_train.values.ravel(), y_pred_train_clf)
# print("on test set::")
# clf_eval(clf, y_test.values.ravel(), y_pred_clf.ravel())
# np.set_printoptions(precision=2)
#
# print("Optimized case - Classifier: ")
# random_forest_clf(X_train, y_train, X_test, y_test, num_estimator=250, min_samples_split =0.01,
#  					max_depth = 10, max_features = 4, random=1,tune=False)
# # cnf_matrix = confusion_matrix(y_test_np, y_test_np)
# # # pdb.set_trace()
# # plt.figure()
# # plt.matshow(cnf_matrix)
# # plt.title('Confusion matrix of the classifier')
# # plt.colorbar()
# # filename = './output/confusion_matrix.png'
# # plt.savefig(filename)
# # plt.show()
# # pdb.set_trace()
# # plot_confusion_matrix(cnf_matrix, quick_bucket_cutoff_lines, title='Confusion matrix, without normalization')
# # plt.figure()
# # plot_confusion_matrix(cnf_matrix, quick_bucket_cutoff_lines,normalize = True, title='Confusion matrix, without normalization')
# # plt.show()

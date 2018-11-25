#tree_evaluation.py
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pdb
from sklearn import tree
from plot_confusion_matrix import plot_confusion_matrix

##can optimize this to numpy but it works so i'm going to leave it
def mask(df, key, value):
	# print(df[key] > value)
	return df[np.abs(df[key]) > value]


def reg_eval(reg_tree, y_test, y_pred):
	rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
	# print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))
	# print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))
	print('Root Mean Squared Error: ', rmse)
	print('')
	# print('============================== RESULT ===============================')
	#########finding assholes with biggest errors, fyi
	diff = pd.DataFrame({'diff' : (y_test - y_pred)})
	diff = np.array((y_test - y_pred))
	# pd.DataFrame.mask = mask
	# big_errors = diff.mask('diff', 80000) # pick out the ones with errors bigger than 80000
	# print('your big errors culprits:')
	# print(big_errors)

	# for y_test
	plt.figure()
	plt.scatter(range(0,np.shape(diff)[0]),diff, s=1)
	plt.xlabel('Distribution of y- y_pred Error')
	plt.ylabel('prediction error')
	# filename = './output/regressor_result.png'
	# plt.savefig(filename)
	plt.show()
	# # big_errors.index.tolist()
	#

	# # # dope graphics -- NEED LABELS
	# dot_data = tree.export_graphviz(reg, feature_names = labels,
	# 									filled=True,
	# 									special_characters=True)
	# graph = graphviz.Source(dot_data)
	# graph.format = 'png'
	# # graph = pydotplus.graph_from_dot_data(dot_data)
	# # graph.render("./output/salary-predict", view=True) #plt.show

	return rmse

def clf_eval(clf_tree, y_test, y_pred):
	acc_score =accuracy_score(y_test, y_pred)
	# print('accuracy score (train set): {}'.format(accuracy_score(y_train, y_pred_train_clf)))
	print('accuracy score (test set): {}'.format(acc_score))
	# cnf_matrix = confusion_matrix(y_test_np, y_test_np)
	cnf_matrix = confusion_matrix(y_test, y_pred)
	# pdb.set_trace()
	plt.figure()
	plt.matshow(cnf_matrix)
	plt.title('Confusion matrix of the classifier')
	plt.colorbar()
	filename = './output/confusion_matrix.png'
	plt.savefig(filename)

	return acc_score

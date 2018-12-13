import csv
##bread and butter packages
import pdb
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#for tree:
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

#for data visualization & result analysis
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# import pydot, pydotplus
import graphviz
from plot_confusion_matrix import plot_confusion_matrix
from tree_evaluation import reg_eval, clf_eval
from rf_tune import *

def random_forest_reg(x_train, y_train, x_test, y_test,
                        num_estimator=20, min_samples_split =2, max_depth = None,
                        max_features = 'auto', random=1,tune=False):
    reg = RandomForestRegressor(n_estimators = num_estimator, random_state= random,
                                max_depth = max_depth, min_samples_split = min_samples_split,
                                max_features = max_features)

    # print('=================Baseline performance===============')
    reg.fit(x_train, y_train)

    print("on train set::")
    y_pred_train = reg.predict(x_train)
    reg_eval(reg, y_train.values.ravel(), y_pred_train)


    print("on test set::")
    y_pred_reg = reg.predict(x_test)
    y_test_reg = y_test.values.ravel()
    # pdb.set_trace()
    # result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_reg})

    reg_eval(reg, y_test_reg, y_pred_reg)
    if tune == True:
        print('Tuning...')
        find_n_estimators('reg', reg, x_train, x_test, y_train, y_test)
        find_max_depth('reg', reg, x_train, x_test, y_train, y_test)
        find_min_sample_split('reg', reg, x_train, x_test, y_train, y_test)
        find_max_features('reg', reg, x_train, x_test, y_train, y_test)



    # dot_data = tree.export_graphviz(reg, feature_names = labels,
    # 									filled=True,
    # 									special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.format = 'png'
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.render("./output/salary-predict", view=True) #plt.show

    # # dope graphics
    # dot_data = StringIO()t_file=dot_data)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("graph.pdf")

    # pdb.set_trace()

# pdb.set_trace()
def random_forest_clf(x_train, y_train, x_test, y_test,
                        num_estimator=20, min_samples_split =2, max_depth = None,
                        max_features = 'auto', random=1,tune=False):
    clf = RandomForestClassifier(n_estimators = num_estimator, random_state= random,
                                max_depth = max_depth, min_samples_split = min_samples_split,
                                max_features = max_features)

    clf.fit(x_train, y_train)
    # print('============BASELINE=======')
    print("on train set::")
    y_pred_train = clf.predict(x_train)
    clf_eval(clf, y_train.values.ravel(), y_pred_train.ravel())

    print("on test set::")
    y_pred_clf = clf.predict(x_test)
    # y_test_clf = y_test.values.ravel()
    clf_eval(clf, y_test.values.ravel(), y_pred_clf.ravel())
    # pdb.set_trace()
    # result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_clf})
    if tune == True:
        print('Tuning...')
        find_n_estimators('clf', clf, x_train, x_test, y_train, y_test)
        find_max_depth('clf', clf, x_train, x_test, y_train, y_test)
        find_min_sample_split('clf', clf, x_train, x_test, y_train, y_test)
        find_max_features('clf', clf, x_train, x_test, y_train, y_test)

    # pass

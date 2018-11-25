##rf_tune.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tree_evaluation import reg_eval,clf_eval

def find_n_estimators(treetype, tree, x_train, x_test, y_train, y_test):
    print('finding best number of estimators....')
    print('-----------------------------------------')
    # estimators = np.logspace(1, 4, 10).astype(int)
    estimators = np.linspace(1,2000,10).astype(int)

    rmse_train= []
    rmse_test= []

    if treetype == 'reg':
        maketree = RandomForestRegressor
    elif treetype == 'clf':
        maketree = RandomForestClassifier
    else:
        print('wrong tree type')
        exit()

    for est in estimators:
        rf = maketree(n_estimators = est)
        rf.fit(x_train, y_train)

        y_pred_train = rf.predict(x_train)
        rmse_train.append(reg_eval(rf, y_train, y_pred_train))

        y_pred_test = rf.predict(x_test)
        rmse_test.append(reg_eval(rf, y_test, y_pred_test))

    plt.figure()
    x = range(0,len(estimators))
    plt.plot(estimators,rmse_train, 'b', label='train rmse')
    plt.plot(estimators,rmse_test, 'r', label='test rmse')
    plt.legend()
    plt.xlabel('Number of estimators')
    plt.ylabel('RMSE error')
    # plt.show()
    filename = './output/rf_tuning/n_estimators_opt_'+ treetype+'.png'
    plt.savefig(filename)

def find_max_depth(treetype, tree, x_train, x_test, y_train, y_test):
    print('finding maximum depth....')
    print('-----------------------------------------')
    depth_list = np.linspace(1,50,25).astype(int)

    rmse_train= []
    rmse_test= []

    if treetype == 'reg':
        maketree = RandomForestRegressor
    elif treetype == 'clf':
        maketree = RandomForestClassifier
    else:
        print('wrong tree type')
        exit()

    for depth in depth_list:
        rf = maketree(max_depth = depth)
        rf.fit(x_train, y_train)

        y_pred_train = rf.predict(x_train)
        rmse_train.append(reg_eval(rf, y_train, y_pred_train))

        y_pred_test = rf.predict(x_test)
        rmse_test.append(reg_eval(rf, y_test, y_pred_test))

    plt.figure()
    plt.plot(depth_list,rmse_train, 'b', label='train rmse')
    plt.plot(depth_list,rmse_test, 'r', label='test rmse')
    plt.legend()
    plt.xlabel('Maximum Depth')
    plt.ylabel('RMSE error')
    # plt.show()
    filename = './output/rf_tuning/Maximum_depth_opt_'+ treetype+'.png'
    plt.savefig(filename)

def find_min_sample_split(treetype, tree, x_train, x_test, y_train, y_test):
    print('finding minimum sample split...')
    print('-----------------------------------------')
    how_many_split = np.linspace(0.1, 1.0, 10)

    rmse_train= []
    rmse_test= []

    if treetype == 'reg':
        maketree = RandomForestRegressor
    elif treetype == 'clf':
        maketree = RandomForestClassifier
    else:
        print('ERROR:: wrong tree type')
        exit()

    for split in how_many_split:
        rf = maketree(min_samples_split = split)
        rf.fit(x_train, y_train)

        y_pred_train = rf.predict(x_train)
        rmse_train.append(reg_eval(rf, y_train, y_pred_train))

        y_pred_test = rf.predict(x_test)
        rmse_test.append(reg_eval(rf, y_test, y_pred_test))

    plt.figure()
    plt.plot(how_many_split,rmse_train, 'b', label='train rmse')
    plt.plot(how_many_split,rmse_test, 'r', label='test rmse')
    plt.legend()
    plt.xlabel('Minimum Sample Split')
    plt.ylabel('RMSE error')
    # plt.show()
    filename = './output/rf_tuning/min_split_'+ treetype+'.png'
    plt.savefig(filename)

def find_max_features(treetype, tree, x_train, x_test, y_train, y_test):
    print('finding maximum features...')
    print('-----------------------------------------')
    how_many_features = np.linspace(1,np.shape(x_train)[1],np.shape(x_train)[1]).astype(int)

    rmse_train= []
    rmse_test= []

    if treetype == 'reg':
        maketree = RandomForestRegressor
    elif treetype == 'clf':
        maketree = RandomForestClassifier
    else:
        print('ERROR:: wrong tree type')
        exit()

    for ftr in how_many_features:
        rf = maketree(max_features = ftr)
        rf.fit(x_train, y_train)

        y_pred_train = rf.predict(x_train)
        rmse_train.append(reg_eval(rf, y_train, y_pred_train))

        y_pred_test = rf.predict(x_test)
        rmse_test.append(reg_eval(rf, y_test, y_pred_test))

    plt.figure()
    plt.plot(how_many_features,rmse_train, 'b', label='train rmse')
    plt.plot(how_many_features,rmse_test, 'r', label='test rmse')
    plt.legend()
    plt.xlabel('Maximum Number of Features')
    plt.ylabel('RMSE error')
    # plt.show()
    filename = './output/rf_tuning/max_features_'+ treetype+'.png'
    plt.savefig(filename)

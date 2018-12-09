import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#tree_runxgb.py
def runxgb(x,y,ft):
    x_train, x_test, y_train, y_test = train_test_split (x,y, test_size = 0.1)

    xgb_train = xgb.DMatrix(x_train, label = y_train, feature_names = ft)
    xgb_test = xgb.DMatrix(x_test, label=y_test, feature_names=ft)

    params = {'max_depth': 120, 'eta': .8, 'silent': 0, 'objective': 'reg:linear', 'tree_method': 'exact', 'eval_metric': 'rmse'}

    plst = list(params.items())
    # # watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    # num_rounds = 100
    start_time = time.time()
    model = xgb.train(plst, xgb_train)
    #
    print('took', time.time() - start_time)

    # pdb.set_trace()
    print('...training...')
    start_time = time.time()
    # model = xgb.XGBRegressor()
    # model.fit(x_train, y_train)

    print('(', time.time() - start_time, 's)')
    print('--------------------------------------')
    print('Result tree:', model)
    # print('--- train results')
    y_pred_train = model.predict(xgb_train)
    # y_pred_train= model.predict(x_train)
    rmse_train =  np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
    print('Train RMSE == ', rmse_train)
    y_pred = model.predict(xgb_test)
    # y_pred= model.predict(x_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('Test RMSE == ', rmse)

    diff_test = np.array((y_test - y_pred))
    diff_train = np.array((y_train - y_pred_train))

    fig2 = plt.figure()
    plt.scatter(range(0,np.shape(diff_train)[0]),diff_train, s=1)
    plt.title('RMSE error on train set')
    plt.xlabel(' y- y_pred Error')
    plt.ylabel('prediction error')
    filename = './output/xgb_rmse_train.png'
    plt.savefig(filename)


    # plt.show()
    # plt.close()

    fig3 = plt.figure()
    plt.scatter(range(0,np.shape(diff_test)[0]),diff_test, s=1)
    plt.title('RMSE error on test set')
    plt.xlabel(' y- y_pred Error')
    plt.ylabel('prediction error')
    filename = './output/xgb_rmse_test.png'
    plt.savefig(filename)


    # pdb.set_trace()
    # fig1,ax = plt.figure()
    # plt.figure(2)
    # xgb.plot_tree(model, num_trees=0,rankdir='LR')
    #
    # plt.figure(3)
    # xgb.plot_tree(model, num_trees=1,rankdir='LR')
    # plt.figure(4)
    # xgb.plot_tree(model, num_trees=2,rankdir='LR')
    # plt.figure(5)
    # xgb.plot_tree(model, num_trees=3,rankdir='LR')
    # plt.figure(6)
    # xgb.plot_tree(model, num_trees=4,rankdir='LR')

        # plt.show()

    fig4 = plt.figure()
    sns.set(color_codes=True)
    ls_diff = list(diff_train)
    ax = sns.distplot(ls_diff, kde=False, rug=True)
    ax.set_title('RMSE error distribution on train set')
    filename = './output/xgb_rmse_distribution_train.png'
    plt.savefig(filename)
    # plt.show()

    fig5 = plt.figure()
    sns.set(color_codes=True)
    ls_diff = list(diff_test)
    ax = sns.distplot(ls_diff, kde=False, rug=True)
    ax.set_title('RMSE error distribution on test set')
    filename = './output/xgb_rmse_distribution_test.png'
    plt.show()

    # pdb.set_trace()

    return rmse

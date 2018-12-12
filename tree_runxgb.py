import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt

def save_result_to_file(filename, dic, rmse_train, rmse_test):
    f = open(filename,'w')
    f.write(filename+'\n')
    f.write('\n')
    f.write(str(dic))
    f.write('\n')

    f.write('train RMSE: '+ str(rmse_train)+'\n')
    f.write('test RMSE: '+str(rmse_test)+'\n')
    f.close()


def quickplot(y_train_rmse, y_test_rmse):
### for cv
    ax = plt.figure()
    plt.plot(y_train_rmse, 'r-', label='RMSE on train set', linewidth=2)

    # plt.hold(True)
    plt.plot(y_test_rmse, 'bo', label='RMSE on test set')

    ax.legend()

    filename = './output/xgb_cv_result' + time.strftime("%m%d_%H%M") + '.png'
    plt.savefig(filename)
    # plt.show()


#tree_runxgb.py
def runxgb(x,y,ft):
    x_train, x_test, y_train, y_test = train_test_split (x,y, test_size = 0.1)

    xgb_train = xgb.DMatrix(x_train, label = y_train, feature_names = ft)
    xgb_train_tester = xgb.DMatrix(x_train, feature_names = ft)
    xgb_test = xgb.DMatrix(x_test, feature_names=ft)

    # params = {'max_depth': 5, 'gamma': 0.1, 'seed' : 27,
    #         'min_child_weight' : 1, 'subsample': 1, 'scale_pos_weight' : 1,
    #         'lambda': .2, 'eta': .3, 'colsample_bytree': 0.3,
    #         'silent': 0, 'objective': 'reg:linear',
    #         'tree_method': 'exact', 'eval_metric': 'rmse'}

    params = {'max_depth': 5, 'gamma': 0.1, 'seed' : 27,
                'min_child_weight' : 1, 'subsample': .8, 'scale_pos_weight' : 1,
                'lambda': .2, 'eta': .3, 'colsample_bytree': 0.3,
                'silent': 1, 'objective': 'reg:linear',
                'tree_method': 'exact', 'eval_metric': 'rmse'}


    # params = {'max_depth': 15, 'gamma': 0.8, 'seed' : 27,
    #             'min_child_weight' : 1, 'subsample': .8,
    #             'scale_pos_weight' : 1, 'lambda': .5, 'eta': .15,
    #             'colsample_bytree': 0.8, 'silent': 1, 'objective': 'reg:linear',
    #             'tree_method': 'exact', 'eval_metric': 'rmse'}


    # params = {'max_depth': 120, 'eta': .8, 'silent': 1, 'objective': 'reg:linear', 'tree_method': 'exact', 'eval_metric': 'rmse'}


    plst = list(params.items())

    # # watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    # num_rounds = 100
    start_time = time.time()
    model = xgb.train(plst, xgb_train)
    # model = xgb.XGBRegressor(plist)
    # model.fit(x_train, y_train)
    #
    print('took', time.time() - start_time)

    # pdb.set_trace()
    print('...training...')
    start_time = time.time()

    print('(', time.time() - start_time, 's)')
    print('--------------------------------------')
    print('Result tree:', model)
    # print('--- train results')
    y_pred_train = model.predict(xgb_train_tester)
    # y_pred_train= model.predict(x_train)
    rmse_train =  np.sqrt(metrics.mean_squared_error(y_train, y_pred_train))
    mse_train = metrics.mean_squared_error(y_train, y_pred_train)
    print('Train RMSE == ', rmse_train)
    print('Train MSE == ', mse_train)
    y_pred = model.predict(xgb_test)
    # y_pred= model.predict(x_test)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    mse = metrics.mean_squared_error(y_test, y_pred)
    print('Test RMSE == ', rmse)
    print('Test MSE == ', mse)

    filename = './output/xgb_parm_rmse' + time.strftime("%m%d_%H%M") + '.txt'
    save_result_to_file(filename, params, rmse_train, rmse)

    diff_train = np.array((y_train - y_pred_train))
    diff_test = np.array((y_test - y_pred))

    # pdb.set_trace()
    # cv_results = xgb.cv(dtrain=xgb_train, params = params, nfold = 4, num_boost_round=100, early_stopping_rounds = 50, metrics='rmse', seed=123)
    #
    # train_val = cv_results['train-rmse-mean'].tail(1)
    # test_val = cv_results['test-rmse-mean'].tail(1)
    # f=open(filename, "a+")
    # f.write('CV result, train RMSE: '+ str(train_val)+'\n')
    # f.write('CV result, test RMSE: '+str(test_val)+'\n')

    # y2 = list(cv_results['test-rmse-mean']); y1 = list(cv_results['train-rmse-mean'])
    # quickplot(y1, y2)

    return rmse, model, [diff_train, diff_test], filename

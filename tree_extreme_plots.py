#tree_extreme_plots.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time
import pdb

def plotty_plots(diff, model, parm_file):
    diff_train, diff_test = diff

    fig2 = plt.figure()
    plt.scatter(range(0,np.shape(diff_train)[0]),diff_train, s=1)
    plt.title('RMSE error on train set')
    plt.xlabel(' y- y_pred Error')
    plt.ylabel('prediction error')
    filename = "sparse_weight_" + time.strftime("%m%d_%H%M") + ".pkl"
    filename = './output/xgb_rmse_train_' + time.strftime("%m%d_%H%M") + '.png'
    plt.savefig(filename)


    # plt.show()
    # plt.close()

    fig3 = plt.figure()
    plt.scatter(range(0,np.shape(diff_test)[0]),diff_test, s=1)
    plt.title('RMSE error on test set')
    plt.xlabel(' y- y_pred Error')
    plt.ylabel('prediction error')
    filename = './output/xgb_rmse_test_' + time.strftime("%m%d_%H%M") + '.png'
    plt.savefig(filename)


    #### tree plots -- will slow down your system if max depth is too big

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
    fig, ax = plt.subplots(figsize=(100, 100))
    ax = xgb.plot_tree(model, num_trees=5,rankdir='LR', ax = ax)
    filename = './output/xgb_tree_plot' + time.strftime("%m%d_%H%M") + '.png'


    plt.savefig(filename)

    # plt.show()

    fig4 = plt.figure()
    sns.set(color_codes=True)
    ls_diff = list(diff_train)
    ax = sns.distplot(ls_diff,  kde=False, rug=True)
    ax.set_title('RMSE error distribution on train set')
    filename = './output/xgb_rmse_distribution_train_' + time.strftime("%m%d_%H%M") + '.png'
    plt.savefig(filename)


    fig5 = plt.figure()
    sns.set(color_codes=True)
    ls_diff = list(diff_test)
    ax = sns.distplot(ls_diff, kde=False, rug=True)
    ax.set_title('RMSE error distribution on test set')
    filename = './output/xgb_rmse_distribution_test_' + time.strftime("%m%d_%H%M") + '.png'
    plt.savefig(filename)
    # plt.show()

    fig6 = plt.figure()
    xgb.plot_importance(model, max_num_features=10)
    plt.rcParams['figure.figsize'] = [10, 10]
    filename = './output/xgb_importance_plot' + time.strftime("%m%d_%H%M") + '.png'
    plt.savefig(filename)
    # plt.show()



    fig7 = plt.figure()
    feature_important = model.get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())

    importance = pd.DataFrame(data=values, index=keys, columns=["score"])
    # pdb.set_trace()
    importance.nlargest(20, columns='score').plot(kind='barh')
    most_important = list(importance.nlargest(20, columns='score').index)

    f=open(parm_file, "a+")
    f.write('\n')
    f.write('top 20 features:: '+'\n')
    for ff in most_important:
        f.write(ff)
        f.write('\n')
    f.close()

    filename = './output/xgb_importance_plot_by score' + time.strftime("%m%d_%H%M") + '.png'
    plt.savefig(filename)




    # plt.show()

    # pdb.set_trace()

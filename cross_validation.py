##cross_validation.py
from sklearn.model_selection import KFold,cross_val_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from sklearn.grid_search import GridSearchCV

def do_cross_validate(x, y):
    # k_fold = KFold(n_splits = num_folds)
    # cross_val_score(learner, x_train, y_train, scoring=how_to_score)

    train_gs_X, test_gs_X, train_gs_Y, test_gs_Y = train_test_split(x,y,train_size=0.1 )
    gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
                  'max_depth': [4, 6, 8],
                  'min_samples_leaf': [20, 50,100,150],
                  }
    print(gb_grid_params)

    gb_gs = GradientBoostingRegressor(n_estimators = 600)

    reg = grid_search.GridSearchCV(gb_gs,
                                   gb_grid_params,
                                   cv=2,
                                   scoring='rmse',
                                   verbose = 3
                                   );
    reg.fit(train_gs_X, train_gs_Y);

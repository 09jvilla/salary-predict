##cross_validation.py
from sklearn.model_selection import KFold,cross_val_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

def do_cross_validate(num_folds,learner,x_train, y_train, how_to_score):
    k_fold = KFold(n_splits = num_folds)
    cross_val_score(learner, x_train, y_train, scoring=how_to_score)

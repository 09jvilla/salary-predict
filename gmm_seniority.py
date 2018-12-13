from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from statistics import mean
import sys
import operator
import os
import matplotlib.pyplot as plt
import altair as alt

MAX_FEATS = 10

full_data = pd.read_csv('./output/cleaned_data_better.csv')
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']

all_columns = list(full_data.columns)

def plot_gmm_preds(x, z, predictions):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM'.format('Predictions' if predictions else 'Actual'))
    plt.xlabel('max_salary')
    plt.ylabel('min_salary')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'gmm_4class_{}.pdf'.format('predictions' if predictions else 'actual')
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)

# features = ['is_acquired', 'is_public', 'remote_ok', 'NYC', \
# 	'LA', 'SF', 'SEA', 'senior', 'back_end', 'full_stack', 'front_end','total_investments']

num_distros = 0
y_cols = []
for column in all_columns:
    if "gmm_" in column:
        num_distros += 1
        y_cols.append(column)
# print(num_distros)
gmm = GaussianMixture(n_components=num_distros)
features = ["max_salary", "min_salary"]

X = full_data.filter(features)
# for feature in features:
#     max_mean = X[feature].mean()
#     max_std = X[feature].std()

#     X[feature] = (X[feature] - max_mean) / max_std

y = full_data.filter(y_cols).values
y_indexed = [y_i.tolist().index(True) for y_i in y]
X_train, X_test, y_train, y_test = train_test_split(X.values, y_indexed, test_size=0.15, random_state=1)

gmm.fit(X_train)
y_pred = gmm.predict(X_test)
plot_gmm_preds(X_test, y_pred, True)
plot_gmm_preds(X_test, y_test, False)
# plot_gmm_preds(X_train, y_train, False)
# for idx, pred in enumerate(y_pred):
#     print(f"PREDICTION: {pred}\nTRUE VALUE:{y_test[idx]}")
#     if idx > 10:
#         break

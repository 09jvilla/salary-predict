import xgboost as xgb
import pandas as pd
import numpy as np
import pdb
import nltk
from nltk import *

from tree_brute_force_tokenization import tokenize
from tree_runxgb import runxgb
from tree_extreme_plots import plotty_plots
"""
specify what you'd like to train the data on:
"""
WANT = 'max_salary'
# WANT = 'min_salary'

NOT_WANT = 'min_salary'
# NOT_WANT = 'max_salary'

data = pd.read_csv('./output/processed_data.csv')
data = data.drop(['Unnamed: 0', NOT_WANT, 'coolness_reasons', 'job_description', 'company_description'], axis=1)
# data = data.fillna('N/A')
print('XGBoost for ', WANT)

print('dropped company description')


y = data[WANT]
raw_x = data.drop([WANT], axis=1)
labels = list(raw_x)

# additional data processing

x, feature_names = tokenize(raw_x)

rmse, model, diff, filename = runxgb(x,y,feature_names)
plotty_plots(diff, model, filename)
# dtrain = xgb.DMatrix('./data')

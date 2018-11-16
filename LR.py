from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd


data = pd.read_csv('cleaned_data_better.csv')

features = ['is_acquired', 'is_public', 'remote_ok', 'NYC', \
	'LA', 'SF', 'SEA', 'senior', 'back_end', 'full_stack', 'front_end']


X_data = data.filter(features)
y_max_data = data.filter(["max_salary"])
y_min_data = data.filter(["min_salary"])

X_train, X_test, y_train, y_test = train_test_split(X_data.values, y_max_data.values, test_size=0.1, random_state=1)

model = linear_model.LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv = 3)
print(scores)
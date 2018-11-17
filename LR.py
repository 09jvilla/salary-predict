from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from statistics import mean
import sys
import operator
import matplotlib.pyplot as plt

MAX_FEATS = 10

full_data = pd.read_csv('cleaned_data_better.csv')

all_columns = list(full_data.columns)
features = ['is_acquired', 'is_public', 'remote_ok', 'NYC', \
	'LA', 'SF', 'SEA', 'senior', 'back_end', 'full_stack', 'front_end','total_investments']
for column in all_columns:
	if "seniority_" in column or "skills_" in column:
		features.append(column)

print(features)
f = open("output/feature_losses.txt","w")

output = ""
model_features = []
train_scores = []
test_scores = []
for iteration in range(MAX_FEATS):
	print(f"{iteration+1} features:\n",file=f)
	scores = {}
	for column in features:
		model_feats_i = model_features + [column]
		X_i = full_data.filter(model_feats_i)
		y_max_data = full_data.filter(["max_salary"])
		X_train, X_test, y_train, y_test = train_test_split(X_i.values, y_max_data.values, test_size=0.05, random_state=1)
		model = linear_model.LinearRegression()
		cv_score = mean(map(float,cross_val_score(model, X_train, y_train, cv = 10)))
		scores[column] = cv_score
		print(f"Score for feature set: {','.join(model_feats_i)}:\t {cv_score}", file=f)
	max_key = max(scores.items(), key=operator.itemgetter(1))[0]
	model_features.append(max_key)
	x_run = full_data.filter(model_features)
	X_train, X_test, y_train, y_test = train_test_split(x_run.values, y_max_data.values, test_size=0.05, random_state=1)
	model_run = linear_model.LinearRegression()
	model_run.fit(X_train,y_train)
	train_scores.append(scores[max_key])
	test_scores.append(model_run.score(X_test,y_test))
	print(f"Max Score feature set: {','.join(model_features)}\t {scores[max_key]}\n\n", file=f)
	features.remove(max_key)

fig, ax = plt.subplots()
x = range(MAX_FEATS)
ax.plot(x, train_scores)
ax.plot(x, test_scores)
fig.savefig("output/LR_score.png")
plt.show()

print(f"TRAIN SCORES: {train_scores}")
print(f"TEST SCORES: {test_scores}")

# X_data = data.filter(features)
# y_max_data = data.filter(["max_salary"])
# y_min_data = data.filter(["min_salary"])




# print(scores)
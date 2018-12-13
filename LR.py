from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pandas as pd
from statistics import mean
import sys
import operator
import matplotlib.pyplot as plt
import altair as alt

MAX_FEATS = 10

full_data = pd.read_csv('cleaned_data_better.csv')

all_columns = list(full_data.columns)
features = ['is_acquired', 'is_public', 'remote_ok', 'NYC', \
	'LA', 'SF', 'SEA', 'senior', 'back_end', 'full_stack', 'front_end','total_investments']
for column in all_columns:
	if "seniority_" in column or "skills_" in column:
		features.append(column)

# print(features)


max_flag = False
y_string = "max_salary" if max_flag else "min_salary"
y_data = full_data.filter([y_string])

cv_scorers_types ={
	"Mean Abs Error" :'neg_mean_absolute_error', 
	"Mean Sq. Error":'neg_mean_squared_error', 
	"Median Abs Error":'neg_median_absolute_error',
	"R-squared": 'r2'
}

loss_functions = {
	"Mean Abs Error" : mean_absolute_error, 
	"Mean Sq. Error": mean_squared_error, 
	"Median Abs Error": median_absolute_error,
	"R-squared": r2_score
}
final_scores = {}
for score_name, cv_scorer_type in cv_scorers_types.items():
	print(f"Running scores for {score_name}")
	loss_func = loss_functions[score_name]

	f = open(f"output/feature_losses_{cv_scorer_type}_{y_string}.txt","w")
	feat_copy = features.copy()
	output = ""
	model_features = []
	train_scores = []
	test_scores = []
	for iteration in range(MAX_FEATS):
		print(f"{iteration+1} features:\n",file=f)
		scores = {}
		for column in feat_copy:
			model_feats_i = model_features + [column]
			X_i = full_data.filter(model_feats_i)
			X_train, X_test, y_train, y_test = train_test_split(X_i.values, y_data.values, test_size=500, random_state=1991)
			model = linear_model.LinearRegression()
			cv_score = mean(map(float,cross_val_score(model, X_train, y_train, scoring=cv_scorer_type, cv = 10)))
			scores[column] = cv_score
			print(f"Score for feature set: {','.join(model_feats_i)}:\t {cv_score}", file=f)
		max_key = max(scores.items(), key=operator.itemgetter(1))[0]
		model_features.append(max_key)
		x_run = full_data.filter(model_features)
		X_train, X_test, y_train, y_test = train_test_split(x_run.values, y_data.values, test_size=500, random_state=1991)
		model_run = linear_model.LinearRegression()
		model_run.fit(X_train,y_train)
		train_score_i = abs(scores[max_key])**(1/2) if score_name == "Mean Sq. Error" else abs(scores[max_key])
		train_scores.append(train_score_i)
		# y_pred_train = model_run.predict(X_train)
		# train_scores.append(loss_func(y_train, y_pred_train))
		y_pred_test = model_run.predict(X_test)
		test_score_i = loss_func(y_test,y_pred_test)
		test_score_i = test_score_i**(1/2) if score_name == "Mean Sq. Error" else test_score_i
		test_scores.append(test_score_i)
		print(f"Max Score feature set: {','.join(model_features)}\t {scores[max_key]}\n\n", file=f)
		feat_copy.remove(max_key)


	# Plot performance
	final_scores[score_name] = {"train" : train_scores[-1], "test" : test_scores[-1]}
	title = "Maximum" if max_flag else "Minimum"
	title += f" Salary Prediction Performance - {score_name}"
	fig, ax = plt.subplots()
	x = range(MAX_FEATS)
	ax.plot(x, train_scores)
	ax.plot(x, test_scores)
	ax.legend([f"Train {score_name}",f"Test {score_name}"])
	plt.xlabel("Number of Features")
	plt.ylabel(f"{score_name}")
	plt.title(title)
	fig.savefig(f"output/LR_{cv_scorer_type}_{y_string}.png")
	# plt.show()

with open(f"output/results_{y_string}.txt","w") as results_fp:
	for score, scores in final_scores.items():
		print(score, file=results_fp)
		for type_set, final_score in scores.items():
			print(f"\t{type_set}: {final_score}", file=results_fp)

# print(f"TRAIN SCORES: {train_scores}")
# print(f"TEST SCORES: {test_scores}")

# X_data = data.filter(features)
# y_max_data = data.filter(["max_salary"])
# y_min_data = data.filter(["min_salary"])




# print(scores)
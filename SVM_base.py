import csv
import pdb
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/cleaned_data.csv')

#figure out salary buckets
min_sal_sorted_df = df.sort_values(by=['min_salary'])
max_sal_sorted_df = df.sort_values(by=['max_salary'])

plt.plot(min_sal_sorted_df['min_salary'].tolist(), 'bo')
plt.plot(max_sal_sorted_df['max_salary'].tolist(), 'ro')

plt.savefig("salary_scurve.png")

#Ok lets do some bucketing
min_sal_buckets = range(50000,210000,10000)
num_buckets = len(min_sal_buckets)+1

def bucket_to_range(bucket):
	if bucket == 0:
		bucket_str = "<" + str(min_sal_buckets[0])
		return bucket_str

	if bucket == len(min_sal_buckets):
		bucket_str = ">" + str(min_sal_buckets[-1])
		return bucket_str

	bottom = min_sal_buckets[bucket-1]
	top = min_sal_buckets[bucket]

	return str(bottom) + "-" + str(top)

def min_class_bucket(row):
	proper_bucket = -1
	
	if row.min_salary < min_sal_buckets[0]:
		return 0

	elif row.min_salary >= min_sal_buckets[-1]:
		return len(min_sal_buckets)

	for i in range(1,len(min_sal_buckets)):
		
		if row.min_salary >= min_sal_buckets[i-1] and row.min_salary < min_sal_buckets[i]:
			return i


df['salary_bucket'] = df.apply(min_class_bucket, axis=1)
df = df.astype( {'salary_bucket':int } )

df.to_csv("data/train_data_with_salary_buckets.csv")

df_XforSVM = df.filter(['is_acquired', 'is_public', 'remote_ok', 'NYC', \
	'LA', 'SF', 'SEA', 'senior', 'back_end', 'full_stack', 'front_end'], axis=1)
df_YforSVM = df.filter(['salary_bucket'], axis=1 )

clf = LinearSVC(random_state=0, tol=1e-5, max_iter=10000)

X_train, X_test, Y_train, Y_test = train_test_split( df_XforSVM.values, df_YforSVM.values.ravel() , test_size=0.05)

clf.fit(X_train, Y_train)


#Let's also try on a different type of svm with rbf kernel
rbf_clf = SVC(gamma = 'scale')
rbf_clf.fit(X_train,Y_train)


##Let's try testing this:
Y_test_pred = clf.predict(X_test)
Y_test_rbf = rbf_clf.predict(X_test)
#See how correct you are:
test_perf = accuracy_score(Y_test, Y_test_pred)
test_perf_rbf = accuracy_score(Y_test, Y_test_rbf)
print("Linear Performance on test set: " + str(test_perf) )
print("RBF Performance on test set: " + str(test_perf_rbf) )

#Performance on train set
Y_train_pred = clf.predict(X_train)
train_perf = accuracy_score(Y_train, Y_train_pred)

#Lets look at how far we are off
dif_matrix = Y_train - Y_train_pred
unique, counts = np.unique(dif_matrix, return_counts=True)
my_class_counts = dict(zip(unique, counts))

dif_matrix_abs = np.abs(dif_matrix)
unique, counts = np.unique(dif_matrix_abs, return_counts=True)
my_class_counts_abs = dict(zip(unique, counts))

x_count = [int(v) for v in my_class_counts.keys()]
y_count = [my_class_counts[v] for v in my_class_counts.keys()]

x_count_abs = [int(v) for v in my_class_counts_abs.keys()]
y_count_abs = [my_class_counts_abs[v] for v in my_class_counts_abs.keys()]

plt.clf()
plt.bar(x_count,y_count)
plt.xlabel('Error Misclassification')
plt.ylabel('Number of Occurences')
plt.title('Number of Buckets SVM Was Off By')
plt.show()

plt.clf()
plt.bar(x_count_abs,y_count_abs)
plt.xlabel('Magnitude of Error Misclassification')
plt.ylabel('Number of Occurences')
plt.title('Number of Buckets SVM Was Off By')
plt.show()

#total records
total_rec = sum(y_count_abs)
zero_to_one_bucket = sum(y_count_abs[0:2])
print("Percentage off by 1 bucket or less: " + str( zero_to_one_bucket / total_rec) )

##Perform same analysis for test set 
dif_matrix_abs_test = np.abs(Y_test - Y_test_pred)
unique, counts = np.unique(dif_matrix_abs_test, return_counts=True)

plt.clf()
plt.bar(unique,counts)
plt.xlabel('Magnitude of Error Misclassification on Test Set')
plt.ylabel('Number of Occurences')
plt.title('Number of Buckets SVM Was Off By on Test Set')
plt.show()

total_rec = sum(counts)
zero_to_one_bucket = sum(counts[0:2])
print("Percentage off by 1 bucket or less on test set: " + str( zero_to_one_bucket / total_rec) )


Y_train_pred_rbf = rbf_clf.predict(X_train)
train_perf_rbf = accuracy_score(Y_train, Y_train_pred_rbf)
print("Linear performance on train set: " + str(train_perf) )
print("RBF performance on train set: " + str(train_perf_rbf) )



#Lets take a look at what's being predicted; i.e. where am I doing poorly?
#pdb.set_trace()

df_results = pd.DataFrame( {'ytrain': Y_train, 'ypred' : Y_train_pred} )
df_count = df_results.groupby(df_results.columns.tolist()).size().reset_index().rename(columns={0:'count'})

ybase = df_count['ytrain']
ypd = df_count['ypred']
ypairs = list(zip(ybase, ypd))

plt.clf()
y_pos = np.arange(len(ypairs))
plt.bar( y_pos, df_count['count'] )
plt.xticks(y_pos, ypairs )
plt.xticks(rotation=90)
plt.xlabel("Pair (true_class,predicted_class)")
plt.ylabel("Count")
plt.title("Error Analysis for Multi-Class SVM")
plt.savefig("SVM_error_analysis")
plt.show()

##Notes: 
#Only predicting really into 3 classes for anything 0,3,4,6 and sometimes 8. 
#does that mean I should try fewer buckets? 
#1) try to incorporate company size; filter out cases with no size



print("Done!")






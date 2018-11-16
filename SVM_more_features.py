import csv
import pdb
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/train_data_with_salary_buckets.csv')


print("Num datapoints: " + str(len(df)))

##filter out low salaries
df = df[ (df.min_salary > 40000) & (df.max_salary > 40000) ]
print("Num datapoints after removing low salaries: " + str(len(df)))

##filter out companies with no average size
df = df[ df.avg_size.notnull() ]
print("Num datapoints after no num employees: " + str(len(df)))



df_XforSVM = df.filter(['is_acquired', 'is_public', 'remote_ok', 'NYC', \
	'LA', 'SF', 'SEA', 'senior', 'back_end', 'full_stack', 'front_end', 'avg_size'], axis=1)
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
#plt.savefig("SVM_error_analysis")
plt.show()

##Notes: 
#Only predicting really into 3 classes for anything 0,3,4,6 and sometimes 8. 
#does that mean I should try fewer buckets? 
#1) try to incorporate company size; filter out cases with no size



print("Done!")






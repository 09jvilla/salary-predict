import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/cleaned_data.csv')

#figure out salary buckets
min_sal_sorted_df = df.sort_values(by=['min_salary'])
max_sal_sorted_df = df.sort_values(by=['max_salary'])

plt.plot(min_sal_sorted_df['min_salary'].tolist(), 'bo')
plt.plot(max_sal_sorted_df['max_salary'].tolist(), 'ro')

plt.show()
plt.savefig("salary_scurve.png")
plt.clf()

print("Lets clear out giant outlier.")

min_sal_sorted_df = min_sal_sorted_df[ min_sal_sorted_df.min_salary < 500000 ]
max_sal_sorted_df = max_sal_sorted_df[ max_sal_sorted_df.min_salary < 500000 ]

plt.plot(min_sal_sorted_df['min_salary'].tolist(), 'bo')
plt.plot(max_sal_sorted_df['max_salary'].tolist(), 'ro')

plt.xlabel('Job Listings (sorted by Salary)')
plt.ylabel('Salary ($)')
plt.title('S-curve of job listings and salary')
plt.legend(["MinSalary", "MaxSalary"] )
plt.show()



#

import csv
import pdb
import pandas as pd
import numpy as np
import math

df = pd.read_csv('train.csv')

##Filter out positions where min salary is 0 and where max salary is 0

print("Original data size")
print(len(df))
df = df[df.min_salary != 0]
df = df[df.min_salary.notnull()]
print("Filtered out min salary zero or missing. New size " + str(len(df)) )
df = df[df.max_salary != 0]
df = df[df.max_salary.notnull()]
print("Filtered out max salary zero or missing. New size " + str(len(df)) )

##Filter out non-engineering or developer roles
df = df[df['job_title'].str.contains("engineer") | df['job_title'].str.contains("Engineer") \
		| df['job_title'].str.contains("developer") | df['job_title'].str.contains("Developer")]

print("Filtered out non-engineering roles. New size " + str(len(df)) )

def bay_area_filter(row):
	found = False
	found |= 'San Francisco' in row.address
	found |= 'Menlo Park' in row.address
	found |= 'Palo Alto' in row.address
	found |= 'Mountain View' in row.address
	found |= 'Redwood City' in row.address
	return found

#Create New Columns for geographies
df['NYC'] = df.apply(lambda row: 'New York' in row.address, axis=1)
df['LA'] = df.apply(lambda row: 'Los Angeles' in row.address, axis=1)
df['SF'] = df.apply(bay_area_filter, axis=1)
df['SEA'] = df.apply(lambda row: 'Seattle' in row.address, axis=1)

##Create binary variable for seniority
df['senior'] = df.apply(lambda row: row.seniority == 'senior' or row.seniority == 'staff', axis=1)



##first replace any NaN values in this column; instead put in string "N/A"
values = {'roles' : "N/A"}
df = df.fillna(value=values)

##Create binary variable for full_stack, back_end, front_end
df['back_end'] = df.apply(lambda row: 'back_end' in row.roles, axis=1)
df['full_stack'] = df.apply(lambda row: 'full_stack' in row.roles, axis=1)
df['front_end'] = df.apply(lambda row: 'front_end' in row.roles, axis=1)

##Create a column for average company size
#define function to compute average
def avg_size(row):
	if math.isnan(row.max_size) or math.isnan(row.min_size):
		return float('nan')
	else:
		return ((row.max_size)+(row.min_size) ) / 2

#apply function to dataframe
df['avg_size'] = df.apply(avg_size, axis=1)


pdb.set_trace()

#write out new dataset
dataset_name = 'cleaned_data_better.csv'
df.to_csv(dataset_name)

print("Cleaned dataset output as " + dataset_name)



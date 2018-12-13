import csv
import pdb
import pandas as pd
import numpy as np
import math
import sys

df = pd.read_csv('../data/train.csv')
NUM_SKILLS = 15

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
<<<<<<< HEAD
#
# def bay_area_filter(row):
# 	found = False
# 	found |= 'San Francisco' in row.address
# 	found |= 'Menlo Park' in row.address
# 	found |= 'Palo Alto' in row.address
# 	found |= 'Mountain View' in row.address
# 	found |= 'Redwood City' in row.address
# 	return found
#
# #Create New Columns for geographies
# df['NYC'] = df.apply(lambda row: 'New York' in row.address, axis=1)
# df['LA'] = df.apply(lambda row: 'Los Angeles' in row.address, axis=1)
# df['SF'] = df.apply(bay_area_filter, axis=1)
# df['SEA'] = df.apply(lambda row: 'Seattle' in row.address, axis=1)
#
# ##Create binary variable for seniority
# df['senior'] = df.apply(lambda row: row.seniority == 'senior' or row.seniority == 'staff', axis=1)
#
# ##Create binary variable for each seniority bucket
# seniority_types = df.seniority.unique()
# for seniority in seniority_types:
# 	df[f"seniority_{seniority}"] = df.apply(lambda row: row.seniority == seniority, axis=1)
#
# ##Create binary variable for each skill type
# skill_types = df.skills.unique()
# skills = []
# for skill in skill_types:
# 	if type(skill) is float:
# 		continue
# 	skills.extend(skill.split(","))
#
# skills = list(set(skills))
# skill_counts = {skill:0 for skill in skills}
#
# for idx, row in df.iterrows():
# 	if type(row.skills) is float:
# 		continue
# 	for skill in row.skills.split(","):
# 		skill_counts[skill] +=1
#
#
# sorted_skills = [item[0] for item in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)]
#
#
# sorted_skills = sorted_skills[:NUM_SKILLS]
#
# for skill in sorted_skills:
# 	df[f"skills_{skill}"] = df.apply(lambda row: skill in str(row.skills), axis=1)
=======

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

##Create binary variable for each seniority bucket
seniority_types = df.seniority.unique()
seniority_counts = {}
for seniority in seniority_types:
	df[f"seniority_{seniority}"] = df.apply(lambda row: row.seniority == seniority, axis=1)
	s = f"seniority_{seniority}"
	print(f"{seniority}: {df[s].sum()}")


##Seniority categories for GMM
gmm_seniority = {}
gmm_seniority["junior"] = ["junior","intern"]
gmm_seniority["regular"] = ["regular"]
gmm_seniority["senior"] = ["senior","staff","manager"]
gmm_seniority["executive"] = ["director","vp","cxo"]

for category, group in gmm_seniority.items():
	df[f"gmm_{category}"] = df.apply(lambda row: row.seniority in group, axis=1)

##Create binary variable for each skill type
skill_types = df.skills.unique()
skills = []
for skill in skill_types:
	if type(skill) is float:
		continue
	skills.extend(skill.split(","))

skills = list(set(skills))
skill_counts = {skill:0 for skill in skills}

for idx, row in df.iterrows():
	if type(row.skills) is float:
		continue
	for skill in row.skills.split(","):
		skill_counts[skill] +=1


sorted_skills = [item[0] for item in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)]


sorted_skills = sorted_skills[:NUM_SKILLS]

for skill in sorted_skills:
	df[f"skills_{skill}"] = df.apply(lambda row: skill in str(row.skills), axis=1)
>>>>>>> e20d5e0fe2dd946acea04d20645fb228d226f44a



##first replace any NaN values in this column; instead put in string "N/A"
# values = {'roles' : "N/A"}
# df = df.fillna(value=values)
# values = {'stage': 'N/A'}
# #
# ##Create binary variable for full_stack, back_end, front_end
# df['back_end'] = df.apply(lambda row: 'back_end' in row.roles, axis=1)
# df['full_stack'] = df.apply(lambda row: 'full_stack' in row.roles, axis=1)
# df['front_end'] = df.apply(lambda row: 'front_end' in row.roles, axis=1)
#
# ##Create a column for average company size
# #define function to compute average
# def avg_size(row):
# 	if math.isnan(row.max_size) or math.isnan(row.min_size):
# 		return float('nan')
# 	else:
# 		return ((row.max_size)+(row.min_size) ) / 2

#apply function to dataframe
# df['avg_size'] = df.apply(avg_size, axis=1)

#####Todo: feature scaling for faster convergence


#write out new dataset
dataset_name = './output/processed_data.csv'
df.to_csv(dataset_name)

print("Cleaned dataset output as " + dataset_name)

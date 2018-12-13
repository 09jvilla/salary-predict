import pickle
import pdb
import pandas as pd
import pdb
import re

df = pd.read_csv('data/cleaned_data_ENG.csv')

##munge the job title column
df['job_title'] = df['job_title'].str.lower()
df['job_title'] = df['job_title'].str.replace('software engineer', 'softwarengineer')
df['job_title'] = df['job_title'].str.replace('machine learning', 'machinelearning')
df['job_title'] = df['job_title'].str.replace('front end', 'frontend')
df['job_title'] = df['job_title'].str.replace('front-end', 'frontend')
df['job_title'] = df['job_title'].str.replace('back end', 'backend')
df['job_title'] = df['job_title'].str.replace('back-end', 'backend')
df['job_title'] = df['job_title'].str.replace('/', ' ')
df['job_title'] = df['job_title'].str.replace('(', ' ')
df['job_title'] = df['job_title'].str.replace(')', ' ')
df['job_title'] = df['job_title'].str.replace('-', '')
df['job_title'] = df['job_title'].str.replace(':', '')
df['job_title'] = df['job_title'].str.replace(',', ' ')

##munge the address
df["address"].replace( to_replace="^([a-zA-Z\s]+)_.+", value=r"\1", regex=True, inplace=True) 
df["address"].replace( to_replace="\s+", value="", regex=True, inplace=True)
df["address"] = df["address"].str.lower()

##fix the skills column
df["skills"].replace( to_replace="_", value="", regex=True, inplace=True )
df["skills"].replace( to_replace=",", value=" ", regex=True, inplace=True )

##fix the roles column
df["roles"].replace( to_replace="_", value="", regex=True, inplace=True )
df["roles"].replace( to_replace=",", value=" ", regex=True, inplace=True )

##fix the industries column
df["industries"].replace( to_replace="health_care", value="healthcare", regex=True, inplace=True )
df["industries"].replace( to_replace="social_media", value="socialmedia", regex=True, inplace=True )
df["industries"].replace( to_replace="big_data", value="bigdata", regex=True, inplace=True )
df["industries"].replace( to_replace="real_estate", value="realestate", regex=True, inplace=True )
df["industries"].replace( to_replace="machine_learning", value="machinelearning", regex=True, inplace=True )
df["industries"].replace( to_replace="finance_technology", value="fintech", regex=True, inplace=True )
df["industries"].replace( to_replace="_and_", value=" and ", regex=True, inplace=True )
df["industries"].replace( to_replace="artificial_intelligence", value="ai", regex=True, inplace=True )

df["industries"].replace( to_replace="-", value="", regex=True, inplace=True )
df["industries"].replace( to_replace=",", value=" ", regex=True, inplace=True )
df["industries"].replace( to_replace="_", value=" ", regex=True, inplace=True )

##Create new columns for acquired and public
df["acquired_string"] = df.apply(lambda row: "acquired" if row.is_acquired == True else "not acquired" , axis=1) 
df["acquired_string"] = df.apply(lambda row: "acquired" if row.is_acquired == True else "" , axis=1) 
df["public_string"] = df.apply(lambda row: "public-company" if row.is_public == True else "" , axis=1) 
df["remote_string"] = df.apply(lambda row: "work-from-home" if row.remote_ok == True else "" , axis=1) 

#look at funding
df["stage"].replace( to_replace="([A-Za-z])\sRound", value=r"Series\1", regex=True, inplace=True )
df["stage"].replace( to_replace="Pre Seed", value="pre-seed", regex=True, inplace=True )
df["stage"].replace( to_replace="Late Stage", value="late-stage", regex=True, inplace=True )

dataset_name = "./data/data_for_glove_embeddings_ENG.csv"
df.to_csv(dataset_name)
print("Cleaned dataset output as " + dataset_name)


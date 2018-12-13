import pickle
import pandas as pd
import pdb
import csv
import math
import numpy as np

GLOVE_SIZE = 300

df = pd.read_csv('data/data_for_glove_embeddings_ENG.csv')
print("Num records: " + str(len(df)))

df_small = df[["job_title", "address", "skills", "roles", "industries", "acquired_string", "public_string", "remote_string", "stage"   ]]

df_small = df_small.fillna(value="N/A")

list_of_data = df_small.values.tolist()
list_of_tups = []

df_sals = df[["min_salary", "max_salary"]]
list_of_sals = df_sals.values.tolist()

for l in range(len(list_of_data)):
    list_filtered = [x for x in list_of_data[l] if x != "N/A"]
    string_joined = " ".join(list_filtered)
    string_joined = string_joined.lower()
    tup = ( list_of_sals[l][0], list_of_sals[l][1], string_joined )
    list_of_tups.append(tup)

#let's create an embedding matrix per row
maxlen = -1
for l in list_of_tups:
    words = len(l[2].split())
    if words > maxlen:
        maxlen = words


embedding_matrix = np.zeros( (len(list_of_tups), maxlen, GLOVE_SIZE ) )

glove_data_file = "./glove/glove.840B.300d.txt"
words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
    try: 
        x = np.array(words.loc[w].values)
    except KeyError:
        x = np.zeros((GLOVE_SIZE,))
    return x


salary_info = []
for i in range(len(list_of_tups)):
    print("Embedding element: " + str(i) )
    string_val = list_of_tups[i][2]
    split_string = string_val.split()

    sal_info_tup = (list_of_tups[i][0], list_of_tups[i][1], len(split_string) )
    salary_info.append(sal_info_tup)
    
    for j in range(len(split_string)):
        embed = vec(split_string[j])
        embedding_matrix[i,j,:] = embed
    


matrix_output = "./data/glove_embeddings_of_dataset_ENG.mat"
np.save( matrix_output, embedding_matrix ) 


salary_output = "./data/glove_embedding_salinfo_ENG.pkl"
pickle.dump( salary_info, open( salary_output, "wb") )

print("Done")

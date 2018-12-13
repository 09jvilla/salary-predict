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

glove_data_file = "./glove/glove.840B.300d.txt"
words = pd.read_table(glove_data_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

def vec(w):
    try: 
        x = np.array(words.loc[w].values)
    except KeyError:
        x = np.zeros((GLOVE_SIZE,))
    return x

embedding_matrix = np.zeros( (len(list_of_data), GLOVE_SIZE ) )

for l in range(len(list_of_data)):
    print("Starting on example " + str(l) )
    list_filtered = [x for x in list_of_data[l] if x != "N/A"]

    single_embed = np.zeros( (len(list_filtered), GLOVE_SIZE) )
    
    for xi,x in enumerate(list_filtered):
        all_words = x.split(" ")
        
        word_embed = np.zeros( (len(all_words), GLOVE_SIZE) ) 
        for i,w in enumerate(all_words):
            word_embed[i,:] = vec(w)
        
        avg_word_embed = np.mean(word_embed, axis=0)

        single_embed[xi,:] = avg_word_embed


    embedding_matrix[l,:] = np.amax(single_embed , axis=0)

matrix_output = "./data/glove_embeddings_avg_max_ENG.mat"
np.save( matrix_output, embedding_matrix ) 

salary_info = []
for i in range(len(list_of_sals)):
    sal_info_tup = (list_of_sals[i][0], list_of_sals[i][1]  )
    salary_info.append(sal_info_tup)
salary_output = "./data/glove_embedding_salinfo_avg_max_ENG.pkl"
pickle.dump( salary_info, open( salary_output, "wb") )

print("Done")

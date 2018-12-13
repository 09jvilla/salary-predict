#tree_brute_force_tokenization.py

import csv
import pdb
import pandas as pd
import numpy as np
import collections
import math
import sys
from nltk import word_tokenize
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess():
    #adapted from Jen's data_process.py
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


    #write out new dataset
    dataset_name = './output/processed_data.csv'
    df.to_csv(dataset_name)

    print("Cleaned dataset output as " + dataset_name)

def vectorize(x, lab):
    print('CURRENTLY USING DUMB VECTORIZATION')
    return_x = pd.DataFrame()
    # d = collections.defaultdict(lambda: 0)
    feature_set = set()
    # pdb.set_trace()
    for i in range(0,x.shape[0]):
        temp_set = set(x[lab][i])
        feature_set = feature_set.union(temp_set)
    # pdb.set_trace()
    feature_set.remove('.')
    feature_set.remove(':')
    # set.remove('\'s')
    print('adding ', len(feature_set), ' new features: ')

    feature_dict = dict.fromkeys(list(feature_set))
    feature_dict.update((k,0) for k in feature_set)
    # pdb.set_trace()
    for key in feature_dict:
        print(key)
        for j in range(x.shape[0]):
            return_x[key] = pd.Series([0] * x.shape[0])
            if key in x[lab][j]:
                return_x[key][j] +=1

    return return_x

def count_vectorize(x, lab):
    #build corpus:
    corpus = []
    for j in range(x.shape[0]):
        corpus.append(x[lab][j])

    vectorizer = CountVectorizer()
    encoded_x = vectorizer.fit_transform(corpus)
    feat_names = vectorizer.get_feature_names()
    #add label at the end of the feature name to prevent duplicates
    for i in range(len(feat_names)):
        feat_names[i] = feat_names[i] + '_' + lab
    # pdb.set_trace()

    return encoded_x, feat_names
    # pdb.set_trace()
# print(X.toarray())
# [[0 1 1 1 0 0 1 0 1]
#  [0 2 0 1 0 1 1 0 1]
#  [1 0 0 1 1 0 1 1 1]
#  [0 1 1 1 0 0 1 0 1]]

def tfidf_vectorize(x, lab):
    tfidf  = TfidfVectorizer()
    corpus = []
    for j in range(x.shape[0]):
        corpus.append(x[lab][j])
    encoded_x = tfidf.fit_transform(corpus)
    feat_names = tfidf.get_feature_names()
    # pdb.set_trace()
    #add label at the end of the feature name to prevent duplicates
    for i in range(len(feat_names)):
        feat_names[i] = feat_names[i] + '_' + lab

    return encoded_x, feat_names

def tokenize(x):
    print('=========TOKENIZING YOUR DATA===========')
    labels = list(x)
    feature_names = []
    result_x = sp.csr_matrix((x.shape[0],1))

    for lab in labels:
        idx= x[lab].first_valid_index()

        if type(x[lab][idx]) == str:
            print('updating  ', lab)
            x[lab] = x[lab].fillna('N/A')
            if lab == 'address':
                x[lab] = x[lab].str.split(pat='_')
                x[lab] = x[lab].str.join(' ')
                # this_x = vectorize(x,lab)
                # x_part, feat_part= count_vectorize(x, lab)
                x_part, feat_part= tfidf_vectorize(x, lab)

            elif lab == 'job_title' or lab=='company_description':
                x[lab] = x[lab].apply(word_tokenize)
                x[lab] = x[lab].str.join(' ')
                # this_x = vectorize(x,lab)
                # x_part, feat_part= count_vectorize(x, lab)
                x_part, feat_part= tfidf_vectorize(x, lab)
                # pdb.set_trace()
            else:
                x[lab] = x[lab].str.split(pat=',')
                x[lab] = x[lab].str.join(' ')
                # this_x = vectorize(x,lab)
                # x_part, feat_part= count_vectorize(x, lab)
                x_part, feat_part= tfidf_vectorize(x, lab)
            # dict_list.append(d)
            # pdb.set_trace()
            # return_x = pd.concat([return_x, this_x], axis=1, sort=False)
            # result_x = sp.hstack([result_x, x_part])
        else:
            x[lab] = x[lab].fillna(0.0)
            print('NOT updating ', lab)
            x_part = np.expand_dims(x[lab].values, axis=1)
            feat_part = [lab]
            # pdb.set_trace()

        result_x = sp.hstack([result_x, x_part])
        feature_names.extend(feat_part)


    # pdb.set_trace()
    result_x= sp.lil_matrix(sp.csr_matrix(result_x)[:,1:])

    return result_x, feature_names

def hand_picked_x(x, feature_names, filename):
    build = sp.csr_matrix((x.shape[0],1))
    new_feature_names = []
    f = open(filename,'r')
    linecount = 0
    for line in f:
        linecount +=1
        keyword = line[:-1]
        new_feature_names.append(keyword)
        here=feature_names.index(keyword)
        # pdb.set_trace()
        # out1 = x.tocsc()[here,here+1]
        # pdb.set_trace()
        build=sp.hstack([build,x[:,here]])
        # pdb.set_trace()
    print(linecount)

    return sp.lil_matrix(sp.csr_matrix(build)[:,1:]), new_feature_names

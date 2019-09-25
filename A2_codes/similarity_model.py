import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

def get_union(vector1,vector2):
    new_vector1 = vector1[(vector1!=0 )& (vector2 != 0)]
    new_vector2 = vector2[(vector1!=0 )& (vector2 != 0)]
    return new_vector1,new_vector2

def cosine_similarity(vector1, vector2):
    new_vector1, new_vector2 = get_union(vector1,vector2)
    if len(new_vector1)==0 or len(new_vector2)==0:
        return 0
    dot_product = (new_vector1*new_vector2).sum()
    normA = ((new_vector1)**2).sum()
    normB = ((new_vector2)**2).sum()
    return round(dot_product / ((normA**0.5)*(normB**0.5)), 2)

def pearson_similarity(vector1, vector2):
    new_vector1, new_vector2 = get_union(vector1,vector2)
    if len(new_vector1)==0 or len(new_vector2)==0:
        return 0
    new_vector1_mean = new_vector1 - np.mean(new_vector1)
    new_vector2_mean = new_vector2 - np.mean(new_vector2)
    norm = np.linalg.norm(new_vector1_mean)*np.linalg.norm(new_vector2_mean)
    return np.dot(new_vector1_mean,new_vector2_mean)/norm
def user_sim(train_data_matrix, algo = "cosine"):
    user_similarity = np.zeros((train_data_matrix.shape[0], train_data_matrix.shape[0]))
    for i, cur in enumerate(user_similarity):
        for j,v in enumerate(cur):
            if user_similarity[i][j] == 0:
                if algo == "cosine":
                    user_similarity[i][j] = cosine_similarity(train_data_matrix[i], train_data_matrix[j])
                else:
                    user_similarity[i][j] = pearson_similarity(train_data_matrix[i], train_data_matrix[j])
                user_similarity[j][i] = user_similarity[i][j]
    print(user_similarity.max(), user_similarity.min())
    print (user_similarity.shape)
    print(user_similarity)
    return user_similarity
def item_sim(train_data_matrix, algo = "cosine"):
    train_data_matrix_t = train_data_matrix.T
    
    item_similarity = np.zeros((train_data_matrix.shape[1], train_data_matrix.shape[1]))
    for i, cur in enumerate(item_similarity):
        for j,v in enumerate(cur):
            if item_similarity[i][j] == 0:
                if algo == "cosine":
                    item_similarity[i][j] = cosine_similarity(train_data_matrix_t[i], train_data_matrix_t[j])
                else:
                    item_similarity[i][j] = pearson_similarity(train_data_matrix_t[i], train_data_matrix_t[j])
                item_similarity[j][i] = item_similarity[i][j]
    print(item_similarity.max(), item_similarity.min())
    print (item_similarity.shape)
    print(item_similarity)
    return item_similarity

def predict_user(ratings, similarity):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    
    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff)/np.array([np.abs(similarity).sum(axis=1)]).T
    for line in pred:
        line[np.isnan(line)] = np.nanmean(line)
    return pred
    
def predict_item(ratings, similarity):
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    for line in pred:
        line[np.isnan(line)] = np.nanmean(line)
    return pred
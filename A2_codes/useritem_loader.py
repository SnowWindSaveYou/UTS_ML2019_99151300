import numpy as np
import pandas as pd

import torch
import torch.utils.data

class UserItemDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, user_item_pairs, ratings):
        'Initialization'
        self.labels  = ratings
        self.samples = user_item_pairs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)

    def __getitem__(self, index):
        'Generates one sample of data'
        # # Load data and get label
        #print("called get item")
        user_item_pair = self.samples[index].astype('long')
        user_social = np.zeros(64).astype('long') #convert to actual social embeddings later
        user_item_pair_social = np.concatenate((user_item_pair, user_social), axis=None)
        X = user_item_pair_social
        y = self.labels[index]
        return X, y

def prepro(df,user_list=[]):
    df[['rating']]=df[['rating']].astype(float)
    df[['user']]=df[['user']].astype(int)
    df[['movie']]=df[['movie']].astype(int)
    user_id = []
    item_id = []

    if len(user_list) !=0:
        subset = df.merge(user_list,"left",on = "user")[['user_id', 'movie', 'rating']].dropna(axis=0,how='any')
    else: 
        user_id = df[['user']].drop_duplicates().reset_index(drop=True).reset_index()
        item_id = df[['movie']].drop_duplicates()
        user_id.rename(columns={'index':'user_id'},inplace=True)
        user_id["user_id"]+=1
        subset = df.merge(user_id,"left",on = "user")[['user_id', 'movie', 'rating']]
        

    total_ratings = np.array(subset.values)
    user_item_pairs = total_ratings[:,0:2]
    ratings = total_ratings[:,2:3]
    
    if len(user_list) !=0:
        return total_ratings,user_item_pairs,ratings
    else:
        num_users = int(len(user_id))
    #     num_users = int(user_id['user'].max() - user_id['user'].min() + 1)
        num_items = len(item_id['movie'])
        print("num_users: ",num_users,"num_items: ", num_items)
        return total_ratings,user_item_pairs,ratings,num_users,num_items,user_id

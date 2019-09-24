import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def slice_movie(df):
    # find empty rows for slice movies 
    # dataset using movie id as user name for splite rating of each movies
    movies_interval = df[df['rating'].isna()]['user'].reset_index()
    movies_interval['index'] = movies_interval['index'].astype(int)
    
    # create list accroding to movie interval by np
    movie_id = 1
    movie_np = []
    last = None
    for i,r in movies_interval.iterrows():
        if last != None: 
            temp = np.full((1,r['index']-last-1), movie_id)
            movie_np = np.append(movie_np, temp)
            movie_id +=1
        last = r['index']
    # add last movie
    temp = np.full((1,len(df)-last-1), movie_id)
    movie_np = np.append(movie_np, temp)
    
    # removie movie slice row, and combine movie colume
    df = df[pd.notnull(df['rating'])]
    df['movie'] = movie_np.astype(int)
    
    return df

def get_users_df(df):
    # get user list and their rating count
    return df.groupby('user')['user'].agg(['count']).reset_index()

def filter_data_by_user(df,users_df, frac=0.1,count=50):
    # get test users
    test_users_df = users_df[users_df['count']>count].sample(frac=frac)
    # filter user by rating count
    cdf = df.merge(users_df, left_on='user',right_on='user')
    cdf = cdf[cdf['count']>count][['user','movie','rating']]
    
    train_set = cdf[~(cdf['user'].isin(test_users_df['user']))]
    test_set = cdf[cdf['user'].isin(test_users_df['user'])]
    return train_set, test_set

def filter_data_by_date(df,users_df, frac=0.1,count=50,year=2005,month=12,day=1):
    # get test users
    test_users_df = users_df[users_df['count']>count].sample(frac=frac)
    # filter user by rating count
    cdf = df.merge(users_df, on='user')
    cdf = cdf[cdf['count']>count][['user','movie','rating','date']]

    test_set = cdf[cdf['user'].isin(test_users_df['user'])& (cdf['date']> pd.datetime(year,month,day))]
    target_set = cdf[cdf['user'].isin(test_users_df['user'])]
    train_set = cdf[~(cdf.isin(test_set))]
    return train_set[pd.notnull(train_set['user'])], test_set, target_set[pd.notnull(target_set['user'])]

def make_sparse_matrix(df,ITEM_NUM):
    users = df[['user']].drop_duplicates() #用户列表
    users = users['user'].astype(int)
    np_matrix = np.zeros((len(users), ITEM_NUM))
    i = 0
    for user in users:
        user_list = df[df['user']== user] #获取全部用户评分
        np_matrix[i][user_list['movie'].astype(int) -1] = user_list['rating'] #按index插入评分
        i+=1
        # if i%100==0:
        #     print("count: ",i)
    return np_matrix
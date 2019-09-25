import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import scipy

import math
from livelossplot import PlotLosses

from sklearn.metrics import roc_curve, auc
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class MF(torch.nn.Module):
    def __init__(self, num_users, num_items,latent_dim=8):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim,sparse=True)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim,sparse=True)
        
    def forward(self, user_indices, item_indices):
#         print(user_indices)

        user_embedding = self.embedding_user(user_indices)
#         print(user_embedding)
        item_embedding = self.embedding_item(item_indices)
        return (user_embedding*item_embedding).sum(1)

#--------------------------------------------------------------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, num_users, num_items,latent_dim=8,layers = [16,32,16,8]):
        super(MLP, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=layers[-1], out_features=1)
#         self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
#         print("item_embedding")
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
#         print("vector",vector)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = torch.nn.ReLU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)
        out = self.affine_output(vector)
#         rating = self.logistic(logits)
        return out

#--------------------------------------------------------------------------------------------------------------------


def predict(model, generator):
    model.eval()
    y_preds_all = torch.Tensor().to(device) 
    y_labels_all = torch.Tensor().to(device) 
    for local_batch, local_labels in generator:
        local_batch  = torch.tensor(local_batch).type(torch.long).to(device)
        local_labels = local_labels.type(torch.float).to(device)
        with torch.no_grad():
            y_preds = model(local_batch[:,0], local_batch[:,1])
        y_preds_all = torch.cat((y_preds_all,y_preds))
        y_labels_all = torch.cat((y_labels_all,local_labels))
    return y_preds_all, y_labels_all
def evaluate(model, generator):
    y_preds_all, y_labels_all = predict(model, generator)  
    y_preds = list(y_preds_all.view(1, y_preds_all.size()[0]).to("cpu").numpy()[0])
    y_actuals = list(y_labels_all.view(1, y_labels_all.size()[0]).to("cpu").numpy()[0])
    print(np.array([y_preds,y_actuals]))
    #print(type(y_preds), type(y_actuals))
    tmse = sum([(a-b) * (a-b) for a,b in zip(y_preds, y_actuals)])
    rmse = math.sqrt((1.0*tmse)/len(y_preds))
    return rmse

def epoch_run(model, generator, opt, criterion,liveloss,mode="train"):
    running_loss = 0
    if(mode == "train"):
        model.train()
    else:
        model.eval()
    #for local_batch, local_labels in generator:
    i = 0
    for local_batch, local_labels  in generator:
        
        # try:
        local_batch  = torch.tensor(local_batch).type(torch.long).to(device)
        local_labels = local_labels.type(torch.float).to(device)
        
        y_preds = model(local_batch[:,0], local_batch[:,1])
        loss = criterion(y_preds, local_labels)
    
        # except:
        #     # for debug
        #     return local_batch

        running_loss += (loss.item()*local_labels.size()[0])
        if(mode == "train"):
            opt.zero_grad()
            loss.backward()
            opt.step()
            liveloss.update({
                'mse':loss.item()
            })
            liveloss.draw()

    avg_loss = running_loss * 1.0 / (len(generator.dataset))
    return avg_loss

def epoch_run_2(model_1,model_2, generator,test_, opt_1, opt_2, criterion_1,criterion_2,liveloss ):
    running_loss_1 = 0
    running_loss_2 = 0
    model_1.train()
    model_2.train()
    i = 0
    for local_batch, local_labels  in generator: 
        print("batch ",i)
        # try:
        local_batch  = torch.tensor(local_batch).type(torch.long).to(device)
        local_labels = local_labels.type(torch.float).to(device)


        y_preds_1 = model_1(local_batch[:,0], local_batch[:,1])
        loss_1 = criterion_1(y_preds_1, local_labels)
        running_loss_1 += (loss_1.item()*local_labels.size()[0])

        y_preds_2 = model_2(local_batch[:,0], local_batch[:,1])
        loss_2 = criterion_2(y_preds_2, local_labels)
        running_loss_2 += (loss_2.item()*local_labels.size()[0])

        opt_1.zero_grad()
        loss_1.backward()
        opt_1.step()
        opt_2.zero_grad()
        loss_2.backward()
        opt_2.step()

        test_batch = test_[0]
        eval_labels = test_[1]
        test_batch[test_batch<0]=0
        eval_labels[eval_labels<0]=0
        test_batch  = torch.tensor(test_batch).type(torch.long).to(device)
        eval_labels = eval_labels.type(torch.float).to(device)
        t_preds_2 = model_2(test_batch[:,0], test_batch[:,1])
        t_loss_2 = criterion_2(t_preds_2, eval_labels)
        t_preds_1 = model_1(test_batch[:,0], test_batch[:,1])
        t_loss_1 = criterion_1(t_preds_1, eval_labels)

        liveloss.update({
            'mf':loss_1.item(),
            'val_mf':t_loss_1.item(),
            'mlpmf':loss_2.item(),
            'val_mlpmf':t_loss_2.item(),
        })
        liveloss.draw()

#         except:
# #             print("local_batch",local_batch)
# #             print("local_labels",local_labels)
#             pass

    avg_loss_1 = running_loss_1 * 1.0 / (len(generator.dataset))
    avg_loss_2 = running_loss_2 * 1.0 / (len(generator.dataset))
    return avg_loss_1,avg_loss_2
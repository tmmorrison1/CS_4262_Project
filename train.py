#!/usr/bin/env python3

## imports
import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import argparse
import logging
import os
import time
import random

## NN stuff
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

## external files
from LeagueInfo import team_aggregate
from LeagueInfo import team_aggregate_diff

from nn_model import LOL_model

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA


MODELS = (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
          SVC)
MODEL_NAMES = ('LR', 'RF', 'XBoost', 'SVM')

team_data = None
NUMPC = 8

## Ensure Reproducibility
def set_seed(args):
    np.random.seed(args.seed)
    random.seed(args.seed)


## Build input data files, removes miscellaneous stuff and builds train/test sets
## Standarizes our team data columns
## Returns (train, test)
def prep_data(args):
    global team_data

    ## Read in the game data
    game_data = pd.read_csv(args.game_data)
    
    ## Train/test split - default 80%
    ## Note: This shuffles the data for us
    train, test = train_test_split(game_data, test_size=1-args.tt_split)
    
    ## FOR NOW WE WILL BUILD TEAM DATA DYNAMICALLY - fixed seed allows us to work around
    ## but fuck that
    if args.team_data == '' and args.use_differentials:
        team_data = team_aggregate_diff(train)
    elif args.team_data == '' and not args.use_differentials:
        team_data = team_aggregate(train)
    else:
        team_data = pd.read_csv(args.team_data)
    team_data.drop('games', axis=1, inplace=True) ## irrelevent for training - could be useful as a check
    team_data.drop('win_pct', axis=1, inplace=True) ## irrelevant for training
    
    ## scale team data
    team_data = (team_data - team_data.mean())/team_data.std()
    
    ## drop stuff
    train['id'] = train['Unnamed: 0']
    test['id'] = test['Unnamed: 0']
    train = train[['id', 'redTeamTag', 'blueTeamTag', 'rResult', 'gamelength']]
    test = test[['id', 'redTeamTag', 'blueTeamTag', 'rResult', 'gamelength']]
    
    return train.values, test.values


## team1, team2 are np.arrays of team keys
def fetch_game_stats(team1,team2):
    return np.concatenate((fetch_team_stats(team1), fetch_team_stats(team2)), axis=1)


## Uses table of team data to gather a single element of training data
def fetch_team_stats(team_keys):
    ts = team_data.loc[team_keys]
    return np.array([ts.kills, ts.inhibs, ts.dragons, ts.goldDiff,
                     ts.barons, ts.heralds, ts.time_to_first_tower, ts.time_to_first_herald,
                     ts.time_to_first_dragon]).T


def run_pca(data, num_components):
    
    data_vals = data
    
    target_index = len(data_vals[0])-2
    
    
    
    team_1 = fetch_team_stats(data_vals[:,1])
    team_2 = fetch_team_stats(data_vals[:,2])
    
    x_data = np.concatenate((team_1, team_2), axis=1) 
    y_data = data_vals[:,target_index].astype("int")
    
    
    y_df = pd.DataFrame(data = y_data)
 
    
    y_df.rename(columns = {0:'target'},inplace = True)
    
    x_data = StandardScaler().fit_transform(x_data)
    
    lol_pca = PCA(n_components=num_components)
    
    principal_components = lol_pca.fit_transform(x_data)
    
    principal_df = pd.DataFrame(data = principal_components)
    
    finalDf = pd.concat([principal_df, y_df], axis = 1)
    
    
    for i in range(1,1+ num_components):
        print('Principle Component ' + str(i) + ' explains ' + str(lol_pca.explained_variance_ratio_[i-1])+' of the total variance')
    
    print() 
    print('Total Explained Variance: ' + str(sum(lol_pca.explained_variance_ratio_))+' for '+str(num_components)+' principle components')
    
    return finalDf


def evaluate(model, test_data):
    ## TODO: passes all of our training data through the model and computes statistics
    #should be correct now
    test_X = test_data[:,0:NUMPC]
    test_y = test_data[:,NUMPC].astype("int")
    
    predictions = model.predict(test_X)
    
    f1 = metrics.f1_score(test_y, predictions)
    roc_auc = metrics.roc_auc_score(test_y, predictions)
    accuracy = metrics.accuracy_score(test_y, predictions)
    log_loss = metrics.log_loss(test_y, predictions, normalize=True)
    
    results = {}
    
    results['f1'] = f1
    results['roc auc'] = roc_auc
    results['accuracy'] = accuracy
    results['log loss'] = log_loss
    
    return results


def train(model_type, train_data):
    ## Split X, y
    X = train_data[:, 0:NUMPC]
    y = train_data[:, NUMPC].astype('int')
    
    ## Sweep parameters
    Cs = [2e-5, 2e-3, 2e-1, 2e2, 2e3]
    gammas = [0.5, 0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(), param_grid, cv=3)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    #print(best_params)
    
    ## TODO: CV over series of model hyperparameters
    model = SVC(C=best_params['C'], gamma=best_params['gamma'])
    #model = model_type()
    model.fit(X, y)

    tr_acc = model.score(X, y)
    
    return model, tr_acc


def neural_net(args, train_data, test_data):
    X = train_data[:, 0:NUMPC]
    y = train_data[:, NUMPC].astype('int')
    
    ## TODO: for each element of X, grab appropriate features
    X_team = torch.Tensor(X)
    labels = torch.Tensor(y).long()

    train_set = TensorDataset(X_team, labels)

    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler,
                              batch_size=args.batch_size)

    set_seed(args)

    train_it = trange(int(args.epochs), desc="Epoch")
    loss_logged = []
    g_step, tr_loss = 0.0, 0.0
    acc_per_it = []
    labels = []
    
    model = LOL_model(X_team.size()[1])
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in train_it:
        epoch_it = tqdm(train_loader, desc='Iteration')

        for step, batch in enumerate(epoch_it):
            outputs = model(batch[0], batch[1])
            loss = outputs[0]
            
            loss.backward()

            tr_loss += loss.item()
            loss_logged.append(loss.item())

            optimizer.step()
            model.zero_grad()
            g_step += 1
            
            acc_per_it = np.concatenate((acc_per_it,
                                        np.argmax(outputs[1].detach().numpy(), axis=1)),
                                        axis=0)
            labels.extend(batch[1].detach().numpy())
            
    tr_acc = accuracy_score(labels, acc_per_it)

    ## Now compute accuracy
    model.eval()

    test_X = torch.Tensor(test_data[:,0:NUMPC])
    test_y = torch.Tensor(test_data[:,NUMPC].astype("int"))

    test_set = TensorDataset(test_X, test_y)

    eval_sampler = SequentialSampler(test_set)
    eval_loader = DataLoader(test_set, sampler=eval_sampler,
                             batch_size=args.batch_size)

    eval_it = tqdm(eval_loader, desc='Iteration')

    test_acc = []
    labels = []
    for step, batch in enumerate(eval_it):
        outputs = model(batch[0])

        test_acc = np.concatenate((test_acc,
                                   np.argmax(outputs.detach().numpy(), axis=1)),
                                  axis=0)
        labels.extend(batch[1].detach().numpy())

    test_acc = accuracy_score(labels, test_acc)

    return tr_acc, tr_loss, test_acc, loss_logged


## Used for when not running experiments from command line
def manual_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.game_data = 'data/league_data_100.csv'
    args.team_data = ''
    args.output_dir = 'model_states/tmp/'
    args.overwrite_output_dir = True
    args.seed=100
    args.tt_split = .8
    args.results = 'diff'
    args.use_differentials = True
    args.batch_size = 256
    args.epochs = 12
    args.lr = 1e-3
    return args


def dumb_model(args, test_data):
    team_1 = fetch_team_stats(test_data[:,1])
    team_2 = fetch_team_stats(test_data[:,2])
      
    win_num = team_1 - team_2
    win_num = win_num >= 0
    results = win_num.sum(axis=1)

    dumb_predictions  = results > win_num.shape[1]/2
    y_true = test_data[:,3].astype("int")
     
    f1 = metrics.f1_score(y_true, dumb_predictions)
    roc_auc = metrics.roc_auc_score(y_true, dumb_predictions)
    accuracy = metrics.accuracy_score(y_true, dumb_predictions)
    log_loss = metrics.log_loss(y_true, dumb_predictions)
    
    results = {}
    
    results['f1'] = f1
    results['roc auc'] = roc_auc
    results['accuracy'] = accuracy
    results['log loss']  = log_loss
    return results


def get_reduced_data(train_data, test_data):
    num_pcs = NUMPC
    complete_data = np.concatenate((train_data,test_data),axis =0)
    
    pca_df = run_pca(complete_data, num_pcs)

    length = round(len(pca_df)*.8)
    
    train_df = pca_df[:length][:]
    test_df = pca_df[length:][:]
        
    y_train = train_df[:]['target']
    x_train = train_df.drop('target',axis =1)
        
    y_test = test_df[:]['target']
    x_test = test_df.drop('target',axis =1)
        
        
    train = pd.concat([x_train, y_train], axis = 1)
    test = pd.concat([x_test, y_test], axis = 1)

    return np.array(train), np.array(test)


def plot_loss(logged_loss):
    plt.plot(logged_loss)
    plt.title('NN Loss')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.grid(alpha=.3)
    plt.savefig('NN_loss_plot.png')

def main():
    ## Gather arguments and se tthe seed
    args = manual_args()

    results = []
    with open('results_sweep.txt', 'w') as f:
        for i in range(3):
            set_seed(args)
            ## Get data - np.arrays
            train_data, test_data = prep_data(args)
            ## Do PCA
            train_data, test_data = get_reduced_data(train_data, test_data)

            ## Run all of the models at the base level and return train/test accuracy
            ix = 0
            for model_type, model_name in zip(MODELS[3], MODEL_NAMES[3]):
                model, tr_acc = train(model_type, train_data)
                eval_acc = evaluate(model, test_data)['accuracy']

                if i == 0:
                    results.append([tr_acc, eval_acc])
                else:
                    results[ix][0] += tr_acc
                    results[ix][1] += eval_acc
                ix += 1
            
            ## Train and evaluate the nn
            tr_acc, tr_loss, eval_acc, loss_logged = neural_net(args, train_data,
                                                                test_data)
            if i == 0:
                results.append([tr_acc, eval_acc])
            else:
                results[ix][0] += tr_acc
                results[ix][1] += eval_acc

            ## reset seed
            args.seed += 1

        print(np.array(results)/3, file=f)
   
    ## Now select the best models manually and train hyperparameters

def sweep_hyperparams():
    args = manual_args()

    results = []
    with open('results_sweep.txt', 'w') as f:
        for i in range(3):
            set_seed(args)
            ## Get data - np.arrays
            train_data, test_data = prep_data(args)
            ## Do PCA
            train_data, test_data = get_reduced_data(train_data, test_data)

            ## Train and evaluate the nn
            tr_acc, tr_loss, eval_acc, loss_logged = neural_net(args, train_data,
                                                                test_data)
            if i == 0:
                results.append([tr_acc, eval_acc])
            else:
                results[0][0] += tr_acc
                results[0][1] += eval_acc

            args.seed += 1
        print(np.array(results)/3, file=f)
        print(np.array(results)/3)
   
    ## Now select the best models manually and train hyperparameters

sweep_hyperparams()   
#main()

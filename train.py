#!/usr/bin/env python3

## imports
import pandas as pd
import matplotlib as plt
import sys
import numpy as np
import argparse
import logging
import os
import time
import random

from LeagueInfo import team_aggregate

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

MODELS = (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier,
          SVC)

team_data = None

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
    if args.team_data == '':
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
    return np.array([ts.kills, ts.towers, ts.inhibs, ts.dragons, ts.goldDiff]).T


def sample_champions(team_key1, team_key2):
    ## FIXME: we did no analysis of this before but if we appended the information
    ## per team to the team_data collection function then we could easily implement an
    ## algorithm here that does the work of simulating champion bans/selections.

    ## TODO: Aggregate champion data and bans
    ## TODO: Write sampler that simulates pregame champion stuff

    return


def evaluate(model, test_data):
    ## TODO: passes all of our training data through the model and computes statistics
    #should be correct now
    team_1 = fetch_team_stats(test_data[:,1]).T
    team_2 = fetch_team_stats(test_data[:,2]).T
    
    
    y_column = [3]
    
 
    test_x = np.concatenate((team_1,team_2),axis=1)
    test_y = test_data[:,3].astype("int")
    
    predictions = model.predict(test_x)
    
    f1 = metrics.f1_score(test_y, predictions)
    roc_auc = metrics.roc_auc_score(test_y, predictions)
    accuracy = metrics.accuracy_score(test_y, predictions)
    log_loss = metrics.log_loss(test_y, predictions)
    
    results = {}
    
    results['f1'] = f1
    results['roc auc'] = roc_auc
    results['accuracy'] = accuracy
    results['log loss']  = log_loss
    
    return results


def train(model, train_data):
    ## Split X, y
    X = train_data[:, (1,2)]
    y = train_data[:, 3]

    ## TODO: for each element of X, grab appropriate features
    X_team = fetch_game_stats(X[:, 0], X[:, 1])
    
    ## fetch team 1/2 data
    
    ## TODO: CV over series of model hyperparameters
    
    return


## Used for when not running experiments from command line
def manual_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.game_data = 'data/league_data_100.csv'
    args.team_data = ''
    args.output_dir = 'model_states/tmp/'
    args.overwrite_output_dir = True
    args.seed=69
    args.tt_split = .8
    
    return args

## Builds the arguments from a bash file and stores in args
def parse_args():
    parser = argparse.ArgumentParser()

    ## add our arguments
    ## Required arguments
    parser.add_argument('--game_data', type=str, required=True,
                        help='File of games played (training data)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Dir where output data will be stored, to overwrite use --overwrite_output_dir')

    ## Optional arugments
    parser.add_argument('--team_data', default='', type=str,
                        help='File of team tabular data')
    parser.add_argument('--seed', default=69, type=int,
                        help='set the seed for experiments')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Set this flag to overwrite the results dir')
    parser.add_argument('--tt_split', default=.8, type=float,
                        help='Perentage of train data from original game data')
    
    return parser.parse_args()


def main():
    ## Gather arguments and se tthe seed
    args = manual_args()
    set_seed(args)

    ## Get data - np.arrays
    train_data, test_data = prep_data(args)

    ## Model Selection - FIXME: This needs to be based on arguments
    ## FIXME: Need a robust global parameter of different models that can be selected
    model = SVC(train_data)

    #tr_acc = train(model, train_data)
    
    return 
    

if __name__ == '__main__':
    main()

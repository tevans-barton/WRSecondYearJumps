#@Tommy Evans-Barton

import os
import pandas as pd
import numpy as np

TOP_PATH = os.environ['PWD']
OUTPATH = TOP_PATH + '/data/interim'

#Helper function to transform the player names into the advanced stat formats
def clean_player_name(players):
    #Use regex pattern matching to remove markers (such as * or +)
    players = players.str.replace(' *(\*|\+)', '')
    #Remove Jr. from player names in order to match the advanced stats
    players = players.str.replace(' Jr.', '')
    #Change the format of the names to match the advanced stats ('O.Beckham')
    players = [(s[0] + '.' + s[s.find(' ') + 1 : ]) for s in players]
    return players


def clean_receivers():
    #Read in data
    try:
        receivers = pd.read_csv(TOP_PATH + '/data/raw/RECEIVERS.csv')
    except:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Drop the position column, as all players are just WR by nature of dataset
    receivers.drop(['Pos'], axis = 1, inplace = True)
    #Clean the player names to be like the advanced stat format
    receivers['Player'] = clean_player_name(receivers['Player'])
    #Remove the Year column and make a first year and second year in the league column for merging
    receivers['First Year'] = receivers['YEAR']
    receivers['Second Year'] = receivers['YEAR'] + 1
    receivers.drop(['YEAR'], axis = 1, inplace = True)
    #Save the dataframe to  a CSV in the data/interim directory
    if not os.path.exists(OUTPATH):
        os.mkdir(OUTPATH)
    receivers.to_csv(OUTPATH + '/RECEIVERS_CLEANED.csv')
    return receivers
    

def clean_stats():
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Drop the Position and Fumble stats
    #Position because all are receivers
    #Fumbles because most fumbles were due to special teams, not indicative of receiver play
    rec_stats.drop(['Pos', 'Fmb'], inplace = True, axis = 1)
    #Clean the player names to be like the advanced stat format
    rec_stats['Player'] = clean_player_name(rec_stats['Player'])
    #Edit the Catch Rate column into a float
    rec_stats['Ctch%'] = pd.to_numeric(rec_stats['Ctch%'].str.replace('%', ''))
    #Save the dataframe to  a CSV in the data/interim directory
    if not os.path.exists(OUTPATH):
        os.mkdir(OUTPATH)
    receivers.to_csv(OUTPATH + '/REC_STATS_CLEANED.csv')
    return receivers


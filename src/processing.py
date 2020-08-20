#@Tommy Evans-Barton

import os
import pandas as pd
import numpy as np

TOP_PATH = os.environ['PWD']
INTERIM_OUTPATH = TOP_PATH + '/data/interim'
FINAL_OUTPATH = TOP_PATH + '/data/final'
PTS_FOR_TD = 6
YDS_PER_POINT = 10
CURRENT_YEAR = 2020

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
    if not os.path.exists(INTERIM_OUTPATH):
        os.mkdir(INTERIM_OUTPATH)
    receivers.to_csv(INTERIM_OUTPATH + '/RECEIVERS_CLEANED.csv', index = False)
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
    #Want a column for slightly altered fantasy points, taking into account only yards and touchdowns
    rec_stats['Rec Pts'] = rec_stats['TD'] * PTS_FOR_TD + rec_stats['Yds'] / YDS_PER_POINT
    #Save the dataframe to  a CSV in the data/interim directory
    if not os.path.exists(INTERIM_OUTPATH):
        os.mkdir(INTERIM_OUTPATH)
    rec_stats.to_csv(INTERIM_OUTPATH + '/REC_STATS_CLEANED.csv', index = False)
    return rec_stats

def clean_adv_stats():
    #Read in data
    try:
        adv_stats = pd.read_csv(TOP_PATH + '/data/raw/ADV_REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Edit the Player column to change a few instances where the formatting didn't conform
    players = adv_stats['Player']
    adv_stats['Player'] = [(s[0] + '.' + s[s.find('.') + 1 : ]) for s in players]
    #Unifying format of team names with other datasets
    team_dict = {
        'GB' : 'GNB',
        'JAC' : 'JAX',
        'KC' : 'KAN',
        'NO' : 'NOR',
        'NE' : 'NWE',
        'LACH' : 'LAC',
        'LARM' : 'LAR',
        'SD' : 'LAC',
        'SF' : 'SFO',
        'TB' : 'TAM',
        '2TM' : np.nan
    }
    teams = adv_stats['Team']
    adv_stats['Team'] = teams.replace(team_dict)
    #Changing DVOA and VOA into numeric types
    adv_stats['DVOA'] = pd.to_numeric(adv_stats['DVOA'].str.replace('%', ''))
    adv_stats['VOA'] = pd.to_numeric(adv_stats['VOA'].str.replace('%', ''))
    #Split the DPI column into DPI Penalties and DPI Yards, in order to make them numeric values for analysis
    adv_stats['DPI Pens'] = [int(s[0 : s.find('/')]) for s in adv_stats['DPI']]
    adv_stats['DPI Yds'] = [int(s[s.find('/') + 1 : ]) for s in adv_stats['DPI']]
    adv_stats.drop(['DPI'], axis = 1, inplace = True)
    if not os.path.exists(INTERIM_OUTPATH):
        os.mkdir(INTERIM_OUTPATH)
    adv_stats.to_csv(INTERIM_OUTPATH + '/ADV_REC_STATS_CLEANED.csv', index = False)
    return adv_stats

def clean_all_data():
    clean_receivers()
    clean_stats()
    clean_adv_stats()
    return

def merge_data():
    #Read in data
    try:
        receivers = pd.read_csv(TOP_PATH + '/data/interim/RECEIVERS_CLEANED.csv')
        rec_stats = pd.read_csv(TOP_PATH + '/data/interim/REC_STATS_CLEANED.csv')
        adv_stats = pd.read_csv(TOP_PATH + '/data/interim/ADV_REC_STATS_CLEANED.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data and processing.clean_all_data)')
        return
    #Take our dataframe of receivers in the first three rounds over past 10 years and merge in their first year stats
    temp = receivers.merge(rec_stats, how = 'left', left_on = ['Player', 'Tm', 'First Year'], 
                                        right_on = ['Player', 'Tm', 'YEAR'], suffixes = [' Draft', ' First Season'])
    #Remove all receivers who did not have rookie year data
    temp = temp[~temp.isnull().any(axis = 1)].reset_index(drop = True)
    #Take our dataframe with first year stats and merge in second year Rec Pts
    sec_yr = rec_stats[['Player', 'Tm', 'YEAR', 'Rec Pts']]
    df = temp.merge(sec_yr, how = 'left', left_on = ['Player', 'Tm', 'Second Year'], 
                                        right_on = ['Player', 'Tm', 'YEAR'], suffixes = [' First Season', ' Second Season'])
    #Get rid of entries where the receiver should have had their second year, but has no stats for that year
    df = df[~((df['Second Year'] < CURRENT_YEAR) & (df.isnull().any(axis = 1)))].reset_index(drop = True)
    df['Rec Pts Jump'] = df['Rec Pts Second Season'] - df['Rec Pts First Season']
    if not os.path.exists(FINAL_OUTPATH):
        os.mkdir(FINAL_OUTPATH)
    df.to_csv(FINAL_OUTPATH + '/FINAL_DATA.csv', index = False)
    return df
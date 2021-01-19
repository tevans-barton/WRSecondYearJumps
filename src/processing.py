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
#Dictionary for cleaning the team names
TEAM_DICT = {
        'GB' : 'GNB',
        'JAC' : 'JAX',
        'KC' : 'KAN',
        'NO' : 'NOR',
        'NE' : 'NWE',
        'LACH' : 'LAC',
        'LARM' : 'LAR',
        'SD' : 'LAC',
        'SDG' : 'LAC',
        'STL' : 'LAR',
        'SF' : 'SFO',
        'TB' : 'TAM',
        '2TM' : np.nan,
        '3TM' : np.nan,
        '4TM' : np.nan
    }
#Helper function to transform the player names into the advanced stat formats
def clean_player_name(players):
    #Use regex pattern matching to remove markers (such as * or +)
    players = players.str.replace(' *(\*|\+)', '')
    #Remove Jr. from player names in order to match the advanced stats
    players = players.str.replace(' Jr.', '')
    #Change the format of the names to match the advanced stats ('O.Beckham')
    players = [(s[0] + '.' + s[s.find(' ') + 1 : ]) for s in players]
    return players

#Function for cleaning the receivers dataframe
def clean_receivers():
    #Read in data
    try:
        receivers = pd.read_csv(TOP_PATH + '/data/raw/RECEIVERS.csv')
    except:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Drop the position column, as all players are just WR by nature of dataset
    receivers.drop(['Pos'], axis = 1, inplace = True)
    #Clean the team names (namely for the Chargers and Rams)
    teams = receivers['Tm']
    receivers['Tm'] = teams.replace(TEAM_DICT)
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



#A helper function to compute what team a receiver is on in the following year
def next_yrs_team(df):
    #Sort the values by Player Name and year
    df = df.sort_values(['Player', 'YEAR']).reset_index(drop = True)
    #Loop through the dataframe and create a column that indicates where a player played the following
    #year, based on stats recorded for that following year, NaN if there are no stats recorded the following year
    #for a player
    new_team_list = []
    for i in range(len(df) - 1):
        if (df.iloc[i]['Player'] == df.iloc[i + 1]['Player']) & (df.iloc[i]['YEAR'] == df.iloc[i + 1]['YEAR'] - 1):
            new_team_list.append(df.iloc[i + 1]['Tm'])
        elif df.iloc[i]['YEAR'] == 2019:
            new_team_list.append(df.iloc[i]['Tm'])
        else:
            new_team_list.append(np.NaN)
    if df.iloc[len(df) - 1]['YEAR'] == 2019:
        new_team_list.append(df.iloc[len(df) - 1]['Tm'])
    else:
        new_team_list.append(np.NaN)
    df['Next Years Team'] = new_team_list
    return df

#Helper funciton to create what share of the WR stats each player had for their team
#Each equation is, for example: (yards for player x) / (sum of yards for receivers on their team for that year)
def tm_share_cols(df):
    #Make dictionaries for each statistic of the form {(<TEAM>, <YEAR>) : STAT}
    wr_tgt_dict = df[['Tm', 'YEAR', 'Tgt']].groupby(['Tm', 'YEAR']).sum()['Tgt'].to_dict()
    wr_yards_dict = df[['Tm', 'YEAR', 'Yds']].groupby(['Tm', 'YEAR']).sum()['Yds'].to_dict()
    wr_td_dict = df[['Tm', 'YEAR', 'TD']].groupby(['Tm', 'YEAR']).sum()['TD'].to_dict()
    wr_rec_dict = df[['Tm', 'YEAR', 'Rec']].groupby(['Tm', 'YEAR']).sum()['Rec'].to_dict()
    #Create series of the stats by teams in order to create the share columns
    tgt_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_tgt_dict))
    yds_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_yards_dict))
    td_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_td_dict))
    rec_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_rec_dict))
    #Create the columns for each player's share of these stats for their teams
    df['WR Tgt Share'] = df['Tgt'] / tgt_by_team_series
    df['WR Yds Share'] = df['Yds'] / yds_by_team_series
    df['WR TD Share'] = df['TD'] / td_by_team_series
    df['WR Rec Share'] = df['Rec'] / rec_by_team_series
    return df


#Create the incoming/outgoing stat shares columns
def in_out_stats_cols(df):
    #Make dictionaries for each statistic of the form {(<TEAM>, <YEAR>) : STAT}
    wr_tgt_dict = df[['Tm', 'YEAR', 'Tgt']].groupby(['Tm', 'YEAR']).sum()['Tgt'].to_dict()
    wr_yards_dict = df[['Tm', 'YEAR', 'Yds']].groupby(['Tm', 'YEAR']).sum()['Yds'].to_dict()
    wr_td_dict = df[['Tm', 'YEAR', 'TD']].groupby(['Tm', 'YEAR']).sum()['TD'].to_dict()
    wr_rec_dict = df[['Tm', 'YEAR', 'Rec']].groupby(['Tm', 'YEAR']).sum()['Rec'].to_dict()
    #Create series of the stats by teams in order to create the share columns
    tgt_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_tgt_dict))
    yds_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_yards_dict))
    td_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_td_dict))
    rec_by_team_series = pd.Series(df.set_index(['Tm', 'YEAR']).index.map(wr_rec_dict))
    #Create a temporary dataframe that only holds the relevant stats of the entries where a receiver is switching teams
    #the following year
    temp = df[~(df['Tm'] == df['Next Years Team'])][['Player', 'WR Tgt Share', 'WR Yds Share', 
                                    'WR TD Share', 'WR Rec Share', 'Tm', 'Next Years Team', 'YEAR']].reset_index(drop = True)
    #Create dictionaries for how many of each stat will be changing for each team each year
    out_stats = temp.groupby(['Tm', 'YEAR']).sum().to_dict()
    in_stats = temp.groupby(['Next Years Team', 'YEAR']).sum().to_dict()
    #Create a series for each of the outgoing stat shares
    wr_tgt_share_departing = df.set_index(['Tm', 'YEAR']).index.map(out_stats['WR Tgt Share']).fillna(0)
    wr_yds_share_departing = df.set_index(['Tm', 'YEAR']).index.map(out_stats['WR Yds Share']).fillna(0)
    wr_td_share_departing = df.set_index(['Tm', 'YEAR']).index.map(out_stats['WR TD Share']).fillna(0)
    wr_rec_share_departing = df.set_index(['Tm', 'YEAR']).index.map(out_stats['WR Rec Share']).fillna(0)
    #Create a series for each of the incoming stat shares
    wr_tgt_share_incoming = df.set_index(['Next Years Team', 'YEAR']).index.map(in_stats['WR Tgt Share']).fillna(0)
    wr_yds_share_incoming = df.set_index(['Next Years Team', 'YEAR']).index.map(in_stats['WR Yds Share']).fillna(0)
    wr_td_share_incoming = df.set_index(['Next Years Team', 'YEAR']).index.map(in_stats['WR TD Share']).fillna(0)
    wr_rec_share_incoming = df.set_index(['Next Years Team', 'YEAR']).index.map(in_stats['WR Rec Share']).fillna(0)
    #Make columns for each of the projected stat shares, based on offseason moves
    df['Projected Tgt Share'] = wr_tgt_share_departing - wr_tgt_share_incoming + df['WR Tgt Share']
    df['Projected Yds Share'] = wr_yds_share_departing - wr_yds_share_incoming + df['WR Yds Share']
    df['Projected TD Share'] = wr_td_share_departing - wr_td_share_incoming + df['WR TD Share']
    df['Projected Rec Share'] = wr_rec_share_departing - wr_rec_share_incoming + df['WR Rec Share']
    #Make columns for each of the projected stats, based on offseason moves and team stats
    df['Projected Tgt'] = df['Projected Tgt Share'] * tgt_by_team_series
    df['Projected Yds'] = df['Projected Yds Share'] * yds_by_team_series
    df['Projected TD'] = df['Projected TD Share'] * td_by_team_series
    df['Projected Rec'] = df['Projected Rec Share'] * rec_by_team_series
    return df

#Function for cleaning the receivers statistics dataframe
def clean_stats():
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Read in the uploaded 2020 Free Agent Wide Receiver data and combine it in
    temp = pd.read_csv(TOP_PATH + '/data/upload/2020_FA_WR.csv')
    df = pd.concat([rec_stats, temp])[rec_stats.columns]
    #Take out non-alphabetic characters from the player names
    df['Player'] = df['Player'].str.replace(' *(\*|\+)', '')
    #Change all of the position entries to upper case
    df['Pos'] = df['Pos'].str.upper()
    #Find the names of every player who ever played receiver over these years
    rec_names = list(df[df['Pos'] == 'WR']['Player'].unique())
    #Fill the non-entries for players who at some point played receiver with 'WR'
    df['Pos'] = df.apply(lambda row : 'WR' if row['Player'] in rec_names else np.NaN, axis = 1)
    #Drop all other entries that don't have a position entry
    df = df.dropna(subset = ['Pos']).reset_index(drop = True)
    #Map the team names so that they are consistent from dataset to dataset
    df['Tm'] = df['Tm'].replace(TEAM_DICT)
    #Create each players share of their WR stats
    df = tm_share_cols(df)
    #Create the 'Next Years Team' column
    df = next_yrs_team(df)
    #Create the change in stat shares columns
    df = in_out_stats_cols(df)
    #Had to take a slightly brittle approach with Jaron Brown, as his inclusion caused overlap issues with John Brown
    df = df[df['Player'] != 'Jaron Brown'].reset_index(drop = True)
    #Clean the player names to be like the advanced stat format
    df['Player'] = clean_player_name(df['Player'])
    #Edit the Catch Rate column into a float, and then make it into a decimal, not a percentage
    #and then rename it to catch rate
    df['Ctch%'] = pd.to_numeric(df['Ctch%'].str.replace('%', ''))
    df['Ctch%'] = df['Ctch%'] / 100
    df = df.rename({'Ctch%' : 'Catch Rate'}, axis = 1)
    #Want a column for slightly altered fantasy points, taking into account only yards and touchdowns
    df['Rec Pts'] = df['TD'] * PTS_FOR_TD + df['Yds'] / YDS_PER_POINT
    #Want a column for slightly altered fantasy points per game, taking into account only yards and touchdowns
    df['Rec Pts/G'] = df['Rec Pts'] / df['G']
    #Save the dataframe to  a CSV in the data/interim directory
    if not os.path.exists(INTERIM_OUTPATH):
        os.mkdir(INTERIM_OUTPATH)
    df.to_csv(INTERIM_OUTPATH + '/REC_STATS_CLEANED.csv', index = False)
    return df

def clean_adv_stats():
    #Read in data
    try:
        adv_stats = pd.read_csv(TOP_PATH + '/data/raw/ADV_REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Had to take a slightly brittle approach with Jaron Brown, as his inclusion caused overlap issues with John Brown
    adv_stats = adv_stats[adv_stats['Player'] != 'Ja.Brown'].reset_index(drop = True)
    #Edit the Player column to change a few instances where the formatting didn't conform
    players = adv_stats['Player'].str.replace(' ', '.')
    adv_stats['Player'] = [(s[0] + '.' + s[s.rfind('.') + 1 : ]) for s in players]
    #Unifying format of team names with other datasets
    teams = adv_stats['Team']
    adv_stats['Team'] = teams.replace(TEAM_DICT)
    #Changing DVOA and VOA into numeric types
    adv_stats['DVOA'] = pd.to_numeric(adv_stats['DVOA'].str.replace('%', ''))
    adv_stats['VOA'] = pd.to_numeric(adv_stats['VOA'].str.replace('%', ''))
    #Split the DPI column into DPI Penalties and DPI Yards, in order to make them numeric values for analysis
    adv_stats['DPI Pens'] = [int(s[0 : s.find('/')]) for s in adv_stats['DPI']]
    adv_stats['DPI Yds'] = [int(s[s.find('/') + 1 : ]) for s in adv_stats['DPI']]
    adv_stats.drop(['DPI'], axis = 1, inplace = True)
    #Save the dataframe to  a CSV in the data/interim directory
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
    sec_yr = rec_stats[['Player', 'Tm', 'YEAR', 'Rec Pts', 'Rec Pts/G']]
    df = temp.merge(sec_yr, how = 'left', left_on = ['Player', 'Tm', 'Second Year'], 
                                        right_on = ['Player', 'Tm', 'YEAR'], suffixes = [' First Season', ' Second Season'])
    #Get rid of entries where the receiver should have had their second year, but has no stats for that year
    df = df[~((df['Second Year'] < CURRENT_YEAR) & (df['Rec Pts Second Season'].isnull()))].reset_index(drop = True)
    #Merge in the advanced stats from each player's first year
    df = df.merge(adv_stats, how = 'left', left_on = ['Player', 'Tm', 'First Year'], right_on = ['Player', 'Team', 'YEAR'])
    #Drop rows with key null entries
    df = df[~df[['Rnd', 'Pick', 'Team', 'Player', 'First Year', 'Age Draft', 'G', 'GS', 'Tgt', 
                     'Rec', 'Catch Rate', 'Yds', 'Y/R', 'TD', '1D', 'Lng', 'Y/Tgt', 'R/G', 'Y/G', 
                     'DYAR', 'YAR', 'DVOA', 'VOA', 'EYds', 'DPI Pens', 
                     'DPI Yds', 'Rec Pts First Season']].isnull().any(axis = 1)].reset_index(drop = True)
    #Drop Eddie Royal from the Dataframe, as he is a very large outlier
    df = df[df['Player'] != 'E.Royal']
    #Create an EYds/G Stat
    df['EYds/G'] = df['EYds'] / df['G']
    #Remove redundant columns and put the remaining ones in an aesthetic order
    col_order = ['Rnd', 'Pick', 'Team', 'Player', 'First Year', 'Age Draft', 'G', 'GS', 'Tgt', 'WR Tgt Share',
                    'Rec', 'WR Rec Share', 'Catch Rate', 'Yds', 'WR Yds Share', 'Y/R', 'TD', 'WR TD Share', '1D', 
                    'Lng', 'Y/Tgt', 'R/G', 'Y/G', 'DYAR', 'YAR', 'DVOA', 'VOA', 'EYds', 'EYds/G', 'DPI Pens', 
                    'DPI Yds', 'Projected Tgt Share', 'Projected Tgt', 'Projected Rec Share', 'Projected Rec', 
                    'Projected Yds Share', 'Projected Yds', 'Projected TD Share', 'Projected TD', 
                    'Rec Pts First Season', 'Rec Pts/G First Season', 'Rec Pts Second Season', 
                    'Rec Pts/G Second Season']
    df = df[col_order].reset_index(drop = True)
    #Save the dataframe to  a CSV in the data/interim directory
    if not os.path.exists(FINAL_OUTPATH):
        os.mkdir(FINAL_OUTPATH)
    df.to_csv(FINAL_OUTPATH + '/FINAL_DATA.csv', index = False)
    return df
#@Tommy Evans-Barton

import os
import pandas as pd
import numpy as np
import json
import urllib

TOP_PATH = os.environ['PWD']

def get_receivers(year):
    #Want URL to have form 'https://www.pro-football-reference.com/years/2019/draft.htm'
    DRAFT_URL_PREFIX = 'https://www.pro-football-reference.com/years/'
    url = DRAFT_URL_PREFIX + str(year) + '/draft.htm'
    #Gets the DataFrame from this website, 0 because there are multiple dataframes on this page
    df = pd.read_html(url)[0]
    #Dropping link to college stats column
    df = df.drop('Unnamed: 28_level_0', axis = 1, errors = 'ignore')
    #Changing column names to more normal form and dropping multi-column index
    col_names = []
    for col in df.columns:
        if 'Unnamed' in col[0] or 'Misc' in col[0]:
            col_names.append(str(col[1]))
        else:    
            col_names.append(str(col[0]) + ' ' + str(col[1]))
    df.columns = col_names
    #Since data on website is broken up by round, 'Pick' appears as a value in the column
    #Except for years where draft has not occurred yet
    try:
        df = df[df['Pick'] != 'Pick']
    except KeyError:
        print('No picks for ' + str(year))
        pass
    #Want a column in the table with the year for easier merging later
    df['YEAR'] = [year] * len(df)
    df = df.drop(['Approx Val CarAV', 'Approx Val DrAV'], axis = 1, errors = 'ignore')
    #Only want receivers drafted on first two days (first three rounds)
    df = df[df['Pos'].str.lower() == 'wr'].reset_index(drop = True)
    df = df[[int(x) <= 3 for x in df['Rnd']]]
    #Drop columns that I won't use for this analysis
    df = df[['Rnd', 'Pick', 'Tm', 'Player', 'Pos', 'Age', 'YEAR']]
    return df

def get_receiving_stats(year):
    #Want url to have form 'https://www.pro-football-reference.com/years/2019/receiving.htm'
    REC_URL_PREFIX = 'https://www.pro-football-reference.com/years/'
    url = REC_URL_PREFIX + str(year) + '/receiving.htm'
    #Gets the DataFrame from this website, 0 because there are multiple dataframes on this page
    df = pd.read_html(url)[0]
    #Take only receivers
    df = df[df['Pos'].str.lower() == 'wr'].reset_index(drop = True)
    df.drop(['Rk'], inplace = True, axis = 1)
    #Want a column in the table with the year for easier merging later
    df['YEAR'] = [year] * len(df)
    return df

def get_adv_receiving_stats(year):
    #Want url to have form 'https://www.footballoutsiders.com/stats/nfl/wr/2010'
    ADV_URL_PREFIX = 'https://www.footballoutsiders.com/stats/nfl/wr/'
    url = ADV_URL_PREFIX + str(year)
    #Gets the first DataFrame from this website, receivers with at least 50 passes
    df_1 = pd.read_html(url, header = 0)[0]
    #Gets the second DataFrame from this website, receivers with 10 to 49 passes
    df_2 = pd.read_html(url, header = 0)[1]
    #Remove ranking columns
    cols = [c for c in df_1.columns if c.lower()[:2] != 'rk']
    df_1 = df_1[cols]
    df_2 = df_2[cols]
    #Remove redundant columns (columns present in other data sets used)
    df_1.drop(['TD', 'Catch  Rate', 'Catch Rate', 'CatchRate', 'FUM', 'Passes', 'Yards'], inplace = True, axis = 1, errors = 'ignore')
    df_2.drop(['TD', 'Catch  Rate', 'Catch Rate', 'CatchRate', 'FUM', 'Passes', 'Yards'], inplace = True, axis = 1, errors = 'ignore')
    #Want a column in the table with the year for easier merging later
    df_1['YEAR'] = [year] * len(df_1)
    df_2['YEAR'] = [year] * len(df_2)
    df = pd.concat([df_1, df_2]).reset_index(drop = True)
    #Remove non-data columns
    df = df[df['Team'] != 'Team'].reset_index(drop = True)
    return df

def get_data(years, outpath):
    #Check if the outpath exists, if not, make it
    if not os.path.exists(TOP_PATH + outpath):
        os.mkdir(TOP_PATH + outpath)
    #Instantiate lists to append dataframes and later concatenate them
    receivers_df_list = []
    rec_stats_df_list = []
    adv_rec_df_list = []
    #Get receivers and their stats for each year passed and append them to appropriate lists
    for y in years:
        try:
            receivers_df_list.append(get_receivers(y))
        except urllib.error.HTTPError:
            print('No receivers for draft for {year}'.format(year = y))
            pass
        try:
            rec_stats_df_list.append(get_receiving_stats(y))
        except urllib.error.HTTPError:
            print('No receiver stats for {year}'.format(year = y))
            pass
        try:
            adv_rec_df_list.append(get_adv_receiving_stats(y))
        except urllib.error.HTTPError:
            print('No advanced receiving stats for {year}'.format(year = y))
            pass
    #Concatenate dataframes and save them into .CSV format
    pd.concat(receivers_df_list).reset_index(drop = True).to_csv(TOP_PATH + outpath + '/RECEIVERS.csv', index = False)
    pd.concat(rec_stats_df_list).reset_index(drop = True).to_csv(TOP_PATH + outpath + '/REC_STATS.csv', index = False)
    pd.concat(adv_rec_df_list).reset_index(drop = True).to_csv(TOP_PATH + outpath + '/ADV_REC_STATS.csv', index = False)
    return
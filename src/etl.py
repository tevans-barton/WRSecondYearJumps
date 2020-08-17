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
    df = df[df['Pos'] == 'WR'].reset_index(drop = True)
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
    df = df[df['Pos'] == 'WR'].reset_index(drop = True)
    df.drop(['Rk'], inplace = True, axis = 1)
    return df

def get_adv_receiving_stats(year):
    #Want url to have form 'https://www.footballoutsiders.com/stats/nfl/wr/2010'
    ADV_URL_PREFIX = 'https://www.footballoutsiders.com/stats/nfl/wr/'
    url = ADV_URL_PREFIX + str(year)
    #Gets the DataFrame from this website, 0 because there are multiple dataframes on this page
    df = pd.read_html(url, header = 0)[0]
    #Remove ranking columns
    cols = [c for c in df.columns if c.lower()[:2] != 'rk']
    df = df[cols]
    #Remove redundant columns
    df.drop(['TD', 'Catch  Rate', 'Catch Rate', 'CatchRate', 'FUM'], inplace = True, axis = 1, errors = 'ignore')
    return df

def get_data(years, outpath):
    #Check if the outpath exists, if not, make it
    if not os.path.exists(TOP_PATH + outpath):
        os.mkdir(TOP_PATH + outpath)
    #Get receivers and their stats for each year passed 
    for y in years:
        try:
            get_receivers(y).to_csv(TOP_PATH + outpath + '/RECEIVERS_' + str(y) + '.csv', index = False)
        except urllib.error.HTTPError:
            print('No receivers for draft for {year}'.format(year = y))
            pass
        try:
            get_receiving_stats(y).to_csv(TOP_PATH + outpath + '/REC_STATS_' + str(y) + '.csv', index = False)
        except urllib.error.HTTPError:
            print('No receiver stats for {year}'.format(year = y))
            pass
        try:
            get_adv_receiving_stats(y).to_csv(TOP_PATH + outpath + '/ADV_REC_STATS_' + str(y) + '.csv', index = False)
        except urllib.error.HTTPError:
            print('No advanced receiving stats for {year}'.format(year = y))
            pass
    return
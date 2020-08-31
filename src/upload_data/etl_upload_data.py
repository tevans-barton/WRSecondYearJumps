#@Tommy Evans-Barton
#This code was written to scrape spotrac (as allowed by their robots.txt) for the data
#that will be uploaded with this project, as HTML scraping can be semi-discontinuous,
#and it is only a small filesize. Essentially this code is only included in the project as
#elementary/educational/due diligence

import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

TOP_PATH = os.environ['PWD']
OUTPATH = TOP_PATH + '/data/upload'

def get_upload_data():
    #Use url for spotrac to get free agents from 2020
    url = 'https://www.spotrac.com/nfl/free-agents/2020/all/wide-receiver/'
    r = requests.get(url)
    url_text = r.text
    #Create a beautiful soup object in order to parse the HTML text
    soup_2020 = BeautifulSoup(url_text)

    #Find all table entries with the columns that we wanted
    table_entries = soup_2020.find_all('td', attrs = {'class' : ['center', 'player']})

    #Get the text from each table entry and append it to a list
    tbl_list = []
    for t in table_entries:
        text = t.text
        tbl_list.append(text)

    #Convert the table list to a pandas series for easier editing
    tbl_series = pd.Series(tbl_list)
    #Make some string alterations of the entries for formatting
    tbl_series = tbl_series.str.replace('\$', '')
    tbl_series = tbl_series.str.replace('\,', '')
    tbl_series = tbl_series.str.replace(' ', '', 1)
    #Keep only the entries that should be strings
    tbl_series = tbl_series[tbl_series.str.slice(0,1).str.isalpha()]

    #Loop through the list and nest them every four (e.g. [[1, 2, 3, 4], [5, 6, 7, 8]])
    #so that it can be made into a dataframe that represents the table we want
    full_list = []
    small_list = []
    count = 0
    for t in tbl_series:
        small_list.append(t)
        count += 1
        if count % 4 == 0:
            full_list.append(small_list)
            small_list = []
    #Make a dataframe of the nested list
    df = pd.DataFrame(full_list)
    #Name the columns manually
    df.columns = ['Player', 'Pos', 'From', 'To']
    return df

def clean_download_upload_data():
    #Read in data
    df = get_upload_data()
    #Get the player column
    players = df['Player']
    #Get rid of any 'Jr' entries as they aren't in our other data
    players = players.str.replace(' Jr', '')
    #Find the indexes of the capital letters to fix the names (Currently look like 'CobbRandall Cobb')
    cap_ind = [[i for i, c in enumerate(s) if c.isupper()] for s in players]
    #Slice so that names are just first_namelast_name ('RandallCobb')
    players = [players[i][cap_ind[i][len(cap_ind[i]) // 2]:].replace(' ', '') for i in range(len(cap_ind))]
    #Find the new capital letter indices and add a space between the first and second
    cap_ind = [[i for i, c in enumerate(s) if c.isupper()] for s in players]
    players = [players[i][0 : cap_ind[i][1]] + ' ' + players[i][cap_ind[i][1] : ] for i in range(len(cap_ind))]
    df['Player'] = players
    #Change the team names to be standard with the other data
    team_dict = {
        'GB' : 'GNB',
        'JAC' : 'JAX',
        'KC' : 'KAN',
        'LVR' : 'OAK',
        'NE' : 'NWE',
        'NO' : 'NOR',
        'SF' : 'SFO',
        'TB' : 'TAM'
    }
    df['Tm'] = df['To'].replace(team_dict)
    #Drop unnecessary columns
    df.drop(['From', 'To'], axis = 1, inplace = True)
    #Make directory and save dataframe as CSV file
    df['YEAR'] = [2020] * len(df)
    if not os.path.exists(TOP_PATH + '/data'):
        os.mkdir(TOP_PATH + '/data')
    if not os.path.exists(OUTPATH):
        os.mkdir(OUTPATH)
    df.to_csv(OUTPATH + '/2020_FA_WR.csv', index = False)
    return df
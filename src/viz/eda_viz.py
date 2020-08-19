#@Tommy Evans-Barton

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
TOP_PATH = os.environ['PWD']
VIZ_OUTPATH = TOP_PATH + '/visualizations'
EDA_OUTPATH = TOP_PATH + '/visualizations/eda'

def round_distribution(savefig = False):
    #Read in data
    try:
        receivers = pd.read_csv(TOP_PATH + '/data/raw/RECEIVERS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the count of receivers drafted per round
    receivers['Rnd'].value_counts().sort_index().plot(kind = 'bar')
    #Label the plot
    plt.title('Distribution of Receivers by Round', fontsize = 18)
    plt.xlabel('Round', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/round_distribution.png')
    plt.show()

def pick_distribution(savefig = False):
    #Read in data
    try:
        receivers = pd.read_csv(TOP_PATH + '/data/raw/RECEIVERS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot a histogram of the receiver picks with bins of size 10
    receivers['Pick'].hist(bins = range(1, 112, 10))
    #Label the plot
    plt.title('Distribution of Receivers by 10 Picks', fontsize = 18)
    plt.xlabel('Pick', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(1, 112, 10), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/pick_distribution.png')
    plt.show()


def age_distribution(savefig = False):
    #Read in data
    try:
        receivers = pd.read_csv(TOP_PATH + '/data/raw/RECEIVERS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the receivers by draft age
    receivers['Age'].value_counts().sort_index().plot(kind = 'bar')
    #Label the plot
    plt.title('Distribution of Receivers by Draft Age', fontsize = 18)
    plt.xlabel('Age', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/age_distribution.png')
    plt.show()
    return

def game_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the games active for each receiver
    rec_stats['G'].hist(bins = range(0,19,2))
    #Label the plot
    plt.title('Distribution of Games Active', fontsize = 18)
    plt.xlabel('Games', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0,18,2), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/games_distribution.png')
    plt.show()
    return

def gs_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the games started for each receiver
    rec_stats['GS'].hist(bins = range(0,19,2))
    #Label the plot
    plt.title('Distribution of Games Started', fontsize = 18)
    plt.xlabel('Games Started', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0,18,2), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/games_started_distribution.png')
    plt.show()
    return

def tgt_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the targets for each receiver
    rec_stats['Tgt'].hist(bins = range(0, 226, 25))
    #Label the plot
    plt.title('Distribution of Targets', fontsize = 18)
    plt.xlabel('Targets', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 226, 25), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/targets_distribution.png')
    plt.show()
    return

def rec_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the receptions for each receiver
    rec_stats['Rec'].hist(bins = range(0, 161, 20))
    #Label the plot
    plt.title('Distribution of Receptions', fontsize = 18)
    plt.xlabel('Receptions', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 161, 20), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/receptions_distribution.png')
    plt.show()
    return

def yards_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the receiving yards for each receiver
    rec_stats['Yds'].hist(bins = range(0, 2001, 200))
    #Label the plot
    plt.title('Distribution of Receiving Yards', fontsize = 18)
    plt.xlabel('Receiving Yards', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 2001, 200), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/rec_yards_distribution.png')
    plt.show()
    return

def ypr_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the receiving yards for each receiver
    rec_stats['Y/R'].hist(bins = range(0, 26, 2))
    #Label the plot
    plt.title('Distribution of Yards per Reception', fontsize = 18)
    plt.xlabel('Yards per Reception', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 26, 2), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/ypr_distribution.png')
    plt.show()
    return

def td_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the touchdowns for each receiver
    rec_stats['TD'].hist(bins = range(0, 17))
    #Label the plot
    plt.title('Distribution of Touchdowns', fontsize = 18)
    plt.xlabel('Touchdowns', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 17, 2), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/td_distribution.png')
    plt.show()
    return

def fd_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the first downs for each receiver
    rec_stats['1D'].hist(bins = range(0, 101, 10))
    #Label the plot
    plt.title('Distribution of First Downs', fontsize = 18)
    plt.xlabel('First Downs', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 101, 10), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/fd_distribution.png')
    plt.show()
    return

def long_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the longest receptions for each receiver
    rec_stats['Lng'].hist(bins = range(0, 101, 10))
    #Label the plot
    plt.title('Distribution of Longest Receptions', fontsize = 18)
    plt.xlabel('Longest Reception (Yds)', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 101, 10), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/lng_distribution.png')
    plt.show()
    return

def ypt_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Yards per Target for each receiver
    rec_stats['Y/Tgt'].hist(bins = range(0, 19))
    #Label the plot
    plt.title('Distribution of Yards per Target', fontsize = 18)
    plt.xlabel('Yards per Target', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 19), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/ypt_distribution.png')
    plt.show()
    return

def rpg_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Receptions per Game for each receiver
    rec_stats['R/G'].hist(bins = range(0, 11))
    #Label the plot
    plt.title('Distribution of Receptions per Game', fontsize = 18)
    plt.xlabel('Receptions per Game', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 11), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/rpg_distribution.png')
    plt.show()
    return

def ypg_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Yards per Game for each receiver
    rec_stats['Y/G'].hist(bins = range(0, 136, 15))
    #Label the plot
    plt.title('Distribution of Yards per Game', fontsize = 18)
    plt.xlabel('Yards per Game', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 136, 15), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/ypg_distribution.png')
    plt.show()
    return

def fmb_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Fumbles for each receiver
    rec_stats['Fmb'].hist(bins = range(0, 9, 1))
    #Label the plot
    plt.title('Distribution of Fumbles', fontsize = 18)
    plt.xlabel('Fumbles', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 9, 1), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/fmb_distribution.png')
    plt.show()
    return

def ctr_distribution(savefig = False):
    #Read in data
    try:
        rec_stats = pd.read_csv(TOP_PATH + '/data/raw/REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Get the catch rates, remove the percent signs, and convert them to floats
    ctch_rt = pd.to_numeric(rec_stats['Ctch%'].str.replace('%', ''))
    #Plot the histogram of these values
    ctch_rt.hist(bins = range(0, 101, 10))
    #Label the plot
    plt.title('Distribution of Catch Rates', fontsize = 18)
    plt.xlabel('Catch Rates (%)', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 101, 10), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/ctr_distribution.png')
    plt.show()
    return

def dyar_distribution(savefig = False):
    #Read in data
    try:
        adv_stats = pd.read_csv(TOP_PATH + '/data/raw/ADV_REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of DYAR
    adv_stats['DYAR'].hist(bins = range(-300, 601, 100))
    #Label the plot
    plt.title('Distribution of DYAR', fontsize = 18)
    plt.xlabel('DYAR', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-300, 601, 100), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/dyar_distribution.png')
    plt.show()
    return

def yar_distribution(savefig = False):
    #Read in data
    try:
        adv_stats = pd.read_csv(TOP_PATH + '/data/raw/ADV_REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of YAR
    adv_stats['YAR'].hist(bins = range(-200, 601, 100))
    #Label the plot
    plt.title('Distribution of YAR', fontsize = 18)
    plt.xlabel('YAR', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-200, 601, 100), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/yar_distribution.png')
    plt.show()
    return

def eyds_distribution(savefig = False):
    #Read in data
    try:
        adv_stats = pd.read_csv(TOP_PATH + '/data/raw/ADV_REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of EYds
    adv_stats['EYds'].hist(bins = range(0, 2251, 250))
    #Label the plot
    plt.title('Distribution of Effective Yards', fontsize = 18)
    plt.xlabel('EYds', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 2251, 250), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/eyds_distribution.png')
    plt.show()
    return

def dvoa_distribution(savefig = False):
    #Read in data
    try:
        adv_stats = pd.read_csv(TOP_PATH + '/data/raw/ADV_REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Convert dvoa to a format where distribution can be plotted
    dvoa = pd.to_numeric(adv_stats['DVOA'].str.replace('%', ''))
    #Plot the distribution of DVOA
    dvoa.hist(bins = range(-50, 71, 10))
    #Label the plot
    plt.title('Distribution of DVOA', fontsize = 18)
    plt.xlabel('DVOA (%)', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-50, 71, 10), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/dvoa_distribution.png')
    plt.show()
    return

def voa_distribution(savefig = False):
    #Read in data
    try:
        adv_stats = pd.read_csv(TOP_PATH + '/data/raw/ADV_REC_STATS.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data)')
        return
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Convert dvoa to a format where distribution can be plotted
    voa = pd.to_numeric(adv_stats['VOA'].str.replace('%', ''))
    #Plot the distribution of VOA
    voa.hist(bins = range(-60, 71, 10))
    #Label the plot
    plt.title('Distribution of VOA', fontsize = 18)
    plt.xlabel('VOA (%)', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-60, 71, 10), fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_OUTPATH):
            os.mkdir(EDA_OUTPATH)
        plt.savefig(EDA_OUTPATH + '/voa_distribution.png')
    plt.show()
    return
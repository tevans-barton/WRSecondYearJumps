#@Tommy Evans-Barton

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
TOP_PATH = os.environ['PWD']
VIZ_OUTPATH = TOP_PATH + '/visualizations'
EDA_POST_OUTPATH = TOP_PATH + '/visualizations/eda_post_cleaning'

def game_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the games active for each receiver
    df_model['G'].hist(bins = range(0,19,2))
    #Label the plot
    plt.title('Distribution of Games Active', fontsize = 18)
    plt.xlabel('Games', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0,18,2), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/games_distribution_modeling.png')
    plt.show()
    return

def gs_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the games started for each receiver
    df_model['GS'].hist(bins = range(0,19,2))
    #Label the plot
    plt.title('Distribution of Games Started', fontsize = 18)
    plt.xlabel('Games Started', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0,18,2), fontsize = 12, rotation = 0)
    plt.yticks(range(0,26,5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/gs_distribution_modeling.png')
    plt.show()
    return

def tgt_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the targets for each receiver
    df_model['Tgt'].hist(bins = range(0, 141, 20))
    #Label the plot
    plt.title('Distribution of Targets', fontsize = 18)
    plt.xlabel('Targets', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0,141,20), fontsize = 12, rotation = 0)
    plt.yticks(range(0,31,5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/tgt_distribution_modeling.png')
    plt.show()
    return

def tgt_share_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the WR target share for each receiver
    df_model['WR Tgt Share'].hist(bins = np.arange(0, .701, .05))
    #Label the plot
    plt.title('Distribution of WR Target Share', fontsize = 18)
    plt.xlabel('WR Target Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(0, .701, .1), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 31, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/tgtshare_distribution_modeling.png')
    plt.show()
    return

def rec_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the receptions for each receiver
    df_model['Rec'].hist(bins = range(0, 101, 10))
    #Label the plot
    plt.title('Distribution of Receptions', fontsize = 18)
    plt.xlabel('Receptions', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 101, 20), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 26, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/rec_distribution_modeling.png')
    plt.show()
    return

def recshare_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the WR reception share for each receiver
    df_model['WR Rec Share'].hist(bins = np.arange(0, .701, .05))
    #Label the plot
    plt.title('Distribution of WR Reception Share', fontsize = 18)
    plt.xlabel('WR Reception Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(0, .701, .1), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 26, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/recshare_distribution_modeling.png')
    plt.show()
    return


def catchrate_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the catch percentage for each receiver
    df_model['Catch Rate'].hist(bins = np.arange(0, 1.01, .1))
    #Label the plot
    plt.title('Distribution of Catch Rate', fontsize = 18)
    plt.xlabel('Catch Rate', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(0, 1.01, .2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 61, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/catchrate_distribution_modeling.png')
    plt.show()
    return

def yds_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the receiving yards for each receiver
    df_model['Yds'].hist(bins = range(0, 1401, 100))
    #Label the plot
    plt.title('Distribution of Receiving Yards', fontsize = 18)
    plt.xlabel('Receiving Yards', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 1401, 200), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 21, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/yds_distribution_modeling.png')
    plt.show()
    return

def ydsshare_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the WR yards share for each receiver
    df_model['WR Yds Share'].hist(bins = np.arange(0, .801, .05))
    #Label the plot
    plt.title('Distribution of WR Yards Share', fontsize = 18)
    plt.xlabel('WR Yards Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(0, .801, .1), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 21, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ydsshare_distribution_modeling.png')
    plt.show()
    return

def ypr_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the yards per reception for each receiver
    df_model['Y/R'].hist(bins = range(4, 21, 2))
    #Label the plot
    plt.title('Distribution of Yards per Reception', fontsize = 18)
    plt.xlabel('Yards per Reception', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(4, 21, 2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 41, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ypr_distribution_modeling.png')
    plt.show()
    return

def td_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the touchdowns for each receiver
    df_model['TD'].hist(bins = range(0, 13, 2))
    #Label the plot
    plt.title('Distribution of Touchdowns', fontsize = 18)
    plt.xlabel('Touchdowns', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 13, 2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 41, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/td_distribution_modeling.png')
    plt.show()
    return

def tdshare_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the WR touchdown share for each receiver
    df_model['WR TD Share'].hist(bins = np.arange(0, .81, .1))
    #Label the plot
    plt.title('Distribution of WR Touchdown Share', fontsize = 18)
    plt.xlabel('WR Touchdown Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(0, .81, .1), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 36, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/tdshare_distribution_modeling.png')
    plt.show()
    return

def firstdown_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the first downs for each receiver
    df_model['1D'].hist(bins = range(0, 66, 5))
    #Label the plot
    plt.title('Distribution of First Downs', fontsize = 18)
    plt.xlabel('First Downs', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 66, 5), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 26, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/firstdown_distribution_modeling.png')
    plt.show()
    return

def longrec_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the longest reception for each receiver
    df_model['Lng'].hist(bins = range(0, 101, 10))
    #Label the plot
    plt.title('Distribution of Longest Reception', fontsize = 18)
    plt.xlabel('Longest Reception', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 101, 10), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 31, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/longrec_distribution_modeling.png')
    plt.show()
    return

def ypt_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the yards per target for each receiver
    df_model['Y/Tgt'].hist(bins = range(0, 13, 2))
    #Label the plot
    plt.title('Distribution of Yards per Target', fontsize = 18)
    plt.xlabel('Yards per Target', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 13, 2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 51, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ypt_distribution_modeling.png')
    plt.show()
    return

def recpergame_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the receptions per game for each receiver
    df_model['R/G'].hist(bins = range(0, 9))
    #Label the plot
    plt.title('Distribution of Receptions per Game', fontsize = 18)
    plt.xlabel('Receptions per Game', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 9), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 36, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/recpergame_distribution_modeling.png')
    plt.show()
    return

def ypg_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the yards per game for each receiver
    df_model['Y/G'].hist(bins = range(0, 121, 20))
    #Label the plot
    plt.title('Distribution of Yards per Game', fontsize = 18)
    plt.xlabel('Yards per Game', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 141, 20), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 51, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ypg_distribution_modeling.png')
    plt.show()
    return

def dyar_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the DYAR for each receiver
    df_model['DYAR'].hist(bins = range(-151, 451, 50))
    #Label the plot
    plt.title('Distribution of DYAR', fontsize = 18)
    plt.xlabel('DYAR', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-150, 451, 50), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 36, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dyar_distribution_modeling.png')
    plt.show()
    return

def yar_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the YAR for each receiver
    df_model['YAR'].hist(bins = range(-151, 451, 50))
    #Label the plot
    plt.title('Distribution of YAR', fontsize = 18)
    plt.xlabel('YAR', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-150, 451, 50), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 36, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/yar_distribution_modeling.png')
    plt.show()
    return

def dvoa_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the DVOA for each receiver
    df_model['DVOA'].hist(bins = range(-80, 61, 20))
    #Label the plot
    plt.title('Distribution of DVOA', fontsize = 18)
    plt.xlabel('DVOA', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-80, 61, 20), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 51, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dvoa_distribution_modeling.png')
    plt.show()
    return

def voa_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the VOA for each receiver
    df_model['VOA'].hist(bins = range(-80, 61, 20))
    #Label the plot
    plt.title('Distribution of VOA', fontsize = 18)
    plt.xlabel('VOA', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-80, 61, 20), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 51, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/voa_distribution_modeling.png')
    plt.show()
    return

def eyds_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the EYds for each receiver
    df_model['EYds'].hist(bins = range(0, 1501, 100))
    #Label the plot
    plt.title('Distribution of EYds', fontsize = 18)
    plt.xlabel('EYds', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 1501, 200), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 21, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/eyds_distribution_modeling.png')
    plt.show()
    return

def dpipens_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the DPI Pens for each receiver
    df_model['DPI Pens'].hist(bins = range(0, 9))
    #Label the plot
    plt.title('Distribution of DPI Pens', fontsize = 18)
    plt.xlabel('DPI Pens', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 9), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 51, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dpipens_distribution_modeling.png')
    plt.show()
    return

def dpiyds_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the DPI Yds for each receiver
    df_model['DPI Yds'].hist(bins = range(0, 176, 25))
    #Label the plot
    plt.title('Distribution of DPI Yards', fontsize = 18)
    plt.xlabel('DPI Yards', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 176, 25), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 81, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dpiyds_distribution_modeling.png')
    plt.show()
    return

def projtgtshare_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected Target Share for each receiver
    df_model['Projected Tgt Share'].hist(bins = np.arange(-.4, 1.01, .1))
    #Label the plot
    plt.title('Distribution of Projected WR Target Share', fontsize = 18)
    plt.xlabel('Projected WR Target Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(-.4, 1.01, .2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 31, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtgtshare_distribution_modeling.png')
    plt.show()
    return

def projtgt_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected Targets for each receiver
    df_model['Projected Tgt'].hist(bins = range(-150, 351, 50))
    #Label the plot
    plt.title('Distribution of Projected Targets', fontsize = 18)
    plt.xlabel('Projected Targets', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-150, 351, 50), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 36, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtgt_distribution_modeling.png')
    plt.show()
    return

def projrecshare_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected WR Reception Share for each receiver
    df_model['Projected Rec Share'].hist(bins = np.arange(-.4, 1.01, .2))
    #Label the plot
    plt.title('Distribution of Projected WR Reception Share', fontsize = 18)
    plt.xlabel('Projected WR Reception Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(-.4, 1.01, .2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 51, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projrecshare_distribution_modeling.png')
    plt.show()
    return

def projrec_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected Receptions for each receiver
    df_model['Projected Rec'].hist(bins = range(-75, 176, 25))
    #Label the plot
    plt.title('Distribution of Projected Receptions', fontsize = 18)
    plt.xlabel('Projected Receptions', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-75, 176, 25), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 31, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projrec_distribution_modeling.png')
    plt.show()
    return

def projydsshare_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected WR Yard Share for each receiver
    df_model['Projected Yds Share'].hist(bins = np.arange(-.4, 1.01, .2))
    #Label the plot
    plt.title('Distribution of Projected WR Yard Share', fontsize = 18)
    plt.xlabel('Projected WR Yard Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(-.4, 1.01, .2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 41, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projydsshare_distribution_modeling.png')
    plt.show()
    return

def projyds_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected Yards for each receiver
    df_model['Projected Yds'].hist(bins = range(-1000, 3001, 500))
    #Label the plot
    plt.title('Distribution of Projected Yards', fontsize = 18)
    plt.xlabel('Projected Yards', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-1000, 3001, 500), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 41, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projyds_distribution_modeling.png')
    plt.show()
    return

def projtdshare_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected WR TD Share for each receiver
    df_model['Projected TD Share'].hist(bins = np.arange(-.6, 1.21, .2))
    #Label the plot
    plt.title('Distribution of Projected WR TD Share', fontsize = 18)
    plt.xlabel('Projected WR TD Share', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(np.arange(-.6, 1.21, .2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 36, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtdshare_distribution_modeling.png')
    plt.show()
    return

def projtd_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Projected TDs for each receiver
    df_model['Projected TD'].hist(bins = range(-10, 21, 5))
    #Label the plot
    plt.title('Distribution of Projected Touchdowns', fontsize = 18)
    plt.xlabel('Projected Touchdowns', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(-10, 21, 5), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 71, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtd_distribution_modeling.png')
    plt.show()
    return

def recptsfirst_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Receiving Points in their First Year for each receiver
    df_model['Rec Pts First Season'].hist(bins = range(0, 226, 25))
    #Label the plot
    plt.title('Distribution of Rec Points in First Season', fontsize = 18)
    plt.xlabel('Rec Points in First Season', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 226, 25), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 31, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/recptsfirst_distribution_modeling.png')
    plt.show()
    return

def recptspergamefirst_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Receiving Points per game in their First Year for each receiver
    df_model['Rec Pts/G First Season'].hist(bins = range(0, 13, 2))
    #Label the plot
    plt.title('Distribution of Rec Points per Game in First Season', fontsize = 18)
    plt.xlabel('Rec Points per Game in First Season', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 13, 2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 41, 10), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/recptspergamefirst_distribution_modeling.png')
    plt.show()
    return

def recptssecond_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Receiving Points in their Second Year for each receiver
    df_model['Rec Pts Second Season'].hist(bins = range(0, 226, 25))
    #Label the plot
    plt.title('Distribution of Rec Points in Second Season', fontsize = 18)
    plt.xlabel('Rec Points in Second Season', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 226, 25), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 26, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/recptssecond_distribution_modeling.png')
    plt.show()
    return

def recptspergamesecond_distribution(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the distribution of the Receiving Points per game in their Second Year for each receiver
    df_model['Rec Pts/G Second Season'].hist(bins = range(0, 17, 2))
    #Label the plot
    plt.title('Distribution of Rec Points per Game in Second Season', fontsize = 18)
    plt.xlabel('Rec Points per Game in Second Season', fontsize = 14)
    plt.ylabel('Count', fontsize = 14)
    plt.xticks(range(0, 17, 2), fontsize = 12, rotation = 0)
    plt.yticks(range(0, 31, 5), fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/recptspergamesecond_distribution_modeling.png')
    plt.show()
    return
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
    plt.yticks(range(0,31,5), fontsize = 12, rotation = 0)
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
    plt.yticks(range(0,26,5), fontsize = 12, rotation = 0)
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
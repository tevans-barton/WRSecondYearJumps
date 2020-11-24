import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



plt.style.use('ggplot')
TOP_PATH = os.environ['PWD']
VIZ_OUTPATH = TOP_PATH + '/visualizations'
EDA_POST_OUTPATH = TOP_PATH + '/visualizations/eda_post_cleaning'

def round_median_targets(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Group by round and find the median of the target for each round
    grouped = df_model[['Rnd', 'Rec Pts/G Second Season']].groupby('Rnd').median()
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the different median targets for each round
    plt.bar(grouped.index.astype('str'), grouped['Rec Pts/G Second Season'])
    #Label the plot
    plt.title('Median Rec Pts/G Second Year by Round', fontsize = 18)
    plt.xlabel('Round', fontsize = 14)
    plt.ylabel('Median Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/rnd_vs_target.png')
    plt.show()
    return

def pick_vs_target(savefig = False):
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
    #Plot the draft selection vs. rec pts/g second season
    plt.scatter(df_model['Pick'], df_model['Rec Pts/G Second Season'])
    #Label the plot
    plt.title('Draft Pick vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Pick', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/pick_vs_target.png')
    plt.show()
    return

def age_median_targets(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Group by round and find the median of the target for each round
    grouped = df_model[['Age Draft', 'Rec Pts/G Second Season']].groupby('Age Draft').median()
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the different median targets for each draft age
    plt.bar(grouped.index.astype(str), grouped['Rec Pts/G Second Season'])
    #Label the plot
    plt.title('Median Rec Pts/G Second Year by Age Drafted', fontsize = 18)
    plt.xlabel('Age Drafted', fontsize = 14)
    plt.ylabel('Median Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/age_vs_target.png')
    plt.show()
    return

def g_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    grouped = df_model[['G', 'Rec Pts/G Second Season']].groupby('G').median()
    plt.figure(figsize = (8.5, 5.5))
    #Plot the games active vs. rec pts/g second season
    plt.bar(grouped.index.astype('int').astype('str'), grouped['Rec Pts/G Second Season'])
    #Label the plot
    plt.title('Games Active vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Games Active', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/g_vs_target.png')
    plt.show()
    return
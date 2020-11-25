import os
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.linear_model import LinearRegression
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
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Pick']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Pick']), np.max(df_model['Pick']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Pick vs. rec pts/g second season
    plt.scatter(df_model['Pick'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Pick']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Draft Selection vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Pick', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
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

def year_median_targets(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Group by round and find the median of the target for each round
    grouped = df_model[['First Year', 'Rec Pts/G Second Season']].groupby('First Year').median()
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the different median targets for each draft age
    plt.bar(grouped.index.astype(str), grouped['Rec Pts/G Second Season'])
    #Label the plot
    plt.title('Median Rec Pts/G Second Year by First Year in the League', fontsize = 18)
    plt.xlabel('First Year in the League', fontsize = 14)
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
        plt.savefig(EDA_POST_OUTPATH + '/year_vs_target.png')
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

def gs_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Change the size of the figure
    grouped = df_model[['GS', 'Rec Pts/G Second Season']].groupby('GS').median()
    plt.figure(figsize = (8.5, 5.5))
    #Plot the games started vs. rec pts/g second season
    plt.bar(grouped.index.astype('int').astype('str'), grouped['Rec Pts/G Second Season'])
    #Label the plot
    plt.title('Games Started vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Games Started', fontsize = 14)
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
        plt.savefig(EDA_POST_OUTPATH + '/gs_vs_target.png')
    plt.show()
    return

def tgt_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Tgt']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Tgt']), np.max(df_model['Tgt']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the targets vs. rec pts/g second season
    plt.scatter(df_model['Tgt'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Tgt']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Targets vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Targets', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/tgt_vs_target.png')
    plt.show()
    return

def sqrt_tgtshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['WR Tgt Share'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['WR Tgt Share'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['WR Tgt Share'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of WR Target Share vs. rec pts/g second season
    plt.scatter(df_model['WR Tgt Share'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['WR Tgt Share']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root WR Target Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root WR Target Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_tgtshare_vs_target.png')
    plt.show()
    return

def sqrt_rec_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Rec'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Rec'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['Rec'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of Receptions vs. rec pts/g second season
    plt.scatter(df_model['Rec'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Rec']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root Receptions vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root Receptions', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_rec_vs_target.png')
    plt.show()
    return

def sqrt_recshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['WR Rec Share'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['WR Rec Share'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['WR Rec Share'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of WR Rec Share vs. rec pts/g second season
    plt.scatter(df_model['WR Rec Share'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['WR Rec Share']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root WR Rec Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root WR Rec Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_recshare_vs_target.png')
    plt.show()
    return

def ctchrate_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Catch Rate']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Catch Rate']), np.max(df_model['Catch Rate']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Catch Rate vs. rec pts/g second season
    plt.scatter(df_model['Catch Rate'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Catch Rate']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Catch Rate vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Catch Rate', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ctchrate_vs_target.png')
    plt.show()
    return

def sqrt_yds_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Yds'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Yds'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['Yds'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of Yards vs. rec pts/g second season
    plt.scatter(df_model['Yds'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Yds']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root Yards vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root Yards', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_yds_vs_target.png')
    plt.show()
    return

def sqrt_ydsshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['WR Yds Share'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['WR Yds Share'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['WR Yds Share'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of WR Yards Share vs. rec pts/g second season
    plt.scatter(df_model['WR Yds Share'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['WR Yds Share']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root WR Yards Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root WR Yards Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_ydsshare_vs_target.png')
    plt.show()
    return

def ypr_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Y/R']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Y/R']), np.max(df_model['Y/R']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the yards per reception vs. rec pts/g second season
    plt.scatter(df_model['Y/R'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Y/R']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Yards per Reception vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Yards per Reception', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ypr_vs_target.png')
    plt.show()
    return

def td_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['TD']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['TD']), np.max(df_model['TD']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the yards per reception vs. rec pts/g second season
    plt.scatter(df_model['TD'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['TD']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Touchdowns vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Touchdowns', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/td_vs_target.png')
    plt.show()
    return

def tdshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['WR TD Share']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['WR TD Share']), np.max(df_model['WR TD Share']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the yards per reception vs. rec pts/g second season
    plt.scatter(df_model['WR TD Share'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['WR TD Share']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('WR Touchdown Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('WR Touchdown Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/tdshare_vs_target.png')
    plt.show()
    return

def sqrt_1d_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['1D'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['1D'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['1D'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of First Downs vs. rec pts/g second season
    plt.scatter(df_model['1D'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['1D']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root First Downs vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root First Downs', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_1d_vs_target.png')
    plt.show()
    return

def long_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Lng']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Lng']), np.max(df_model['Lng']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the longest reception vs. rec pts/g second season
    plt.scatter(df_model['Lng'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Lng']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Longest Reception vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Longest Reception', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/long_vs_target.png')
    plt.show()
    return

def ypt_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Y/Tgt']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Y/Tgt']), np.max(df_model['Y/Tgt']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Yards per Target vs. rec pts/g second season
    plt.scatter(df_model['Y/Tgt'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Y/Tgt']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Yards per Target vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Yards per Target', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ypt_vs_target.png')
    plt.show()
    return

def sqrt_rpg_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['R/G'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['R/G'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['R/G'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of Receptions per Game vs. rec pts/g second season
    plt.scatter(df_model['R/G'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['R/G']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root Receptions per Game vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root Receptions per Game', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_rpg_vs_target.png')
    plt.show()
    return

def ypg_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Y/G']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Y/G']), np.max(df_model['Y/G']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Yards per Game vs. rec pts/g second season
    plt.scatter(df_model['Y/G'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Y/G']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Yards per Game vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Yards per Game', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/ypg_vs_target.png')
    plt.show()
    return

def dyar_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['DYAR']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['DYAR']), np.max(df_model['DYAR']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the DYAR vs. rec pts/g second season
    plt.scatter(df_model['DYAR'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['DYAR']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('DYAR vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('DYAR', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dyar_vs_target.png')
    plt.show()
    return

def yar_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['YAR']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['YAR']), np.max(df_model['YAR']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the YAR vs. rec pts/g second season
    plt.scatter(df_model['YAR'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['YAR']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('YAR vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('YAR', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/yar_vs_target.png')
    plt.show()
    return

def dvoa_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['DVOA']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['DVOA']), np.max(df_model['DVOA']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the DVOA vs. rec pts/g second season
    plt.scatter(df_model['DVOA'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['DVOA']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('DVOA vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('DVOA', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dvoa_vs_target.png')
    plt.show()
    return

def voa_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['VOA']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['VOA']), np.max(df_model['VOA']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the VOA vs. rec pts/g second season
    plt.scatter(df_model['VOA'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['VOA']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('VOA vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('VOA', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/voa_vs_target.png')
    plt.show()
    return

def sqrt_eyds_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['EYds'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['EYds'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['EYds'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of EYds vs. rec pts/g second season
    plt.scatter(df_model['EYds'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['EYds']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root EYds vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root EYds', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_eyds_vs_target.png')
    plt.show()
    return

def dpipens_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['DPI Pens']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['DPI Pens']), np.max(df_model['DPI Pens']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the DPI Pens vs. rec pts/g second season
    plt.scatter(df_model['DPI Pens'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['DPI Pens']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('DPI Pens vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('DPI Pens', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dpipens_vs_target.png')
    plt.show()
    return

def dpiyds_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['DPI Yds']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['DPI Yds']), np.max(df_model['DPI Yds']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the DPI Yards vs. rec pts/g second season
    plt.scatter(df_model['DPI Yds'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['DPI Yds']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('DPI Yards vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('DPI Yards', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/dpiyds_vs_target.png')
    plt.show()
    return

def projtgtshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected Tgt Share']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected Tgt Share']), np.max(df_model['Projected Tgt Share']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Target Share vs. rec pts/g second season
    plt.scatter(df_model['Projected Tgt Share'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected Tgt Share']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Target Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Target Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtgtshare_vs_target.png')
    plt.show()
    return

def projtgt_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected Tgt']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected Tgt']), np.max(df_model['Projected Tgt']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Targets vs. rec pts/g second season
    plt.scatter(df_model['Projected Tgt'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected Tgt']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Targets vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Targets', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtgt_vs_target.png')
    plt.show()
    return

def projrecshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected Rec Share']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected Rec Share']), np.max(df_model['Projected Rec Share']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Reception Share vs. rec pts/g second season
    plt.scatter(df_model['Projected Rec Share'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected Rec Share']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Reception Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Reception Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projrecshare_vs_target.png')
    plt.show()
    return

def projrec_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected Rec']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected Rec']), np.max(df_model['Projected Rec']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Receptions vs. rec pts/g second season
    plt.scatter(df_model['Projected Rec'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected Rec']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Receptions vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Receptions', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projrec_vs_target.png')
    plt.show()
    return

def projydsshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected Yds Share']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected Yds Share']), np.max(df_model['Projected Yds Share']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Yards Share vs. rec pts/g second season
    plt.scatter(df_model['Projected Yds Share'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected Yds Share']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Yards Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Yards Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projydsshare_vs_target.png')
    plt.show()
    return

def projyds_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected Yds']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected Yds']), np.max(df_model['Projected Yds']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Yards vs. rec pts/g second season
    plt.scatter(df_model['Projected Yds'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected Yds']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Yards vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Yards', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projyds_vs_target.png')
    plt.show()
    return

def projtdshare_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected TD Share']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected TD Share']), np.max(df_model['Projected TD Share']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Touchdown Share vs. rec pts/g second season
    plt.scatter(df_model['Projected TD Share'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected TD Share']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Touchdown Share vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Touchdown Share', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtdshare_vs_target.png')
    plt.show()
    return

def projtd_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Projected TD']).reshape(-1, 1), np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Projected TD']), np.max(df_model['Projected TD']))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Projected Touchdowns vs. rec pts/g second season
    plt.scatter(df_model['Projected TD'], df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Projected TD']).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Projected Touchdowns vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Projected Touchdowns', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/projtd_vs_target.png')
    plt.show()
    return

def sqrt_recptsfirst_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Rec Pts First Season'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Rec Pts First Season'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['Rec Pts First Season'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of Rec Pts First Season vs. rec pts/g second season
    plt.scatter(df_model['Rec Pts First Season'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Rec Pts First Season']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root Rec Pts First Season vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root Rec Pts First Season', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_recptsfirst_vs_target.png')
    plt.show()
    return

def sqrt_recppgfirst_vs_target(savefig = False):
    #Read in data
    try:
        df = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
    except FileNotFoundError:
        print('File Not Found Error (try running etl.get_data, processing.clean_all_data, and processing.merge_data')
        return
    #Get the subset of the data that will be used to build the model
    df_model = df[df['First Year'] < 2019].reset_index(drop = True)
    #Make the linear model to plot the line
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(df_model['Rec Pts/G First Season'].apply(lambda x : x ** .5)).reshape(-1, 1), 
                                    np.array(df_model['Rec Pts/G Second Season']))
    x_reg = np.linspace(np.min(df_model['Rec Pts/G First Season'].apply(lambda x : x ** .5)), 
                                    np.max(df_model['Rec Pts/G First Season'].apply(lambda x : x ** .5)))
    y_reg = lin_reg.coef_[0] * x_reg + lin_reg.intercept_
    #Change the size of the figure
    plt.figure(figsize = (8.5, 5.5))
    #Plot the Square Root of Rec Pts/G First Season vs. rec pts/g second season
    plt.scatter(df_model['Rec Pts/G First Season'].apply(lambda x : x **.5), df_model['Rec Pts/G Second Season'])
    #Plot the regression line
    (plt.plot(x_reg, y_reg, c = 'b', label = 'y $\\approx$ {m}x + {b}\n$r^2$ = {corr}'.format(m = round(lin_reg.coef_[0], 5), 
                                                    b = round(lin_reg.intercept_, 5), 
                                                    corr = lin_reg.score(np.array(df_model['Rec Pts/G First Season']
                                                    .apply(lambda x : x ** .5)).reshape(-1, 1), 
                                                    df_model['Rec Pts/G Second Season']))))
    #Label the plot
    plt.title('Square Root Rec Pts/G First Season vs. Rec Pts/G Second Season', fontsize = 18)
    plt.xlabel('Square Root Rec Pts/G First Season', fontsize = 14)
    plt.ylabel('Rec Pts/G in Second Year', fontsize = 14)
    plt.xticks(fontsize = 12, rotation = 0)
    plt.yticks(fontsize = 12, rotation = 0)
    plt.legend()
    #Tight layout to get it to save the figure correctly
    plt.tight_layout()
    #If safefig passed as true, save the figure to the eda visualizations folder
    if savefig:
        if not os.path.exists(VIZ_OUTPATH):
            os.mkdir(VIZ_OUTPATH)
        if not os.path.exists(EDA_POST_OUTPATH):
            os.mkdir(EDA_POST_OUTPATH)
        plt.savefig(EDA_POST_OUTPATH + '/sqrt_recppgfirst_vs_target.png')
    plt.show()
    return
#@Tommy Evans-Barton

import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import scipy.stats

TOP_PATH = os.environ['PWD']
YEAR_CUTOFF = 2019

REG_FEATURES = ['Tgt', 'Rec', 'Catch Rate', 'Yds', 'Y/R', 
                'TD', 'Y/Tgt', 'R/G','Y/G']
REG_FEATURES_SQRT = ['Rec', 'Yds', 'R/G']
REG_FEATURES_NO_TRANSFORM= ['Tgt', 'Catch Rate', 'Y/R', 'TD', 'Y/Tgt', 'Y/G']

TARGET = 'Rec Pts/G Second Season'

TEST_SIZE = .3

ADV_FEATURES = ['DYAR', 'YAR', 'DVOA', 'VOA', 'EYds', 'EYds/G']
ADV_FEATURES_SQRT = ['EYds', 'EYds/G']
ADV_FEATURES_NO_TRANSFORM = ['DYAR', 'YAR', 'DVOA', 'VOA']

DF = pd.read_csv(TOP_PATH + '/data/final/FINAL_DATA.csv')
DF_MODEL = DF[DF['First Year'] < YEAR_CUTOFF].reset_index(drop = True)

sqrt_transformer = Pipeline(steps = [
    ('sqrt', FunctionTransformer(lambda x : x**.5))
])

no_transform = Pipeline(steps=[
    ('none', FunctionTransformer(lambda x : x))
])

preproc_regular = (ColumnTransformer(transformers=[('none', no_transform, REG_FEATURES_NO_TRANSFORM), 
                ('sqrt', sqrt_transformer, REG_FEATURES_SQRT)]))

preproc_advanced = (ColumnTransformer(transformers=[('none', no_transform, ADV_FEATURES_NO_TRANSFORM), 
                                        ('sqrt', sqrt_transformer, ADV_FEATURES_SQRT)]))

reg_model = Pipeline(steps=[('preprocessor', preproc_regular), ('regressor', LinearRegression(normalize = True))])
adv_model = Pipeline(steps=[('preprocessor', preproc_advanced), ('regressor', LinearRegression(normalize = True))])


def run_simulations(N):
    reg_r2_scores = []
    adv_r2_scores = []
    for i in range(N):
        X_train, X_test, y_train, y_test = (train_test_split(DF_MODEL[REG_FEATURES + ADV_FEATURES], 
                                            DF_MODEL['Rec Pts/G Second Season'], test_size = TEST_SIZE, random_state = i))
        reg_model.fit(X_train[REG_FEATURES], y_train)
        adv_model.fit(X_train[ADV_FEATURES], y_train)
        reg_r2_scores.append(reg_model.score(X_test[REG_FEATURES], y_test))
        adv_r2_scores.append(adv_model.score(X_test[ADV_FEATURES], y_test))
    return (reg_r2_scores, adv_r2_scores)


# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import time
import logging
from loguru import logger

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import yaml
from sacred import Experiment
from math import sqrt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)

ex = Experiment('House Price Prediction', interactive= True)
ex.add_config('config.yaml')


# %%
with open('config.yaml', 'r') as f:
    config = yaml.load(f)

logging.basicConfig(filename=os.path.join(config['PATH']['LOG_PATH'], "newfile.log"), format='%(asctime)s %(message)s', filemode='w', level= logging.INFO)
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

# %%
def fill_categorical_na(df, var_list):
    X = df.copy()

    X[var_list] = df[var_list].fillna('Missing')
    return X

def elapsed_years(df, var):
    df[var] = df['YrSold'] - df[var]
    return df

def find_freq_labels(df, var, rare_pct):
    df = df.copy()
    tmp = df[var].value_counts(normalize = True)
    return tmp[tmp > rare_pct].index

def replace_categories(train, test, var, target):
    train = train.copy()
    test = test.copy()

    ordered_labels = train.groupby(var)[target].mean().sort_values().index
    ordinal_label = {k:i for i,k in enumerate(ordered_labels, 0)}

    train[var] = train[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)
    return ordinal_label, train, test


# %%
@ex.automain
def main():
    ## Reading the original dataframe
    logging.info('READING THE DATAFRAME !!!')
    logger.info('READING THE DATAFRAME !!!')
    train = pd.read_csv(os.path.join(config['PATH']['DATA_PATH'], 'train.csv'))
    ## Splitting the dataset into train and test
    logging.info('SPLITTING THE DATAFRAME')
    logger.info('SPLITTING THE DATAFRAME')
    X_train, X_test, y_train, y_test = train_test_split(train, train['SalePrice'], test_size = 0.1, random_state = 0)
    ## Loading the feature list 
    logging.info('LOADING THE FEATURE LIST')
    logger.info('LOADING THE FEATURE LIST')
    selected_feat = pd.read_csv(os.path.join(config['PATH']['DATA_PATH'], 'selected_features.csv'))
    features = list(selected_feat['0']) + ['LotFrontage']
    ## 1. filling missing values in categorical variables with 'missing'
    logging.info('MISSING VALUES IMPUTATION IN CATEGORICAL VARIABLES')
    logger.info('MISSING VALUES IMPUTATION IN CATEGORICAL VARIABLES')
    vars_with_na = [var for var in features if X_train[var].isnull().sum() > 0 and X_train[var].dtypes == 'O']

    X_train = fill_categorical_na(X_train, vars_with_na)
    X_test = fill_categorical_na(X_test, vars_with_na)

    ## filling missing values in categorical variables with 'mode'
    logging.info('MISSING VALUES IMPUTATION IN CONTINUOUS VARIABLES')
    logger.info('MISSING VALUES IMPUTATION IN CONTINUOUS VARIABLES')
    vars_with_na = [var for var in features if X_train[var].isnull().sum() > 0 and X_train[var].dtypes != 'O']

    mean_var_dict = {}

    for var in vars_with_na:
        mode_val = X_train[var].mode()[0]
        mean_var_dict[var] = mode_val

        X_train[var].fillna(mode_val, inplace = True)

        X_test[var].fillna(mode_val, inplace = True)

    np.save(os.path.join(config['PATH']['DATA_PATH'], 'mean_var_dict.npy'), mean_var_dict)

    ## 2. Temporal variables
    logging.info('TEMPORAL VARIABLES TREATMENT')
    logger.info('TEMPORAL VARIABLES TREATMENT')
    X_train = elapsed_years(X_train, 'YearRemodAdd')
    X_test = elapsed_years(X_test, 'YearRemodAdd')

    ## 3. Numerical variables: Gaussian tansformation
    logging.info('TRANSFORMING NUMERICAL VARIABLES TO GAUSSIAN')
    logger.info('TRANSFORMING NUMERICAL VARIABLES TO GAUSSIAN')
    for var in ['LotFrontage', '1stFlrSF', 'GrLivArea', 'SalePrice']:
        X_train[var] = np.log(X_train[var])
        X_test[var] = np.log(X_test[var])
    
    ## 4. Categorical variables : Treating rare labels
    logging.info('TREATING CATEGORICAL VARIABLES FOR RARE LABELS')
    logger.info('TREATING CATEGORICAL VARIABLES FOR RARE LABELS')
    cat_vars = [var for var in features if X_train[var].dtypes == 'O']

    frequent_labels_dict = {}

    for var in cat_vars:
        frequent_ls = find_freq_labels(X_train, var, 0.01)

        frequent_labels_dict[var] = frequent_ls

        X_train[var] = np.where(X_train[var].isin(frequent_ls), X_train[var], 'Rare')
        X_test[var] = np.where(X_test[var].isin(frequent_ls), X_test[var], 'Rare')

    np.save(os.path.join(config['PATH']['DATA_PATH'], 'FrequentLabels.npy'), frequent_labels_dict)

    ## 5. Categorical variables: replace strings with numbers
    logging.info('REPLACING STRINGS WITH NUMBERS IN CATEGORICAL VARIABLES')
    logger.info('REPLACING STRINGS WITH NUMBERS IN CATEGORICAL VARIABLES')
    ordinal_label_dict = {}
    for var in cat_vars:
        ordinal_label , X_train, X_test = replace_categories(X_train, X_test, var, 'SalePrice')
        ordinal_label_dict[var] = ordinal_label

    np.save(os.path.join(config['PATH']['DATA_PATH'], 'OrdinalLabels.npy'), ordinal_label_dict)

    ## 6. Feature Scaling
    logging.info('FEATURE SCALING')
    logger.info('FEATURE SCALING')
    y_train = X_train['SalePrice']
    y_test = X_test['SalePrice']

    scaler = MinMaxScaler()
    scaler.fit(X_train[features])

    joblib.dump(scaler, os.path.join(config['PATH']['DATA_PATH'], 'scaler.pkl'))

    X_train = pd.DataFrame(scaler.transform(X_train[features]), columns = features)
    X_test = pd.DataFrame(scaler.transform(X_test[features]), columns = features)

    logging.info('TRAINING THE MODEL !!!')
    logger.info('TRAINING THE MODEL !!!')
    #lin_model = Lasso(alpha = config['LASSO']['ALPHA'], random_state = 0)
    #lin_model = LinearRegression()
    lin_model = RandomForestRegressor(max_depth= config['RANDOM_FOREST']['max_depth'], random_state = 0)
    lin_model.fit(X_train, y_train)

    train_mse = mean_squared_error(y_train, lin_model.predict(X_train))
    train_rmse = sqrt(mean_squared_error(y_train, lin_model.predict(X_train)))
    train_mae = mean_absolute_error(y_train, lin_model.predict(X_train))

    test_mse = mean_squared_error(y_test, lin_model.predict(X_test))
    test_rmse = sqrt(mean_squared_error(y_test, lin_model.predict(X_test)))
    test_mae = mean_absolute_error(y_test, lin_model.predict(X_test))

    logging.info('SAVING THE MODEL !!!')
    logger.info('SAVING THE MODEL !!!')
    joblib.dump(lin_model, os.path.join(config['PATH']['MODELS_PATH'], '{}.pkl'.format(config['RANDOM_FOREST']['MODEL_FILE_NAME'])))

    logger.info('train_mse:{}  train_rmse:{} train_mae:{}', np.round(train_mse, 4) , np.round(train_rmse, 4) , np.round(train_mae, 4))
    logger.info('test_mse: {}  test_rmse: {} test_mae: {}', np.round(test_mse, 4) , np.round(test_rmse, 4) , np.round(test_mae, 4))

    ex.log_scalar("train_mse", np.round(train_mse, 4))        
    ex.log_scalar("train_rmse", np.round(train_rmse, 4))        
    ex.log_scalar("train_mae", np.round(train_mae, 4))

    ex.log_scalar("test_mse", np.round(test_mse, 4))        
    ex.log_scalar("test_rmse", np.round(test_rmse, 4))        
    ex.log_scalar("test_mae", np.round(test_mae, 4))


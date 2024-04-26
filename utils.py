from os import path
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
import pickle

# z-score normalization
def normalize(data):
    data = MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(data)
    return data


def features_target_split(data, target_column):
    features = data.drop(target_column, axis=1)
    features.drop('DATE', axis=1, inplace=True)
    features = normalize(features)
    target = data[target_column]
    return features, target

# split data, according to date. yyyy-mm--dd
def train_valid_test_split(data, train_start_date, valid_start_date, test_start_date, end_date):
    target_column = "RET"

    # split the dataset into train, valid, test
    train_data = data.loc[(data['DATE'] >= train_start_date) & (
        data['DATE'] < valid_start_date)].reset_index(drop=True)
    valid_data = data.loc[(data['DATE'] >= valid_start_date) & (
        data['DATE'] < test_start_date)].reset_index(drop=True)
    test_data = data.loc[(data['DATE'] >= test_start_date) & (
        data['DATE'] <= end_date)].reset_index(drop=True)

    # split the dataset into features and target
    train_features, train_target = features_target_split(
        train_data, target_column)
    del train_data
    valid_features, valid_target = features_target_split(
        valid_data, target_column)
    del valid_data
    test_features, test_target = features_target_split(
        test_data, target_column)
    del test_data

    return train_features, train_target, valid_features, valid_target, test_features, test_target,

# r2 score function
def r2_score(target, pred):
    target, pred = np.array(target), np.array(pred).flatten()
    pred = np.where(pred < 0, 0, pred)
    RSS = np.dot(target-pred, target-pred)
    TSS = np.dot(target, target)
    return 1 - RSS / TSS

# mse score function
def mse_score(target, pred):
    return mean_squared_error(target, pred)

# validation for fine tune
def validation(model, sup_pars: dict, X_trn, y_trn, X_vld, y_vld, train_start_date, is_nn=False):
    best_r2s = None
    sup_pars_list = list(ParameterGrid(sup_pars))
    output_raws = []

    for sup_par in sup_pars_list:
        if is_nn:
            mod = model().set_params(**sup_par).fit(X_trn, y_trn, X_vld, y_vld)
        else:
            mod = model().set_params(**sup_par).fit(X_trn, y_trn)
        y_pred = mod.predict(X_vld)
        r2s = r2_score(y_vld, y_pred)

        if best_r2s == None or r2s > best_r2s:
            best_r2s = r2s
            best_sup_par = sup_par
            best_model = mod

        output_raws.append([train_start_date,sup_par, f'{r2s*100:.5f}%'])

    # last line is best super params and best r2 score
    output_raws.append([train_start_date, best_sup_par, f'{best_r2s*100:.5f}%'])
    return best_model, output_raws

# read model from file
def read_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def evaluate(target, pred):
    r2 = r2_score(target, pred)
    mse = mse_score(target, pred)
    print(f'R2 score: {r2*100:.2f}%')
    print(f'MSE score: {mse:.3f}')
    return r2, mse

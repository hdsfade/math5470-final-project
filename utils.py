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
    
    train_features = data.drop(target_column, axis=1)
    train_features.drop('DATE', axis=1, inplace=True)
    train_features = normalize(train_features)
    train_target = data[target_column]
    return train_features, train_target

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
    train_features, train_target = features_target_split(train_data, target_column)
    del train_data
    valid_features, valid_target = features_target_split(valid_data, target_column)
    del valid_data
    test_features, test_target = features_target_split(test_data, target_column)
    del test_data

    return train_features, train_target, valid_features, valid_target, test_features, test_target,

# r2 score function
def r2_score(target, pred):
    target, pred = np.array(target), np.array(pred).flatten()
    RSS = np.sum(np.square(target-pred))
    TSS = np.sum(np.square(target))
    return 1 - RSS / TSS

# mse score function
def mse_score(target, pred):
    return mean_squared_error(target, pred)

# validation for fine tune
def validation(model, sup_pars: dict, X_trn, y_trn, X_vld, y_vld, train_start_date,
               base_model_save_path, base_result_save_path, is_nn=False):
    print(sup_pars, train_start_date,
               base_model_save_path, base_result_save_path)
    model_save_path = path.join(base_model_save_path, train_start_date)
    result_save_path = path.join(base_result_save_path, train_start_date)

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

        output_raws.append([sup_par, f'{r2s*100:.5f}%'])
    
    # last line is best super params and best r2 score
    output_raws.append([best_sup_par, f'{best_r2s*100:.5f}%'])
    with open(result_save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['super parameters', 'r2 score'])
        writer.writerows(output_raws)
        
    # store best model
    with open(model_save_path, 'wb') as file:
        pickle.dump(best_model, file)

# read model from file        
def read_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model
            
def evaluate(target, pred):
    print(f'R2 score: {r2_score(target,pred)*100:.2f}%')
    print(f'MSE score: {mse_score(target, pred):.3f}')

    
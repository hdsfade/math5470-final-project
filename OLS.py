import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import train_valid_test_split, validation, evaluate
from sklearn.inspection import permutation_importance


# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = "./results/OLS/train_result.csv"
pred_result_save_path = "./results/OLS/pred_result.csv"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']
base_importance_save_path = "./results/OLS/importance_result"

start_date = "1957-01-01"  # start date
end_date = "2020-12-31"  # end date
train_start_date = start_date
valid_start_date = "1975-01-01"
test_start_date = "1987-01-01"

print('---------OLS--------')
dataset = pd.read_csv(data_path)
while test_start_date <= end_date:
    # split the dataset into train, valid, test
    train_features, train_target, valid_features, valid_target, test_features, test_target = \
        train_valid_test_split(dataset, train_start_date,
                               valid_start_date, test_start_date, end_date)

    OLS = LinearRegression

    # validation
    best_model, info = validation(OLS, {}, train_features, train_target, valid_features, valid_target,
                                  train_start_date)
    pd.DataFrame(info).to_csv(train_result_save_path,
                              index=False, mode='a', header=False)

    # evaluate
    pred = best_model.predict(test_features)
    r2, mse = evaluate(test_target, pred)
    pd.DataFrame([[train_start_date, r2, mse]]).to_csv(
        pred_result_save_path, index=False, mode='a', header=False)

    # important features
    print(test_features.columns)
    result = permutation_importance(
        best_model, test_features, test_target, n_repeats=2, random_state=42, n_jobs=2)
    importances = pd.Series(result.importances_mean,
                            index=test_features.columns)
    importance_save_path = base_importance_save_path+train_start_date+'csv'
    pd.DataFrame([test_features.columns, importances]).to_csv(
        importance_save_path, index=False, mode='a', header=False)

    train_start_date = (pd.to_datetime(
        train_start_date) + pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    valid_start_date = (pd.to_datetime(
        valid_start_date) + pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    test_start_date = (pd.to_datetime(test_start_date) +
                       pd.DateOffset(years=1)).strftime("%Y-%m-%d")
print('---------OLS--------')

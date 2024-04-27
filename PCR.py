import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import work, r2_score


# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = "./results/PCR/train_result.csv"
pred_result_save_path = "./results/PCR/pred_result.csv"
base_importance_save_path = "./results/PCR/importance_result"
base_value_save_path = "./results/PCR/pred_value"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']

start_date = "2020-01-01"  # start date
end_date = "2020-12-31"  # end date
train_start_date = start_date
valid_start_date = "2020-06-01"
test_start_date = "2020-08-01"


class PCRegressor:

    def __init__(self, n_PCs=1, loss='mse'):
        self.n_PCs = n_PCs
        if loss not in ['huber', 'mse']:
            raise AttributeError(
                f"The loss should be either 'huber' or 'mse', but {
                    loss} is given"
            )
        else:
            self.loss = loss

    def set_params(self, **params):
        for param in params.keys():
            setattr(self, param, params[param])
        return self

    def fit(self, X, y):
        X = np.array(X)
        N, K = X.shape
        y = np.array(y).reshape((N, 1))
        self.mu = np.mean(X, axis=0).reshape((1, K))
        self.sigma = np.std(X, axis=0).reshape((1, K))
        self.sigma = np.where(self.sigma == 0, 1, self.sigma)
        X = (X-self.mu)/self.sigma
        pca = PCA()
        X = pca.fit_transform(X)[:, :self.n_PCs]
        self.pc_coef = pca.components_.T[:, :self.n_PCs]
        if self.loss == 'mse':
            self.model = LinearRegression().fit(X, y)
        else:
            self.model = HuberRegressor().fit(X, y)
        return self

    def predict(self, X):
        X = np.array(X)
        X = (X-self.mu)/self.sigma
        X = X @ self.pc_coef
        return self.model.predict(X).squeeze()

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        score = r2_score(y_test, y_pred)
        return score


params = {'n_PCs': [1, 3, 5, 7, 10, 50]}

print('---------PCR--------')
dataset = pd.read_csv(data_path)
PCR = PCRegressor
work(PCR, dataset, params, train_result_save_path, pred_result_save_path, base_importance_save_path, base_value_save_path,
     train_start_date, valid_start_date, test_start_date, end_date)
print('---------PCR--------')

import argparse
import random

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from utils import work
parser = argparse.ArgumentParser(description='nn params')
parser.add_argument('--layers', type=int, help='network layer num', default=1)
# parse args
args = parser.parse_args()

# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = f"./results/nn{args.layers}/train_result.csv"
pred_result_save_path = f"./results/nn{args.layers}/pred_result.csv"
base_importance_save_path = f"./results/nn{args.layers}/importance_result"
base_pred_value_save_path = f"./results/nn{args.layers}/pred_value"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']

start_date = "2020-01-01"  # start date
end_date = "2020-4-30"  # end date
train_start_date = start_date
valid_start_date = "2020-08-01"
test_start_date = "2020-12-31"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def r2_loss(target, pred):
    RSS = torch.square(target - pred)
    TSS = torch.square(target)
    return 1 - RSS / TSS


class NeuralNetwork(nn.Module):
    def __init__(self, seed=10086, n_layers=1, base_neurons=5, learning_rate=0.01, l1=1e-5, l2=0, batch_size=10000, epochs=20):
        super(NeuralNetwork, self).__init__()
        self.seed = seed
        self.n_layers = n_layers
        self.base_neurons = base_neurons
        self.learning_rate = learning_rate
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        self.epochs = epochs

    def set_params(self, **params):
        for param in params.keys():
            setattr(self, param, params[param])
        return self

    def fit(self, X_trn, y_trn, X_vld, y_vld):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        setup_seed(self.seed)
        self.model = nn.Sequential()
        self.model.add_module('input', nn.Linear(
            X_trn.shape[1], self.base_neurons))
        self.model.add_module(f'hidden_{0}', nn.ReLU())

        # add layers
        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                self.model.add_module(
                    f'hidden_{i}', nn.Linear(self.base_neurons, 2**i))
            else:
                self.model.add_module(
                    f'hidden_{i}', nn.Linear(2**(i+1), 2**i))

        self.model.add_module('output', nn.Linear(2, 1))

        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate, weight_decay=self.l2)
        loss = nn.MSELoss()
        # Convert data to tensors
        X_trn_tensor = torch.from_numpy(np.array(X_trn)).float().to(device)
        y_trn_tensor = torch.from_numpy(
            np.array(y_trn)).float().view(-1, 1).to(device)
        X_vld_tensor = torch.from_numpy(np.array(X_vld)).float().to(device)
        y_vld_tensor = torch.from_numpy(
            np.array(y_vld)).float().view(-1, 1).to(device)

        # Create data loaders
        train_data = torch.utils.data.TensorDataset(X_trn_tensor, y_trn_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=False)
        vld_data = torch.utils.data.TensorDataset(X_vld_tensor, y_vld_tensor)
        vld_loader = torch.utils.data.DataLoader(
            vld_data, batch_size=100000, shuffle=False)

        no_improve_time = 0
        last_loss = None
        current_loss = None
        for epoch in range(self.epochs):
            for X, y in train_loader:
                l = loss(self.model(X), y)
                optimizer.zero_grad()

                l1_reg = torch.tensor(0.0).to(device)
                for para in self.model.parameters():
                    l1_reg += torch.norm(para, 1)
                l += self.l1 * l1_reg

                l.backward()
                optimizer.step()

            if epoch >= 5:
                current_loss = self.validation(vld_loader, r2_loss)

                if last_loss is not None and current_loss.sum() > last_loss.sum() - 1e-3:
                    no_improve_time += 1

            if no_improve_time >= self.patience:
                return self
            if current_loss is not None:
                last_loss = current_loss
        return self

    def validation(self, valid_loader, loss_func):
        self.model.eval()
        l = 0

        with torch.no_grad():
            for X, y in valid_loader:
                l_tmp= loss_func(self.model(X), y)
                l += l_tmp.mean()

        return l / len(valid_loader)

    def predict(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.from_numpy(np.array(X)).float().to(device)
        y = self.model(X).cpu()
        return y.detach().squeeze().numpy()


print('---------nn--------')
dataset = pd.read_csv(data_path)

params = {
    'n_layers': [args.layers],
    'l1': [1e-5, 1e-3],
    'learning_rate': [.001, .01],
    'epochs': [20],
    'BatchNormalization': [True],
    'patience': [5],
    'seed': [10086]
}

print('GPU is available: ', torch.cuda.is_available())

work(NeuralNetwork, dataset, params, train_result_save_path, pred_result_save_path, base_importance_save_path, base_pred_value_save_path,
     train_start_date, valid_start_date, test_start_date, end_date, model_with_importance=False, is_nn=True)
print('---------nn--------')

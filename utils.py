import csv
import os
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
def get_Data():
    train_data = pd.read_csv('./data/train.csv')
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.width', 1000)  # 设置显示的宽度
    print(train_data.shape)
    print(train_data.head(2))


    #去除sold price和summary属性，生成新的数据
    train_data_ = train_data.loc[:, train_data.columns != 'Sold Price']
    all_features = train_data_.loc[:, train_data_.columns != 'Summary']
    # # 数据处理
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # # 在标准化数据之后，所有数据都意味着消失，因此我们可以将缺失值设置为0
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    print(all_features[numeric_features].head())
    # `Dummy_na=True` 将“na”（缺失值）视为有效的特征值，并为其创建指示符特征。
    all_features = pd.get_dummies(all_features[numeric_features], dummy_na=True)
    print(all_features.shape)
    labels = train_data['Sold Price']
    print(labels.shape)
    return all_features.values,labels.values


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        mse_loss=log_rmse(outputs, y_batch, criterion)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss


def evaluate(model, dataloader, criterion, device):


    return epoch_loss,mse


def log_rmse(output, labels, loss):

    return rmse
#保存训练模型
def save_checkpoint(state, filename='checkpoint.pth'):

#加载训练模型
def load_checkpoint(model, optimizer, filename='checkpoint.pth'):

if __name__ == '__main__':
    get_Data()
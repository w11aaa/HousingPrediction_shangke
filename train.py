import torch
from torch import nn,optim
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import numpy as np
from housepredicting.model import HousingModel
from housepredicting.housingDataset import HousingDataset
from torch.utils.data import DataLoader
from housepredicting.utils import *

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义Dataset和DataLoader
X,y=get_Data()

# 分割数据集为训练集和测试集（80%训练，20%测试）


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 转换为 float32 类型
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32).reshape(-1,1)
y_test = y_test.astype(np.float32).reshape(-1,1)


# 创建 Dataset 对象
train_dataset = HousingDataset(X_train, y_train)
test_dataset = HousingDataset(X_test, y_test)

# 创建 DataLoader 对象
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# 创建模型
model=HousingModel(19)
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义训练参数
num_epochs = 100
checkpoint_path = 'mlp_model_checkpoint.pth'

# 尝试加载检查点
start_epoch = 0
if os.path.exists(checkpoint_path):
    start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

# 训练循环
best_mse = float('inf')  # 初始化最佳损失为无穷大
for epoch in range(start_epoch, num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss,mse = evaluate(model, test_loader, criterion, device)

    print(f"Epoch [{epoch + 1}/{num_epochs}] 训练损失: {train_loss:.4f} 测试损失: {test_loss:.4f}")
    if best_mse < mse:
        best_mse = mse
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss,
        }, checkpoint_path)
        print(f"保存最佳模型，测试损失: {best_mse:.4f} 在 Epoch {epoch + 1}")
    # 保存检查点
    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss,
    }, checkpoint_path)
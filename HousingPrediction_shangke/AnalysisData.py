import pandas as pd
def get_Data():
    train_data = pd.read_csv('./data/train.csv')
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.width', 1000)  # 设置显示的宽度
    print(train_data.shape)
    print(train_data.head(2))


    # #去除sold price和summary属性，生成新的数据
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
if __name__ == '__main__':
    train_x,train_label=get_Data()
    print(train_x.shape,train_label.shape)
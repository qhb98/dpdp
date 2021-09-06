# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import numpy as np
import pandas as pd
import os
from src.conf.configs import Configs
import seaborn as sns

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def draw_heatmap():
    pd.set_option('display.max_columns', None)
    node_str2int = {}  # 讲复杂的原id映射成序号，便于观察
    node_int2str = []
    factory_info_file_path = os.path.join(Configs.benchmark_folder_path, Configs.factory_info_file)
    df_factory = pd.read_csv(factory_info_file_path)
    for index, row in df_factory.iterrows():
        factory_id_str = str(row['factory_id'])
        node_int2str.append(factory_id_str)
        node_str2int[factory_id_str] = index

    df = pd.read_csv('./benchmark/instance_50/3000_2.csv')
    # print(df.columns)
    # print(df.info())
    df = df[['demand', 'creation_time', 'pickup_id', 'delivery_id']]
    df['time'] = df['creation_time'].apply(lambda x: 0.1 + int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    bins = [x for x in range(0, 60 * 24 + 10, 10)]
    df['interval'] = pd.cut(df.time, bins=bins, labels=[x for x in range(1, 144 + 1)])
    df['pickup'] = df['pickup_id'].map(node_str2int)
    df['delivery'] = df['delivery_id'].map(node_str2int)
    df = df[['demand', 'interval', 'pickup', 'delivery']]
    print(df.head(10))
    print(df.tail())
    # ts_matrix = np.zeros((144//2, len(node_int2str)))
    pickup_matrix = np.zeros((len(node_int2str), 144))
    delivery_matrix = np.zeros((len(node_int2str), 144))
    for index, row in df.iterrows():
        demand = float(row['demand'])
        t = int(row['interval']) - 1
        id1 = int(row['pickup']) + 1
        id2 = int(row['delivery']) + 1
        pickup_matrix[id1][t] += demand
        delivery_matrix[id2][t] += demand
        # pickup_matrix[0][t] += demand
    print(pickup_matrix)
    plt.figure(dpi=200)
    plt.figure(figsize=(20, 10))
    sns.heatmap(pickup_matrix, cmap='Reds')
    plt.show()

if __name__ == "__main__":
    # if you want to traverse all instances, set the selected_instances to []
    print(111)

    pd.set_option('display.max_columns', None)
    node_str2int = {}  # 讲复杂的原id映射成序号，便于观察
    node_int2str = []
    factory_info_file_path = os.path.join(Configs.benchmark_folder_path, Configs.factory_info_file)
    df_factory = pd.read_csv(factory_info_file_path)
    for index, row in df_factory.iterrows():
        factory_id_str = str(row['factory_id'])
        node_int2str.append(factory_id_str)
        node_str2int[factory_id_str] = index

    # 读取订单信息
    df = pd.read_csv('./benchmark/instance_30/500_6.csv')
    df = df[['demand', 'creation_time', 'pickup_id', 'delivery_id']]
    df['time'] = df['creation_time'].apply(lambda x: 0.1+int(x.split(':')[0])*60 + int(x.split(':')[1]) )
    bins = [x for x in range(0, 60*24+10, 10)]
    df['interval'] = pd.cut(df.time, bins=bins, labels=[x for x in range(1, 144+1)])
    df['pickup'] = df['pickup_id'].map(node_str2int)
    df['delivery'] = df['delivery_id'].map(node_str2int)
    df = df[['demand', 'interval', 'pickup', 'delivery']]
    print(df.head(10))
    print(df.tail())
    number = df.shape[0]
    print('原始订单数量', df.shape[0])
    print('0-1的订单 {} 占比 {:.2f}%'.format(df[df['demand'] <= 1].shape[0], 100*df[df['demand'] <= 1].shape[0]/number))
    print('0-3的订单 {} 占比 {:.2f}%'.format(df[df['demand'] <= 3].shape[0], 100*df[df['demand'] <= 3].shape[0]/number))
    print('0-5的订单 {} 占比 {:.2f}%'.format(df[df['demand'] <= 5].shape[0], 100 * df[df['demand'] <= 5].shape[0] / number))
    print('>5的订单 {} 占比 {:.2f}%'.format(df[df['demand'] > 5].shape[0], 100 * df[df['demand'] > 5].shape[0] / number))
    g = df.groupby(['pickup', 'delivery']).agg('count')
    # g = df.groupby(['pickup', 'delivery', 'interval']).count().dropna()
    print(g)
    df.drop_duplicates(subset=['pickup', 'delivery'], keep='last', inplace=True)
    print('去重后订单数量', df.shape[0])

    draw_heatmap()
    print("Happy Ending")

import os
import pandas as pd
import matplotlib.pyplot as plt


# loss in p_net_1
loss_dir1 = 'D:/桌面/result/gnn_p_net_1/log'

# test in p_net_1
test_dir1 = 'D:/桌面/result/gnn_p_net_1/test'
test_dir2 = 'D:/桌面/result/cnn_p_net_1/test'
test_dir3 = 'D:/桌面/result/grc_p_net_1/test'
test_dir4 = 'D:/桌面/result/ffd_p_net_1/test'
test_dir5 = 'D:/桌面/result/random_p_net_1/test'

# test in p_net_2
test_dir6 = 'D:/桌面/result/gnn_p_net_2/test'
test_dir7 = 'D:/桌面/result/cnn_p_net_2/test'
test_dir8 = 'D:/桌面/result/grc_p_net_2/test'
test_dir9 = 'D:/桌面/result/ffd_p_net_2/test'
test_dir10 = 'D:/桌面/result/random_p_net_2/test'
loss_paths = [loss_dir1]
test_paths = [test_dir1, test_dir2, test_dir3, test_dir4, test_dir5]

colors = ['red', 'blue', 'green', 'tan', 'black']
linestyles = []
markers = ['s', '^', '1', '*', 'o']
labels = ['GNN-A2C', 'RLA', 'GRC', 'FFD', 'Random']

def figure_loss(folder_paths, markers, factor, x_label, y_label):
    for folder_path, marker in zip(folder_paths, markers):
        if not os.path.exists(folder_path):
            print('no such path:{}'.format(folder_path))
            continue
        parts = folder_path.split('/')
        x = []
        y = []
        step = 500
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                print(type(df[factor].iloc[0]))
                for x_id in range(0, len(df[factor])):
                    if (x_id % step) == 0:
                        # match = re.search(r'\((.*?)\)', df[factor].iloc[x_id])
                        # if match:
                        #     result = float(match.group(1))
                        x.append(x_id)
                        y.append(df[factor].iloc[x_id])
        plt.plot(x, y, label='GNN-A2C', marker=marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

# 画出单一属性的均值
def figure_avg_factor(factor, folder_path=None):
    col_name = factor  # 指定要提取的列的名称
    if folder_path is None:
        folder_paths = [test_dir1, test_dir2, test_dir3, test_dir4, test_dir5]
    labels = ['GNN-A2C', 'RLA', 'GRC', 'FFD', 'Random']
    markers = ['s', '^', '1', '*', 'o']
    for folder_path, marker, label in zip(folder_paths, markers, labels):
        folder_path = folder_path  # 指定包含 csv 文件的目录
        step = 100  # 指定步长
        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                data.append(df[col_name].iloc[::step])

        mean_values = pd.concat(data, axis=1).mean(axis=1)
        plt.plot(mean_values, label=label, marker=marker)
    plt.xlabel('processing times')
    plt.ylabel(f'long-term r2c in p_net_1')
    plt.legend()
    plt.show()

# 画出两属性比值的均值
def figure_two_factor(factor1, factor2, folder_path=None):
    col1_name = factor1  # 指定要提取的列的名称
    col2_name = factor2
    if folder_path is None:
        folder_paths = [test_dir1, test_dir2, test_dir3, test_dir4, test_dir5]
    labels = ['GNN-A2C', 'RLA', 'GRC', 'FFD', 'Random']
    markers = ['s', '^', '1', '*', 'o']
    for folder_path, marker, label in zip(folder_paths, markers, labels):
        folder_path = folder_path  # 指定包含 csv 文件的目录
        step = 100  # 指定步长
        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                result = df[col1_name].iloc[::step] / df[col2_name].iloc[::step]
                data.append(result)

        mean_values = pd.concat(data, axis=1).mean(axis=1)
        plt.plot(mean_values, label=label, marker=marker)
    plt.xlabel('processing times')
    plt.ylabel(f'accept ratio in p_net_1')
    plt.legend()
    plt.show()

def figure_diff_r2c(factor):
    col_name = factor  # 指定要提取的列的名称
    folder_paths = [test_dir1, test_dir6, test_dir2, test_dir7,test_dir8]
    labels = ['GNN-A2C-1', 'GNN-A2C-2', 'RLA-1', 'RLA-2','GRC']
    markers = ['s', 'D', '^', 'v','o']
    for folder_path, marker, label in zip(folder_paths, markers, labels):
        folder_path = folder_path  # 指定包含 csv 文件的目录
        step = 100  # 指定步长
        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                data.append(df[col_name].iloc[::step])

        mean_values = pd.concat(data, axis=1).mean(axis=1)
        plt.plot(mean_values, label=label, marker=marker)
    plt.xlabel('processing times')
    plt.ylabel(f'long-term r2c')
    plt.legend()
    plt.show()

def figure_diff_ac(factor1, factor2):
    col1_name = factor1  # 指定要提取的列的名称
    col2_name = factor2
    folder_paths = [test_dir1, test_dir6, test_dir2, test_dir7, test_dir8]
    labels = ['GNN-A2C-1', 'GNN-A2C-2', 'RLA-1', 'RLA-2', 'GRC']
    markers = ['s', 'D', '^', 'v','o']
    for folder_path, marker, label in zip(folder_paths, markers, labels):
        folder_path = folder_path  # 指定包含 csv 文件的目录
        step = 100  # 指定步长
        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                result = df[col1_name].iloc[::step] / df[col2_name].iloc[::step]
                data.append(result)

        mean_values = pd.concat(data, axis=1).mean(axis=1)
        plt.plot(mean_values, label=label, marker=marker)
    plt.xlabel('processing times')
    plt.ylabel(f'accept ratio')
    plt.legend()
    plt.show()

# figure_loss(loss_paths, markers=['s','^'],factor='loss/loss',x_label='update time', y_label='loss')
figure_avg_factor('total_r2c')
figure_two_factor('success_count', 'v_net_count')
figure_diff_r2c('total_r2c')
figure_diff_ac('success_count', 'v_net_count')

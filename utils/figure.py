import os

import numpy as np
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
    folder_paths = [test_dir1, test_dir6, test_dir2, test_dir7, test_dir8]
    labels = ['GNN-A2C-1', 'GNN-A2C-2', 'RLA-1', 'RLA-2', 'GRC']
    markers = ['s', 'D', '^', 'v', 'o']
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
    markers = ['s', 'D', '^', 'v', 'o']
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
# figure_avg_factor('total_r2c')
# figure_two_factor('success_count', 'v_net_count')
# figure_diff_r2c('total_r2c')
# figure_diff_ac('success_count', 'v_net_count')


def figure_two_factor_pretty(factor1, factor2, folder_paths=None):
    """
    绘制不同算法在相同物理拓扑下的切片请求接受率对比图（平滑曲线 + 阴影置信区间）
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})

    if folder_paths is None:
        folder_paths = [test_dir1, test_dir2, test_dir3, test_dir4, test_dir5]

    labels = ['RAS', 'CNN', 'GRC', 'First', 'Random']
    markers = ['s', '^', '1', '*', 'o']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F5B041', '#7D3C98']

    step = 100
    plt.figure(figsize=(7, 4))
    for folder_path, marker, label, color in zip(folder_paths, markers, labels, colors):
        if not os.path.exists(folder_path):
            print('no such path:', folder_path)
            continue

        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, file_name))
                result = df[factor1].iloc[::step] / df[factor2].iloc[::step]
                data.append(result)

        y_values = pd.concat(data, axis=1)
        mean = y_values.mean(axis=1)
        std = y_values.std(axis=1)
        x = np.arange(len(mean))

        plt.plot(x, mean, label=label, color=color, marker=marker, linewidth=2, markersize=5)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    plt.xlabel('Slice Requests Processed (#)')
    plt.ylabel('Average Accept Ratio')
    plt.title('Comparison of Slice Request Acceptance Ratios', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig('accept_ratio_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()


def figure_diff_ac_pretty(factor1, factor2):
    """
    绘制算法泛化性能（跨拓扑）的对比图：
    - 以条形图展示不同算法在两个物理拓扑上的接受率下降幅度
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})

    # 假设路径分别是算法1在p_net_A、p_net_B的测试结果
    folder_paths = [test_dir1, test_dir6, test_dir2, test_dir7, test_dir8]
    labels = ['RAS', 'CNN', 'GRC']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    # 每个算法两组结果（train_net, test_net）
    train_results, test_results = [], []
    step = 100

    for idx in range(0, len(folder_paths), 2):
        train_path = folder_paths[idx]
        test_path = folder_paths[idx + 1] if idx + 1 < len(folder_paths) else None

        # 计算train拓扑均值
        train_data = []
        for file_name in os.listdir(train_path):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(train_path, file_name))
                result = df[factor1].iloc[::step] / df[factor2].iloc[::step]
                train_data.append(result)
        train_mean = pd.concat(train_data, axis=1).mean(axis=1).mean()

        # 计算test拓扑均值
        if test_path:
            test_data = []
            for file_name in os.listdir(test_path):
                if file_name.endswith('.csv'):
                    df = pd.read_csv(os.path.join(test_path, file_name))
                    result = df[factor1].iloc[::step] / df[factor2].iloc[::step]
                    test_data.append(result)
            test_mean = pd.concat(test_data, axis=1).mean(axis=1).mean()
        else:
            test_mean = train_mean

        train_results.append(train_mean)
        test_results.append(test_mean)

    # 绘制条形对比图
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, train_results, width, label='Train Topology (p_net_A)', color='#3498DB')
    plt.bar(x + width / 2, test_results, width, label='Test Topology (p_net_B)', color='#E74C3C')

    for i, (a, b) in enumerate(zip(train_results, test_results)):
        drop = (a - b) / a * 100
        plt.text(i, max(a, b) + 0.005, f'-{drop:.1f}%', ha='center', fontsize=10)

    plt.xticks(x, labels)
    plt.ylabel('Average Accept Ratio')
    plt.title('Generalization Accept Ratio Across Physical Net', fontsize=14)
    plt.grid(alpha=0.3, axis='y', linestyle='--')
    plt.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig('generalization_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()


def figure_diff_r2c_pretty(factor):
    """
    绘制不同算法在两个物理拓扑上的长期收益（r2c）曲线对比图。
    改进点：
    - 平滑曲线 + 阴影置信区间
    - 相同算法用相同颜色，不同拓扑用虚实线区分
    - 论文风格（Times New Roman, Seaborn-paper）
    """

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})

    # 数据路径与算法标签（成对出现：拓扑A, 拓扑B）
    folder_paths = [test_dir1, test_dir6, test_dir2, test_dir7, test_dir8]
    labels = ['RAS (A)', 'RAS (B)', 'CNN (A)', 'CNN (B)', 'GRC']
    line_styles = ['-', '--', '-', '--', '-.']
    colors = ['#E74C3C', '#E74C3C', '#3498DB', '#3498DB', '#2ECC71']
    markers = ['s', 'D', '^', 'v', 'o']

    plt.figure(figsize=(7, 4))
    step = 100

    for folder_path, marker, label, color, ls in zip(folder_paths, markers, labels, colors, line_styles):
        if not os.path.exists(folder_path):
            print('no such path:', folder_path)
            continue

        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, file_name))
                data.append(df[factor].iloc[::step])

        # 合并并计算均值与标准差
        y_values = pd.concat(data, axis=1)
        mean = y_values.mean(axis=1)
        std = y_values.std(axis=1)
        x = np.arange(len(mean))

        # 可选平滑：moving average，减少RL噪声
        window = 5
        mean_smooth = mean.rolling(window=window, min_periods=1).mean()
        std_smooth = std.rolling(window=window, min_periods=1).mean()

        plt.plot(x, mean_smooth, label=label, color=color, linestyle=ls, marker=marker,
                 markersize=8,  # ← 符号更大
                 markeredgewidth=1.2,  # ← 符号边框更厚
                 markeredgecolor=color,  # ← 边框与线同色（黑白中也能区分）
                 linewidth=2.2,  # ← 线条更粗
                 )
        plt.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
                         color=color, alpha=0.15)

    plt.xlabel('Slice Requests Processed')
    plt.ylabel('Long-term R2C')
    plt.title('Generalization of Total R2C Across Physical Net', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(frameon=False, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig('generalization_r2c_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()


def figure_avg_factor_pretty(factor, folder_paths=None):
    """
    绘制单一指标（如 total_r2c）在不同算法间的平均变化趋势。
    改进点：
    - 平滑曲线 + 阴影置信区间
    - Times New Roman + Seaborn-paper 论文风格
    - 自动保存高分辨率图片
    """

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})

    if folder_paths is None:
        folder_paths = [test_dir1, test_dir2, test_dir3, test_dir4, test_dir5]

    labels = ['RAS', 'CNN', 'GRC', 'First', 'Random']
    markers = ['s', '^', '1', '*', 'o']
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F5B041', '#7D3C98']

    step = 100
    plt.figure(figsize=(7, 4))

    for folder_path, marker, label, color in zip(folder_paths, markers, labels, colors):
        if not os.path.exists(folder_path):
            print('no such path:', folder_path)
            continue

        # 读取所有实验数据
        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(folder_path, file_name))
                data.append(df[factor].iloc[::step])

        # 合并计算平均与标准差
        y_values = pd.concat(data, axis=1)
        mean = y_values.mean(axis=1)
        std = y_values.std(axis=1)
        x = np.arange(len(mean))

        # 平滑处理（移动平均）
        window = 5
        mean_smooth = mean.rolling(window=window, min_periods=1).mean()
        std_smooth = std.rolling(window=window, min_periods=1).mean()

        # 绘制曲线 + 阴影置信区间
        plt.plot(x, mean_smooth, label=label, color=color,
                 marker=marker,
                 markersize=8,  # ← 符号更大
                 markeredgewidth=1.2,  # ← 符号边框更厚
                 markeredgecolor=color,  # ← 边框与线同色（黑白中也能区分）
                 linewidth=2.2,  # ← 线条更粗
                 )
        plt.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, color=color, alpha=0.15)

    plt.xlabel('Slice Requests Processed')
    plt.ylabel(f'Average {factor.replace("_", " ").title()}')
    plt.title(f'Comparison of {factor.replace("_", " ").title()} Across Algorithms', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{factor}_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()


# figure_two_factor_pretty('success_count', 'v_net_count')
figure_diff_ac_pretty('success_count', 'v_net_count')
figure_diff_r2c_pretty('total_r2c')
figure_avg_factor_pretty('total_r2c')

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
        step = 50  # 指定步长
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
        folder_paths = [test_dir1, test_dir2, test_dir3]

    labels = ['RAS', 'CNN', 'GRC']
    markers = ['s', '^', '1']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

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

        # plt.plot(x, mean, label=label, color=color, marker=marker, linewidth=2, markersize=5)
        # plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

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


# def figure_diff_r2c_pretty(factor):
#     """
#     绘制不同算法在两个物理拓扑上的长期收益（r2c）曲线对比图。
#     改进点：
#     - 平滑曲线 + 阴影置信区间
#     - 相同算法用相同颜色，不同拓扑用虚实线区分
#     - 论文风格（Times New Roman, Seaborn-paper）
#     """
#
#     plt.style.use('seaborn-v0_8-paper')
#     plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})
#
#     # 数据路径与算法标签（成对出现：拓扑A, 拓扑B）
#     folder_paths = [test_dir1, test_dir6, test_dir2, test_dir7, test_dir8]
#     labels = ['RAS (A)', 'RAS (B)', 'CNN (A)', 'CNN (B)', 'GRC']
#     line_styles = ['-', '--', '-', '--', '-.']
#     colors = ['#E74C3C', '#E74C3C', '#3498DB', '#3498DB', '#2ECC71']
#     markers = ['s', 'D', '^', 'v', 'o']
#
#     plt.figure(figsize=(7, 4))
#     step = 100
#
#     for folder_path, marker, label, color, ls in zip(folder_paths, markers, labels, colors, line_styles):
#         if not os.path.exists(folder_path):
#             print('no such path:', folder_path)
#             continue
#
#         data = []
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('.csv'):
#                 df = pd.read_csv(os.path.join(folder_path, file_name))
#                 data.append(df[factor].iloc[::step])
#
#         # 合并并计算均值与标准差
#         y_values = pd.concat(data, axis=1)
#         mean = y_values.mean(axis=1)
#         std = y_values.std(axis=1)
#         x = np.arange(len(mean))
#
#         # 可选平滑：moving average，减少RL噪声
#         window = 5
#         mean_smooth = mean.rolling(window=window, min_periods=1).mean()
#         std_smooth = std.rolling(window=window, min_periods=1).mean()
#
#         plt.plot(x, mean_smooth, label=label, color=color, linestyle=ls, marker=marker,
#                  markersize=8,  # ← 符号更大
#                  markeredgewidth=1.2,  # ← 符号边框更厚
#                  markeredgecolor=color,  # ← 边框与线同色（黑白中也能区分）
#                  linewidth=2.2,  # ← 线条更粗
#                  )
#         plt.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth,
#                          color=color, alpha=0.15)
#
#     plt.xlabel('Slice Requests Processed')
#     plt.ylabel('Long-term R2C')
#     plt.title('Generalization of Total R2C Across Physical Net', fontsize=14)
#     plt.grid(alpha=0.3, linestyle='--')
#     plt.legend(frameon=False, loc='upper right', ncol=2)
#     plt.tight_layout()
#     plt.savefig('generalization_r2c_comparison.png', dpi=600, bbox_inches='tight')
#     plt.show()


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

    labels = ['RAS', 'CNN', 'GRC']
    markers = ['s', '^', '1']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

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


def figure_diff_r2c_pretty(factor):
    """
    绘制算法泛化性能（跨拓扑）的 Total R2C 条形对比图：
    - 每个算法两根柱子（Train/Test topology）
    - 显示性能下降百分比
    """

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})

    # 路径（Train/Test成对）
    folder_paths = [test_dir1, test_dir6, test_dir2, test_dir7, test_dir8]

    labels = ['RAS', 'CNN', 'GRC']

    train_results = []
    test_results = []

    step = 100

    for idx in range(0, len(folder_paths), 2):

        train_path = folder_paths[idx]
        test_path = folder_paths[idx + 1] if idx + 1 < len(folder_paths) else None

        # ---------- Train topology ----------
        train_data = []

        for file_name in os.listdir(train_path):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(train_path, file_name))

                # 取采样点
                result = df[factor].iloc[::step]

                train_data.append(result)

        train_mean = pd.concat(train_data, axis=1).mean(axis=1).mean()

        # ---------- Test topology ----------
        if test_path:

            test_data = []

            for file_name in os.listdir(test_path):
                if file_name.endswith('.csv'):
                    df = pd.read_csv(os.path.join(test_path, file_name))

                    result = df[factor].iloc[::step]

                    test_data.append(result)

            test_mean = pd.concat(test_data, axis=1).mean(axis=1).mean()

        else:
            test_mean = train_mean

        train_results.append(train_mean)
        test_results.append(test_mean)

    # ---------- Plot ----------
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))

    plt.bar(
        x - width / 2,
        train_results,
        width,
        label='Train Topology (p_net_A)',
        color='#3498DB'
    )

    plt.bar(
        x + width / 2,
        test_results,
        width,
        label='Test Topology (p_net_B)',
        color='#E74C3C'
    )

    # 标注下降比例
    for i, (a, b) in enumerate(zip(train_results, test_results)):
        drop = (a - b) / a * 100 if a != 0 else 0

        plt.text(
            i,
            max(a, b) * 1.02,
            f'-{drop:.1f}%',
            ha='center',
            fontsize=10
        )

    plt.xticks(x, labels)

    plt.ylabel('Average Total R2C')

    plt.title(
        'Generalization of Total R2C Across Physical Net',
        fontsize=14
    )

    plt.grid(alpha=0.3, axis='y', linestyle='--')

    plt.legend(frameon=False, loc='upper right')

    plt.tight_layout()

    plt.savefig(
        'generalization_r2c_bar.png',
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()


def figure_loss_pretty(loss_dir, step=50, smooth_window=50):
    """
    绘制Actor-Critic训练损失变化趋势图（可调采样步长）

    参数:
        loss_dir : 训练目录
        step : 下采样步长（控制点密度）
        smooth_window : 平滑窗口大小

    数据:
        training_info.csv
            loss/actor_loss
            loss/critic_loss
    """

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 13,
        'font.family': 'Times New Roman'
    })

    file_path = os.path.join(loss_dir, 'training_info.csv')

    if not os.path.exists(file_path):
        print("No such file:", file_path)
        return

    df = pd.read_csv(file_path)

    # ===== 下采样 =====
    actor_loss = df['loss/actor_loss'].iloc[::step]
    critic_loss = df['loss/critic_loss'].iloc[::step]

    x = np.arange(len(actor_loss))

    # ===== 平滑 =====
    actor_smooth = actor_loss.rolling(
        window=smooth_window,
        min_periods=1
    ).mean()

    critic_smooth = critic_loss.rolling(
        window=smooth_window,
        min_periods=1
    ).mean()

    # ===== 绘图 =====

    plt.figure(figsize=(7, 4))

    plt.plot(
        x,
        actor_smooth,
        label='Actor Loss',
        color='#E74C3C',
        linestyle='-',
        marker='s',
        markersize=7,
        markeredgewidth=1.2,
        markeredgecolor='#E74C3C',
        linewidth=2.2
    )

    plt.plot(
        x,
        critic_smooth,
        label='Critic Loss',
        color='#3498DB',
        linestyle='--',
        marker='o',
        markersize=7,
        markeredgewidth=1.2,
        markeredgecolor='#3498DB',
        linewidth=2.2
    )

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')

    plt.title(
        'Actor-Critic Training Stability',
        fontsize=14
    )

    plt.grid(alpha=0.3, linestyle='--')

    plt.legend(frameon=False, loc='upper right')

    plt.tight_layout()

    plt.savefig(
        'actor_critic_loss.png',
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()


# figure_two_factor_pretty('success_count', 'v_net_count')
# figure_diff_ac_pretty('success_count', 'v_net_count')
# figure_diff_r2c_pretty('total_r2c')
# figure_avg_factor_pretty('total_r2c')


# gemini修改后
def figure_avg_factor_pretty_improved(factor, folder_paths=None):
    """
    绘制单一指标在不同算法间的平均变化趋势（整合健壮性读取版）。
    """

    # 1. 设置整体风格与字体
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})

    if folder_paths is None:
        # 替换为你的实际文件夹路径变量
        folder_paths = [test_dir1, test_dir2, test_dir3]

    labels = ['RAS', 'CNN', 'GRC']
    markers = ['s', '^', 'o']  # 区分度更高的实心标记
    linestyles = ['-', '--', '-.']  # 区分度更高的线型
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    step = 100

    # 2. 创建主图和局部放大图对象
    fig, ax = plt.subplots(figsize=(8, 5))
    # 局部放大图位置 [x起点, y起点, 宽, 高] (相对于主图0-1的比例)
    axins = ax.inset_axes([0.15, 0.15, 0.35, 0.3])

    for folder_path, marker, linestyle, label, color in zip(folder_paths, markers, linestyles, labels, colors):
        if not os.path.exists(folder_path):
            print(f'Warning: No such path: {folder_path}')
            continue

        # 3. 健壮的数据读取逻辑
        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(folder_path, file_name))

                    # 关键修改：重置索引，确保对齐拼接时不按原文件行号错位
                    series_data = df[factor].iloc[::step].reset_index(drop=True)
                    data.append(series_data)

                except KeyError:
                    print(f"Warning: 找不到列 '{factor}' 在 {file_name} 中，已跳过。")
                    continue

        if not data:
            print(f"Error: {folder_path} 中没有读到有效数据。")
            continue

        # 4. 合并与统计计算
        y_values = pd.concat(data, axis=1)
        mean = y_values.mean(axis=1)
        std = y_values.std(axis=1)

        # 设定 X 轴。如果希望 X 轴显示真实请求数量，改为: x = np.arange(len(mean)) * step
        x = np.arange(len(mean))

        # 计算标准误 (SEM) 替代标准差，收窄阴影
        n_experiments = len(data)
        sem = std / np.sqrt(n_experiments) if n_experiments > 0 else std

        # 平滑处理（移动平均）
        window = 5
        mean_smooth = mean.rolling(window=window, min_periods=1).mean()
        sem_smooth = sem.rolling(window=window, min_periods=1).mean()

        # 5. 绘制主图
        ax.plot(x, mean_smooth, label=label, color=color,
                marker=marker, linestyle=linestyle,
                markersize=8, markeredgewidth=1.2,
                markeredgecolor=color, linewidth=2.2)

        ax.fill_between(x, mean_smooth - sem_smooth, mean_smooth + sem_smooth, color=color, alpha=0.15)

        # 6. 绘制局部放大图 (代码与主图几乎一致，只是画在 axins 上)
        axins.plot(x, mean_smooth, color=color, marker=marker,
                   linestyle=linestyle, markersize=5, linewidth=1.5)
        axins.fill_between(x, mean_smooth - sem_smooth, mean_smooth + sem_smooth, color=color, alpha=0.15)

    # 7. 主图格式设置
    ax.set_xlabel('Slice Requests Processed')
    # 如果上文 x 乘以了 step，这里最好加上单位，例如 'Slice Requests Processed (x100)' 或真实数值
    ax.set_ylabel(f'Average {factor.replace("_", " ").title()}')
    ax.set_title(f'Comparison of {factor.replace("_", " ").title()} Across Algorithms', fontsize=14)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=False, loc='upper right')

    # 8. 局部放大图格式与范围设置
    # 注意：这里的 10 和 19 是根据 x = np.arange(len(mean)) 估算的。
    # 如果你前面修改成了 x = np.arange(len(mean)) * step，这里也要改成 10*step 和 19*step
    axins.set_xlim(10, 19)
    axins.set_ylim(0.485, 0.505)

    axins.grid(alpha=0.2, linestyle=':')
    axins.tick_params(axis='both', which='major', labelsize=9)

    # 绘制连接线框
    ax.indicate_inset_zoom(axins, edgecolor="gray")

    # 保存并展示
    plt.tight_layout()
    plt.savefig(f'{factor}_comparison_final.png', dpi=600, bbox_inches='tight')
    plt.show()


def figure_diff_r2c_pretty_bw(factor):
    """
    绘制跨拓扑的 Total R2C 条形对比图（适配黑白打印与数据修复版）
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 13, 'font.family': 'Times New Roman'})

    # 请务必在此处填入正确的测试文件夹变量名！
    algorithm_paths = {
        'RAS': {'train': test_dir1, 'test': test_dir6},
        'CNN': {'train': test_dir2, 'test': test_dir7},
        'GRC': {'train': test_dir3, 'test': test_dir8}
    }

    labels = list(algorithm_paths.keys())
    train_results = []
    test_results = []
    step = 100
    tail_points = 5  # 取曲线最后5个点求稳态平均

    for algo in labels:
        paths = algorithm_paths[algo]

        # 读取 Train 数据
        train_data = []
        for file_name in os.listdir(paths['train']):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(paths['train'], file_name))
                train_data.append(df[factor].iloc[::step].reset_index(drop=True))

        if train_data:
            train_mean = pd.concat(train_data, axis=1).mean(axis=1).iloc[-tail_points:].mean()
        else:
            train_mean = 0

        # 读取 Test 数据
        test_data = []
        if paths['test'] and os.path.exists(paths['test']):
            for file_name in os.listdir(paths['test']):
                if file_name.endswith('.csv'):
                    df = pd.read_csv(os.path.join(paths['test'], file_name))
                    test_data.append(df[factor].iloc[::step].reset_index(drop=True))

            if test_data:
                test_mean = pd.concat(test_data, axis=1).mean(axis=1).iloc[-tail_points:].mean()
            else:
                test_mean = train_mean
        else:
            test_mean = train_mean

        train_results.append(train_mean)
        test_results.append(test_mean)

    # ---------- 开始绘图 ----------
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))

    # 添加 hatch 填充条纹，加上 edgecolor 增加黑白对比度
    ax.bar(x - width / 2, train_results, width,
           label='Train Topology (p_net_A)',
           color='#3498DB', edgecolor='black', linewidth=1, hatch='//')

    ax.bar(x + width / 2, test_results, width,
           label='Test Topology (p_net_B)',
           color='#E74C3C', edgecolor='black', linewidth=1, hatch='\\\\')

    # 标注性能降幅比例
    for i, (a, b) in enumerate(zip(train_results, test_results)):
        # (原值 - 新值) / 原值 * 100
        drop = (a - b) / a * 100 if a != 0 else 0

        # 动态箭头显示，避免负号歧义
        if drop > 0:
            text_str = f'↓ {drop:.1f}%'  # 下降
        elif drop < 0:
            text_str = f'↑ {abs(drop):.1f}%'  # 如果有意外的反弹上升，显示向上的箭头
        else:
            text_str = '0.0%'

        # 稍微调高一点标注文本的 Y 坐标，防止和柱子顶部太拥挤
        ax.text(i, max(a, b) + 0.015, text_str, ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f'Steady-State Average {factor.replace("_", " ").title()}')
    ax.set_title('Generalization of Total R2C Across Physical Net', fontsize=14)

    # 将 y 轴上限稍微放大一点，给上方的降幅百分比文字留出空间
    max_val = max(max(train_results), max(test_results))
    ax.set_ylim(0, max_val * 1.15)

    ax.grid(alpha=0.3, axis='y', linestyle='--')
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig('generalization_r2c_bar_bw.png', dpi=600, bbox_inches='tight')
    plt.show()


# figure_avg_factor_pretty_improved('total_r2c')

# figure_diff_r2c_pretty_bw('total_r2c')


def figure_avg_factor_pretty_cn(factor, folder_paths=None):
    """
    绘制单一指标在不同算法间的平均变化趋势（中文论文版）
    """
    # 设置风格
    plt.style.use('seaborn-v0_8-paper')

    # 【关键修改】配置中文字体和公式字体，支持中文和负号显示
    # Windows系统一般用 'SimSun' (宋体) 或 'SimHei' (黑体)
    # Mac系统一般用 'Arial Unicode MS' 或 'Songti SC'
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if folder_paths is None:
        folder_paths = [test_dir1, test_dir2, test_dir3]  # 替换为实际路径

    labels = ['RAS', 'CNN', 'GRC']
    markers = ['s', '^', 'o']
    linestyles = ['-', '--', '-.']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    step = 100

    # 中英文指标映射字典（根据你的实际情况添加）
    factor_name_cn = {
        'total_r2c': '总收益开销比 (Total R2C)',
        'revenue': '总收益 (Revenue)',
        'acceptance_ratio': '请求接受率 (Acceptance Ratio)'
    }.get(factor, factor)  # 如果字典里没有，就原样输出

    fig, ax = plt.subplots(figsize=(8, 5))
    axins = ax.inset_axes([0.15, 0.15, 0.35, 0.3])

    for folder_path, marker, linestyle, label, color in zip(folder_paths, markers, linestyles, labels, colors):
        if not os.path.exists(folder_path):
            continue

        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(folder_path, file_name))
                    series_data = df[factor].iloc[::step].reset_index(drop=True)
                    data.append(series_data)
                except KeyError:
                    continue

        if not data: continue

        y_values = pd.concat(data, axis=1)
        mean = y_values.mean(axis=1)
        std = y_values.std(axis=1)
        x = np.arange(len(mean))

        n_experiments = len(data)
        sem = std / np.sqrt(n_experiments) if n_experiments > 0 else std

        window = 5
        mean_smooth = mean.rolling(window=window, min_periods=1).mean()
        sem_smooth = sem.rolling(window=window, min_periods=1).mean()

        ax.plot(x, mean_smooth, label=label, color=color,
                marker=marker, linestyle=linestyle,
                markersize=8, markeredgewidth=1.2,
                markeredgecolor=color, linewidth=2.2)
        ax.fill_between(x, mean_smooth - sem_smooth, mean_smooth + sem_smooth, color=color, alpha=0.15)

        axins.plot(x, mean_smooth, color=color, marker=marker,
                   linestyle=linestyle, markersize=5, linewidth=1.5)
        axins.fill_between(x, mean_smooth - sem_smooth, mean_smooth + sem_smooth, color=color, alpha=0.15)

    # 【关键修改】中文标签
    ax.set_xlabel('已处理的切片请求数(x50)', fontsize=15)
    ax.set_ylabel(f'{factor_name_cn}', fontsize=15)
    # ax.set_title(f'不同算法下R2C的变化趋势对比', fontsize=20, pad=15)

    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=False, loc='upper right', fontsize=15)

    # 局部放大图设置 (请根据数据微调)
    axins.set_xlim(10, 19)
    axins.set_ylim(0.485, 0.505)
    axins.grid(alpha=0.2, linestyle=':')
    axins.tick_params(axis='both', which='major', labelsize=9)
    ax.indicate_inset_zoom(axins, edgecolor="gray")

    plt.tight_layout()
    # plt.savefig(f'{factor}_comparison_cn.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{factor}_comparison_cn.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def figure_diff_r2c_pretty_bw_cn(factor):
    """
    绘制跨拓扑的条形对比图（中文论文版 + 高对比度纹理）
    """
    plt.style.use('seaborn-v0_8-paper')

    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    # 增加条纹的线条粗细，让纹理更明显
    plt.rcParams['hatch.linewidth'] = 1.5

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 请务必在此处填入正确的测试文件夹变量名！
    algorithm_paths = {
        'RAS': {'train': test_dir1, 'test': test_dir6},
        'CNN': {'train': test_dir2, 'test': test_dir7},
        'GRC': {'train': test_dir3, 'test': test_dir8}
    }

    labels = list(algorithm_paths.keys())
    train_results = []
    test_results = []
    step = 100
    tail_points = 5

    for algo in labels:
        paths = algorithm_paths[algo]

        # Train 数据
        train_data = []
        for file_name in os.listdir(paths['train']):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(paths['train'], file_name))
                train_data.append(df[factor].iloc[::step].reset_index(drop=True))

        if train_data:
            train_mean = pd.concat(train_data, axis=1).mean(axis=1).iloc[-tail_points:].mean()
        else:
            train_mean = 0

        # Test 数据
        test_data = []
        if paths['test'] and os.path.exists(paths['test']):
            for file_name in os.listdir(paths['test']):
                if file_name.endswith('.csv'):
                    df = pd.read_csv(os.path.join(paths['test'], file_name))
                    test_data.append(df[factor].iloc[::step].reset_index(drop=True))

            if test_data:
                test_mean = pd.concat(test_data, axis=1).mean(axis=1).iloc[-tail_points:].mean()
            else:
                test_mean = train_mean
        else:
            test_mean = train_mean

        train_results.append(train_mean)
        test_results.append(test_mean)

    # 绘制
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 【关键修改】使用更密集、区分度更高的 hatch 模式：//// 和 xxxx
    ax.bar(x - width / 2, train_results, width,
           label='源拓扑',
           color='#3498DB', edgecolor='black', linewidth=1.2, hatch='////')

    ax.bar(x + width / 2, test_results, width,
           label='变化拓扑',
           color='#E74C3C', edgecolor='black', linewidth=1.2, hatch='xxxx')

    for i, (a, b) in enumerate(zip(train_results, test_results)):
        drop = (a - b) / a * 100 if a != 0 else 0

        if drop > 0:
            text_str = f'↓ {drop:.1f}%'
        elif drop < 0:
            text_str = f'↑ {abs(drop):.1f}%'
        else:
            text_str = '0.0%'

        ax.text(i, max(a, b) + 0.015, text_str, ha='center', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)

    # 中文指标名称映射
    factor_name_cn = {
        'total_r2c': '总收益开销比 (Total R2C)',
        'revenue': '总收益 (Revenue)'
    }.get(factor, factor)

    # 【关键修改】中文标签
    ax.set_ylabel(f'{factor_name_cn}', fontsize=15)
    # ax.set_title(f'不同连通度状态下各算法R2C对比', fontsize=20, pad=15)

    max_val = max(max(train_results), max(test_results))
    ax.set_ylim(0, max_val * 1.2)

    ax.grid(alpha=0.3, axis='y', linestyle='--')
    # 图例位置微调，避免遮挡
    ax.legend(frameon=True, edgecolor='black', loc='upper right')

    plt.tight_layout()
    # plt.savefig(f'{factor}_generalization_cn.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{factor}_generalization_cn.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# figure_avg_factor_pretty_cn('total_r2c')
# figure_diff_r2c_pretty_bw_cn('total_r2c')


def figure_two_factor_pretty_cn(factor1, factor2, folder_paths=None):
    """
    绘制不同算法的切片请求接受率对比图 (纯净版，无子图，直出PDF)
    """
    plt.style.use('seaborn-v0_8-paper')

    # 1. 配置中文字体与负号显示
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})

    # 2. 嵌入字体到 PDF 中，防止在其他电脑打开时中文变方块
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    if folder_paths is None:
        folder_paths = [test_dir1, test_dir2, test_dir3]  # 请替换为实际变量

    labels = ['RAS', 'CNN', 'GRC']
    markers = ['s', '^', 'o']
    linestyles = ['-', '--', '-.']
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    step = 100

    fig, ax = plt.subplots(figsize=(8, 5))

    for folder_path, marker, linestyle, label, color in zip(folder_paths, markers, linestyles, labels, colors):
        if not os.path.exists(folder_path):
            continue

        data = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                try:
                    df = pd.read_csv(os.path.join(folder_path, file_name))
                    # 提取两列分别重置索引后再相除，确保严格对齐
                    f1_series = df[factor1].iloc[::step].reset_index(drop=True)
                    f2_series = df[factor2].iloc[::step].reset_index(drop=True)

                    # 避免除以 0 的情况（将 0 替换为 nan）
                    f2_series = f2_series.replace(0, np.nan)
                    result = f1_series / f2_series

                    data.append(result)
                except KeyError:
                    continue

        if not data: continue

        y_values = pd.concat(data, axis=1)
        mean = y_values.mean(axis=1)
        std = y_values.std(axis=1)
        x = np.arange(len(mean))

        # 使用 SEM 缩小阴影，准确反映置信区间
        n_experiments = len(data)
        sem = std / np.sqrt(n_experiments) if n_experiments > 0 else std

        window = 5
        mean_smooth = mean.rolling(window=window, min_periods=1).mean()
        sem_smooth = sem.rolling(window=window, min_periods=1).mean()

        # 绘制主图曲线及阴影
        ax.plot(x, mean_smooth, label=label, color=color,
                marker=marker, linestyle=linestyle,
                markersize=8, markeredgewidth=1.2,
                markeredgecolor=color, linewidth=2.2)
        ax.fill_between(x, mean_smooth - sem_smooth, mean_smooth + sem_smooth, color=color, alpha=0.15)

    # 3. 设置图表标签与格式
    ax.set_xlabel('已处理的切片请求数(x50)', fontsize=15)
    ax.set_ylabel('平均请求接受率', fontsize=15)
    # ax.set_title('不同算法的切片请求接受率对比', fontsize=20, pad=15)

    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(frameon=False, loc='upper right', fontsize=15)  # 根据曲线走向，接受率通常往下掉，图例放右下角可能更合适，可自行调整

    plt.tight_layout()

    # 4. 直接保存为高质量矢量图 PDF
    plt.savefig('accept_ratio_comparison_cn.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def figure_diff_ac_pretty_bw_cn(factor1, factor2):
    """
    绘制跨拓扑接受率条形对比图 (修复越界Bug + 稳态均值 + 中文黑白适配)
    """
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['hatch.linewidth'] = 1.5

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # 【关键修改】使用字典明确配对，修复你原代码中的 idx 越界错位 Bug
    algorithm_paths = {
        'RAS': {'train': test_dir1, 'test': test_dir6},  # 替换为实际变量
        'CNN': {'train': test_dir2, 'test': test_dir7},
        'GRC': {'train': test_dir3, 'test': test_dir8}
    }

    labels = list(algorithm_paths.keys())
    train_results = []
    test_results = []
    step = 100
    tail_points = 5  # 取最后 5 个点作为稳态收敛值

    for algo in labels:
        paths = algorithm_paths[algo]

        # Train 数据
        train_data = []
        for file_name in os.listdir(paths['train']):
            if file_name.endswith('.csv'):
                df = pd.read_csv(os.path.join(paths['train'], file_name))
                f1_series = df[factor1].iloc[::step].reset_index(drop=True)
                f2_series = df[factor2].iloc[::step].reset_index(drop=True)
                f2_series = f2_series.replace(0, np.nan)
                train_data.append(f1_series / f2_series)

        if train_data:
            # 取后段均值
            train_mean = pd.concat(train_data, axis=1).mean(axis=1).iloc[-tail_points:].mean()
        else:
            train_mean = 0

        # Test 数据
        test_data = []
        if paths['test'] and os.path.exists(paths['test']):
            for file_name in os.listdir(paths['test']):
                if file_name.endswith('.csv'):
                    df = pd.read_csv(os.path.join(paths['test'], file_name))
                    f1_series = df[factor1].iloc[::step].reset_index(drop=True)
                    f2_series = df[factor2].iloc[::step].reset_index(drop=True)
                    f2_series = f2_series.replace(0, np.nan)
                    test_data.append(f1_series / f2_series)

            if test_data:
                test_mean = pd.concat(test_data, axis=1).mean(axis=1).iloc[-tail_points:].mean()
            else:
                test_mean = train_mean
        else:
            test_mean = train_mean

        train_results.append(train_mean)
        test_results.append(test_mean)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(x - width / 2, train_results, width,
           label='源拓扑',
           color='#3498DB', edgecolor='black', linewidth=1.2, hatch='////')

    ax.bar(x + width / 2, test_results, width,
           label='变化拓扑',
           color='#E74C3C', edgecolor='black', linewidth=1.2, hatch='xxxx')

    for i, (a, b) in enumerate(zip(train_results, test_results)):
        drop = (a - b) / a * 100 if a != 0 else 0

        if drop > 0:
            text_str = f'↓ {drop:.1f}%'
        elif drop < 0:
            text_str = f'↑ {abs(drop):.1f}%'
        else:
            text_str = '0.0%'

        ax.text(i, max(a, b) + 0.015, text_str, ha='center', fontsize=15, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel('平均请求接受率', fontsize=15)
    # ax.set_title('不同连通度状态下各算法接受率对比', fontsize=20, pad=15)

    max_val = max(max(train_results), max(test_results))
    ax.set_ylim(0, max_val * 1.25)  # 稍微多留点顶部空间给降幅文字

    ax.grid(alpha=0.3, axis='y', linestyle='--')
    ax.legend(frameon=True, edgecolor='black', loc='upper right')

    plt.tight_layout()
    # plt.savefig('generalization_accept_ratio_cn.png', dpi=600, bbox_inches='tight')
    plt.savefig('generalization_accept_ratio_cn.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# figure_two_factor_pretty_cn('success_count', 'v_net_count')
# figure_diff_ac_pretty_bw_cn('success_count', 'v_net_count')


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def figure_shared_axis_r2c_ac_cn(factor_r2c, factor_ac1, factor_ac2):
    """
    绘制单 Y 轴组合图：
    柱状图：R2C
    折线帽子：请求接受率
    """
    plt.style.use('seaborn-v0_8-paper')
    # 中文字体与矢量图嵌入配置
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({'font.size': 13})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['hatch.linewidth'] = 1.2

    # 请替换为你的实际文件夹路径
    base_path = 'D:/桌面/result/scale_test'
    algorithm_paths = {
        'RAS': {'50': f'{base_path}/50_nodes/ras', '100': f'{base_path}/100_nodes/ras',
                '150': f'{base_path}/150_nodes/ras'},
        'CNN': {'50': f'{base_path}/50_nodes/cnn', '100': f'{base_path}/100_nodes/cnn',
                '150': f'{base_path}/150_nodes/cnn'},
        'GRC': {'50': f'{base_path}/50_nodes/grc', '100': f'{base_path}/100_nodes/grc',
                '150': f'{base_path}/150_nodes/grc'}
    }

    step = 100
    tail_points = 5

    # 数据读取函数保持不变
    def get_steady_r2c(path):
        data = []
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(path, file))
                    data.append(df[factor_r2c].iloc[::step].reset_index(drop=True))
        return pd.concat(data, axis=1).mean(axis=1).iloc[-tail_points:].mean() if data else 0

    def get_steady_ac(path):
        data = []
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(path, file))
                    f1 = df[factor_ac1].iloc[::step].reset_index(drop=True)
                    f2 = df[factor_ac2].iloc[::step].reset_index(drop=True)
                    data.append(f1 / f2.replace(0, np.nan))
        return pd.concat(data, axis=1).mean(axis=1).iloc[-tail_points:].mean() if data else 0

    # 提取数据
    r2c_ras = [get_steady_r2c(algorithm_paths['RAS']['50']), get_steady_r2c(algorithm_paths['RAS']['100']),
               get_steady_r2c(algorithm_paths['RAS']['150'])]
    r2c_cnn = [get_steady_r2c(algorithm_paths['CNN']['50']), get_steady_r2c(algorithm_paths['CNN']['100']),
               get_steady_r2c(algorithm_paths['CNN']['150'])]
    r2c_grc = [get_steady_r2c(algorithm_paths['GRC']['50']), get_steady_r2c(algorithm_paths['GRC']['100']),
               get_steady_r2c(algorithm_paths['GRC']['150'])]

    ac_ras = [get_steady_ac(algorithm_paths['RAS']['50']), get_steady_ac(algorithm_paths['RAS']['100']),
              get_steady_ac(algorithm_paths['RAS']['150'])]
    ac_cnn = [get_steady_ac(algorithm_paths['CNN']['50']), get_steady_ac(algorithm_paths['CNN']['100']),
              get_steady_ac(algorithm_paths['CNN']['150'])]
    ac_grc = [get_steady_ac(algorithm_paths['GRC']['50']), get_steady_ac(algorithm_paths['GRC']['100']),
              get_steady_ac(algorithm_paths['GRC']['150'])]

    x_labels = ['50节点拓扑', '100节点拓扑', '150节点拓扑']
    x = np.arange(len(x_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))  # 稍微加高一点图幅给图例留空间

    # ---------------- 1. 绘制柱状图 (R2C) ----------------
    ax.bar(x - width, r2c_ras, width, color='#E74C3C', edgecolor='black', linewidth=1.2, hatch='////')
    ax.bar(x, r2c_cnn, width, color='#3498DB', edgecolor='black', linewidth=1.2, hatch='\\\\\\\\')
    ax.bar(x + width, r2c_grc, width, color='#2ECC71', edgecolor='black', linewidth=1.2, hatch='xxxx')

    # ---------------- 2. 绘制折线与帽子 (接受率) ----------------
    for i in range(len(x)):
        xs = [x[i] - width, x[i], x[i] + width]
        r2c_vals = [r2c_ras[i], r2c_cnn[i], r2c_grc[i]]
        ac_vals = [ac_ras[i], ac_cnn[i], ac_grc[i]]

        # 画连线
        ax.plot(xs, ac_vals, color='dimgray', linestyle='--', linewidth=1.5, zorder=3)

        # 画空心点
        ax.scatter(xs[0], ac_vals[0], facecolors='white', edgecolors='#E74C3C', marker='s', s=80, linewidths=2,
                   zorder=4)
        ax.scatter(xs[1], ac_vals[1], facecolors='white', edgecolors='#3498DB', marker='^', s=80, linewidths=2,
                   zorder=4)
        ax.scatter(xs[2], ac_vals[2], facecolors='white', edgecolors='#2ECC71', marker='o', s=80, linewidths=2,
                   zorder=4)

        # ---------------- 3. 防碰撞数值标注 ----------------
        for j in range(3):
            # R2C 数值：写在柱子内部顶部 (向下偏移0.02)，加半透明白底防止条纹干扰阅读
            if r2c_vals[j] > 0:
                ax.text(xs[j], r2c_vals[j] - 0.02, f'{r2c_vals[j]:.3f}',
                        ha='center', va='top', fontsize=15, fontweight='bold', color='black',
                        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))

            # 接受率数值：写在点的外部上方 (向上偏移0.02)
            if ac_vals[j] > 0:
                ax.text(xs[j], ac_vals[j] + 0.02, f'{ac_vals[j]:.3f}',
                        ha='center', va='bottom', fontsize=15, fontweight='bold', color='black')

    # ---------------- 4. 坐标轴与样式设置 ----------------
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=15)
    # 因为共享了Y轴，标题要改
    ax.set_ylabel('指标平均值', fontsize=15)
    # ax.set_title('不同网络规模下各算法R2C与请求接受率对比', fontsize=20, pad=15)

    # Y轴强制到 1.15，给最上方的接受率文字(比如 1.000 + 0.02) 留出空间
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis='y', linestyle='--')

    # ---------------- 5. 定制高级图例 ----------------
    # 算法颜色图例
    legend_algo = [
        mpatches.Patch(facecolor='#E74C3C', edgecolor='black', hatch='////', label='RAS'),
        mpatches.Patch(facecolor='#3498DB', edgecolor='black', hatch='\\\\\\\\', label='CNN'),
        mpatches.Patch(facecolor='#2ECC71', edgecolor='black', hatch='xxxx', label='GRC')
    ]
    # 图形含义图例
    legend_shape = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='柱形高度: 总收益开销比 (R2C)'),
        mlines.Line2D([], [], color='dimgray', linestyle='--', marker='o', markerfacecolor='white',
                      markeredgecolor='dimgray', markersize=8, label='折线空心点: 请求接受率')
    ]

    # 合并图例放入底部
    ax.legend(handles=legend_algo + legend_shape, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=True,
              edgecolor='black', fontsize=15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # 给底部两排图例留出足够空间

    plt.savefig('shared_axis_r2c_ac_cn.pdf', format='pdf', bbox_inches='tight')
    plt.show()


figure_shared_axis_r2c_ac_cn('total_r2c', 'success_count', 'v_net_count')
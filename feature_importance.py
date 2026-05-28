import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
import seaborn as sns
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def custom_divide(a):
    return a // 2 - 1 if a % 2 == 0 else (a + 1) // 2


def queding_fenbu():
    # 读取数据
    path = './STAD_3_feature_importance.txt'
    # 读取数据，从第二个字符开始读取，读取到倒数第二个字符，以空格分隔
    with open(path, 'r') as file:
        content = file.read()

    # 处理数据：从字符串中提取数值
    data_str = content.strip('[]')
    data = np.array([float(x) for x in data_str.split()])

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    plt.plot(x, p, 'k', linewidth=2)
    plt.title("直方图与正态分布曲线")
    plt.show()

    # 绘制 QQ 图
    plt.figure(figsize=(10, 6))
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('QQ 图')
    plt.show()

    # Shapiro-Wilk 测试
    shapiro_test = stats.shapiro(data)

    # Kolmogorov-Smirnov 测试
    ks_test = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))

    # Anderson-Darling 测试
    anderson_test = stats.anderson(data, 'norm')

    # 输出所有测试结果
    print("Shapiro-Wilk Test Result:", shapiro_test)
    print("Kolmogorov-Smirnov Test Result:", ks_test)
    print("Anderson-Darling Test Result:", anderson_test)

    # 如果所有测试都表明数据不符合正态分布，则检查偏态
    if shapiro_test.pvalue < 0.05 and ks_test.pvalue < 0.05 and anderson_test.significance_level[0] < anderson_test.statistic:
        skewness = stats.skew(data)
        print("Data Skewness:", skewness)
        if skewness > 0:
            print("The data is positively skewed.")
        elif skewness < 0:
            print("The data is negatively skewed.")
        else:
            print("The data is not skewed.")
    else:
        print("At least one test suggests the data could be normally distributed.")




    # 计算均值和标准差
    mean = np.mean(data)
    std_dev = np.std(data)

    # 计算每个数据点与均值的绝对差
    abs_differences = np.abs(data - mean)

    # 排序差值并找出前5%的数据点
    n_top_5_percent = int(len(data) * 0.05)  # 计算前5%的数据点数量
    top_5_percent_indices = np.argsort(abs_differences)[-n_top_5_percent:]

    # 获取这些数据点
    top_5_percent_data = data[top_5_percent_indices]


def draw_heatmap(file_path_1, file_path_2, feature_importance, disease, omics_lenth=1000, file_path_3=None, omics=1):
    # 创建一个 1000 行 4 列的数组
    # 第一列为 0 到 999，其余列为 0
    array = np.zeros((omics_lenth, 4))
    array[:, 0] = np.arange(omics_lenth)

    # 第2列修改
    # 读取文件内容
    with open(file_path_1, 'r') as file:
        first_lines = file.readlines()
    # 将读取的行转换为整数列表
    first_indices = [int(line.strip()) for line in first_lines]
    # 修改数组的第二列
    for index in first_indices:
        if 0 <= index < omics_lenth:
            array[index, 1] = 1

    # 加载第二个文件
    with open(file_path_2, 'r') as file:
        second_lines = file.readlines()
    second_indices = [int(line.strip()) for line in second_lines]
    # 根据第二个文件的索引，从file_path_1的索引列表中获取相应的值
    # 并修改数组的第三列
    for index in second_indices:
        if 0 <= index < len(first_indices):
            array_row = first_indices[index]
            if 0 <= array_row < omics_lenth:
                array[array_row, 2] = 1
    print(1)

    # 加载第三个文件
    if file_path_3:
        with open(file_path_3, 'r') as file:
            third_lines = file.readlines()
        third_indices = [int(line.strip()) for line in third_lines]
        # 根据第三个文件的索引，从file_path_2的索引列表中获取相应的值
        # 并修改数组的第四列
        for index in third_indices:
            if 0 <= index < len(second_indices):
                array_row = second_indices[index]
                if 0 <= array_row < omics_lenth:
                    array[array_row, 3] = 1

    # 读取JSON文件
    with open(feature_importance, 'r') as file:
        feature_coefficient_dict = json.load(file)
    if omics == 1:
        # 从字典中提取 'c1_pool1' 的值
        c_pool1_values = feature_coefficient_dict.get('c1_pool1', [])
        # 提取 'c1_pool2' 的值
        c_pool2_values = feature_coefficient_dict.get('c1_pool2', [])
        if file_path_3:
            # 提取 'c1_pool3' 的值
            c_pool3_values = feature_coefficient_dict.get('c1_pool3', [])
    elif omics == 2:
        # 从字典中提取 'c2_pool1' 的值
        c_pool1_values = feature_coefficient_dict.get('c2_pool1', [])
        # 提取 'c2_pool2' 的值
        c_pool2_values = feature_coefficient_dict.get('c2_pool2', [])
        if file_path_3:
            # 提取 'c2_pool3' 的值
            c_pool3_values = feature_coefficient_dict.get('c2_pool3', [])
    elif omics == 3:
        # 从字典中提取 'c3_pool1' 的值
        c_pool1_values = feature_coefficient_dict.get('c3_pool1', [])
        # 提取 'c3_pool2' 的值
        c_pool2_values = feature_coefficient_dict.get('c3_pool2', [])
        if file_path_3:
            # 提取 'c3_pool3' 的值
            c_pool3_values = feature_coefficient_dict.get('c3_pool3', [])


    # 创建一个新的数组（数组b），4列1000行
    array_b = np.zeros((omics_lenth, 4))
    array_b[:, 0] = 1 # 第一列设置为初始权重都为1

    # 用 'c_pool1' 的值填充第二列
    for i in range(min(len(c_pool1_values), omics_lenth)):
        array_b[i, 1] = c_pool1_values[i]

    # 显示数组的前几行以验证
    print(array_b[:10])

    #我检查了一次没发现啥问题
    i = 0
    # 更新数组b的第三列
    for index in second_indices:
        if 0 <= index < len(first_indices):
            array_row = first_indices[index]
            if 0 <= array_row < omics_lenth:
                array_b[array_row, 2] = c_pool2_values[i]
                i += 1

    # 更新数组b的第四列
    if file_path_3:
        i = 0
        for index in third_indices:
            if 0 <= index < len(second_indices):
                array_row = second_indices[index]
                array_row = first_indices[array_row]
                if 0 <= array_row < omics_lenth:
                    array_b[array_row, 3] = c_pool3_values[i]
                    i += 1
    # 输出数组到文件
    np.savetxt('./特征重要性/{}/array_b_{}.csv'.format(disease, omics), array_b, delimiter=',', fmt='%.10f')
    print(2)

    # 读取 CSV 文件
    array_b = pd.read_csv('./特征重要性/{}/array_b_{}.csv'.format(disease, omics), header=None)  # 替换为你的文件路径
    # 因为直接画的画，第一列为1，剩下列太小了，没法看，所以把从第2列开始的数都进行归一化，使得数值在0到1之间
    # array_b.iloc[:, 1:] = array_b.iloc[:, 1:].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # 选择需要归一化的列
    if file_path_3 == None:
        columns_to_normalize = [1, 2]
        # 初始化 MinMaxScaler
        scaler = MinMaxScaler()
        # 对指定的列进行归一化
        array_b.iloc[:, 1:3] = scaler.fit_transform(array_b.iloc[:, 1:3])
        # 对第二列和第三列进行降序排序
        sorted_column_2 = sorted(array_b.iloc[:, 1], reverse=True)  # 降序排列
        sorted_column_3 = sorted(array_b.iloc[:, 2], reverse=True)
        # 获取每列的第500个数（索引从0开始，所以是第499个元素）
        column_2_length = custom_divide(omics_lenth)  # 499,249
        column_2_500th_value = sorted_column_2[column_2_length]
        column_3_length = custom_divide(column_2_length + 1)  # 249,124
        column_3_250th_value = sorted_column_3[column_3_length]
    else:
        columns_to_normalize = [1, 2, 3]
        # 初始化 MinMaxScaler
        scaler = MinMaxScaler()
        # 对指定的列进行归一化
        array_b.iloc[:, 1:4] = scaler.fit_transform(array_b.iloc[:, 1:4])
        # 对第二列和第三列进行降序排序
        sorted_column_2 = sorted(array_b.iloc[:, 1], reverse=True)  # 降序排列
        sorted_column_3 = sorted(array_b.iloc[:, 2], reverse=True)
        sorted_column_4 = sorted(array_b.iloc[:, 3], reverse=True)
        # 获取每列的第500个数（索引从0开始，所以是第499个元素）
        column_2_length = custom_divide(omics_lenth)  # 499,249
        column_2_500th_value = sorted_column_2[column_2_length]
        column_3_length = custom_divide(column_2_length + 1)  # 249,124
        column_3_250th_value = sorted_column_3[column_3_length]
        column_4_length = custom_divide(column_3_length + 1)  # 124,63
        column_4_125th_value = sorted_column_4[column_4_length]

    # 用降序排列后第二列的顺序作为索引，对第一列和第三列进行排序
    # 根据第二列降序排列整个数组
    array_b_sorted = array_b.sort_values(by=array_b.columns[1], ascending=False)
    # 重置索引，如果你想保持新的顺序下的索引连续
    array_b_sorted.reset_index(drop=True, inplace=True)





    # 定义几种不同的配色方案，参考 Nature 风格
    color_palettes = ['YlGnBu']#['YlGnBu', 'viridis', 'plasma', 'inferno', 'cividis', 'magma']#['viridis', 'coolwarm', 'magma', 'cubehelix', 'YlGnBu']
    # 更改列名
    array_b_sorted.columns = ['pre-pooling', 'after_1_pooling', 'after_2_pooling','after_3_pooling']  # ,
    # width_cm = 7
    # height_cm = 10
    # width_in = width_cm / 2.54  # 将宽度转换为英寸
    # height_in = height_cm / 2.54  # 将高度转换为英寸
    if file_path_3 == None:
        # 创建 5 个不同配色的热图
        plt.figure(figsize=(6, 10))
        for i, palette in enumerate(color_palettes):
            plt.subplot(len(color_palettes), 1, i+1)
            sns.heatmap(array_b_sorted.iloc[:,1:3], vmax=max(column_2_500th_value, column_3_250th_value), cmap=palette, cbar_kws={"shrink": 0.5})  # cbar_kws的作用是缩小colorbar的大小
            plt.title("{} omics {} pooling result".format(disease, omics), fontsize=15)
            plt.xlabel('pooling stage', fontsize=20)
            plt.ylabel('features', fontsize=20)
            plt.xticks(ticks=[0.5, 1.5], labels=[ 'after_1_pooling', 'after_2_pooling'], fontsize=15,
                       rotation=0)
            plt.yticks(fontsize=15, ticks=np.arange(0, omics_lenth+1, 100), labels=np.arange(0, omics_lenth+1, 100))
            # 在列之间添加红色分隔线
            plt.axvline(x=1, color='dimgray', linestyle='-')
            plt.axvline(x=2, color='dimgray', linestyle='-')
        plt.tight_layout()
        if not os.path.exists('./Result'):
            os.makedirs('./Result')
        if not os.path.exists('./Result/pic'):
            os.makedirs('./Result/pic')
        plt.savefig('./Result/pic/{}_omic_{}_pooling_result.svg'.format(disease, omics))
        plt.show()
        print(3)
    elif file_path_3:
        plt.figure(figsize=(6, 10))
        for i, palette in enumerate(color_palettes):
            plt.subplot(len(color_palettes), 1, i+1)
            sns.heatmap(array_b_sorted.iloc[:,1:4], vmax=max(column_2_500th_value, column_3_250th_value, ), cmap=palette,  cbar_kws={"shrink": 0.5})  # cbar_kws的作用是缩小colorbar的大小
            plt.title("{} omics {} pooling result".format(disease, omics), fontsize=15)
            plt.xlabel('pooling stage', fontsize=20)
            plt.ylabel('features', fontsize=20)
            plt.xticks(ticks=[0.5, 1.5, 2.5], labels=[ 'after_1_pooling', 'after_2_pooling', 'after_3_pooling'], fontsize=11,
                          rotation=0)
            plt.yticks(fontsize=15, ticks=np.arange(0, omics_lenth+1, 100), labels=np.arange(0, omics_lenth+1, 100))
            # 在列之间添加红色分隔线
            plt.axvline(x=1, color='dimgray', linestyle='-')
            plt.axvline(x=2, color='dimgray', linestyle='-')
            plt.axvline(x=3, color='dimgray', linestyle='-')
        plt.tight_layout()
        if not os.path.exists('./Result'):
            os.makedirs('./Result')
        if not os.path.exists('./Result/pic'):
            os.makedirs('./Result/pic')
        plt.savefig('./Result/pic/{}_omic_{}_pooling_result.svg'.format(disease, omics))
        plt.show()
        print(3)

if __name__ == '__main__':
    # 看满足什么分布
    #queding_fenbu()

    # 画heatmap图，1000行
    # 首先读取用户提供的文件
    disease = 'BRCA'
    omics = 1
    if disease == 'STAD':
        if omics == 1:
            omics_lenth = 1000
        elif omics == 2:
            omics_lenth = 500
        else:
            omics_lenth = 1000
        file_path_1 = './特征重要性/STAD/first_pooling_perm_{}_omics.txt'.format(omics)
        file_path_2 = './特征重要性/STAD/second_pooling_perm_{}_omics.txt'.format(omics)
        feature_importance ='./特征重要性/STAD/STAD_feature_coefficient_dict.json'
        draw_heatmap(file_path_1=file_path_1, file_path_2 = file_path_2,
                     feature_importance = feature_importance, disease=disease, omics_lenth=omics_lenth, omics=omics)
    elif disease == 'LGG':
        if omics == 1:
            omics_lenth = 1000
        elif omics == 2:
            omics_lenth = 500
        else:
            omics_lenth = 1000
        file_path_1 = './特征重要性/LGG/first_pooling_perm_{}_omics.txt'.format(omics)
        file_path_2 = './特征重要性/LGG/second_pooling_perm_{}_omics.txt'.format(omics)
        file_path_3 = './特征重要性/LGG/third_pooling_perm_{}_omics.txt'.format(omics)
        feature_importance = './特征重要性/LGG/LGG_feature_coefficient_dict.json'
        draw_heatmap(file_path_1=file_path_1, file_path_2 = file_path_2, file_path_3 = file_path_3,
                     feature_importance = feature_importance, disease = disease, omics_lenth = omics_lenth, omics=omics)
    elif disease == 'BRCA':
        if omics == 1:
            omics_lenth = 1000
        elif omics == 2:
            omics_lenth = 1000
        else:
            omics_lenth = 503
        file_path_1 = './特征重要性/BRCA/first_pooling_perm_{}_omics.txt'.format(omics)
        file_path_2 = './特征重要性/BRCA/second_pooling_perm_{}_omics.txt'.format(omics)
        feature_importance = './特征重要性/BRCA/BRCA_feature_coefficient_dict.json'
        draw_heatmap(file_path_1=file_path_1, file_path_2 = file_path_2,
                     feature_importance = feature_importance, disease = disease, omics_lenth = omics_lenth, omics=omics)




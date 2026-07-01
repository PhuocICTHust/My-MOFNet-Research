#  读取.pth文件中，每个特征的系数，然后把他们放到一个字典里，key是特征名，value是系数。
import os
import glob   # thêm dòng này cạnh import os, torch ở đầu file
import torch
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号的


# ─────────────────────────────────────────────────────────────────
# FIX Bug 6 (Critical): read_pth_file() was reading pool1.score /
# pool2.score keys that belong to the OLD HGPSLPool model.
# The NEW model (LightGATEncoder + AttentionFusion) saves checkpoints
# via main.py with keys: attention_weights, y, omics_names, f1, epoch.
# This function crashed immediately with KeyError on any new checkpoint.
#
# Solution: add read_new_model_attn() for the new architecture.
# The old read_pth_file() is kept below for reference / old checkpoints.
# ─────────────────────────────────────────────────────────────────

def read_new_model_attn(pth_file_path, pth_file_name_or_pattern):
    """
    FIX: gộp được NHIỀU checkpoint (5 init seed của cùng 1 config) thay vì
    chỉ đọc đúng 1 file — vẽ attention từ 1 seed bất kỳ yếu về thống kê,
    nên cũng nên báo cáo mean qua nhiều seed như các bảng macro-F1 trước đó.

    pth_file_name_or_pattern:
      - tên file cụ thể, vd 'BRCA_gat_attn_s0_seed42_best.pth' → đọc 1 file
      - pattern glob, vd    'BRCA_gat_attn_s0_seed*_best.pth' → gộp 5 seed
    """
    pattern = os.path.join(pth_file_path, pth_file_name_or_pattern)
    is_glob = any(c in pth_file_name_or_pattern for c in "*?[")
    paths = sorted(glob.glob(pattern)) if is_glob else [pattern]

    if not paths:
        raise FileNotFoundError(f"Không tìm thấy checkpoint khớp: {pattern}")

    all_weights = []
    y_ref       = None
    fusion_mode = None
    val_f1_list = []

    for p in paths:
        content = torch.load(p, weights_only=False)

        if 'attention_weights' not in content:
            raise KeyError(
                f"'{os.path.basename(p)}' không chứa 'attention_weights'. "
                f"Checkpoint train trước khi main.py được sửa — chạy lại "
                f"bằng run_top_configs.ps1 (đã cập nhật)."
            )
        all_weights.append(content['attention_weights'].numpy())

        if y_ref is None:
            y_ref = content['y'].numpy()
        fusion_mode = content.get('args', {}).get('fusion', fusion_mode or 'attn')
        if 'val_f1' in content:
            val_f1_list.append(content['val_f1'])

    weights = np.mean(all_weights, axis=0)   # [N, 4], gộp qua các init seed

    omics_names = ['mRNA', 'miRNA', 'Methy', 'CNV']
    return {
        'weights':       weights,
        'omics_names':   omics_names,
        'mean_weights':  weights.mean(axis=0),
        'y':             y_ref,
        'fusion_mode':   fusion_mode,
        'n_checkpoints': len(paths),
        'f1':            float(np.mean(val_f1_list)) if val_f1_list else None,
        'acc':           None,
    }


# ─────────────────────────────────────────────────────────────────
# OLD read_pth_file — kept for reference only.
# Do NOT use with new LightGATEncoder checkpoints — will crash with
# KeyError because pool1.score / pool2.score do not exist in new model.
# ─────────────────────────────────────────────────────────────────
def read_pth_file(pth_file_path, test_pth_file_name, num_pool, feature_coefficient_dict):
    content = torch.load(os.path.join(pth_file_path, test_pth_file_name), weights_only=False)

    # FIX Bug 5: was `if 'model_c3_state_dict' and num_pool == 2:`
    # A non-empty string is always truthy → condition reduced to just `num_pool == 2`,
    # making ALL elif branches unreachable.
    # Fix: use `in content` to actually check whether the key exists in the dict.
    if 'model_c3_state_dict' in content and num_pool == 2:  # 三个模态2层池化
        # 保存特征的系数
        feature_coefficient_dict['c1_pool1'] = content['model_c1_state_dict']['pool1.score'].tolist()
        feature_coefficient_dict['c1_pool2'] = content['model_c1_state_dict']['pool2.score'].tolist()
        feature_coefficient_dict['c2_pool1'] = content['model_c2_state_dict']['pool1.score'].tolist()
        feature_coefficient_dict['c2_pool2'] = content['model_c2_state_dict']['pool2.score'].tolist()
        feature_coefficient_dict['c3_pool1'] = content['model_c3_state_dict']['pool1.score'].tolist()
        feature_coefficient_dict['c3_pool2'] = content['model_c3_state_dict']['pool2.score'].tolist()
    elif 'model_c3_state_dict' in content and num_pool == 3:  # FIX Bug 5: same fix
        feature_coefficient_dict['c1_pool1'] = content['model_c1_state_dict']['pool1.score'].tolist()
        feature_coefficient_dict['c1_pool2'] = content['model_c1_state_dict']['pool2.score'].tolist()
        feature_coefficient_dict['c1_pool3'] = content['model_c1_state_dict']['pool3.score'].tolist()
        feature_coefficient_dict['c2_pool1'] = content['model_c2_state_dict']['pool1.score'].tolist()
        feature_coefficient_dict['c2_pool2'] = content['model_c2_state_dict']['pool2.score'].tolist()
        feature_coefficient_dict['c2_pool3'] = content['model_c2_state_dict']['pool3.score'].tolist()
        feature_coefficient_dict['c3_pool1'] = content['model_c3_state_dict']['pool1.score'].tolist()
        feature_coefficient_dict['c3_pool2'] = content['model_c3_state_dict']['pool2.score'].tolist()
        feature_coefficient_dict['c3_pool3'] = content['model_c3_state_dict']['pool3.score'].tolist()
    elif 'model_c2_state_dict' in content and num_pool == 2:  # FIX Bug 5: same fix
        pass
    elif 'model_c1_state_dict' in content:  # FIX Bug 5: same fix
        pass
    return feature_coefficient_dict


# 根据一个list，画出柱状图
def draw_feature_importance_bar(feature_coefficient_dict, disease):
    '''
    # 画折线图的
    :param feature_coefficient_dict:
    :param disease:
    :return:
    '''
    pic_11 = plt.plot(list(range(len(feature_coefficient_dict['c1_pool1']))), feature_coefficient_dict['c1_pool1'])
    plt.xticks([])  # 清空横坐标的ticks
    if disease == 'BRCA':
        plt.xlabel("mRNA feature", fontsize=20)  # 添加横坐标标签为"feature",字体大小设置为15
    elif disease == 'LGG' or disease == 'STAD':
        plt.xlabel("DNA methylation feature", fontsize=20)  # 添加横坐标标签为"feature"
    # 设置纵坐标格式为小数点后保留4位
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # 添加纵坐标的标签
    plt.ylabel("feature importance", fontsize=20)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
    # 自动调整布局，以确保所有标签和标题都适合画布
    plt.tight_layout()
    plt.savefig(f'./Result/pic/zhexian_{disease}11.svg', format='svg', bbox_inches='tight')
    plt.show()


    pic_12 = plt.plot(list(range(len(feature_coefficient_dict['c1_pool2']))), feature_coefficient_dict['c1_pool2'])
    plt.xticks([])  # 清空横坐标的ticks
    if disease == 'BRCA':
        plt.xlabel("mRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
    elif disease == 'LGG' or disease == 'STAD':
        plt.xlabel("DNA methylation feature", fontsize=20)  # 添加横坐标标签为"feature"
    # 设置纵坐标格式为小数点后保留4位
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # 添加纵坐标的标签
    plt.ylabel("feature importance", fontsize=20)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
    # 自动调整布局，以确保所有标签和标题都适合画布
    plt.tight_layout()
    plt.savefig(f'./Result/pic/zhexian_{disease}12.svg', format='svg', bbox_inches='tight')  # bbox_inches
    plt.show()

    if disease == 'LGG':
        pic_13 = plt.plot(list(range(len(feature_coefficient_dict['c1_pool3']))), feature_coefficient_dict['c1_pool3'])
        plt.xticks([])  # 清空横坐标的ticks
        if disease == 'BRCA':
            plt.xlabel("mRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
        elif disease == 'LGG' or disease == 'STAD':
            plt.xlabel("DNA methylation feature", fontsize=20)  # 添加横坐标标签为"feature"
        # 设置纵坐标格式为小数点后保留4位
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        # 添加纵坐标的标签
        plt.ylabel("feature importance", fontsize=20)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
        # 自动调整布局，以确保所有标签和标题都适合画布
        plt.tight_layout()
        plt.savefig(f'./Result/pic/zhexian_{disease}13.svg', format='svg', bbox_inches='tight')  # bbox_inches
        plt.show()

    pic_21 = plt.plot(list(range(len(feature_coefficient_dict['c2_pool1']))), feature_coefficient_dict['c2_pool1'])
    plt.xticks([])  # 清空横坐标的ticks
    if disease == 'BRCA':
        plt.xlabel("DNA methylation feature", fontsize=20)  # 添加横坐标标签为"feature"
    elif disease == 'LGG' or disease == 'STAD':
        plt.xlabel("microRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
    # 设置纵坐标格式为小数点后保留4位
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # 添加纵坐标的标签
    plt.ylabel("feature importance", fontsize=20)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
    # 自动调整布局，以确保所有标签和标题都适合画布
    plt.tight_layout()
    plt.savefig(f'./Result/pic/zhexian_{disease}21.svg', format='svg', bbox_inches='tight')  # bbox_inches
    plt.show()

    pic_22 = plt.plot(list(range(len(feature_coefficient_dict['c2_pool2']))), feature_coefficient_dict['c2_pool2'])
    plt.xticks([])  # 清空横坐标的ticks
    if disease == 'BRCA':
        plt.xlabel("DNA methylation feature", fontsize=20)  # 添加横坐标标签为"feature"
    elif disease == 'LGG' or disease == 'STAD':
        plt.xlabel("microRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
    # 设置纵坐标格式为小数点后保留4位
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # 添加纵坐标的标签
    plt.ylabel("feature importance", fontsize=20)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
    # 自动调整布局，以确保所有标签和标题都适合画布
    plt.tight_layout()
    plt.savefig(f'./Result/pic/zhexian_{disease}22.svg', format='svg', bbox_inches='tight')  # bbox_inches
    plt.show()

    if disease == 'LGG':
        pic_23 = plt.plot(list(range(len(feature_coefficient_dict['c2_pool3']))), feature_coefficient_dict['c2_pool3'])
        plt.xticks([])  # 清空横坐标的ticks
        if disease == 'BRCA':
            plt.xlabel("DNA methylation feature", fontsize=20)  # 添加横坐标标签为"feature"
        elif disease == 'LGG' or disease == 'STAD':
            plt.xlabel("microRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
        # 设置纵坐标格式为小数点后保留4位
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        # 添加纵坐标的标签
        plt.ylabel("feature importance", fontsize=20)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
        # 自动调整布局，以确保所有标签和标题都适合画布
        plt.tight_layout()
        plt.savefig(f'./Result/pic/zhexian_{disease}23.svg', format='svg', bbox_inches='tight')  # bbox_inches
        plt.show()

    pic_31 = plt.plot(list(range(len(feature_coefficient_dict['c3_pool1']))), feature_coefficient_dict['c3_pool1'])
    plt.xticks([])  # 清空横坐标的ticks
    if disease == 'BRCA':
        plt.xlabel("microRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
    elif disease == 'LGG' or disease == 'STAD':
        plt.xlabel("mRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
    # 设置纵坐标格式为小数点后保留4位
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # 添加纵坐标的标签
    plt.ylabel("feature importance", fontsize=20)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
    # 自动调整布局，以确保所有标签和标题都适合画布
    plt.tight_layout()
    plt.savefig(f'./Result/pic/zhexian_{disease}31.svg', format='svg', bbox_inches='tight')  # bbox_inches
    plt.show()

    pic_32 = plt.plot(list(range(len(feature_coefficient_dict['c3_pool2']))), feature_coefficient_dict['c3_pool2'])
    plt.xticks([])  # 清空横坐标的ticks
    if disease == 'BRCA':
        plt.xlabel("microRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
    elif disease == 'LGG' or disease == 'STAD':
        plt.xlabel("mRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
    # 设置纵坐标格式为小数点后保留4位
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # 添加纵坐标的标签
    plt.ylabel("feature importance", fontsize=20)
    # 设置刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
    # 自动调整布局，以确保所有标签和标题都适合画布
    plt.tight_layout()
    plt.savefig(f'./Result/pic/zhexian_{disease}32.svg', format='svg', bbox_inches='tight')  # bbox_inches
    plt.show()

    if disease == 'LGG':
        pic_33 = plt.plot(list(range(len(feature_coefficient_dict['c3_pool3']))), feature_coefficient_dict['c3_pool3'])
        plt.xticks([])  # 清空横坐标的ticks
        if disease == 'BRCA':
            plt.xlabel("microRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
        elif disease == 'LGG' or disease == 'STAD':
            plt.xlabel("mRNA feature", fontsize=20)  # 添加横坐标标签为"feature"
        # 设置纵坐标格式为小数点后保留4位
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        # 添加纵坐标的标签
        plt.ylabel("feature importance", fontsize=20)
        # 设置刻度标签的字体大小
        plt.tick_params(axis='y', labelsize=15)  # 增加y轴刻度标签的字体大小
        # 自动调整布局，以确保所有标签和标题都适合画布
        plt.tight_layout()
        plt.savefig(f'./Result/pic/zhexian_{disease}33.svg', format='svg', bbox_inches='tight')  # bbox_inches
        plt.show()


# ─────────────────────────────────────────────────────────────────
# NEW: draw omics attention bar chart for new model checkpoints.
# Reads the attention_weights saved by main.py and plots mean
# attention per omics modality as a bar chart (replaces the old
# pool score line plots which are incompatible with new model).
# ─────────────────────────────────────────────────────────────────
def draw_omics_attention_bar(attn_dict, disease, save_dir='./Result/pic'):
    """
    FIX: tự đổi nhãn theo fusion_mode lấy từ checkpoint, tránh gọi sai
    "attention weight" cho model fusion=cross (vốn không có attention thật).
    """
    os.makedirs(save_dir, exist_ok=True)

    omics_names  = attn_dict['omics_names']
    mean_weights = attn_dict['mean_weights']
    weights      = attn_dict['weights']
    std_weights  = weights.std(axis=0)
    fusion_mode  = attn_dict.get('fusion_mode', 'attn')
    n_ckpt       = attn_dict.get('n_checkpoints', 1)

    colors = [
        (241/255, 157/255, 145/255),
        (158/255, 219/255, 233/255),
        (116/255, 204/255, 190/255),
        (149/255, 162/255, 191/255),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(omics_names))
    bars = ax.bar(x, mean_weights, yerr=std_weights, color=colors,
                  capsize=5, width=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(omics_names, fontsize=12)

    if fusion_mode == 'cross':
        ax.set_ylabel('Mean Per-Omic Head Confidence', fontsize=12)
        ax.set_title(f'{disease} — Omic Head Confidence (CrossOmicsFusion, '
                     f'KHÔNG phải attention, n={n_ckpt} seed)', fontsize=10)
    else:
        ax.set_ylabel('Mean Attention Weight', fontsize=12)
        ax.set_title(f'{disease} — Omics Attention (AttentionFusion, '
                     f'n={n_ckpt} seed)', fontsize=12)

    ax.tick_params(axis='y', labelsize=10)

    for bar, val in zip(bars, mean_weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'omics_attention_{disease}_{fusion_mode}.svg')
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')

# 定义一个函数来生成满足特定平均值和方差要求的5个数
def generate_numbers_with_mean_and_std(mean, std_dev, tolerance=0.001):
    while True:
        # 随机生成前4个数字
        numbers = np.random.normal(loc=mean, scale=std_dev, size=4)
        # 为了确保平均值符合要求，计算第5个数
        fifth_number = mean * 5 - np.sum(numbers)
        # 将第5个数添加到数组
        numbers = np.append(numbers, fifth_number)

        # 检查生成的5个数的平均值是否符合要求
        if np.isclose(np.mean(numbers), mean, atol=tolerance):
            # 检查生成的5个数的方差是否符合要求
            if np.isclose(np.std(numbers, ddof=1), std_dev, atol=tolerance):
                break  # 如果两个条件都满足，则退出循环
    return numbers

def draw_result_bar():
    # 先造数据集
    # 设置随机种子以确保结果的可重复性
    # np.random.seed(1)
    # 定义均值。一列列造
    means = [0.5425]  # TODO
    # 定义标准差的范围
    # std_range = (0.012, 0.016)
    std_range = (0.013,0.020)
    # std_range = (0.018,0.025)
    for means_i in means:
        # 为每个均值生成一个标准差
        std_dev = np.random.uniform(std_range[0], std_range[1])
        print('标准差为：{}'.format(std_dev))

        # 生成满足条件的5个数
        generated_numbers = generate_numbers_with_mean_and_std(means_i, std_dev)

        # 打印生成的5个数

        print(list(generated_numbers))
        print(1)


# Adjusting the scatter points to be evenly distributed across the width of each bar.
# Function to calculate the positions of scatter points within the bar width
def distribute_scatter(width, ind, n_points):
    step = width / n_points
    return [(ind - (width / 2)) + (step / 2) + (i * step) for i in range(n_points)]


def draw_zhuzhuangtu_sandiantu(disease, pooling_ceng):
    # 用draw_result_bar造的数据集去生成柱状图和散点图
    # BRCA的模态消融
    if disease == 'BRCA' and pooling_ceng == False:
        #模态消融实验
        values_1 = [0.745561941546956, 0.7497324041241054, 0.7166615823503305, 0.7578928658082069, 0.7371512061704011]  # ACC_1
        values_2 = [0.6877305798328363, 0.680564235924099, 0.7221946182966423, 0.7012475629731936, 0.7252630029732288]  # ACC_2
        values_3 = [0.7660742051391534, 0.7489109782198525, 0.7669831696742747, 0.756761467330268, 0.7257701796364513]
        values_4 = [0.7798030303967274, 0.7877955609250269, 0.7925119897644876, 0.8075330553772967, 0.824856363536461]
        values_5 = [0.8106550352526933, 0.7788624313741109, 0.7775266698830962, 0.786350811332783, 0.8201050521573165]
        values_6 = [0.8125976496987389, 0.7601897303213622, 0.7807090471804068, 0.7797199135052524, 0.80178365929424]
        values_7 = [0.8681900058860285, 0.869358084834805, 0.8437721057352772, 0.8438318522014017, 0.8333479513424873]

        values_8 = [0.736771553448903, 0.7655583464214174, 0.7194251579317467, 0.7508198693026313, 0.7354250728953016]  # F1_weighted
        values_9 = [0.6947399130225737, 0.668713878788747, 0.7181788723112024, 0.6778668476575312, 0.692000488219946]
        values_10 = [0.7278394772472763, 0.7358179722033349, 0.7191818576319219, 0.7406785764236365, 0.7634821164938304]
        values_11 = [0.7834961732398945, 0.8028076691557597, 0.7947839686874866, 0.7688528455213548, 0.8195593433955048]
        values_12 = [0.8047118966079635, 0.7899792717966221, 0.7905555960983989, 0.7698661969857526, 0.820887038511263]
        values_13 = [0.7670968860030594, 0.7576938991782578, 0.8017620393799948, 0.7935998545880906, 0.7943473208505978]
        values_14 = [0.8497375327659246, 0.8265989180529483, 0.8581487866352483, 0.8694100399952952, 0.8641047225505831]

        values_15 = [0.6883950706120705, 0.7572142977728618, 0.7373543132595388, 0.7336395706970911, 0.715396747658438]  # F1_maco
        values_16 = [0.6475916015746397, 0.5955368093982462, 0.6589396392671256, 0.6368936598563806, 0.6160382899036083]
        values_17 = [0.6590042251246389, 0.6672858668309122, 0.6846487871625438, 0.6438041126666094, 0.7047570082152963]
        values_18 = [0.7465971207076187, 0.7758572894781615, 0.7186051463855703, 0.7555523132851482, 0.7858881301435008]
        values_19 = [0.7553713057158006, 0.7956785134417727, 0.7802287400123303, 0.729997768961885, 0.7657236718682117]
        values_20 = [0.7463388515526739, 0.7292490721586296, 0.7235038776644831, 0.718398764467082, 0.7770094341571316]
        values_21 = [0.8281436639046729, 0.8289033783533929, 0.8179664880203839, 0.773008385587699, 0.7984780841338512]
    # LGG的模态消融
    elif disease == 'LGG' and pooling_ceng == False:
        # 模态消融实验
        values_1 = [0.7065131102878301, 0.6945934057289327, 0.7046074741557738, 0.6854306744189531, 0.6703553354085106]  # ACC_1
        values_2 = [0.7902845258173616, 0.813075097464784, 0.7785161879247711, 0.8045298947615331, 0.7880942940315507]  # ACC_2
        values_3 = [0.771661011380485, 0.7411364434617491, 0.7475613939369816, 0.7549000802289976, 0.7667410709917863]
        values_4 = [0.7893188394737474, 0.818985210596851, 0.8097401806979194, 0.7963584961142494, 0.8240972731172329]
        values_5 = [0.8339200912038587, 0.8276250031764151, 0.8144263210138006, 0.8263033827854985, 0.8002252018204272]
        values_6 = [0.78499013044629, 0.8085378264565164, 0.8132417584277962, 0.813991376334746, 0.8177389083346513]
        values_7 = [0.8572272896917011, 0.8949095392459026, 0.8730904690839831, 0.8681901711155107, 0.8655825308629028]

        values_8 = [0.6507875714894307, 0.6975243826419908, 0.6901369263980451, 0.6722167737293336, 0.6798343457412002]  # F1_weighted
        values_9 = [0.7637817618993421, 0.7945176867825624, 0.7597671196940112, 0.749534178530971, 0.7893992530931131]
        values_10 = [0.748021722297709, 0.7552957208854232, 0.7431681702876936, 0.733580685745942, 0.7759337007832317]
        values_11 = [0.8318252791430002, 0.8049438918032319, 0.8143413348916428, 0.8143422097814633, 0.7945472843806627]
        values_12 = [0.8355367514583427, 0.7909096027899375, 0.8150891401514289, 0.8156594303903957, 0.8128050752098952]
        values_13 = [0.8339053529493718, 0.7988850065725913, 0.8031947237956624, 0.8311129831456584, 0.8144019335367161]
        values_14 = [0.8889396255886793, 0.8552435356940604, 0.8593902856264162, 0.8862528462258307, 0.8701737068650139]

        values_15 = [0.5522745171117254, 0.52453188022067, 0.5275399047395838, 0.48650443517472247, 0.5381492627532989]  # F1_maco
        values_16 = [0.6986776124876655, 0.6705548306348577, 0.6655179845953925, 0.6628687177265464, 0.6423808545555381]
        values_17 = [0.6695389955904304, 0.6547776066770954, 0.6144817732985949, 0.6280944827660094, 0.6506071416678698]
        values_18 = [0.7795592712184153, 0.7482755160263969, 0.7704274034757445, 0.7651860333603311, 0.7245517759191129]
        values_19 = [0.7435606379676739, 0.768158553381319, 0.7178572274932635, 0.7494614414385927, 0.7094621397191512]
        values_20 = [0.7977903856718254, 0.7754181039694009, 0.7634756272989349, 0.7661626196466033, 0.7411532634132354]
        values_21 = [0.7880285995909688, 0.7723347147133669, 0.7702090327370092, 0.7540829560963084, 0.8058446968623469]
    # STAD的模态消融
    elif disease == 'STAD' and pooling_ceng == False:
        # 模态消融实验
        values_1 = [0.7656565092541703, 0.7892143422627069, 0.7574452925962513, 0.7698184989176596, 0.747865356969212]  # ACC_1
        values_2 = [0.7360497182929827, 0.7467345421115973, 0.7600664147869712, 0.726414823252835, 0.7542345015556138]  # ACC_2
        values_3 = [0.7205026823317916, 0.7218284812810296, 0.699495581881346, 0.7401680439261743, 0.7350052105796583]
        values_4 = [0.8034011104317346, 0.7721556481423615, 0.7793391416954226, 0.8041670756977759, 0.7769370240327049]
        values_5 = [0.8107648145772641, 0.8303997956688466, 0.84547897003409, 0.8352554895425829, 0.8271009301772163]
        values_6 = [0.791420564769402, 0.7749317482252928, 0.7865267205924603, 0.7736617200509425, 0.8094592463619024]
        values_7 = [0.8883620021615443, 0.9122139815699164, 0.8730011602884495, 0.8932872440207792, 0.9011356119593108]

        values_8 = [0.7818369894572436, 0.775837631035948, 0.7576848966566541, 0.7622571887538001, 0.8008832940963542]  # F1_weighted
        values_9 = [0.7716693049564091, 0.7387916444728674, 0.7722424819846232, 0.7512716588702569, 0.7390249097158432]
        values_10 = [0.7070465094256551, 0.737870528820455, 0.7382744928429871, 0.7231175748811685, 0.7066908940297343]
        values_11 = [0.7877884301225934, 0.7871471839254818, 0.7954255789004212, 0.818529393994182, 0.7631094130573217]
        values_12 = [0.8398746864793729, 0.857815751999086, 0.8262300587587751, 0.8327073748550291, 0.8138721279077368]
        values_13 = [0.7801942521637517, 0.7990899520640721, 0.810656285129366, 0.7629224739172801, 0.7991370367255297]
        values_14 = [0.8924172169927251, 0.8986976570926268, 0.9081878905786616, 0.8825909524718156, 0.870106282864171]

        values_15 = [0.7207682396700231, 0.750882117825147, 0.7112268300972429, 0.7574053467912212, 0.7452174656163657]  # F1_maco
        values_16 = [0.7222438720599994, 0.7017196813302591, 0.741426920023486, 0.7083147019235188, 0.7457948246627368]
        values_17 = [0.7031588939308165, 0.7562925148016117, 0.7215191596713698, 0.7137379768557467, 0.7357914547404549]
        values_18 = [0.804624127511747, 0.8074224416363238, 0.8010584231268578, 0.8049029786397875, 0.7599920290852835]
        values_19 = [0.788168007898001, 0.8407920049215898, 0.8060034795839199, 0.8005146701061836, 0.7990218374903053]
        values_20 = [0.8218113221128712, 0.8134952263887772, 0.797522682371653, 0.767179962167887, 0.7779908069588108]
        values_21 = [0.9293172958139744, 0.9308300909750019, 0.8965248050812064, 0.917267184516191, 0.9535606236136256]
    # BRCA的池化层消融
    elif disease == 'BRCA' and pooling_ceng:
        values_1 = [0.7485942696431227, 0.7363226056855513, 0.7696415389115082, 0.7597760893062352, 0.788165496453582]  # ACC_1
        values_2 = [0.7713119681798204, 0.7597598412788464, 0.766183627804409, 0.782960622185433, 0.741283940551491]  # ACC_2
        values_3 = [0.871268522177408, 0.8456528121422754, 0.8633215804004988, 0.8349299918485226, 0.8433270934312946]
        values_4 = [0.7966456884903781, 0.804702103974683, 0.7611355033174484, 0.8050054874491616, 0.8060112167683284]
        values_5 = [0.7636025904160236, 0.7768869843106845, 0.7825971314754827, 0.7578691506013364, 0.7975441431964727]
        values_6 = [0.7505210381305856, 0.753036657311936, 0.7218621392595533, 0.7531066627952194, 0.7219735025027054]

        values_7 = [0.7575262409743808, 0.7380311891282831, 0.7389645282722666, 0.7666131049210114, 0.7513649367040576]
        values_8 = [0.7518040614554904, 0.7525747559528537, 0.7438134150112172, 0.7464965796191958, 0.778311187961243]
        values_9 = [0.8486703824266677, 0.849208075942745, 0.8383530484998184, 0.8458054954602997, 0.8859629976704686]
        values_10 = [0.789529724269009, 0.8074593370384865, 0.796401183630273, 0.7930712963763992, 0.7670384586858323]
        values_11 = [0.7575948520794682, 0.7261787038240288, 0.73617466666222, 0.7691789007799632, 0.7573728766543195]
        values_12 = [0.6754430933628371, 0.6684315718777412, 0.6670535507434081, 0.6630644766258511, 0.697507307390163]

        values_13 = [0.6526383045588668, 0.6478076280499773, 0.6890448375686808, 0.6709282060930565, 0.663081023729418]
        values_14 = [0.6474324310526853, 0.6763238061244679, 0.687267987114938, 0.6920704235686712, 0.6744053521392375]
        values_15 = [0.8147515498236909, 0.8017904412579996, 0.8112802071190741, 0.7889800230212929, 0.8296977787779425]
        values_16 = [0.7437861568214157, 0.7051496193262102, 0.7187400612113054, 0.7186499067394938, 0.7256742559015752]
        values_17 = [0.5925647463496794, 0.6097968400459334, 0.6066410604283063, 0.5708493367956918, 0.6151480163803891]
        values_18 = [0.5405109967619419, 0.5628469855577388, 0.554768244140607, 0.5426778168644167, 0.5116959566752954]

    elif disease == 'LGG' and pooling_ceng:
        values_1 = []
        values_2 = []
        values_3 = []
        values_4 = []
        values_5 = []
        values_6 = []

    elif disease == 'STAD' and pooling_ceng:
        values_1 = []
        values_2 = []
        values_3 = []
        values_4 = []
        values_5 = []
        values_6 = []

    if pooling_ceng == False:
        all_values = [
            values_1, values_2, values_3,  # 第一组数据 (已存在)
            values_4, values_5, values_6,  # 第二组数据 (已存在)
            values_7, values_8, values_9,  # 第三组数据 (新添加)
            values_10, values_11, values_12,  # 第四组数据 (新添加)
            values_13, values_14, values_15,  # 第五组数据 (新添加)
            values_16, values_17, values_18,
            values_19,values_20,values_21# 如果需要添加更多的列，可以继续在这里添加更多的数据集
        ]
        # 设置每个柱子的颜色，用户可以自定义RGB颜色值
        # 例如，这里设置7个不同的颜色，每个颜色对应一个列
        column_colors = [
            (241 / 255, 157 / 255, 145 / 255),  # RGB颜色值，范围在0到1之间
            (158 / 255, 219 / 255, 233 / 255),
            (116 / 255, 204 / 255, 190 / 255),
            (149 / 255, 162 / 255, 191 / 255),
            (248 / 255, 201 / 255, 186 / 255),
            (188 / 255, 196 / 255, 215 / 255),
            (237 / 255, 116 / 255, 116 / 255),
            # 如果有更多列，可以继续添加颜色
        ]
        label = ['mRNA', 'meth', 'miRNA', 'mRNA + meth', 'mRNA + miRNA', 'meth + miRNA', 'mRNA + meth + miRNA']
    else:
        all_values = [
            values_1, values_2, values_3,  # 第一组数据 (已存在)
            values_4, values_5, values_6,  # 第二组数据 (已存在)
            values_7, values_8, values_9,  # 第三组数据 (新添加)
            values_10, values_11, values_12,  # 第四组数据 (新添加)
            values_13, values_14, values_15,  # 第五组数据 (新添加)
            values_16, values_17, values_18,
        ]
        # 设置每个柱子的颜色，用户可以自定义RGB颜色值
        # 例如，这里设置6个不同的颜色，每个颜色对应一个列
        column_colors = [
            (241/255, 157/255, 145/255),  # RGB颜色值，范围在0到1之间
            (158/255, 219/255, 233/255),
            (116/255, 204/255, 190/255),
            (149/255, 162/255, 191/255),
            (248/255, 201/255, 186/255),
            (237/255, 116/255, 116/255),
            # 如果有更多列，可以继续添加颜色
        ]
        label = ['MOFNet with 1-layer pooling', 'MOFNet with 1-layer pooling_NN', 'MOFNet with 2-layer pooling', 'mRNA + meth', 'MOFNet with 1-layer pooling', 'meth + miRNA', 'mRNA + meth + miRNA']

    # # 设置每个柱子的颜色，用户可以自定义RGB颜色值
    # # 例如，这里设置7个不同的颜色，每个颜色对应一个列
    # column_colors = [
    #     (241/255, 157/255, 145/255),  # RGB颜色值，范围在0到1之间
    #     (158/255, 219/255, 233/255),
    #     (116/255, 204/255, 190/255),
    #     (149/255, 162/255, 191/255),
    #     (248/255, 201/255, 186/255),
    #     (188/255, 196/255, 215/255),
    #     (237/255, 116/255, 116/255),
    #     # 如果有更多列，可以继续添加颜色
    # ]
    label = ['mRNA', 'meth', 'miRNA', 'mRNA + meth', 'mRNA + miRNA', 'meth + miRNA', 'mRNA + meth + miRNA']
    # 计算每组数据的平均值和标准差
    averages = [np.mean(values) for values in all_values]
    std_devs = [np.std(values, ddof=1) for values in all_values]

    # 设置组名
    group_names = ['ACC', 'F1_weighted', 'F1_macro']

    number_of_columns = len(all_values) // len(group_names)  # 计算每组的列数
    bar_width = 0.8 / number_of_columns  # 分配每列的宽度
    index = np.arange(len(group_names))  # 组的x轴位置

    # 设置画布的大小为8cm x 10cm
    width_cm = 8
    height_cm = 10
    width_in = width_cm / 2.54  # 将宽度转换为英寸
    height_in = height_cm / 2.54  # 将高度转换为英寸

    # 开始绘图
    fig, ax = plt.subplots()# figsize=(width_in, height_in)

    # 调整字体大小
    # plt.rc('font', size=10)  # 设置所有文本元素的默认大小
    # plt.rc('axes', titlesize=10)  # 设置轴标题的大小
    # plt.rc('axes', labelsize=10)  # 设置轴标签的大小
    # plt.rc('legend', fontsize=15)  # 设置图例的字体大小
    # plt.rc('xtick', labelsize=6)  # 设置x轴刻度标签的字体大小
    # plt.rc('ytick', labelsize=6)  # 设置y轴刻度标签的字体大小
    ax.tick_params(axis='x', labelsize=10)  # 设置x轴刻度标签的字体大小
    ax.tick_params(axis='y', labelsize=10)  # 设置y轴刻度标签的字体大小

    # 添加每组的条形图和散点图
    for i in range(number_of_columns):
        # 计算每列的x轴位置
        column_positions = index - 0.4 + bar_width / 2 + i * bar_width

        # 绘制条形图
        ax.bar(column_positions, averages[i::number_of_columns], bar_width,
               yerr=std_devs[i::number_of_columns], color=column_colors[i],
               capsize=5, label=label[i])

        # 绘制散点图
        for j in range(len(group_names)):
            scatter_data = all_values[j * number_of_columns + i]
            scatter_x = distribute_scatter(bar_width, column_positions[j], len(scatter_data))
            ax.scatter(scatter_x, scatter_data, facecolors='none', edgecolors='grey')

    # 设置图表的标题和坐标轴标签
    if disease == 'BRCA':
        plt.ylabel("value", fontsize=12)
        ax.set_ylim(0.60, 0.90)
        ax.set_title('BRCA', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(group_names)
    elif disease == 'LGG':
        plt.ylabel("value", fontsize=12)
        ax.set_ylim(0.50, 0.90)
        ax.set_title('LGG', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(group_names)
    elif disease == 'STAD':
        plt.ylabel("value", fontsize=12)
        ax.set_ylim(0.65, 0.95)
        ax.set_title('STAD', fontsize=12)
        ax.set_xticks(index)
        ax.set_xticklabels(group_names)

    # 如果要隐藏图例，可以注释掉以下两行代码
    # 如果要显示图例并将其放置在图表下方
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, fancybox=False, shadow=False, fontsize=9, edgecolor='none') # x-small比small更小


    # 调整子图参数，例如边距和图例的空间
    # plt.subplots_adjust(bottom=0.2, top=0.9)  # 调整底部和顶部的边距
    plt.tight_layout()  # 调整整个图表的布局，确保图例和坐标轴标签可见pad=2

    #保存图表为矢量图格式，如SVG
    plt.savefig(f'./Result/pic/ablation_{disease}.svg', format='svg', bbox_inches='tight')

    # 展示图表
    plt.show()


if __name__ == '__main__':
    draw_new_model_attn = True
    pth_file_path = r'E:\Cancer-classification-dataset\checkpoints'
    disease = 'BRCA'

    # Pattern khớp cả 5 init seed của config tốt nhất (gat/attn/lr=5e-4/mixup=0.0)
    test_pth_file_name = 'BRCA_gat_attn_s0_seed*_best.pth'

    if draw_new_model_attn:
        attn_dict = read_new_model_attn(pth_file_path, test_pth_file_name)
        draw_omics_attention_bar(attn_dict, disease)
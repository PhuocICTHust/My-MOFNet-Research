
import os
import subprocess  # FIX Bug 7: was missing — used on line 21 but never imported
import pickle
import statistics
import json
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
import io


def load_KEGG(kegg_file='/home/zcx/project/MOFNet/KEGG_all_pathway.pkl'):
    '''
        load kegg pathway
    '''
    if os.path.exists(kegg_file):
        # 如果 pkl 文件存在，则加载它
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    else:
        # 如果 pkl 文件不存在，则运行 KEGG.py 文件
        # FIX Bug 7: subprocess now properly imported above
        subprocess.call(['python', '/home/wcy/Diffusion/pathway/kegg/KEGG_process.py'])
        # 加载生成的 pkl 文件
        with open(kegg_file, 'rb') as f:
            KEGG = pickle.load(f)
    return KEGG


# 定义一个函数来添加或更新键值对
def add_or_append_to_dict(dictionary, key, value):
    if key in dictionary:
        # 如果键已存在，则在现有值后面添加
        dictionary[key].append(value)
    else:
        # 如果键不存在，则添加新的键值对
        dictionary[key] = [value]


if __name__ == '__main__':
    disease = 'STAD'
    omics = 3
    only_one_disease = False
    all_diseases = True
    jiaozheng = False
    out_jiaozheng_result = False
    KEGG = load_KEGG(kegg_file='/home/zcx/project/MOFNet/KEGG_all_pathway.pkl')  # 加载 KEGG 数据库
    if disease == 'BRCA':
        cancer_pathway = KEGG['hsa05224']  # BRCA 的 KEGG 通路（gene symbol格式）
        raw_data_path = f'/home/zcx/project/MOFNet/KEGG富集分析/BRCA_features_{omics}.csv'
        aft_train_rank = f'/home/zcx/project/MOFNet/特征重要性/BRCA/array_b_{omics}.csv'
    elif disease == 'STAD':
        cancer_pathway = KEGG['hsa05226']  # STAD 的 KEGG 通路（gene symbol格式）
        # 把数据集中的mRNA的基因名转换成KEGG数据库中的基因名，就是ENSG格式转换成gene symbol格式https://biodbnet-abcc.ncifcrf.gov/db/db2db.php,然后再网页中选输出到xls文件
        if omics == 3:
            raw_data_path = f'/home/zcx/project/MOFNet/KEGG富集分析/STAD_mRNA_gene_symbol_name.xls'
        else:
            raw_data_path = f'/home/zcx/project/MOFNet/KEGG富集分析/STAD_features_{omics}.csv'
        aft_train_rank = f'/home/zcx/project/MOFNet/特征重要性/STAD/array_b_{omics}.csv'
    elif disease == 'LGG':
        cancer_pathway = KEGG['hsa05214']  # LGG 的 KEGG 通路（gene symbol格式）
        raw_data_path = f'/home/zcx/project/MOFNet/KEGG富集分析/LGG_features_{omics}.csv'  # LGG的mRNA数据集格式直接就是gene symbol格式的
        aft_train_rank = f'/home/zcx/project/MOFNet/特征重要性/LGG/array_b_{omics}.csv'


    # 数据集第一行为表头，第一列为ENSG格式的基因名，第二列为gene symbol格式的基因名
    overlap_cancer_pathway = []
    overlap_all_pathway = {}
    no_gene_symbol_name = 0
    if only_one_disease:
        with open(raw_data_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:-1]:
                line = line.strip().split('\t')
                gene_symbol_name = line[1]
                if gene_symbol_name == '-':
                    no_gene_symbol_name += 1
                # 先判断所有的mRNA（1000个）是否在KEGG数据库中，如果在，就把它的通路存到一个列表中(overlap_cancer_pathway)
                if gene_symbol_name in cancer_pathway:
                    overlap_cancer_pathway.append(gene_symbol_name)
                    # 输出overlap_cancer_pathway的长度，即为数据集中的mRNA也在KEGG通路中的数量，以及占lines长度的比例
            print(len(overlap_cancer_pathway), len(overlap_cancer_pathway) / len(lines[1:]))
            print(1)
    if all_diseases:
        if disease == 'STAD' and omics == 3:  # 因为STAD的mRNA数据是从TCGA下载的，所以第一列是ENSG格式的基因名，第二列是gene symbol格式的基因名，需要转置
            with open(raw_data_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:-1]:  # 第一行为表头，最后一行为空行
                    line = line.strip().split('\t')
                    gene_symbol_name = line[1]
                    if gene_symbol_name == '-':
                        no_gene_symbol_name += 1
                    # 先判断所有的mRNA（1000个）是否在KEGG数据库中，如果在，就把它的通路存到一个列表中(overlap_cancer_pathway)
                    for key, value in KEGG.items():
                        if gene_symbol_name in KEGG[key]:
                            overlap_cancer_pathway.append(gene_symbol_name)
                            add_or_append_to_dict(overlap_all_pathway, key, gene_symbol_name)
        else:#if disease == 'LGG' or disease == 'BRCA':
            with open(raw_data_path, 'r') as f:
                lines = f.readlines()
                for line in lines[0:-1]:  # 最后一行是空行
                    gene_symbol_name = line.strip().split('.')[0]
                    if gene_symbol_name == '-':
                        no_gene_symbol_name += 1
                    # 先判断所有的mRNA（1000个）是否在KEGG数据库中，如果在，就把它的通路存到一个列表中(overlap_cancer_pathway)
                    for key, value in KEGG.items():
                        if gene_symbol_name in KEGG[key]:
                            overlap_cancer_pathway.append(gene_symbol_name)
                            add_or_append_to_dict(overlap_all_pathway, key, gene_symbol_name)

        # 输出overlap_cancer_pathway的长度，即为数据集中的mRNA也在KEGG通路中的数量，以及占lines长度的比例
        print(len(overlap_cancer_pathway), len(overlap_cancer_pathway) / len(lines[1:]))
        lengths = [len(v) for v in overlap_all_pathway.values()]
        # print(f'池化前：最大长度：{max(lengths)},最小长度：{min(lengths)},平均长度：{sum(lengths) / len(lengths)},中位数长度：{statistics.median(lengths)}, 长度大于10的通路数量：{len([i for i in lengths if i > 10])}')

        # 按照每个键对应的值的长度降序排列字典
        sorted_overlap_all_pathway = dict(
            sorted(overlap_all_pathway.items(), key=lambda item: len(item[1]), reverse=True))

        # 将字典写入JSON文件
        with open(f'./KEGG富集分析/{disease}_{omics}_sorted_overlap_all_pathway.json', 'w') as file:
            json.dump(sorted_overlap_all_pathway, file, indent=4)
        print(2)

        data = pd.read_csv(aft_train_rank, header=None)

        # 根据第3列(索引为2)的值进行降序排序，并获取前250个行索引  TODO 这里其实写错了，250不一定，比如BRCA的microRNA就只有503个，两次池化后应该是503/2/2=125个，所以这里应该是取前125个,但是因为画网络图的时候连250个都找不到一条边，不改就不改了
        top_250_indices = data.iloc[:, 2].sort_values(ascending=False).head(250).index

        # 行索引是从0开始的，调整为1开始的索引
        top_250_indices_adjusted = top_250_indices + 1

        top_250_indices_adjusted.tolist()
        print(3)

        count_no_gene_symbol = 0
        aft_pooling_valid_gene_symbols = []
        aft_pooling_overlap_cancer_pathway = []
        aft_pooling_overlap_all_pathway = {}
        with open(raw_data_path, 'r') as f:
            lines = f.readlines()
            # 提取对应行的内容,selected_lines就是挑选出来的前250个
            # 因为lines的第一行是表头，所以index是+1也就是从1开始的
            selected_lines = [lines[index].strip().split('.')[0] for index in top_250_indices]

        # 将列表转换为DataFrame(做富集分析用的）
        df = pd.DataFrame([line.strip() for line in selected_lines], columns=['Gene'])
        # 移除DataFrame中'-'的行(做富集分析用的）
        df_clean = df[df['Gene'] != '-']
        # 输出到CSV文件(做富集分析用的）STAD_3_aft_pooling_delete__.csv
        csv_file_path = f'./KEGG富集分析/{disease}_{omics}_aft_pooling_delete__.csv'
        df_clean.to_csv(csv_file_path, index=False)

        # 遍历选定的行
        for line in selected_lines:
            # 分割每行并获取基因符号（位于第二列）
            gene_symbol = line.strip()
            if gene_symbol == '-':
                count_no_gene_symbol += 1
            else:
                aft_pooling_valid_gene_symbols.append(gene_symbol)  # 这个是按得分降序排列的
        print(4)

        for valid_gene_symbol in aft_pooling_valid_gene_symbols:
            for key, value in KEGG.items():
                if valid_gene_symbol in KEGG[key]:
                    aft_pooling_overlap_cancer_pathway.append(valid_gene_symbol)
                    # aft_pooling_overlap_all_pathway就是经过池化后剩下的基因，它们在KEGG数据库中的通路
                    add_or_append_to_dict(aft_pooling_overlap_all_pathway, key, valid_gene_symbol)
        lengths = [len(v) for v in aft_pooling_overlap_all_pathway.values()]
        # print(f'池化后：最大长度：{max(lengths)},最小长度：{min(lengths)},平均长度：{sum(lengths) / len(lengths)},中位数长度：{statistics.median(lengths)}, 长度大于10的通路数量：{len([i for i in lengths if i > 10])}')
        # 按照每个键对应的值的长度降序排列字典
        aft_pooling_sorted_overlap_all_pathway = dict(
            sorted(aft_pooling_overlap_all_pathway.items(), key=lambda item: len(item[1]), reverse=True))

        # 将字典写入JSON文件
        with open(f'./KEGG富集分析/{disease}_{omics}_aft_pooling_sorted_overlap_all_pathway.json', 'w') as file:
            json.dump(aft_pooling_sorted_overlap_all_pathway, file, indent=4)  # indent=4表示缩进4个空格
        print(5)
    if jiaozheng:


        # Assuming the P-values are stored in a text file, we will read the file and load the values into a DataFrame
        # p_values = [  # 1000个基因作为background的时候的p值
        #     0.214613443, 0.219381588, 0.302766831, 0.309504104, 0.309504104, 0.467779029, 0.478717344,
        # 0.479525862, 0.479525862, 0.479525862, 0.479525862, 0.479525862, 0.51111864, 0.578380104,
        # 0.594245278, 0.624780012, 0.624780012, 0.625193152, 0.625193152, 0.625193152, 0.631754914,
        # 0.695816936, 0.730422835, 0.730422835, 0.730422835, 0.730422835, 0.730422835, 0.730422835,
        # 0.730422835, 0.730422835, 0.730422835, 0.731287713, 0.766158453, 0.806348049, 0.806348049,
        # 0.806348049, 0.817916338, 0.817916338, 0.817916338, 0.818793623, 0.837906339, 0.857517743,
        # 0.859285543, 0.859285543, 0.86106241, 0.86106241, 0.86106241, 0.86106241, 0.86106241,
        # 0.86106241, 0.87066521, 0.891979264, 0.897512707, 0.90044281, 0.90044281, 0.90044281,
        # 0.90044281, 0.90044281, 0.90044281, 0.90044281, 0.921300801, 0.928751404, 0.928751404,
        # 0.928751404, 0.940268229, 0.949075523, 0.949075523, 0.949075523, 0.949075523, 0.949075523,
        # 0.963648765, 0.963648765, 0.974085088, 0.974085088, 0.981549253, 0.981549253, 0.9868808,
        # ]
        if disease == 'LGG':
            a = pd.read_excel('/home/zcx/project/MOFNet/KEGG富集分析/LGG_师兄帮选_全人类背景_metascape_result.xlsx', sheet_name='Enrichment', header=0)
        elif disease == 'BRCA':
            # a = pd.read_excel('/home/zcx/project/MOFNet/KEGG富集分析/BRCA_全人类背景_metascape_result.xlsx', sheet_name='Enrichment', header=0)
            a = pd.read_excel('/home/zcx/project/MOFNet/KEGG富集分析/BRCA_全人类背景_改array_b为1（改正错误）.xlsx',
                              sheet_name='Enrichment', header=0)
        # 取出第10列的值，即p值
        p_values = a.iloc[:, 10].tolist()

        # p_values = [  # 所有人类基因作为background的时候的p值
        #     0.02325984, 0.041898866, 0.049146916, 0.05539562, 0.083382001, 0.119267934, 0.137533977,
        # 0.180454959, 0.198085115, 0.201417165, 0.221751671, 0.233162814, 0.244110935, 0.24676604,
        # 0.248433039, 0.271397012, 0.288654167, 0.305586345, 0.311571242, 0.318689058, 0.348071489,
        # 0.365983549, 0.385983631, 0.404858785, 0.435011674, 0.44555439, 0.463727041, 0.469287187,
        # 0.473080668, 0.480237035, 0.516831324, 0.526810636, 0.531723591, 0.5555419, 0.591183122,
        # 0.603807026, 0.604086303, 0.620041147, 0.623995812, 0.627994384, 0.635617241, 0.641154364,
        # 0.643166813, 0.64688339, 0.64688339, 0.654202097, 0.654202097, 0.657805002, 0.657805002,
        # 0.675270209, 0.675270209, 0.68787678, 0.691853521, 0.698249857, 0.703806173, 0.721505035,
        # 0.752828812, 0.760503843, 0.765489126, 0.784424636, 0.786682634, 0.795483129, 0.799747773,
        # 0.801847002, 0.803924469, 0.805980397, 0.827235659, 0.830844955, 0.83784135, 0.865945795,
        # 0.867357393, 0.879418162, 0.891550251, 0.905533669, 0.925250493, 0.939756194, 0.993702917,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        # 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        # ]

        # Perform Bonferroni correction
        bonferroni_correction = multipletests(p_values, method='bonferroni')

        # Perform Benjamini-Hochberg correction
        benjamini_correction = multipletests(p_values, method='fdr_bh')

        # Holm-Bonferroni correction
        holm_correction = multipletests(p_values, method='holm')

        # Hochberg correction
        # hochberg_correction = multipletests(p_values, method='hochberg')

        # Benjamini-Yekutieli correction
        benjamini_yekutieli_correction = multipletests(p_values, method='fdr_by')

        sidak_correction = multipletests(p_values, method='sidak')
        holm_sidak_correction = multipletests(p_values, method='holm-sidak')
        simes_hochberg_correction = multipletests(p_values, method='simes-hochberg')
        hommel_correction = multipletests(p_values, method='hommel')
        fdr_bh_correction = multipletests(p_values, method='fdr_bh')
        fdr_by_correction = multipletests(p_values, method='fdr_by')
        fdr_tsbh_correction = multipletests(p_values, method='fdr_tsbh')
        fdr_tsbky_correction = multipletests(p_values, method='fdr_tsbky')


        # Combine the results in a DataFrame
        corrections = pd.DataFrame({
            'P-Value': p_values,
            'Bonferroni Adjusted': bonferroni_correction[1],
            'Benjamini-Hochberg Adjusted': benjamini_correction[1],
            'holm Adjusted': holm_correction[1],
            # 'hochberg Adjusted': hochberg_correction[1],
            'Benjamini-Yekutieli Adjusted': benjamini_yekutieli_correction[1],
            'sidak Adjusted': sidak_correction[1],
            'holm-sidak Adjusted': holm_sidak_correction[1],
            'simes-hochberg Adjusted': simes_hochberg_correction[1],
            'hommel Adjusted': hommel_correction[1],
            'fdr_bh Adjusted': fdr_bh_correction[1],
            'fdr_by Adjusted': fdr_by_correction[1],
            'fdr_tsbh Adjusted': fdr_tsbh_correction[1],
            'fdr_tsbky Adjusted': fdr_tsbky_correction[1],
        })

        # Let's see the first few rows of the DataFrame
        print(corrections)
        if out_jiaozheng_result:
            # 输出corrections到csv文件
            csv_file_path = f'/home/zcx/project/MOFNet/KEGG富集分析/{disease}_jiaozheng_result.csv'
            corrections.to_csv(csv_file_path, index=False)
        print(6)
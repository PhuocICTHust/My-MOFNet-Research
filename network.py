import pandas as pd
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(df):
    """
    计算DataFrame中每一行之间的余弦相似度。

    参数:
    df : pandas.DataFrame
        输入的DataFrame，每一行代表一个向量。

    返回:
    pandas.DataFrame
        包含余弦相似度的DataFrame，其中元素A_ij代表第i行和第j行之间的相似度。
    """
    # 使用sklearn的cosine_similarity计算余弦相似度
    similarity_matrix = cosine_similarity(df)

    # 将相似度矩阵转换成DataFrame格式，并设置行和列的名字与输入DataFrame相同
    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)

    # 将对角线元素设置为0，因为它们代表向量与自己的相似度
    np.fill_diagonal(similarity_df.values, 0)

    return similarity_df


if __name__ == '__main__':
    all_pathway = False
    out_top_pathwey_json = False
    specific_pathway = True
    if specific_pathway:
        specific_pathway_name = 'GO:1903047'
        rna_rna = False
        rna = False
        rna_dna_methylation = False
        rna_microrna = False
        rna_250rna = False
        node_correlation = True
        if node_correlation:
            pearson = False
            cosine = True

    disease = 'BRCA'
    background_data_path = '/home/zcx/project/MOFNet/网络图/2022.human.source'
    if disease == 'BRCA':  # 都是mRNA,只不过STAD和LGG是模态3为mRNA,而BRCA是模态1为mRNA
        disease_data_path = '/home/zcx/project/MOFNet/KEGG富集分析/BRCA_1_aft_pooling_delete__.csv'
        dna_methylation_data_path = '/home/zcx/project/MOFNet/KEGG富集分析/BRCA_2_aft_pooling_delete__.csv'
        microrna_data_path = '/home/zcx/project/MOFNet/KEGG富集分析/BRCA_3_aft_pooling_delete__.csv'
    elif disease == 'LGG':
        disease_data_path = '/home/zcx/project/MOFNet/KEGG富集分析/LGG_3_aft_pooling_delete__.csv'
    elif disease == 'STAD':
        disease_data_path = '/home/zcx/project/MOFNet/KEGG富集分析/STAD_3_aft_pooling_delete__.csv'
    # 读取source类型的文件
    background_data = pd.read_csv(background_data_path, sep='\t', header=None)

    if all_pathway:
        # 读取disease类型的文件
        disease_data = pd.read_csv(disease_data_path, sep='\t', header=0)
        # 保存disease类型的文件中的数据
        disease_data = disease_data.iloc[:, 0].values

        # 遍历background_data中的每一行,如果该行的第1个元素在disease_data中,而且该行的第3个元素也在disease_data中,则保存该行的第1个元素和第3个元素到comfirmed_data中
        comfirmed_data = []
        for i in range(len(background_data)):
            if background_data.iloc[i, 0] in disease_data and background_data.iloc[i, 2] in disease_data:
                comfirmed_data.append([background_data.iloc[i, 0], background_data.iloc[i, 2]])
        comfirmed_data = pd.DataFrame(comfirmed_data)
        comfirmed_data.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_comfirmed_data.csv'.format(disease, specific_pathway_name), index=False, header=False)
        print('done')
    if out_top_pathwey_json:
        if disease == 'BRCA':
            matascape_data = pd.read_excel('/home/zcx/project/MOFNet/KEGG富集分析/BRCA_全人类背景_改array_b为1（改正错误）.xlsx', header=0, sheet_name='Enrichment')

            # 新建一个字典，用来存储pathway名称和对应的基因（挑出来的）
            pathway_gene_dict = {}
            # pathway名称在matascape_data的第3列，包含的基因在matascape_data的第8列
            for i in range(len(matascape_data)):
                pathway_name = matascape_data.iloc[i, 2]
                pathway_gene = matascape_data.iloc[i, 7]
                # 把pathway_gene从字符串转换成列表
                pathway_gene = pathway_gene.split(',')
                pathway_gene_dict[pathway_name] = pathway_gene

            # 输出pathway_gene_dict到json文件
            with open('/home/zcx/project/MOFNet/KEGG富集分析/BRCA_top_metascape_pathway_gene.json', 'w') as f:
                json.dump(pathway_gene_dict, f, indent=4)  # indent=4 更加美观显示，每个键值对占一行，不加则所有键值对在一行显示
    if specific_pathway:
        # 读取BRCA_top_metascape_pathway_gene.json文件，并取键=specific_pathway_name的值
        with open('/home/zcx/project/MOFNet/KEGG富集分析/BRCA_top_metascape_pathway_gene.json', 'r') as f:
            pathway_gene_dict = json.load(f)
        specific_pathway_gene = pathway_gene_dict[specific_pathway_name]
        if rna_rna:
            # 读取background_data中的第1列和第3列,如果第1列和第3列中的元素都在specific_pathway_gene中,则保存到specific_pathway_mrna_mrna_line中
            specific_pathway_mrna_mrna_line = []
            for i in range(len(background_data)):
                if background_data.iloc[i, 0] in specific_pathway_gene and background_data.iloc[i, 2] in specific_pathway_gene:
                    specific_pathway_mrna_mrna_line.append([background_data.iloc[i, 0], background_data.iloc[i, 2]])
            print(f'mrna_mrna共有{len(specific_pathway_mrna_mrna_line)}条线，分别是{specific_pathway_mrna_mrna_line}')
            # 把specific_pathway_mrna_mrna_line保存到csv文件中
            specific_pathway_mrna_mrna_line = pd.DataFrame(specific_pathway_mrna_mrna_line)
            specific_pathway_mrna_mrna_line.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_mrna_mrna_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
            specific_pathway_mrna_mrna_line = np.array(specific_pathway_mrna_mrna_line)
            print('done_mRNA_mRNA')
        if rna:
            # 读取BRCA_GO:1903047_specific_pathway_mrna_mrna_line.csv文件的内容作为specific_pathway_mrna_mrna_line
            specific_pathway_mrna_mrna_line = pd.read_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_mrna_mrna_line.csv'.format(disease, specific_pathway_name), sep=',', header=None)
            specific_pathway_mrna_mrna_line = np.array(specific_pathway_mrna_mrna_line)
            # 读取background_data中的第1列和第3列,如果第1列和第3列中的元素有任意一个在specific_pathway_gene中,则保存到specific_pathway_mrna_disease_line中，但是要去掉specific_pathway_mrna_mrna_line中的元素
            # 现在发现如果这么写会有问题，那么就分开两个保存，俩基因symbol只有前边的在specific_pathway_gene的保存在一个文件中，只有后边在specific_pathway_gene中的保存在另一个文件中
            specific_pathway_qian_mrna_disease_line = []
            specific_pathway_hou_mrna_disease_line = []
            specific_pathway_mrna_250mrna_line = []
            for i in range(len(background_data)):
                if background_data.iloc[i, 0] in specific_pathway_gene:
                    if [background_data.iloc[i, 0], background_data.iloc[i, 2]] not in specific_pathway_mrna_mrna_line:
                        specific_pathway_qian_mrna_disease_line.append([background_data.iloc[i, 0], background_data.iloc[i, 2]])
                        # 如果前node在通路中，后node还在250个mrna中，则挑出
                        if background_data.iloc[i, 2] in pd.read_csv(disease_data_path, sep='\t', header=0).iloc[:, 0].values:
                            specific_pathway_mrna_250mrna_line.append([background_data.iloc[i, 0], background_data.iloc[i, 2]])
                elif background_data.iloc[i, 2] in specific_pathway_gene:
                    if [background_data.iloc[i, 0], background_data.iloc[i, 2]] not in specific_pathway_mrna_mrna_line:
                        specific_pathway_hou_mrna_disease_line.append([background_data.iloc[i, 0], background_data.iloc[i, 2]])
                        if background_data.iloc[i, 0] in pd.read_csv(disease_data_path, sep='\t', header=0).iloc[:, 0].values:
                            specific_pathway_mrna_250mrna_line.append([background_data.iloc[i, 0], background_data.iloc[i, 2]])
            print(f'mrna_disease共有{len(specific_pathway_qian_mrna_disease_line) + len(specific_pathway_hou_mrna_disease_line)}条线，'
                  f'分别是{specific_pathway_qian_mrna_disease_line}和{specific_pathway_hou_mrna_disease_line}')
            print(f'mrna_250mrna共有{len(specific_pathway_mrna_250mrna_line)}条线，分别是{specific_pathway_mrna_250mrna_line}')
            # 把specific_pathway_qian/hou_mrna_disease_line保存到csv文件中
            specific_pathway_qian_mrna_disease_line = pd.DataFrame(specific_pathway_qian_mrna_disease_line)
            specific_pathway_hou_mrna_disease_line = pd.DataFrame(specific_pathway_hou_mrna_disease_line)
            specific_pathway_mrna_250mrna_line = pd.DataFrame(specific_pathway_mrna_250mrna_line)
            specific_pathway_qian_mrna_disease_line.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_qian_mrna_disease_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
            specific_pathway_hou_mrna_disease_line.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_hou_mrna_disease_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
            specific_pathway_mrna_250mrna_line.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_mrna_250mrna_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
            specific_pathway_qian_mrna_disease_line = np.array(specific_pathway_qian_mrna_disease_line)
            specific_pathway_hou_mrna_disease_line = np.array(specific_pathway_hou_mrna_disease_line)
            print('done_one_mRNA')
        if rna_dna_methylation:
            # 读取BRCA_GO:1903047_specific_pathway_mrna_disease_line.csv文件的内容作为specific_pathway_mrna_disease_line
            specific_pathway_qian_mrna_disease_line = pd.read_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_qian_mrna_disease_line.csv'.format(disease, specific_pathway_name), sep=',', header=None)
            specific_pathway_qian_mrna_disease_line = np.array(specific_pathway_qian_mrna_disease_line)
            specific_pathway_hou_mrna_disease_line = pd.read_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_hou_mrna_disease_line.csv'.format(disease, specific_pathway_name), sep=',', header=None)
            specific_pathway_hou_mrna_disease_line = np.array(specific_pathway_hou_mrna_disease_line)
            # 遍历
            # 在specific_pathway_mrna_disease_line的基础上，看两个元素中，不在background_data中的元素，是否在dna_methylation_data_path中，如果在，则保存到specific_pathway_mrna_dna_methylation_line中
            specific_pathway_mrna_dna_methylation_line = []
            # 分别遍历specific_pathway_qian_mrna_disease_line和specific_pathway_hou_mrna_disease_line
            for i in range(len(specific_pathway_qian_mrna_disease_line)):
                # 如果specific_pathway_mrna_disease_line的两个元素中，有1个不在background_data的第1列或第3列中，则判断该元素是否在dna_methylation_data_path中，如果在，则保存到specific_pathway_mrna_dna_methylation_line中
                if specific_pathway_qian_mrna_disease_line[i][1] in pd.read_csv(dna_methylation_data_path, sep='\t', header=0).iloc[:, 0].values:
                    specific_pathway_mrna_dna_methylation_line.append(specific_pathway_qian_mrna_disease_line[i])
            for i in range(len(specific_pathway_hou_mrna_disease_line)):
                # 如果specific_pathway_mrna_disease_line的两个元素中，有1个不在background_data的第1列或第3列中，则判断该元素是否在dna_methylation_data_path中，如果在，则保存到specific_pathway_mrna_dna_methylation_line中
                if specific_pathway_hou_mrna_disease_line[i][0] in pd.read_csv(dna_methylation_data_path, sep='\t', header=0).iloc[:, 0].values:
                    specific_pathway_mrna_dna_methylation_line.append(specific_pathway_hou_mrna_disease_line[i])
            print(f'mrna_dna_methylation共有{len(specific_pathway_mrna_dna_methylation_line)}条线，分别是{specific_pathway_mrna_dna_methylation_line}')
            # 把specific_pathway_mrna_dna_methylation_line保存到csv文件中
            specific_pathway_mrna_dna_methylation_line = pd.DataFrame(specific_pathway_mrna_dna_methylation_line)
            specific_pathway_mrna_dna_methylation_line.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_mrna_dna_methylation_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
            specific_pathway_mrna_dna_methylation_line = np.array(specific_pathway_mrna_dna_methylation_line)
            print('done_mRNA_DNA_methylation')
        if rna_microrna:
            # 读取BRCA_GO:1903047_specific_pathway_mrna_disease_line.csv文件的内容作为specific_pathway_mrna_disease_line
            specific_pathway_qian_mrna_disease_line = pd.read_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_qian_mrna_disease_line.csv'.format(disease, specific_pathway_name), sep=',', header=None)
            specific_pathway_qian_mrna_disease_line = np.array(specific_pathway_qian_mrna_disease_line)
            specific_pathway_hou_mrna_disease_line = pd.read_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_hou_mrna_disease_line.csv'.format(disease, specific_pathway_name), sep=',', header=None)
            specific_pathway_hou_mrna_disease_line = np.array(specific_pathway_hou_mrna_disease_line)
            # 在specific_pathway_mrna_disease_line的基础上，看两个元素中，不在background_data中的元素，是否在microrna_data_path中，如果在，则保存到specific_pathway_mrna_microrna_line中
            specific_pathway_mrna_microrna_line = []
            for i in range(len(specific_pathway_qian_mrna_disease_line)):
                # 如果specific_pathway_mrna_disease_line的两个元素中，有1个不在background_data的第1列或第3列中，则判断该元素是否在dna_methylation_data_path中，如果在，则保存到specific_pathway_mrna_dna_methylation_line中
                if specific_pathway_qian_mrna_disease_line[i][1] in pd.read_csv(microrna_data_path, sep='\t', header=0).iloc[:, 0].values:
                    specific_pathway_mrna_microrna_line.append(specific_pathway_qian_mrna_disease_line[i])
            for i in range(len(specific_pathway_hou_mrna_disease_line)):
                # 如果specific_pathway_mrna_disease_line的两个元素中，有1个不在background_data的第1列或第3列中，则判断该元素是否在dna_methylation_data_path中，如果在，则保存到specific_pathway_mrna_dna_methylation_line中
                if specific_pathway_hou_mrna_disease_line[i][0] in pd.read_csv(microrna_data_path, sep='\t', header=0).iloc[:, 0].values:
                    specific_pathway_mrna_microrna_line.append(specific_pathway_hou_mrna_disease_line[i])

            print(f'mrna_microrna共有{len(specific_pathway_mrna_microrna_line)}条线，分别是{specific_pathway_mrna_microrna_line}')
            # 把specific_pathway_mrna_microrna_line保存到csv文件中
            specific_pathway_mrna_microrna_line = pd.DataFrame(specific_pathway_mrna_microrna_line)
            specific_pathway_mrna_microrna_line.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_specific_pathway_mrna_microrna_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
            specific_pathway_mrna_microrna_line = np.array(specific_pathway_mrna_microrna_line)
            print('done_mRNA_microRNA')
        if node_correlation:
            if pearson:
                # 读取/home/zcx/project/MOFNet/网络图/计算相关性.xlsx文件,计算每一行之间的皮尔森相关系数
                node_correlation_data = pd.read_excel('/home/zcx/project/MOFNet/网络图/计算相关性.xlsx', header=None)
                # 转置 DataFrame，以便行变成列
                transposed_data = node_correlation_data.T
                # 添加行索引和列索引，以便计算相关系数矩阵
                transposed_data.columns = ['BIRC5', 'BBS4', 'BCL2', 'CDC20', 'CKS2', 'FOXM1', 'STMN1', 'MAD2L1', 'MCM3', 'MCM6', 'MYBL2', 'SKP2', 'AURKA', 'TTK', 'XPC', 'CCNB2', 'GINS1', 'SPRY2', 'DBF4', 'UBE2C', 'KIF4A', 'RACGAP1', 'GPSM2', 'GINS3', 'SKA1']
                # transposed_data.index = ['BIRC5', 'BBS4', 'BCL2', 'CDC20', 'CKS2', 'FOXM1', 'STMN1', 'MAD2L1', 'MCM3', 'MCM6', 'MYBL2', 'SKP2', 'AURKA', 'TTK', 'XPC', 'CCNB2', 'GINS1', 'SPRY2', 'DBF4', 'UBE2C', 'KIF4A', 'RACGAP1', 'GPSM2', 'GINS3', 'SKA1']
                # 计算相关系数矩阵
                corr_matrix = transposed_data.corr()
                # 添加行索引和列索引
                corr_matrix.columns = ['BIRC5', 'BBS4', 'BCL2', 'CDC20', 'CKS2', 'FOXM1', 'STMN1', 'MAD2L1', 'MCM3', 'MCM6', 'MYBL2', 'SKP2', 'AURKA', 'TTK', 'XPC', 'CCNB2', 'GINS1', 'SPRY2', 'DBF4', 'UBE2C', 'KIF4A', 'RACGAP1', 'GPSM2', 'GINS3', 'SKA1']
                corr_matrix.index = ['BIRC5', 'BBS4', 'BCL2', 'CDC20', 'CKS2', 'FOXM1', 'STMN1', 'MAD2L1', 'MCM3', 'MCM6', 'MYBL2', 'SKP2', 'AURKA', 'TTK', 'XPC', 'CCNB2', 'GINS1', 'SPRY2', 'DBF4', 'UBE2C', 'KIF4A', 'RACGAP1', 'GPSM2', 'GINS3', 'SKA1']
                # 如果两个相关系数得分＞0.7的，就取出对应的行索引和列索引保存在node_correlation_line中
                node_correlation_line = []
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        if corr_matrix.iloc[i, j] > 0.7:
                            node_correlation_line.append([corr_matrix.index[i], corr_matrix.columns[j]])

                # 输出node_correlation_line到csv文件中
                node_correlation_line = pd.DataFrame(node_correlation_line)
                node_correlation_line.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_node_correlation_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
                print("done_node_correlation")
            if cosine:
                node_cos_correlation_data = pd.read_excel('/home/zcx/project/MOFNet/网络图/计算相关性.xlsx', header=None)
                node_cos_correlation_data.index = ['BIRC5', 'BBS4', 'BCL2', 'CDC20', 'CKS2', 'FOXM1', 'STMN1', 'MAD2L1', 'MCM3', 'MCM6', 'MYBL2', 'SKP2', 'AURKA', 'TTK', 'XPC', 'CCNB2', 'GINS1', 'SPRY2', 'DBF4', 'UBE2C', 'KIF4A', 'RACGAP1', 'GPSM2', 'GINS3', 'SKA1']
                # 调用函数计算相似度
                similarity_node_cos_correlation_data = calculate_cosine_similarity(node_cos_correlation_data)
                # 计算最大的前30%的值的个数
                top_10_percent_count = int(np.ceil(0.3 * similarity_node_cos_correlation_data.size))  # ceil()向上取整, floor()向下取整
                # 将DataFrame的值转换为一维数组
                values_flat = similarity_node_cos_correlation_data.values.flatten()
                # 获取排序后的索引
                sorted_indices = np.argsort(values_flat)[::-1]  #argsort函数返回的是数组值从小到大的索引值, [::-1]表示逆序,即从大到小
                # 获取前30%的索引
                top_indices = sorted_indices[:top_10_percent_count]
                # 获取行和列名
                top_row_col_names = [(similarity_node_cos_correlation_data.index[i // similarity_node_cos_correlation_data.shape[1]],
                                      similarity_node_cos_correlation_data.columns[i % similarity_node_cos_correlation_data.shape[1]])
                                     for i in top_indices]
                top_row_col_names = pd.DataFrame(top_row_col_names)
                top_row_col_names.to_csv('/home/zcx/project/MOFNet/网络图/{}_{}_node_cos_correlation_line.csv'.format(disease, specific_pathway_name), index=False, header=False)
                print('done_node_cos_correlation')


















import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cancer", type=str, default="BRCA")
parser.add_argument("--data_path", type=str, default="E:\\Cancer-classification-dataset")
parser.add_argument("--num_classes", type=int, default=5)
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--device', type=str, default='cuda:1', help='specify cuda devices')# todo
parser.add_argument('--epochs', type=int, default=3000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=9999, help='patience for early stopping')
# parser.add_argument('--my_dataset', default=['BRCA_1','BRCA_2','BRCA_3'] , nargs='+')
# parser.add_argument('--my_dataset_test', default=['BRCA_test_1','BRCA_test_2','BRCA_test_3'] , nargs='+')
# parser.add_argument('--my_dataset', default=['LIHC_1','LIHC_2','LIHC_3'] , nargs='+')
# parser.add_argument('--my_dataset_test', default=['LIHC_test_1','LIHC_test_2','LIHC_test_3'] , nargs='+')
# parser.add_argument('--my_dataset', default=['LGG_1','LGG_2','LGG_3'] , nargs='+')
# parser.add_argument('--my_dataset_test', default=['LGG_test_1','LGG_test_2','LGG_test_3'] , nargs='+')
parser.add_argument('--my_dataset', default=['STAD_1','STAD_2','STAD_3'] , nargs='+')
parser.add_argument('--my_dataset_test', default=['STAD_test_1','STAD_test_2','STAD_test_3'] , nargs='+')
parser.add_argument('--num_epoch_pretrain', default=0) # todo
parser.add_argument('--num_features', default=1) # todo
parser.add_argument('--num_nodes', default=[1000,500,1000])#BRCA[1000, 1000, 503（5类）] zhangheng_LGG[1000,500,1000(4类)] ZHANGHENG_LIHC[1000, 600, 1000] ZHANGHENG_STAD[1000, 500, 1000（4类）]

parser.add_argument('--only_test', type=bool, default=True, help='train or test')
parser.add_argument('--continue_training', type=bool, default=False, help='continue training')
parser.add_argument('--continue_epoch', type=str, default='0', help='continue training')




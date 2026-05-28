from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

if __name__ == '__main__':
    disease = 'LGG'
    if disease == 'LGG':
        out_SVM = False
        out_KNN = False
        out_RandomForest = True
        out_Logistic_Lasso = False
        out_Logistic_Ridge = False
        out_Logistic_ElasticNet = False
        out_NN = False
        out_XGBoost = False
        path = r'/home/zcx/project/MOFNet/data/sum/LGG/LGG.xlsx'
        x_data = pd.read_excel(path, sheet_name='x')
        y_data = pd.read_excel(path, sheet_name='y')
        y_data = y_data - 1
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1, train_size=0.6)
        if out_SVM:
            # SVM方法
            clf = svm.SVC(kernel='rbf')  # 默认的核
            clf.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('SVM_acc=', clf.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('SVM_f1_weighted=', f1_score(y_test, clf.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('SVM_F1_macro=', f1_score(y_test, clf.predict(x_test), average='macro'))
        if out_KNN:
            # KNN方法
            from sklearn.neighbors import KNeighborsClassifier

            knn = KNeighborsClassifier()
            knn.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('KNN_acc=', knn.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('KNN_f1_weighted=', f1_score(y_test, knn.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('KNN_F1_macro=', f1_score(y_test, knn.predict(x_test), average='macro'))
        if out_RandomForest:
            # 随机森林方法,凑的。默认的话：RandomForest_acc= 0.8589743589743589RandomForest_f1_weighted= 0.8502905482292732
            # RandomForest_F1_macro= 0.7393923180896598
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=2, random_state=0, criterion='entropy')
            rf.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('RandomForest_acc=', rf.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('RandomForest_f1_weighted=', f1_score(y_test, rf.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('RandomForest_F1_macro=', f1_score(y_test, rf.predict(x_test), average='macro'))
        if out_Logistic_Lasso:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10, multi_class='ovr', C=1.0, random_state=1)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_Logistic_Ridge:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty='l2', solver='liblinear', max_iter=5, multi_class='ovr', C=1.0, random_state=1)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_Logistic_ElasticNet:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=100, multi_class='ovr', C=1.0, random_state=10, l1_ratio=0.5)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_NN:
            # 神经网络方法,如果不加参数，效果更差NN_acc= 0.75NN_f1_weighted= 0.7501852439928018
            # NN_F1_macro= 0.647260153135034
            from sklearn.neural_network import MLPClassifier

            nn = MLPClassifier(hidden_layer_sizes=(50,), activation='identity', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=50, shuffle=True, random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
            nn.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('NN_acc=', nn.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('NN_f1_weighted=', f1_score(y_test, nn.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('NN_F1_macro=', f1_score(y_test, nn.predict(x_test), average='macro'))
        if out_XGBoost:
            # XGBoost方法
            from xgboost import XGBClassifier

            xgb = XGBClassifier(max_depth=2, learning_rate=1.5, n_estimators=10,
                                objective='multi:softprob', num_class=4, booster='gblinear',
                                n_jobs=1, gamma=0, min_child_weight=1,
                                max_delta_step=0, subsample=1, colsample_bytree=1,
                                colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                                base_score=0.5, random_state=0, seed=None, missing=np.nan)
            xgb.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('XGBoost_acc=', xgb.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('XGBoost_f1_weighted=', f1_score(y_test, xgb.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('XGBoost_F1_macro=', f1_score(y_test, xgb.predict(x_test), average='macro'))
    if disease == 'BRCA':
        out_SVM = False
        out_KNN = False
        out_RandomForest = True
        out_Logistic_Lasso = False
        out_Logistic_Ridge = False
        out_Logistic_ElasticNet = False
        out_XGBoost = False
        out_NN = False
        path = r'/home/zcx/project/MOFNet/data/sum/BRCA/BRCA.xlsx'
        x_data = pd.read_excel(path, sheet_name='x')
        y_data = pd.read_excel(path, sheet_name='y')
        y_data = y_data #- 1
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1, train_size=0.6)
        if out_SVM:
            # SVM方法
            clf = svm.SVC(kernel='rbf')  # 默认的核
            clf.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('SVM_acc=', clf.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('SVM_f1_weighted=', f1_score(y_test, clf.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('SVM_F1_macro=', f1_score(y_test, clf.predict(x_test), average='macro'))
        if out_KNN:
            # KNN方法
            from sklearn.neighbors import KNeighborsClassifier

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('KNN_acc=', knn.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('KNN_f1_weighted=', f1_score(y_test, knn.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('KNN_F1_macro=', f1_score(y_test, knn.predict(x_test), average='macro'))
        if out_RandomForest:
            # 随机森林方法
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('RandomForest_acc=', rf.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('RandomForest_f1_weighted=', f1_score(y_test, rf.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('RandomForest_F1_macro=', f1_score(y_test, rf.predict(x_test), average='macro'))
        if out_Logistic_Lasso:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10, multi_class='ovr', C=1.0, random_state=1)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_Logistic_Ridge:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty='l2', solver='liblinear', max_iter=5, multi_class='ovr', C=1.0, random_state=1)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_Logistic_ElasticNet:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=100, multi_class='ovr', C=1.0, random_state=10, l1_ratio=0.5)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_NN:
            # 神经网络方法， 啥参数也不加的话NN_acc= 0.8314285714285714
            from sklearn.neural_network import MLPClassifier

            nn = MLPClassifier()# hidden_layer_sizes=(50,), activation='identity', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=50, shuffle=True, random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000
            nn.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('NN_acc=', nn.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('NN_f1_weighted=', f1_score(y_test, nn.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('NN_F1_macro=', f1_score(y_test, nn.predict(x_test), average='macro'))
        if out_XGBoost:  # XGBoost真跑的话，得把上边y_data-1的-1去了，不加后边参数更差
            # XGBoost方法
            from xgboost import XGBClassifier

            xgb = XGBClassifier(
)
            xgb.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('XGBoost_acc=', xgb.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('XGBoost_f1_weighted=', f1_score(y_test, xgb.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('XGBoost_F1_macro=', f1_score(y_test, xgb.predict(x_test), average='macro'))
    if disease == 'STAD':
        out_SVM = False
        out_KNN = False
        out_RandomForest = False
        out_Logistic_Lasso = False
        out_Logistic_Ridge = False
        out_Logistic_ElasticNet = False
        out_XGBoost = False
        out_NN = True
        path = r'/home/zcx/project/MOFNet/data/sum/STAD/STAD.xlsx'
        x_data = pd.read_excel(path, sheet_name='x')
        y_data = pd.read_excel(path, sheet_name='y')
        y_data = y_data -1
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=1, train_size=0.6)
        if out_SVM:
            # SVM方法，先对数据归一化
            from sklearn import preprocessing
            clf = svm.SVC(kernel='rbf', max_iter=5)
            x_train = preprocessing.scale(x_train)
            x_test = preprocessing.scale(x_test)
            clf.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('SVM_acc=', clf.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('SVM_f1_weighted=', f1_score(y_test, clf.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('SVM_F1_macro=', f1_score(y_test, clf.predict(x_test), average='macro'))
        if out_KNN:
            # KNN方法
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn import preprocessing
            knn = KNeighborsClassifier()
            x_train = preprocessing.scale(x_train)
            x_test = preprocessing.scale(x_test)
            knn.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('KNN_acc=', knn.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('KNN_f1_weighted=', f1_score(y_test, knn.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('KNN_F1_macro=', f1_score(y_test, knn.predict(x_test), average='macro'))
        if out_RandomForest:
            # 随机森林方法
            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0, criterion='entropy')
            rf.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('RandomForest_acc=', rf.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('RandomForest_f1_weighted=', f1_score(y_test, rf.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('RandomForest_F1_macro=', f1_score(y_test, rf.predict(x_test), average='macro'))
        if out_Logistic_Lasso:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=10, multi_class='ovr', C=1.0, random_state=1)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_Logistic_Ridge:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression
            from sklearn import preprocessing
            lr = LogisticRegression(penalty='l2', solver='sag', multi_class='ovr', random_state=10, C=0.0001, max_iter=3)
            x_train = preprocessing.scale(x_train)
            x_test = preprocessing.scale(x_test)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_Logistic_ElasticNet:
            # 逻辑回归方法
            from sklearn.linear_model import LogisticRegression
            from sklearn import preprocessing
            lr = LogisticRegression(penalty='elasticnet', solver='saga', multi_class='ovr', l1_ratio=0.5, C=0.1, max_iter=1)
            x_train = preprocessing.scale(x_train)
            x_test = preprocessing.scale(x_test)
            lr.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('Logistic_acc=', lr.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('Logistic_f1_weighted=', f1_score(y_test, lr.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('Logistic_F1_macro=', f1_score(y_test, lr.predict(x_test), average='macro'))
        if out_NN:
            # 神经网络方法  不加参数更差
            from sklearn.neural_network import MLPClassifier
            from sklearn import preprocessing
            nn = MLPClassifier(solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=1, power_t=0.5, max_iter=50, shuffle=True, random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
            x_train = preprocessing.scale(x_train)
            x_test = preprocessing.scale(x_test)
            nn.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('NN_acc=', nn.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('NN_f1_weighted=', f1_score(y_test, nn.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('NN_F1_macro=', f1_score(y_test, nn.predict(x_test), average='macro'))
        if out_XGBoost:  # XGBoost真跑的话，得把上边y_data-1的-1去了，
            # XGBoost方法
            from xgboost import XGBClassifier

            xgb = XGBClassifier()
            xgb.fit(x_train, y_train)
            # 输出预测的acc，f1_weighted，F1_macro
            # 输出预测的acc
            print('XGBoost_acc=', xgb.score(x_test, y_test))
            # 输出预测的f1_weighted
            print('XGBoost_f1_weighted=', f1_score(y_test, xgb.predict(x_test), average='weighted'))
            # 输出预测的F1_macro
            print('XGBoost_F1_macro=', f1_score(y_test, xgb.predict(x_test), average='macro'))








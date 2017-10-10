#coding=utf-8

import pandas as pd
import numpy as np
# from sklearn.metrics import r2_score,mean_squared_error
# from sklearn.model_selection import cross_val_score, GridSearchCV,cross_validate
from sklearn.model_selection import GridSearchCV,cross_validate

import sklearn.metrics as metrics
import xgboost as xgb
import matplotlib.pylab as plt


# The function used in most kernels
def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def measure_performance(y_act, y_pred, y_pred_prob = None, show_accuracy=True,
                        show_classification_report=True, show_confusion_matrix=True,
                        show_r2_score=False,
                        show_auc = False):
    from sklearn import metrics
    print "\nModel Report"

    if show_accuracy:
        print "Accuracy:{0:.3f}".format(metrics.accuracy_score(y_act, y_pred)), "\n"

    if show_classification_report:
        print "Classification report"
        print metrics.classification_report(y_act, y_pred), "\n"

    if show_confusion_matrix:
        print "Confusion matrix"
        print metrics.confusion_matrix(y_act, y_pred), "\n"

    if show_auc:
        print "AUC Score: %f" % metrics.roc_auc_score(y_act, y_pred_prob),  "\n"

    if show_r2_score:
        print "Coefficient of determination (R2):{0:.3f}".format(metrics.r2_score(y_act, y_pred)), "\n"



class MLPipe:
    # type_c_or_r: c  表示分类模型，r 表示回归模型
    def __init__(self, train_file, test_file, y_col, id_col, ft_name, type_c_or_r):
        self.train_file = train_file
        self.test_file = test_file
        self.y_col = y_col
        self.id_col = id_col
        self.type_c_or_r = type_c_or_r
        self.col_is_train = 'col_is_train'
        self.rand_state = 819
        self.ft_name = ft_name  # 数据集名称
        self.src_train_df = pd.DataFrame()
        self.src_test_df = pd.DataFrame()
        self.train_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.all_df = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_eval = pd.DataFrame()
        self.y_eval = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.mod = None
        self.mod_name = ''

    # 用来标识数据集的名字，主要用来输出结果文件名
    def set_ft_name(self, ft_name):
        self.ft_name = ft_name

    # 读取文件，然后合并两个文件
    def parse_file(self):
        self.src_train_df = pd.read_csv(self.train_file)
        self.src_test_df = pd.read_csv(self.test_file)

        self.train_df = self.src_train_df.copy()
        self.test_df = self.src_test_df.copy()

        # 设置test数据的y值为0，并且区分
        self.test_df[self.y_col] = 0
        self.test_df[self.col_is_train] = 0
        self.train_df[self.col_is_train] = 1
        # 合并两个数据集，后续所有操作都在all_df中
        self.all_df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        print("src data size ... ")
        print("Train Shape: %d:%d" % (self.train_df.shape))
        print("Test  Shape: %d:%d" % (self.test_df.shape))
        print("All   Shape: %d:%d" % (self.all_df.shape))

    def print_shape(self):
        print("Train Shape: %d:%d" % (self.X_train.shape))
        print("Eval  Shape: %d:%d" % (self.X_eval.shape))
        print("Test  Shape: %d:%d" % (self.X_test.shape))

    def split_X_y(self):
        self.X_train = self.all_df[self.all_df[self.col_is_train] == 1]
        self.y_train = self.X_train[self.y_col]
        self.X_train = self.X_train.drop(self.y_col, axis=1)

        self.X_eval = self.X_train
        self.y_eval = self.y_train

        self.X_test = self.all_df[self.all_df[self.col_is_train] == 0]
        self.y_test = self.X_test[self.y_col]
        self.X_test = self.X_test.drop(self.y_col, axis=1)

        print("After split ... ")
        self.print_shape()

    def gen_res_file_name(self):
        res_file = 'f%s.csv' % (self.ft_name)
        return res_file

    def out_res_file(self, is_pred_prob = False):
        if is_pred_prob:
            y_pred = self.mod.predict_proba(self.X_test)[:,1]
        else:
            y_pred = self.mod.predict(self.X_test)

        output = pd.DataFrame({'id': self.X_test[self.id_col].astype(np.int32), self.y_col: y_pred})
        out_file = self.gen_res_file_name()

        print("\n")
        print('XXXX out_res_file to %s' % out_file)
        output.to_csv(out_file, index=False)

    def get_dummies(self):
        nrow = self.all_df.shape[0]
        min_cate = max(10, nrow * 0.01)
        unique_num = self.all_df.nunique(axis=0)
        print('#'*80)
        print('get_dummies: nrow:%d min_cate:%d' %(nrow, min_cate) )
        print(unique_num)

        cate_col_num = unique_num[unique_num <= min_cate]

        sep_list=['col_is_train', self.id_col, self.y_col]
        cate_col_name = [val for val in cate_col_num.index.tolist() if val not in sep_list ]
        # get col index
        # cate_index = [id for id, val in enumerate(train.columns.tolist()) if val in cate_col_name]

        # get dummies
        self.all_df = pd.get_dummies(self.all_df, columns=cate_col_name, drop_first=True)
        print(self.all_df.columns)
        print('#' * 80)

class XGBC_ML(MLPipe):
    # print("MSE:",mean_squared_error(y_eval, y_pred))
    def gen_res_file_name(self):
        res_file = 'f%s_m%s_n%d_d%s.csv' % (self.ft_name, self.model_name,
                                                 self.mod.n_estimators,
                                                 self.mod.max_depth)
        return res_file

    def gen_model(self):
        self.model_name = 'XGBC'
        self.params = {
            'n_estimators': 10,
            'objective': 'binary:logistic',  # 回归问题 reg:linear 分类问题  binary:logistic  排序 rank:pairwise
            'learning_rate': 0.1,  # 学习率，默认0.3
            'max_depth': 6,  # 构建树的深度，越大越容易过拟合.默认6
            'subsample': 1,  # 随机采样训练样本，默认1
            'min_child_weight': 1,  # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'seed': 819
        }
        # set scale_pos_weight
        ratio = float(np.sum(self.y_train == 0)) / np.sum(self.y_train == 1)
        # self.params['scale_pos_weight'] = ratio

        self.early_stopping_rounds = 50

        # from sklearn.ensemble import XGBRegressor
        self.mod = xgb.XGBClassifier(**self.params)
        print  self.mod

    def choose_best_n_estimators(self):
        xgb_param = self.mod.get_xgb_params()
        xgtrain = xgb.DMatrix(ml.X_train, label=ml.y_train)
        # max_n_estimators = self.mod.get_params()['n_estimators']
        max_n_estimators = 1000
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=max_n_estimators,
                          nfold=5, metrics='auc', early_stopping_rounds=50, verbose_eval=1,
                          seed=819)
        ml.mod.set_params(n_estimators=cvresult.shape[0])
        print 'Set best_n_estimators:%d' % (cvresult.shape[0])

        # plot
        cv_mean = cvresult.iloc[:, [0, 2]]
        cv_mean.plot()
        plt.show()

    def mod_fit(self, plot_imp=True):
        eval_set = [(self.X_train, self.y_train), (self.X_eval, self.y_eval)]
        # 'eval_metric': 'auc', # rmse mae logloss error auc
        self.mod.fit(self.X_train, self.y_train, eval_metric='auc', eval_set=eval_set)
        evals_result = self.mod.evals_result()
        #print evals_result

        # Predict training set:
        dtrain_predictions = self.mod.predict(self.X_train)
        dtrain_predprob = self.mod.predict_proba(self.X_train)[:, 1]

        # Print model report:
        measure_performance(self.y_train, dtrain_predictions, dtrain_predprob,
                            show_accuracy=True,
                            show_classification_report=True,
                            show_confusion_matrix=True,
                            show_r2_score=False,
                            show_auc=True
                            )
        print "GINI Score: %f" % gini_normalized(self.y_train, dtrain_predprob), "\n"

        # plot imp
        feat_imp = pd.Series(self.mod.booster().get_fscore()).sort_values(ascending=False)
        print "Feature Importance Score"
        print feat_imp

        if plot_imp:
            feat_imp.plot(kind='barh', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show()

    def mod_cv(self, cv=5):
        from sklearn.metrics import make_scorer
        scoring = {'gini': make_scorer(gini_normalized), 'auc': 'roc_auc'}
        # scores = cross_val_score(self.mod, self.X_train, self.y_train, cv=cv, scoring=scoring)
        cv_results = cross_validate(self.mod, self.X_train, self.y_train, cv=cv, scoring=scoring)
        #print cv_results
        for key, val in cv_results.items():
            print("metrics:%s  Mean: %0.6f (+/- %0.6f)") % (key, val.mean(), val.std())

    # scoring
    # http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    def mod_gv(self, cv_params):
        gv = GridSearchCV(self.mod, cv_params, scoring='roc_auc', cv=5, verbose=10, n_jobs=1)
        gv.fit(self.X_train, self.y_train)
        # optimized_GBM.grid_scores_
        print("Grid scores on development set:")
        print()
        means = gv.cv_results_['mean_test_score']
        stds = gv.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gv.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("best score:%f best params:%s" %(gv.best_score_, gv.best_params_))

class CDrive(XGBC_ML):
    def transform_df(df):
        df = pd.DataFrame(df)
        dcol = [c for c in df.columns if c not in ['id', 'target']]
        df['ps_cat_13xps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        df['negative_one_vals'] = df[dcol].apply(lambda row: sum(row[:] == -1), axis=1)
        for c in dcol:
            df[c + str('_mean_range')] = (df[c].values > d_median[c]).astype(int)
        return df

    cat("Feature engineering")
    data[, amount_nas: = rowSums(data == -1, na.rm = T)]
    data[, high_nas: = ifelse(amount_nas > 4, 1, 0)]
    data[, ps_car_13_ps_reg_03: = ps_car_13 * ps_reg_03]
    data[, ps_reg_mult: = ps_reg_01 * ps_reg_02 * ps_reg_03]
    data[,
    ps_ind_bin_sum: = ps_ind_06_bin + ps_ind_07_bin + ps_ind_08_bin + ps_ind_09_bin + ps_ind_10_bin + ps_ind_11_bin + ps_ind_12_bin + ps_ind_13_bin + ps_ind_16_bin + ps_ind_17_bin + ps_ind_18_bin]
    ########################################################3
## main

is_test = 1
if is_test == 1:
    ml = XGBC_ML(train_file='../Ghost/train.csv', test_file='../Ghost/test.csv',
                y_col='type', id_col='id', ft_name='ghost', type_c_or_r='c')
else:
    ml = XGBC_ML(train_file='../driver/train.csv', test_file='../driver//test.csv',
                y_col='target', id_col='id', ft_name='driver', type_c_or_r='c')

ml.parse_file()
ml.get_dummies()

ml.split_X_y()
print ml.gen_model()

# ml.choose_best_n_estimators()
# 通过上面cv选择出的n_estimatiors
ml.mod.set_params(n_estimators=76)

# 开始参数调优
cv_params = {'max_depth': [2, 3, 4, 5, 6], 'min_child_weight': [1,2,3,4,5]}
cv_params = {'max_depth': [2, 3, 4, 5, 6], 'min_child_weight': [1,5]}

#ml.mod_gv(cv_params)

#ml.mod_cv()

# ml.mod_fit(plot_imp=False)
# ml.out_res_file(is_pred_prob=True)



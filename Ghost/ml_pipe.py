#!/root/anaconda2/bin/python
#coding=utf-8

import pandas as pd
import numpy as np
# from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV,cross_validate
import sklearn.metrics as metrics
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pylab as plt

from sklearn.model_selection import StratifiedKFold


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

def gini_xgb(pred, dtrain):
    y = dtrain.get_label()
    return 'gini', gini_normalized(y, pred)

def gini_lgb(pred, dtrain):
    y = list(dtrain.get_label())
    score = gini_normalized(y, pred)
    return 'gini', score, True


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
    def parse_file(self, is_test = 0):
        if is_test == 1:
            nrows = 1000
        else:
            nrows = None

        self.src_train_df = pd.read_csv(self.train_file, nrows=nrows )
        self.src_test_df = pd.read_csv(self.test_file, nrows=nrows )

        self.train_df = self.src_train_df.copy()
        self.test_df = self.src_test_df.copy()

        # 设置test数据的y值为0，并且区分
        self.test_df[self.y_col] = 0
        self.test_df[self.col_is_train] = 0
        self.train_df[self.col_is_train] = 1
        # 合并两个数据集，后续所有操作都在all_df中
        self.all_df = pd.concat([self.train_df, self.test_df], ignore_index=True)

        for c in self.all_df.select_dtypes(include=['float64']).columns:
            self.all_df[c] = self.all_df[c].astype(np.float32)

        for c in self.all_df.select_dtypes(include=['int64']).columns[2:]:
            self.all_df[c] = self.all_df[c].astype(np.int8)

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
            'learning_rate': 0.03,  # 学习率，默认0.3
            'max_depth': 4,  # 构建树的深度，越大越容易过拟合.默认6
            'subsample': 1,  # 随机采样训练样本，默认1
            'min_child_weight': 1,  # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
            # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
            # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
            'seed': 819
        }
        # set scale_pos_weight
        ratio = float(np.sum(self.y_train == 0)) / np.sum(self.y_train == 1)
        #self.params['scale_pos_weight'] = ratio
        self.params['scale_pos_weight'] = 10

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

    def mod_fit_kfold(self, plot_imp=True):
        skf = StratifiedKFold(n_splits=5)
        for train_idx, test_idx in skf.split(self.X_train, self.y_train):
            eval_set = [(self.X_train[train_idx], self.y_train[train_idx]),
                        (self.X_train[test_idx], self.y_train[test_idx])]
        # 'eval_metric': 'auc', # rmse mae logloss error auc
            self.mod.fit(self.X_train[train_idx], self.y_train[train_idx],
                         eval_metric='auc', eval_set=eval_set)
            evals_result = self.mod.evals_result()
        #print evals_result

        # Predict training set:
            #dtrain_predictions = self.mod.predict(self.X_train)
            pred = self.mod.predict_proba(self.X_test)[:, 1]

        # Print model report:

            print "GINI Score: %f" % gini_normalized(self.y_train, dtrain_predprob), "\n"

        # plot imp
            feat_imp = pd.Series(self.mod.booster().get_fscore()).sort_values(ascending=False)
            print "Feature Importance Score"
            print feat_imp


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


def transform_df(df):
    df = pd.DataFrame(df)

    df['amount_nas'] = np.sum((df == -1), axis=1)
    df['high_nas'] = np.where(df['amount_nas'] > 4, 1, 0)

#    df['ps_car_13_ps_reg_01'] = df.ps_car_13 * df.ps_reg_03
    df['ps_reg_mult'] = df.ps_reg_01 * df.ps_reg_02 * df.ps_reg_03

    import re
    ind_bin_ocl = [col for col in df.columns if re.match(r'ps_ind_\d*_bin', col)]
    df['ind_bin_num'] = np.sum(df[ind_bin_ocl], axis=1)

    #dcol = [c for c in df.columns if c not in ['id', 'target']]
    #for c in dcol:
    #    df[c + str('_mean_range')] = (df[c].values > df[c].median()).astype(int)

    #df = df.replace(-1, np.NaN)
    df = df.replace(-1, 0)

    # poly2
    col_int = ['ps_ind_01','ps_ind_03','ps_ind_14','ps_ind_15',
            'ps_reg_01','ps_reg_02','ps_reg_03',
            'ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15']
    for i in range(0, len(col_int)):
        for j in range(i+1, len(col_int)):
            df[col_int[i]+'_' + col_int[j]] = df[col_int[i]] * df[col_int[j]]

    # drop calc feat
    col_to_drop = df.columns[df.columns.str.startswith('ps_calc_')]
    df = df.drop(col_to_drop, axis=1)

    return df

class LGB_ML(MLPipe):
    def gen_res_file_name(self):
        res_file = 'f%s_m%s.csv' % (self.ft_name, self.model_name)
        return res_file

    def gen_model(self):
        self.model_name = 'LGB'
        self.params = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary'}
        self.params['metric'] = 'auc'
        self.mod = None

        all_col = [c for c in self.X_train.columns if c not in [self.id_col, self.y_col]]
        cate_col = [c for c in all_col if '_cat' in c]
        self.train_data = lgb.Dataset(data=self.X_train[all_col], label=self.y_train,
                                      feature_name=all_col,
                                      categorical_feature=cate_col,
                                      free_raw_data=False)
        self.test_data = lgb.Dataset(data=self.X_test[all_col], label=self.y_test,
                                      feature_name=all_col,
                                      categorical_feature=cate_col,
                                      free_raw_data=False)
        self.eval_data = lgb.Dataset(data=self.X_eval[all_col], label=self.y_eval,
                                     feature_name=all_col,
                                     categorical_feature=cate_col,
                                     free_raw_data=False)

    def mod_fit(self, num_round = 10, plot_imp=False):
        self.mod = lgb.train(self.params, self.train_data, num_round,
                        valid_sets=[self.eval_data])

    def mod_cv(self, num_round = 10):
        lgb.cv(self.params, self.train_data, num_round, nfold=5)

## main

def run_one():
    is_test = 0
    is_xgb = 1
    if is_xgb == 1:
        ml = XGBC_ML(train_file='../driver/train.csv', test_file='../driver/test.csv',
             y_col = 'target', id_col = 'id', ft_name = 'driver', type_c_or_r = 'c')
    else:
        ml = LGB_ML(train_file='../driver/train.csv', test_file='../driver/test.csv',
                    y_col='target', id_col='id', ft_name='driver', type_c_or_r='c')


    ml.parse_file(is_test)
    ml.all_df = transform_df(ml.all_df)
    print ml.all_df.describe()

    ml.split_X_y()
    print ml.gen_model()

    #ml.choose_best_n_estimators()

    # 通过上面cv选择出的n_estimatiors
    #ml.mod.set_params(n_estimators=475)

    # 开始参数调优
    cv_params = {'max_depth': [2, 3, 4, 5, 6], 'min_child_weight': [1,2,3,4,5]}
    cv_params = {'max_depth': [2, 3, 4, 5, 6], 'min_child_weight': [1,5]}

    #ml.mod_gv(cv_params)

    #ml.mod_cv()

    ml.mod_fit(plot_imp=True)

    #ml.out_res_file(is_pred_prob=True)
    #ml.out_res_file(is_pred_prob=False)


def run_two():
    is_test = 0
    ml = MLPipe(train_file='../driver/train.csv', test_file='../driver/test.csv',
                y_col='target', id_col='id', ft_name='driver', type_c_or_r='c')

    ml.parse_file(is_test)
    ml.all_df = transform_df(ml.all_df)
    print ml.all_df.describe()

    ml.split_X_y()

    # resample
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=819)
    X_resampled, y_resampled = ros.fit_sample(ml.X_train.values, ml.y_train.values)

    ml.X_train = pd.DataFrame(X_resampled, columns=ml.X_train.columns)
    ml.y_train = pd.Series(y_resampled)

    # xgb
    params = {'eta': 0.02, 'max_depth': 8, 'subsample': 0.9, 'colsample_bytree': 0.9,
              'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

    X = ml.X_train
    features = X.columns
    #
    #features = ['ps_ind_17_bin','ps_ind_05_cat','ps_reg_03_ps_car_13','ps_car_13','ps_car_07_cat','ps_ind_03_ps_car_13','ps_ind_06_bin','ps_car_05_cat','ps_ind_07_bin','ps_ind_04_cat','ps_car_12_ps_car_13','ps_car_11_cat','ps_ind_16_bin','ps_reg_01_ps_car_15','ps_ind_03_ps_car_11','ps_reg_02_ps_car_13','ps_car_03_cat','ps_car_01_cat','ps_reg_03_ps_car_15','ps_ind_01_ps_car_13','ps_ind_02_cat','ps_ind_03_ps_car_12','ps_ind_15_ps_car_14','ps_reg_01_ps_car_14','ps_reg_02_ps_car_15','ps_reg_03','amount_nas']
    #features = X.columns[feat_col]
    features = [c for c in X.columns if c not in [ml.id_col, ml.y_col, ml.col_is_train]]
    X = X[features].values
    y = ml.y_train.values
    test = ml.X_test
    sub = ml.X_test['id'].to_frame()
    sub['target'] = 0

    if is_test == 1:
        nrounds = 20  # need to change to 2000
    else:
        nrounds = 2000
    kfold = 5  # need to change to 5
    skf = StratifiedKFold(n_splits=kfold, random_state=819)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' xgb kfold: {}  of  {} : '.format(i + 1, kfold))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        d_train = xgb.DMatrix(X_train, y_train, feature_names=features)
        d_valid = xgb.DMatrix(X_valid, y_valid, feature_names=features)
        d_eval = xgb.DMatrix(ml.X_eval[features].values, ml.y_eval.values, feature_names=features)

        watchlist = [(d_train, 'train'), (d_valid, 'valid'), (d_eval, 'eval')]
        xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100,
                              feval=gini_xgb, maximize=True, verbose_eval=100)
        sub['target'] += xgb_model.predict(xgb.DMatrix(test[features].values, feature_names=features),
                                           ntree_limit=xgb_model.best_ntree_limit + 50) / (2 * kfold)
        # print
        feat_imp = pd.Series(xgb_model.get_fscore()).sort_values(ascending=False)
        print "Feature Importance Score"
        print feat_imp

    # lgb
    params = {'metric': 'auc', 'learning_rate': 0.01, 'max_depth': 10, 'max_bin': 10,
                'objective': 'binary',
        'feature_fraction': 0.8, 'bagging_fraction':0.9, 'bagging_freq':10, 'min_data': 500}

    skf = StratifiedKFold(n_splits=kfold, random_state=1)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' lgb kfold: {}  of  {} : '.format(i + 1, kfold))
        X_train, X_eval = X[train_index], X[test_index]
        y_train, y_eval = y[train_index], y[test_index]
        lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds,
                                lgb.Dataset(X_eval, label=y_eval), verbose_eval = 100,
                                feval = gini_lgb, early_stopping_rounds = 100)
        sub['target'] += lgb_model.predict(test[features].values,num_iteration = lgb_model.best_iteration) / (2 * kfold)

    sub.to_csv('fdriver_xgb_lgb_4.csv', index=False, float_format='%.5f')


def run_3():
    is_test = 1
    ml = MLPipe(train_file='../driver/train.csv', test_file='../driver/test.csv',
                y_col='target', id_col='id', ft_name='driver', type_c_or_r='c')

    ml.parse_file(is_test)
    ml.all_df = transform_df(ml.all_df)
    print ml.all_df.describe()

    ml.split_X_y()

    # resample
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=819)
    X_resampled, y_resampled = ros.fit_sample(ml.X_train.values, ml.y_train.values)

    ml.X_train = pd.DataFrame(X_resampled, columns=ml.X_train.columns)
    ml.y_train = pd.Series(y_resampled)


    # xgb
    params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,
              'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

    # X = ml.X_train
    X = ml.X_train
    features =  [c for c in X.columns if c not in [ml.id_col, ml.y_col, ml.col_is_train]]
    #
    #features = ['ps_ind_17_bin','ps_ind_05_cat','ps_reg_03_ps_car_13','ps_car_13','ps_car_07_cat','ps_ind_03_ps_car_13','ps_ind_06_bin','ps_car_05_cat','ps_ind_07_bin','ps_ind_04_cat','ps_car_12_ps_car_13','ps_car_11_cat','ps_ind_16_bin','ps_reg_01_ps_car_15','ps_ind_03_ps_car_11','ps_reg_02_ps_car_13','ps_car_03_cat','ps_car_01_cat','ps_reg_03_ps_car_15','ps_ind_01_ps_car_13','ps_ind_02_cat','ps_ind_03_ps_car_12','ps_ind_15_ps_car_14','ps_reg_01_ps_car_14','ps_reg_02_ps_car_15','ps_reg_03','amount_nas']
    #features = X.columns[feat_col]
    X = X[features].values
    y = ml.y_train.values
    test = ml.X_test
    sub = ml.X_test['id'].to_frame()
    sub['target'] = 0

    all_imp = pd.DataFrame()
    nrounds = 20  # need to change to 2000
    kfold = 5  # need to change to 5
    skf = StratifiedKFold(n_splits=kfold, random_state=819)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(' xgb kfold: {}  of  {} : '.format(i + 1, kfold))
        X_train, X_valid = X[train_index], X[test_index]
        y_train, y_valid = y[train_index], y[test_index]
        d_train = xgb.DMatrix(X_train, y_train, feature_names=features)
        d_valid = xgb.DMatrix(X_valid, y_valid, feature_names=features)
        d_eval = xgb.DMatrix(ml.X_eval[features].values, ml.y_eval.values, feature_names=features)

        watchlist = [(d_train, 'train'), (d_valid, 'valid'), (d_eval, 'eval')]
        xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100,
                              feval=gini_xgb, maximize=True, verbose_eval=100)
        sub['target'] += xgb_model.predict(xgb.DMatrix(test[features].values, feature_names=features),
                                           ntree_limit=xgb_model.best_ntree_limit + 50) / (2 * kfold)
        # print
        feat_imp = pd.Series(xgb_model.get_fscore()).sort_values(ascending=False)
        all_imp = pd.concat([all_imp, feat_imp], axis=1)
        print "Feature Importance Score"
        print feat_imp
    print '#'*80
    print "Feature Importance Score"
    all_imp.to_csv('./all_imp.csv')
    print all_imp.sum(axis=1).sort_values(ascending=False)

#run_one()
run_two()
#run_3()

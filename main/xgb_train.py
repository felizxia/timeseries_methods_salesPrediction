import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import statsmodels.api as sm
import datetime as DT
from datetime import datetime
import xgboost as xgb
from eval_metrics.eval import *


class XgbTsPipeline:

    def __init__(self):

        self.data = None
        self.val_mape_res = defaultdict(dict)
        self.val_mase_res = defaultdict(dict)
        self.params = defaultdict(dict)

    def preprocessing(self, smoothing_method='lowess', save=True):
        # get raw data joined
        # temp how to deal with it; promotional effects how to deal with it; see feature importance
        data1 = pd.read_csv('../data/Features data set.csv')
        data2 = pd.read_csv('../data/sales data-set.csv')
        data3 = pd.read_csv('../data/stores data-set.csv')

        # combine tables
        self.data = data2.merge(data1, on=['Store', 'Date', 'IsHoliday'], how='left').merge(data3, on='Store', how='left')
        # fill missing value
        self.data.fillna(-99999, inplace=True) # denote missing value as -99999

        # smoothing
        self.data['y'] = self.data.Weekly_Sales
        if smoothing_method == 'lowess':
            # for each store, dept, we have different smoothing
            for s in self.data.Store.unique():
                for d in self.data.Dept.unique():
                    data_smooth = self.data.loc[(self.data.Store == s) & (self.data.Dept == d)]
                    if len(data_smooth)<10:
                        continue
                    lowess = sm.nonparametric.lowess
                    self.data.loc[(self.data.Store == s) & (self.data.Dept == d), 'y'] = lowess(data_smooth.Weekly_Sales, range(len(data_smooth.Weekly_Sales)), 0.05, 10, return_sorted=False)
        elif smoothing_method == 'sigma':
            # anything larger or smaller than data+=3std, will be replaced with 3std
            for s in self.data.Store.unique():
                for d in self.data.Dept.unique():
                    data_smooth = self.data.loc[(self.data.Store == s) & (self.data.Dept == d)].Weekly_Sales
                    mean_v, std_v = np.mean(data_smooth), np.std(data_smooth)
                    max_bound, min_bound = mean_v + 3*std_v, mean_v - 3*std_v
                    self.data.loc[(self.data.Store == s) & (self.data.Dept == d) & (self.data.Weekly_Sales < min_bound), 'y'] = min_bound
                    self.data.loc[(self.data.Store == s) & (self.data.Dept == d) & (self.data.Weekly_Sales > min_bound), 'y'] = max_bound
        else:
            print('this smoothing method does not exists')
        # add new features
        # turn True and False to 0 or 1
        self.data.loc[self.data.IsHoliday == True, 'IsHoliday'] = 1
        self.data.loc[self.data.IsHoliday == False, 'IsHoliday'] = 0
        self.data['promotion'] = 0
        self.data.promotion.where(self.data.MarkDown1.notnull()==True,-1, inplace = True)
        self.data.promotion.where(self.data.MarkDown2.notnull()==True,-1, inplace = True)
        self.data.promotion.where(self.data.MarkDown3.notnull()==True,-1, inplace = True)
        self.data.promotion.where(self.data.MarkDown4.notnull()==True,-1, inplace = True)
        self.data.promotion.where(self.data.MarkDown5.notnull()==True,-1, inplace = True)

        self.data.Date = pd.to_datetime(self.data.Date, format='%d/%m/%Y')
        self.data['Month'] = self.data.Date.dt.month
        self.data['Year'] = self.data.Date.dt.year
        self.data['day'] = self.data.Date.dt.day
        self.data['dayofweek'] = self.data.Date.dt.dayofweek
        self.data['weekofyear'] = self.data.Date.dt.weekofyear
        if save:
            self.data.to_csv('preprocessed_data.csv', index=False, encoding='utf-8')
        return self.data

    def load_processed_data(self, data_path = '../processed_data/preprocessed_data.csv'):

        self.data = pd.read_csv(data_path)


    def generate_sale_features(self, df, forward_peroid): # either -7, -14 days if day is unit or -1, -2 if week is unit..
        columns = df.columns
        feature_set = []
        new_column_names = []
        for p in forward_peroid:
            for feature in columns:
                if p <= len(df):
                    feature_set += [df.iloc[-p][feature]]
                else:
                    feature_set += [-99999] # mark as nan
                new_column_names += [str(-p) + '_' + str(feature)]
        return feature_set, new_column_names


    def generate_backtest_data(self, df, test_date, time_gap, slide_peroid, slide_step, forward_peroid):
        """
        y is array of **training data**, include first prediction day [t_n, t_{n-1}... t1] (t1, t2, .. t7): prediction day for eg.,
        predict future days means how many future days to predict using machine learning models
        """
        # features for backtest data (train, validation)
        X_train = [[] for _ in range(len(test_date))] # len(test_date)
        X_validation = [[] for _ in range(len(test_date))]
        Y_train = [[] for _ in range(len(test_date))]
        Y_validation = [[] for _ in range(len(test_date))]

        # last train date : array
        last_test_day_date = [(datetime.strptime(t,'%Y-%m-%d') - DT.timedelta(days=time_gap)).strftime('%Y-%m-%d') for t in test_date]
        feature_names = None
        for i in range(len(test_date)): # Generate train data for each day model
            # for each dept,
            for d in df.Dept.unique():
                df_d = df.loc[(df.Dept == d) & (df.Date <= last_test_day_date[i])]

                t = -1 # start from last test day, then last train day is last test day - time_gap
                # use id from last_test_ady_index
                while -(t - slide_peroid) <= len(df_d):
                    test = df_d.iloc[t-slide_peroid-1 : t+1]
                    train = df_d.iloc[t-slide_peroid: t] # train data not included test data
                    other_feature_cols = ['Weekly_Sales', 'IsHoliday', 'Temperature',
               'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
               'MarkDown5', 'CPI', 'Unemployment', 'Size', 'promotion',
               'Month', 'Year', 'day', 'dayofweek', 'y'] # TO GET forward days

                    # X_train
                    Xtrain_y_features = [np.min(train.y), np.max(train.y), np.mean(train.y), np.std(train.y), stats.skew(train.y), stats.kurtosis(train.y)]

                    y_names = ['min_y', 'max_y', 'mean_y', 'std_y', 'skew_y', 'kurtosis_y']
                    Xtrain_other_features, other_names = self.generate_sale_features(train[other_feature_cols],forward_peroid)
                    # Xtest
                    Xtest_y_features = [np.min(test.y), np.max(test.y), np.mean(test.y), np.std(test.y), stats.skew(test.y), stats.kurtosis(test.y)]

                    Xtest_other_features, _ = self.generate_sale_features(test[other_feature_cols], forward_peroid)
                    # Ytrain
                    Y_train[i] += [df_d.iloc[t-1].y]
                    # Ytest
                    Y_validation[i] += [df_d.iloc[t].y]
                    X_train[i] += [Xtrain_y_features + Xtrain_other_features]
                    X_validation[i] += [Xtest_y_features + Xtest_other_features]

                    feature_names = y_names + other_names
                    t -= slide_step

        return X_train, X_validation, Y_train, Y_validation, feature_names


    # generate predict days feature and label

    def generate_test_data(self, df, test_date, time_gap, history_length_period, forward_peroid, generate_way):

        test_feature_names, naive_test, ma_test = None, [], []

        last_test_day_date = [(datetime.strptime(t, '%Y-%m-%d') - DT.timedelta(days=time_gap)).strftime('%Y-%m-%d') for t in
                              test_date]

        if generate_way == 0:  # use last day as feature, thus all future days have same features but different models
            # change it later.
            X_test = []
            Y_test = []

            for d in df.Dept.unique():
                # make sure the id has test data, else continue
                test_Res = df.loc[(df.Dept == d) & (df.Date == test_date[i])].Weekly_Sales
                if not len(test_Res):
                    continue
                df_d = df.loc[(df.Dept == d) & (df.Date == last_test_day_date[i])] # use last training day feature for all model
                t=-1

                train = df_d.iloc[t - history_length_period: t]  # train data not included test data
                other_feature_cols = ['Weekly_Sales', 'IsHoliday', 'Temperature',
                                      'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                                      'MarkDown5', 'CPI', 'Unemployment', 'Size', 'promotion',
                                      'Month', 'Year', 'day', 'dayofweek','y']

                Xtest_y_features = [np.min(train.y), np.max(train.y), np.mean(train.y), np.std(train.y),
                                    stats.skew(train.y), stats.kurtosis(train.y)]
                y_names = ['min_y', 'max_y', 'mean_y', 'std_y', 'skew_y', 'kurtosis_y']
                Xtest_other_features, other_names = self.generate_sale_features(train[other_feature_cols], forward_peroid)
                X_test = [Xtest_y_features + Xtest_other_features]
                test_feature_names = y_names + other_names

        else:

            X_test = [[] for _ in range(len(test_date))]
            Y_test = [[] for _ in range(len(test_date))]
            naive_test = [[] for _ in range(len(test_date))]
            ma_test = [[] for _ in range(len(test_date))]

            for i in range(len(test_date)):  # Generate train data for each day model
                for d in df.Dept.unique():
                    # make sure the id has test data, else continue
                    test_Res = df.loc[(df.Dept == d) & (df.Date == test_date[i])].Weekly_Sales
                    if not len(test_Res):
                        continue
                    df_d = df.loc[(df.Dept == d) & (df.Date <= last_test_day_date[i])]
                    test_Res = test_Res.values[0]
                    Y_test[i] += [test_Res]
                    t = -1
                    train = df_d.iloc[t - history_length_period: t]  # train data not included test data
                    other_feature_cols = ['Weekly_Sales', 'IsHoliday', 'Temperature',
                                          'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
                                          'MarkDown5', 'CPI', 'Unemployment', 'Size', 'promotion',
                                          'Month', 'Year', 'day', 'dayofweek','y']

                    Xtest_y_features = [np.min(train.y), np.max(train.y), np.mean(train.y), np.std(train.y),
                                        stats.skew(train.y), stats.kurtosis(train.y)]
                    y_names = ['min_y', 'max_y', 'mean_y', 'std_y', 'skew_y', 'kurtosis_y']
                    Xtest_other_features, other_names = self.generate_sale_features(train[other_feature_cols], forward_peroid)
                    X_test[i] += [Xtest_y_features + Xtest_other_features]
                    test_feature_names = y_names + other_names

                    # save random walk res and moving average res
                    naive_res = df.loc[(df.Dept == d) & (df.Date == last_test_day_date[i])].y
                    if len(naive_res):
                        naive_test[i] += [naive_res.values[0]]
                    else:
                        naive_test[i] += [-99999]  # nan
                    # monthly average 4 week
                    previous_time = (
                                datetime.strptime(last_test_day_date[i], '%Y-%m-%d') - DT.timedelta(days=4 * 7)).strftime(
                        '%Y-%m-%d')
                    ma_res = df.loc[(df.Dept == d) & (df.Date >= previous_time) & (df.Date <= last_test_day_date[i])].y
                    if len(ma_res):
                        ma_test[i] += [np.mean(ma_res.values)]
                    else:
                        ma_test[i] += [-99999]  # nan

        return X_test, Y_test, test_feature_names, naive_test, ma_test


    def generate_data(self, data_input):
        # for each store, generate data seperately.
        # generate training, validation, test data for each sku = Store + Dept
        # print(self.data.Date.unique()[-1:])
        X_train, X_validation, Y_train, Y_validation, feature_names = \
            self.generate_backtest_data(
            df = data_input.loc[data_input.Date <= data_input.Date.unique()[-2]], #-5
            test_date = self.data.Date.unique()[-1:], # -4 test on one day
            time_gap = 4 * 7,  # week as unit, 4 weeks: 28 days
            slide_peroid = 8,  # each has 8 week range
            slide_step = 2,
            forward_peroid = [1, 2, 3])


        X_test, Y_test, test_feature_names, naive_test, ma_test = \
            self.generate_test_data(
            df = data_input,
            test_date = data_input.Date.unique()[-1:],
            time_gap = 7,
            history_length_period = 143,
            forward_peroid = [1,2,3],
            generate_way = 1)

        return X_train, X_validation, Y_train, Y_validation, feature_names, X_test, Y_test, test_feature_names, naive_test, ma_test

    def xgboost(self, param, stored_param, iter,  X_train, Y_train, X_validation, Y_validation, feature_names, s):

        # store saved params

        param['verbosity'] = 0
        param['objective'] ='reg:linear'
        param['nthread'] = 4
        param['eval_metric'] = 'rmse'
        # store saved params
        self.params[s][iter] = param
        num_round = 100

        # print(feature_names)
        for i in range(len(X_train)):
            # n models for predicting n days
            # validation accuracy output
            dtrain = xgb.DMatrix(np.array(X_train[i]), label=np.array(Y_train[i]), feature_names=feature_names)
            bst = xgb.train(param, dtrain, num_round)
            dvalidation = xgb.DMatrix(np.array(X_validation[i]), feature_names=feature_names)
            validation_pred = bst.predict(dvalidation)
            mape_res = mape(validation_pred, Y_validation[i])
            # print('cross validation mape ..' + str(mape_res))
            self.val_mape_res[s][iter] = mape_res
            mae_res = mae(validation_pred, Y_validation[i])
            self.val_mase_res[s][iter] = mae_res
            # print('cross validation mae ..' + str(mae_res))
            stored_param[i][iter] = [mae_res, param]

        return stored_param

    def run(self, save=True):
        # either load or process data first
        self.load_processed_data()

        for s in self.data.Store.unique():
            data_store = self.data.loc[self.data.Store==s]
            X_train, X_validation, Y_train, Y_validation, feature_names, X_test, Y_test, test_feature_names, naive_test, ma_test = \
                self.generate_data(data_store)

            stored_param = defaultdict(dict)
            # randomly search parameters
            for iters in range(10):
                d = np.random.randint(50)
                eta = 10 ** np.random.uniform(-6, 1)
                par = {}
                par['max_depth']= d
                par['eta'] = eta
                stored_param = self.xgboost(par, stored_param, iters, X_train, Y_train, X_validation, Y_validation,
                                            feature_names, s)
            # find best parameters
            for d in stored_param: # d models
                d_par = stored_param[d]
                final_params = None
                min_cv = 99999
                for k in d_par.keys():
                    if d_par[k][0] < min_cv:
                        min_cv = d_par[k][0]
                        final_params = d_par[k][1]
                # print(final_params)
                dtrain = xgb.DMatrix(np.array(X_train[d]), label=np.array(Y_train[d]), feature_names=feature_names)
                bst = xgb.train(final_params, dtrain)
                dtest = xgb.DMatrix(np.array(X_test[d]), feature_names=feature_names)
                ypred = bst.predict(dtest)

        print(self.val_mape_res)
        print('\n')
        print(self.val_mase_res)
        print('\n')
        print(self.params)

        # save numpy
        np.save('../res/xgb/performance/val_mape_res.npy', self.val_mape_res)
        np.save('../res/xgb/performance/val_mase_res.npy', self.val_mase_res)
        np.save('../res/xgb/performance/xgbparams.npy', self.params)

if __name__ == "__main__":
    Pipeline = XgbTsPipeline()
    Pipeline.run()
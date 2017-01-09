# coding=utf-8
from __future__ import print_function

import re

import cPickle
import numpy as np
import pandas as pd
from sklearn import linear_model, svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def add_rul(df):
    # Remaining useful life
    df['RUL'] = df.groupby('id')['cycle'].transform(lambda grp: grp.max() - grp)
    return df


def add_labels(df, w0=15, w1=30):
    df['label1'] = df['RUL'].apply(lambda x: 1 if x <= w1 else 0)
    df['label2'] = df['RUL'].apply(lambda x: 2 if x <= w0 else (1 if x <= w1 else 0))
    return df


def add_rolling_values(agg_func, result_prefix, df, window_size=5):
    # Sensor data all in s1,s2,s3... pattern
    sensor_colnames = [x for x in df.columns if x == 'id' or re.search('s\d', x) is not None]
    sensor_df = df[sensor_colnames]
    new_df = sensor_df.groupby('id').transform(agg_func).fillna(0)
    # Column names for rolling mean will be a1,a2,a3...
    new_df.columns = [result_prefix + x[1:] for x in new_df.columns]
    df = pd.concat([df, new_df], axis=1)
    return df


def select_top_features(df, target_col='RUL', measure='pearson_correlation', n=35):
    if measure != 'pearson_correlation':
        raise ValueError('Only pearson correlation is supported for now.')
    excluded_colnames = ['id', 'label1', 'label2', 'RUL']
    feature_colnames = [x for x in df.columns if x not in excluded_colnames]
    feature_corr = []
    for col in feature_colnames:
        corr = df[col].corr(df[target_col])
        if not np.isnan(corr):
            feature_corr.append((col, corr))
    top_corr = sorted(feature_corr, key=lambda x: abs(x[1]), reverse=True)[:n]
    return [colname for colname, val in top_corr]


if __name__ == '__main__':
    colnames = ["id", "cycle", "setting1", "setting2", "setting3", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
                "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21"]
    df_train = pd.read_table('PM_train.txt', sep='\s+', names=colnames)
    df_test = pd.read_table('PM_test.txt', sep='\s+', names=colnames)
    df_truth = pd.read_table('PM_truth.txt', sep='\s+', names=['RUL'])

    # Window size for running sum/std
    window_size = 5

    # Prepare training data
    # =====================
    # Add remaining useful life and label
    df_train = add_rul(df_train)
    df_train = add_labels(df_train)
    # Add rolling mean
    df_train = add_rolling_values(lambda grp: grp.rolling(window_size, min_periods=1).mean(), 'a', df_train)
    # Add rolling std
    df_train = add_rolling_values(lambda grp: grp.rolling(window_size, min_periods=1).std(), 'sd', df_train)

    # Create min-max transformer
    cols_for_normalization = [x for x in df_train.columns if x not in ['id', 'label1', 'label2', 'RUL']]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(df_train[cols_for_normalization])

    # Normalize training data
    df_train[cols_for_normalization] = min_max_scaler.transform(df_train[cols_for_normalization])

    # Feature selection
    top_feature_colnames = select_top_features(df_train)

    # Prepare testing data
    # ====================
    # Add rolling mean
    df_test = add_rolling_values(lambda grp: grp.rolling(window_size, min_periods=1).mean(), 'a', df_test)
    # Add rolling std
    df_test = add_rolling_values(lambda grp: grp.rolling(window_size, min_periods=1).std(), 'sd', df_test)

    # For test data, only retrieve rows where id has max cycle
    max_idx = df_test.groupby('id')['cycle'].transform(max) == df_test['cycle']
    df_test = df_test[max_idx]
    df_test.index = range(len(df_test))

    # Append RUL, label1, label2 to test data
    df_truth = add_labels(df_truth)
    df_test = pd.concat([df_test, df_truth], axis=1)
    df_test[cols_for_normalization] = min_max_scaler.transform(df_test[cols_for_normalization])

    # Regresssion, predicting RUL
    train_X = df_train
    train_y = df_train[['RUL', 'label1', 'label2']]
    test_X = df_test
    test_y = df_test[['RUL', 'label1', 'label2']]

    train_data = (train_X, train_y)
    test_data = (test_X, test_y)

    # Save train and test data
    with open('train.pkl', 'wb') as fp:
        cPickle.dump(train_data, fp)
    with open('test.pkl', 'wb') as fp:
        cPickle.dump(test_data, fp)

    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_X.shape)
    print(top_feature_colnames)

    """
    # estimator = RandomForestRegressor(n_estimators=8, max_depth=32)
    estimator = RandomForestRegressor(random_state=10, n_estimators=8, max_depth=32)
    estimator.fit(train_X, train_y)
    print('RandomForest', estimator.score(test_X, test_y))

    estimator = linear_model.Ridge()
    estimator.fit(train_X, train_y)
    print('RidgeRegression', estimator.score(test_X, test_y))

    estimator = svm.SVR()
    estimator.fit(train_X, train_y)
    print('SVM', estimator.score(test_X, test_y))

    for predicted, actual in zip(estimator.predict(test_X), test_y):
        print('{:4d}  {:4d}  {}'.format(int(round(predicted)), int(actual), int(abs(actual - predicted))))
    """

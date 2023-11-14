from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

from multiclass_performanceMetrics import *
from neuralNet_functions import *
from nested_design_analysis import *


def MinMaxSclProcess(data, train_data, test_data):
    # Parameters
    # data: dataset of interest with only selected features from the original data set without the target variable
    # train_data: splitted data(param) for training. test_data splitted data(param) for testing
    # Output: MinMax scaled train data and test data
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_X_train = pd.DataFrame(scaler.transform(train_data))
    scaled_X_train.columns = data.columns
    scaled_X_test = pd.DataFrame(scaler.transform(test_data))
    scaled_X_test.columns = data.columns
    return scaled_X_train, scaled_X_test


def extract_data_by_classes(data, target_name, predictor_features):
    # param: data(pandas DataFrame) is the dataset
    # param: target_name(string) is the name of the target variable
    # param: predictor_features(list) is the list of columns(variables) that will predict the target variable
    # return: list of dataframes each of which is filtered by a class in the set of classes in the target variable
    class_labels = data[target_name].unique()
    dfs_by_class = []
    columns = predictor_features + [target_name]
    for cls in class_labels:
        dfs_by_class.append(data[data[target_name] == cls].loc[:, columns])
    return dfs_by_class


def sampling_data(dfs_by_class):
    # param: dfs_by_class is a list of pandas dataframes
    # each of which is filtered by a class in the set of classes in the target variable
    # return: one merged data consisting equal number of randomly sampled data points from each class
    min_num_rows = float('inf')
    for df in dfs_by_class:
        num_rows = df.shape[0]
        # finds the minimum number of data points in each dataframe
        if num_rows < min_num_rows:
            min_num_rows = num_rows
    # random sample the data such that each dataframe filtered by a class contains the equal number of data points
    # this prevents training bias toward the overrepresented class
    random_seed = np.random.randint(1, 1000)
    sampled_merged_data = dfs_by_class[0].sample(n=min_num_rows, random_state=random_seed)
    for i in range(1, len(dfs_by_class)):
        sampled_class_i_df = dfs_by_class[i].sample(n=min_num_rows, random_state=random_seed)
        # sampled_merged_data = sampled_merged_data.append(sampled_class_i_df)
        sampled_merged_data = pd.concat([sampled_merged_data, sampled_class_i_df], axis=0)
    return sampled_merged_data.sort_index()


def drop_missing_values(data, column_name, missing_value=np.nan):
    # param: data(pandas DataFrame) is the dataset
    # param: column_name(str) is a name of column where missing values are
    # param: missing_value(int, float, str, etc) is the value of missing value. Default numpy.nan
    # return new dataframe where rows with missing value in the column_name are dropped
    return data[data[column_name] != missing_value]



def simpleOversample(dfs_by_class):
    # param: dfs_by_class is a list of pandas dataframes
    # each of which is filtered by a class in the set of classes in the target variable
    # return: one merged data consisting equal number of randomly sampled data points from each class
    max_num_rows = -float('inf')
    for df in dfs_by_class:
        num_rows = df.shape[0]
        # finds the maximum number of data points in each dataframe
        if num_rows > max_num_rows:
            max_num_rows = num_rows
    # random sample the data such that each dataframe filtered by a class contains the equal number of data points
    # this prevents training bias toward the overrepresented class
    #random_seed = np.random.randint(1, 1000)
    # number of oversampled data = # of data in majority - # of data in a class Ck
    sampled_merged_data = dfs_by_class[0].copy()
    oversampled_data_by_class = dfs_by_class[0].sample(frac=1).iloc[:max_num_rows - len(dfs_by_class[0]), :]
    for i in range(1, len(dfs_by_class)):
        # random oversampling data using shuffling method
        sampled_class_i_df = dfs_by_class[i].sample(frac=1).iloc[:max_num_rows - len(dfs_by_class[i]), :]
        oversampled_data_by_class = pd.concat([oversampled_data_by_class, sampled_class_i_df], axis=0)
        sampled_merged_data = pd.concat([sampled_merged_data, dfs_by_class[i].copy()], axis=0)
    sampled_merged_data = pd.concat([sampled_merged_data, oversampled_data_by_class], axis=0, ignore_index=True)
    return sampled_merged_data.sort_index()



from numpy.random import default_rng

rng = default_rng()


def balanced_nested_design_sampling(df, **kwargs):
    # param: dfs_by_class is a list of pandas dataframe by class
    # param: df is a pandas dataframe representing data matrix
    # Beta_ji (influence of subject within a group)
    # return a data matrix after breaking the dependency caused by subject within a group(Beta_ji)
    # based on nested design
    # return: a dataframe after breaking the depdency by Beta_ji

    # break df into dataframes by class
    col_names = list(df.columns)
    target_name = kwargs.get('target_name', col_names[-1])
    num_window = kwargs.get("num_window", 1)  # number of nonoverlapping time window in a signal
    predictor_features = kwargs.get('predictor_features', col_names[:-1])
    dfs_by_class = extract_data_by_classes(df, target_name, predictor_features)
    additional_f = kwargs.get("additional_f", False)  # Bool. If True, subtract contribution of a class in nested design

    num_subjects = []
    for i in range(len(dfs_by_class)):
        # drop class labels from dataset as not used in nested design
        dfs_by_class[i].drop(labels=target_name, axis=1, inplace=True)
        # number of subjects in the order of encoded class labels
        num_subjects.append(int(dfs_by_class[i].shape[0] / num_window))
    min_col = dfs_by_class[0].shape[1]  # number of columns/features in the data

    # balancing the number of samples in the dataset for balanced_nested_design
    yijk = np.zeros((int(min_col * num_window), 1))
    min_num_subjects = min(num_subjects)
    for i, data in enumerate(dfs_by_class):
        x_flat = np.zeros((int(min_col * num_window), num_subjects[i]))
        for j in range(num_subjects[i]):
            temp = data.iloc[j * int(num_window):(j + 1) * int(num_window), :]
            temp = np.array(temp).flatten()  # row by row flattening
            x_flat[:, j] = temp
        # raondomly downsample majority classes to fit in balanced nested design
        idx = rng.choice(num_subjects[i], size=min_num_subjects, replace=False)
        idx.sort()

        x_flat = x_flat[:, idx]
        yijk = np.concatenate((yijk, x_flat), axis=1)
        if i == 0:
            # remove the first zero padded column of yijk that is used as filler to avoid runtime error
            yijk = yijk[:, 1:]

    # break dependency caused by the influence of a subject within a group
    m_size = [len(dfs_by_class), min_num_subjects, int(min_col * num_window)]
    alphas, betas = balanced_nested_design_estimator(yijk, m_size)
    # print(f"yijk shape: {yijk.shape}")  # debugging purpose
    yijk = break_subject_dependency(yijk, m_size, betas)
    if additional_f: yijk = break_class_dependency(yijk, m_size, alphas)

    # bring back to original dataset shape
    yijk = np.reshape(yijk.T, (int(num_window * min_num_subjects * 3), min_col))
    # print(yijk.shape)  # debugging purpose

    # convert data type to pandas dataframe
    nested_design_df = pd.DataFrame(yijk)
    nested_design_df.columns = predictor_features
    # add class labels back to the dataframe
    y_label = [i // int(num_window * min_num_subjects) for i in range(int(num_window * min_num_subjects * 3))]
    nested_design_df[target_name] = y_label
    return nested_design_df


def repeat_sampling_and_training(model_function, model_f_params, data,
                                 target_name, predictor_features, **kwargs):
    # param: model_function is a function object that creates a model
    # param: model_f_params is a list specifying the parameters of model_function (the order of elements in this list matters)
    # param: data(pandas DataFrame) is the dataset
    # param: target_name(string) is the name of the target variable
    # param: predictor_features(list) is the list of columns(variables) that will predict the target variable
    # return: dictionary containing performance metrics for each iteration
    # and confusion matrix of the model with the best test accuracy

    # (Boolean) indicates whether to apply MinMaxScale to data default: no minmax scaling
    doMinMaxScaling = kwargs.get('doMinMaxScaling', False)
    # (int) number of times to repeat sampling data and training model from it. default: 1000 repetitions
    num_repeat = kwargs.get('num_repeat', 1000)
    # if True, do oversampling. Else, do undersample to balance data
    is_oversample = kwargs.get('is_oversample', False)
    # oversampling technique. default) simply increasing the randomly selected minority class data
    # oversample_f must take a dataframe as its parameter
    oversample_f = kwargs.get('oversample_f', None)

    performance_measures_dict = {
        'training_accuracy': np.zeros(num_repeat),
        'testing_accuracy': np.zeros(num_repeat),
        'AUC_score': np.zeros(num_repeat)
    }

    # custom function for data processing to the whole dataset
    data_processing_f = kwargs.get("data_processing_f", None)
    # get a list of dataframe associated with one class
    dfs_lst = extract_data_by_classes(data, target_name, predictor_features)

    avg_cm = np.zeros((len(dfs_lst), len(dfs_lst)))
    for i in range(num_repeat):
        # applying data processing function before any other preprocessing technique
        if data_processing_f:
            num_window = kwargs.get("num_window", 1)
            new_data = data_processing_f(data, target_name=target_name,
                                     predictor_features=predictor_features, num_window=num_window)
            dfs_lst = extract_data_by_classes(new_data, target_name, predictor_features)
        # random sampling data such that the dataset contains the equal number of rows from each class
        if is_oversample:  # do oversampling of minority class
            # oversampling only done on training data to prevent data leakage
            X = data.drop(labels=target_name, axis=1)
            y = data[target_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            train_d = pd.concat([X_train, y_train], axis=1)
            train_d.columns = data.columns
            if oversample_f:  # use requested oversampling technique function
                train_d = oversample_f(train_d)
                X_train, y_train = train_d.drop(labels=target_name, axis=1), train_d[target_name]
            else:
                # default oversampling) simply increasing the randomly selected minority class data
                dfs_lst = extract_data_by_classes(train_d, target_name, predictor_features)
                train_d = simpleOversample(dfs_lst)
                X_train, y_train = train_d.drop(labels=target_name, axis=1), train_d[target_name]
        else:  # do undersampling of majority class
            sampled_data = sampling_data(dfs_lst)
            # split into training and testing data
            sampled_data = sampled_data.sample(frac=1)
            X = sampled_data.drop(labels=target_name, axis=1)
            y = sampled_data[target_name]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        if doMinMaxScaling:
            X_train, X_test = MinMaxSclProcess(pd.concat([X_train, X_test], axis=0), X_train, X_test)

        # fit model to training data
        model = model_function(params=model_f_params)
        model.fit(X_train, y_train)
        # predict y values for training and testing data
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # store performance metrics values of a model for each iteration
        metrics = {"recall": recall, "precision": precision, "f1": f1}
        train_cm = confusion_matrix(y_train, train_pred)  # rows are actual(truth), and columns are predicted
        test_cm = confusion_matrix(y_test, test_pred)  # rows are actual(truth), and columns are predicted
        performance_measures_dict['training_accuracy'][i] = accuracy(train_cm)
        test_acc = accuracy(test_cm)
        performance_measures_dict['testing_accuracy'][i] = test_acc

        for metric_name, metric_f in metrics.items():
            for k in range(len(dfs_lst)):  # per each class
                key_name = metric_name + f"_class{k}"
                try:
                    performance_measures_dict[key_name]
                except KeyError:
                    # raised only when i = 0
                    performance_measures_dict[key_name] = np.zeros(num_repeat)
                performance_measures_dict[key_name][i] = metric_f(test_cm, k)

        y_prob = model.predict_proba(X_test)
        if len(dfs_lst) > 2:  # multiclass AUC score
            performance_measures_dict['AUC_score'] = roc_auc_score(y_test, y_prob,
                                                               multi_class='ovo', average='macro')
        else:  # binary classification AUC score
            performance_measures_dict['AUC_score'] = roc_auc_score(y_test, y_prob[:,1])
        avg_cm = avg_cm + test_cm
    # average the confusion matrix over all repetitions
    avg_cm = pd.DataFrame(avg_cm / num_repeat)
    avg_cm.columns = [f"Predicted Class {i}" for i in range(len(dfs_lst))]
    avg_cm.index = [f"Actual Class {i}" for i in range(len(dfs_lst))]
    return performance_measures_dict, avg_cm


def print_performance_metrics(performance_measures_dict):
    # param: performance_measures_dict(dict) contains performance metrics for each iteration
    for metric_name, metric_vals in performance_measures_dict.items():
        print(f"Mean of {metric_name}: {metric_vals.mean()}")
        print(f"Standard deviation of {metric_name}: {metric_vals.std()}")


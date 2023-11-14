import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def recall(c_mat, class_k):
    # param: c_mat is a confusion matrix of model to be evaluated
    # In the confusion matrix, rows are the actual(truth), and columns are the predicted
    # param: class_k(int) is the class label, for which the metric will be computed
    # return the recall(sensitivity) metric with respect to class label k of the model
    actual_k_total = 0
    for j in range(len(c_mat[0])):  # iterate through all columns
        actual_k_total += c_mat[class_k][j]
    return c_mat[class_k][class_k] / actual_k_total


def accuracy(c_mat):
    # param: c_mat is a confusion matrix of model to be evaluated
    # In the confusion matrix, rows are the actual(truth), and columns are the predicted
    # return the accuracy metric of the model
    return np.sum(c_mat.diagonal()) / np.sum(c_mat)


def precision(c_mat, class_k):
    # param: c_mat is a confusion matrix of model to be evaluated
    # In the confusion matrix, rows are the actual(truth), and columns are the predicted
    # param: class_k(int) is the class label, for which the metric will be computed
    # return the precision metric with respect to class label k of the model
    predicted_k_total = 0
    for i in range(len(c_mat)):  # iterate through all rows
        predicted_k_total += c_mat[i][class_k]
    return c_mat[class_k][class_k] / predicted_k_total if predicted_k_total != 0 else 0.0


def f1(c_mat, class_k):
    # param: c_mat is a confusion matrix of model to be evaluated
    # In the confusion matrix, rows are the actual(truth), and columns are the predicted
    # param: class_k(int) is the class label, for which the metric will be computed
    # return the f1 score metric of the given classl label
    recall_val = recall(c_mat, class_k)
    precision_val = precision(c_mat, class_k)
    return 2 * (recall_val * precision_val) / (recall_val + precision_val)
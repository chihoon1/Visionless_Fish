import numpy as np
import pandas as pd
from scipy.stats import f
from numpy.random import default_rng

rng = default_rng()

# Perform nested design ANOVA for analysis for hypothesis testing
def nested_design_analysis(yijk, m_size):
    # Let i, j, (k) denote group, subject within a group, (subpart) respectively
    # Input:
    #   yijk is a matrix containing the value of j(with k) in group i
    #   size of yijk matrix: n by(a * b) where i = 1:a, j = 1:b, k = 1:n
    # a=num of class, b=num of subject within a class, n=num of observations within a subject
    # m_size is an array containing the size of num of class, num of subject within a class,
    #                                               num of observations within a subject
    # Balanced Nested Model expected in this function
    #
    a, b, n = m_size  # a=num of class, b=num of subject within a class, n=num of observations within a subject
    print(f"nested_design_analysis m_size: {m_size}")

    yij_mean = np.mean(yijk, axis=0)
    yij_mean.shape = (1, a*b)

    y_bar = np.mean(yij_mean)  # matrix of y mean with the same size as yijk
    # print(y_bar.shape)  # debugging purpose

    yij_bar = np.tile(yij_mean, (n,1))  # matrix of y_ij mean with the same size as yijk
    # print(yij_bar.shape)  # debugging purpose

    mm = np.reshape(yij_mean, (a,b)).T
    yi_mean = np.mean(mm, axis=0)
    f = np.tile(yi_mean, (b, 1)).T.flatten()
    yi_bar = np.tile(f, (n, 1))  # matrix of y_i means with the same size as yijk
    # print(yi_bar.shape)  # debugging purpose

    SST = np.sum( np.sum( np.square(yijk - y_bar) ) )
    SSA = np.sum( np.sum( np.square(yi_bar - y_bar) ) )
    SSB_A = np.sum( np.sum( np.square(yij_bar - yi_bar) ) )
    SSE = np.sum( np.sum( np.square(yijk - yij_bar) ) )

    return SST, SSA, SSB_A, SSE


def nested_desgin_hypothesis_testingR(SST, SSA, SSB_A, SSE, m_size, **kwargs):
    # params: SST, SSA, SSB_A, and SSE are integer
    # param: m_size is an array where 0-th element = num of class,
    #                   1st element = num of subject within a class,
    #                   2nd element = num of observations within a subject
    # return whether a null hypothesis is rejected(False) or not rejected(True_

    alfa = kwargs.get('alfa', 0.05)  # significant level alfa value. Default: 0.05

    a, b, n = m_size  # a=num of class, b=num of subject within a class, n=num of observations within a subject
    MSA = SSA / (a - 1)
    MSB_A = SSB_A / (a * (b - 1))
    MSE = SSE / (a * b * (n - 1))

    # i and j are both random or either one is random
    Fstat_alpha = MSA / MSB_A  # f-statistic value for hypothesis testing
    # print(f"F statistic for alpha: {Fstat_alpha}")  # debugging purpose
    pval = 1 - f.cdf(Fstat_alpha, a - 1, a * (b - 1))
    return pval


def no_correction_testing(pvals, alfa=0.05):
    # param: pvals is a numpy array containing p values from multiple hypothesis testing
    # param: alfa(int) is a significance level
    # return a boolean numpy array indicating rejecting null hypothesis(True) or no rejection(False)
    return pvals <= alfa


def false_discovery_rate(pvals, alfa=0.05):
    # After significance level is corrected with false discovery rate method,
    #   comparing pvals with the corrected significance level alfa
    # In order to avoid higher chances of type I error caused by multiple hypothesis testing
    # param: alfa(int) is a significance level before correction
    # param: pvals is a numpy array containing p values from multiple hypothesis testing
    # return a boolean numpy array indicating rejecting null hypothesis(True) or no rejection(False)
    num_testings = len(pvals)

    # sort the p-values from all testings in ascending order
    # the position of a p-value in the sorted p-values will be used to compute the false discovery rate correction
    pvals_sort = np.sort(pvals)

    # i-th element=True if null hypothesis is rejected in i-th testing
    reject_null_np = np.ones(num_testings, dtype='bool')
    for i in range(len(pvals)):
        idx = np.where(pvals_sort == pvals[i])[0][0] + 1  # get the position of a p-value in the sorted order
        fdr_alfa = idx * alfa / num_testings  # new alfa after false discovery rate correction
        # print(f"pval: {pvals[i]}, alfa: {fdr_alfa}")  # debugging purpose
        reject_null_np[i] = pvals[i] <= fdr_alfa
    return reject_null_np


def fish_nested_design_analysis(m, p, s, **kwargs):
    # Perform hypothesis testings to test whether difference between fish types exists for numerous times
    # Null Hypothesis: all fish types are the same. Hence, fish type can not be distinguished with the given data
    # Alternative hypothesis: at least one fish type differs. Hence, fish type can be classified with the given data
    # param: m (matrix) is a data set of Molino(i) with its subjects/fishs(j) and neurons(k)
    # param: p (matrix) is a data set of Pachon(i) with its subjects/fishs(j) and neurons(k)
    # param: s (matrix) is a data set of Surfcace(i) with its subjects/fishs(j) and neurons(k)
    # return number of times null hypothesis is rejected

    # number of iteration to repeat from sampling to the analysis
    num_iter = kwargs.get('num_iter', 100)

    # function for p value correction for multiple testing
    p_correction_f = kwargs.get('p_correction_f', no_correction_testing)

    m_size = kwargs.get("m_size", [3, 11, 275])  # a vetor containg the size of i, j, k

    count = 0
    pvals = np.zeros(num_iter)
    for i in range(num_iter):
        # slicing upto 275 neurons
        min_cols = 275
        # There are 11 subjects in Surface fish, but 16 subjects in each of Molino and Pachon
        # Hence, need to downsample Molino and Pachon to apply balanced nested design
        m_row_idx = rng.choice(16, size=11, replace=False)  # random downsampling
        p_row_idx = rng.choice(16, size=11, replace=False)  # random downsampling
        m_sampled = m.iloc[m_row_idx, :min_cols].T
        p_sampled = p.iloc[p_row_idx, :min_cols].T
        s_sampled = s.iloc[:, :min_cols].T

        # form a nested design matrix by combining the datasets from all fish types
        ijk = pd.concat([m_sampled, p_sampled], axis=1)
        ijk = pd.concat([ijk, s_sampled], axis=1)
        ijk = np.array(ijk)

        # perform a hypothesis testing
        SST, SSA, SSB_A, SSE = nested_design_analysis(ijk, m_size)
        p_val = nested_desgin_hypothesis_testingR(SST, SSA, SSB_A, SSE, m_size)
        pvals[i] = p_val
    # count the number of times rejecting null hypothesis using corrected p-value
    # to avoid higher chances of type I error caused by multiple hypothesis testing
    reject_null_counts = p_correction_f(pvals)
    counts = np.sum(reject_null_counts)
    return counts


def form_timeWindow_yijk(m_size, *args):
    # param: m_size is a vector containing the size of i, j, k, and l(where l is # of non-overlapping time window)
    # param args used for dataset of each fish type
    # return a np 2d array representing a balanced nest design data matrix in the form of i * j columns and k * l rows
    num_class, min_num_subjects, min_col, num_window = m_size
    dfs_by_class = args
    # balancing the number of samples in the dataset for balanced_nested_design
    yijk = np.zeros((int(min_col * num_window), 1))
    # form an array of number of subjects within a fish type
    num_subjects = [int(dfs_by_class[i].shape[0] // num_window) for i in range(len(dfs_by_class))]

    # reshaping the dataset and combining the datasets for all fish types
    # to form a balanced nested design data matrix
    for i, data in enumerate(dfs_by_class):
        x_flat = np.zeros((int(min_col * num_window), num_subjects[i]))
        for j in range(num_subjects[i]):
            temp = data.iloc[j * int(num_window):(j + 1) * int(num_window), :]
            temp = np.array(temp).flatten()  # row by row flattening
            x_flat[:, j] = temp
        # print(x_flat.shape)  # debugging purpose
        # downsample majority classes to fit in balanced nested design
        idx = rng.choice(num_subjects[i], size=min_num_subjects, replace=False)
        idx.sort()

        x_flat = x_flat[:, idx]
        yijk = np.concatenate((yijk, x_flat), axis=1)
        if i == 0:
            # remove the first zero padded column of yijk that is used as filler to avoid runtime error
            yijk = yijk[:, 1:]
    return yijk


def timeWindow_fish_nested_design_analysis(m, p, s, **kwargs):
    # Given the dataset of a fish type where neural signals are split into sub-signals by non-overlapping time window,
    # Perform hypothesis testings to test whether difference between fish types exists for numerous times
    # Null Hypothesis: all fish types are the same. Hence, fish type can not be distinguished with the given data
    # Alternative hypothesis: at least one fish type differs. Hence, fish type can be classified with the given data
    # param: m (matrix) is a data set of type Molino(i) with its fishes(j) and neurons(k) and l num of time windows
    # param: p (matrix) is a data set of type Pachon(i) with its fishes(j) and neurons(k) and l num of time windows
    # param: s (matrix) is a data set of type Surface(i) with its fishes(j) and neurons(k) and l num of time windows
    # return number of times null hypothesis is rejected

    num_iter = kwargs.get('num_iter', 100)  # number of iteration to repeat from sampling to the analysis

    # function for p value correction for multiple testing
    p_correction_f = kwargs.get('p_correction_f', no_correction_testing)

    # a vector containing the size of i, j, k, l(where l is # of non-overlapping time window)
    m_size = kwargs.get("m_size", [3, 11, 275, 11])

    count = 0
    pvals = np.zeros(num_iter)
    # slicing upto 275 neurons
    min_cols = 275
    m = m.iloc[:, :min_cols]
    p = p.iloc[:, :min_cols]
    s = s.iloc[:, :min_cols]
    for i in range(num_iter):
        # construct balanced nested design data matrix
        ijk = form_timeWindow_yijk(m_size, m, p, s)
        # print(ijk.shape, "\n", ijk)  # debugging purpose
        # redefine the size of matrix with the constructed balanced nested design data matrix
        m_size_alt = [m_size[0], m_size[1], int(m_size[2]*m_size[3])]  # a, b, k*l
        # run the hypothesis testing
        SST, SSA, SSB_A, SSE = nested_design_analysis(ijk, m_size_alt)
        p_val = nested_desgin_hypothesis_testingR(SST, SSA, SSB_A, SSE, m_size_alt)
        pvals[i] = p_val
    reject_null_counts = p_correction_f(pvals)
    counts = np.sum(reject_null_counts)
    return counts
    #return counts, avg_SST, avg_SSA, avg_SSBA, avg_SSE


def balanced_nested_design_estimator(yijk, m_size):
    # Let i, j, (k) denote group, subject within a group, (subpart/measurements) respectively
    # Input:
    #   yijk is a matrix containing the value of j(with k) in group i
    #   size of yijk matrix: n by(a * b) where i = 1:a, j = 1:b, k = 1:n
    #   m_size is an array where 0-th element = num of class,
    #                   1st element = num of subject within a class,
    #                   2nd element = num of observations within a subject
    # Balanced Nested Model expected in this function
    # return estimator of alphas(coefficient of groups) and Beta_ji (influence of subject within a group)
    #

    a, b, n = m_size  # a=num of class, b=num of subject within a class, n=num of observations within a subject
    # initialization
    alphas = np.zeros(a)
    betas = np.zeros((a, b))  # row: group index, column: subject index


    yij_mean = np.mean(yijk, axis=0)
    yij_mean.shape = (1, a * b)

    y_mean = np.mean(yij_mean)  # a mean of all yijk
    # print(y_mean.shape)  # debugging purpose

    # compute estimator of alphas
    mm = np.reshape(yij_mean, (a, b)).T
    yi_mean = np.mean(mm, axis=0)
    alphas = yi_mean - y_mean

    # compute estimator of Beta_ji
    reshaped_yi_mean = np.tile(yi_mean, (b,1))
    reshaped_yi_mean = reshaped_yi_mean.T
    reshaped_yi_mean = reshaped_yi_mean.flatten()
    betas = yij_mean - reshaped_yi_mean
    return alphas, betas


def break_subject_dependency(yijk, m_size, betas, **kwargs):
    # Let i, j, (k) denote group, subject within a group, (subpart/measurements) respectively
    # Input:
    #   yijk is a matrix containing the value of j(with k) in group i
    #   size of yijk matrix: n by(a * b) where i = 1:a, j = 1:b, k = 1:n
    #   m_size is an array where 0-th element = num of class,
    #                   1st element = num of subject within a class,
    #                   2nd element = num of observations within a subject
    # Balanced Nested Model expected in this function
    # Beta_ji (influence of subject within a group)
    # return a data matrix after breaking the dependency caused by subject within a group(Beta_ji)
    # based on nested design
    #
    n = m_size[2]
    betas_n_rows = np.tile(betas, (n,1))  # reshape betas to subtract Beta_ji from the data matrix
    # print(betas_n_rows)  # debugging purpose
    return yijk - betas_n_rows


def break_class_dependency(yijk, m_size, alphas, **kwargs):
    # Let i, j, (k) denote group, subject within a group, (subpart/measurements) respectively
    # Input:
    #   yijk is a matrix containing the value of j(with k) in group i
    #   size of yijk matrix: n by(a * b) where i = 1:a, j = 1:b, k = 1:n
    #   m_size is an array where 0-th element = num of class,
    #                   1st element = num of subject within a class,
    #                   2nd element = num of observations within a subject
    # Balanced Nested Model expected in this function
    # alpha_i (influence of a class/group)
    # return a data matrix after breaking the dependency caused by a class(alpha_i)
    # based on nested design
    #
    b = m_size[1]
    n = m_size[2]
    # reshape alphas to subtract alpha_i from the data matrix
    alphas.shape = (1,len(alphas))
    alpha_reshaped = np.tile(alphas.T, (1,int(b/len(alphas)))).flatten()
    alpha_reshaped = np.tile(alpha_reshaped, (n,1))
    # print(alpha_reshaped)  # debugging purpose
    return yijk - alpha_reshaped



if __name__ == '__main__':
    # simple testing of codes with the example found on a textbook
    yijk = [[65, 68, 56, 74, 69, 73, 65, 67, 72, 81, 76, 77],
            [71, 70, 55, 76, 70, 77, 74, 59, 63, 75, 72, 69],
            [63, 64, 65, 79, 80, 77, 70, 61, 64, 77, 79, 74],
            [69, 71, 68, 81, 79, 79, 69, 66, 69, 75, 82, 79],
            [73, 75, 70, 72, 68, 68, 73, 71, 70, 80, 78, 66]]
    yijk = np.array(yijk)
    m_size = [4, 3, 5]
    SST, SSA, SSB_A, SSE = nested_design_analysis(yijk, m_size)
    print(SST, SSA, SSB_A, SSE)
    nested_desgin_hypothesis_testingR(SST, SSA, SSB_A, SSE, m_size)
import json
import numpy as np  # check out how to install numpy
from utils import load, plot_sample

# =========================================
#       Homework on K-Nearest Neighbors
# =========================================
# Course: Introduction to Information Theory
# Lecturer: Haim H. Permuter.
#
# NOTE:
# -----
# Please change the variable ID below to your ID number as a string.
# Please do it now and save this file before doing the assignment

ID = '318416690'

# Loading and plot a sample of the data
# ---------------------------------------
# The MNIST database contains in total of 60000 train images and 10000 test images
# of handwritten digits.
# This database is a benchmark for many classification algorithms, including neural networks.
# For further reading, visit http://yann.lecun.com/exdb/mnist/
#
# You will implement the KNN algorithm to classify two digits from the database: 3 and 5.
# First, we will load the data from .MAT file we have prepared for this assignment: MNIST_3_and_5.mat

Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')


# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#
# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'

# sampleNum = 0
# plot_sample(Xvalid[sampleNum, :], Yvalid[sampleNum, :])

# Build a KNN classifier based on what you've learned in class:
#
# 1. The classifier should be given a train dataset (Xtrain, Ytain),  and a test sample Xtest.
# 2. The classifier should return a class for each row in test sample in a column vector Ytest.
#
# Finally, your results will be saved into a <ID>.txt file, where each row is an element in Ytest.
#
# Note:
# ------
# For you conveniece (and ours), give you classifications in a 1902 x 1 vector named Ytest,
# and set the variable ID at the beginning of this file to your ID.


# < your code here >
def KNNclassf(x_train, y_train, k, x_test, L=None):
    """for each picture(row in the X_test matrix), we calculate the distance(L2 norm) to each picture in X_train
    matrix, then we take the labels of the Kth closest one and compare appearances of 3's against 5's - the bigger one
    will be our classification.
    additional candidate for the metric was L1, after testing for different k is seems that L2 give better results
    """
    y_test = np.zeros((x_test.shape[0], 1), dtype=int)
    for i in range(len(x_test)):
        dis = np.linalg.norm(x_test[i] - x_train,
                             axis=1, ord=L)  # calc the distance between pixels from the test pic to the test picture
        nn = dis.argsort()[:k]  # take the Kth nearset
        nn_labels = y_train[nn].flatten()  # convert it to one dim vector(like matlab squeeze)
        classf_3 = sum([1 for char in nn_labels if char == 3])
        classf_5 = sum([1 for char in nn_labels if char == 5])
        if classf_3 > classf_5:
            y_test[i] = 3
        else:  # for a tie, 5 wins
            y_test[i] = 5
    return y_test


# NOTE simulations for peaking the best K - I ran it for k=1-29, and the best one was 17.
# for L in [1,2]:
#     for k in range(1,30):
#         y_check = KNNclassf(Xtrain,Ytrain,k,Xvalid, L)
#         correct_1 = [1 for i in range(len(Yvalid)) if y_check[i] == Yvalid[i]]
#         print(f'for k={k} we get and norm{L}',sum(correct_1), 'correct classfications')

# Example submission array - comment/delete it before submitting:
# Ytest = KNNclassf(Xtrain, Ytrain, 17, Xtest)
# save classification results
print('saving')
# np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')

# coding:UTF-8
#  http://blog.csdn.net/google19890102/article/details/51743192
import numpy as np
import random as rd

def load_data(file_name):
    data = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            tmp.append(float(x) / 255.0)
        data.append(tmp)
    f.close()
    return data

def sigmrnd(P):
    m, n = np.shape(P)
    X = np.mat(np.zeros((m, n)))
    P_1 = sigm(P)
    for i in xrange(m):
        for j in xrange(n):
            r = rd.random()
            if P_1[i, j] >= r:
                X[i, j] = 1

    return X

def sigm(P):
    return 1.0 / (1 + np.exp(-P))


# step_1: load data
datafile = "b.txt"
data = np.mat(load_data(datafile))
m, n = np.shape(data)

# step_2: initialize
num_epochs = 10
batch_size = 100
input_dim = n

hidden_sz = 100

alpha = 1
momentum = 0.1
W = np.mat(np.zeros((hidden_sz, input_dim)))
vW = np.mat(np.zeros((hidden_sz, input_dim)))
b = np.mat(np.zeros((input_dim, 1)))
vb = np.mat(np.zeros((input_dim, 1)))
c = np.mat(np.zeros((hidden_sz, 1)))
vc = np.mat(np.zeros((hidden_sz, 1)))

# step_3: training
print "Start to train RBM: "

num_batches = int(m / batch_size)
for i in xrange(num_epochs):
    kk = np.random.permutation(range(m))
    err = 0.0

    for j in xrange(num_batches):
        batch = data[kk[j * batch_size:(j + 1) * batch_size], ]

        v1 = batch
        h1 = sigmrnd(np.ones((batch_size, 1)) * c.T + v1 * W.T)
        v2 = sigmrnd(np.ones((batch_size, 1)) * b.T + h1 * W)
        h2 = sigm(np.ones((batch_size, 1)) * c.T + v2 * W.T)

        c1 = h1.T * v1
        c2 = h2.T * v2

        vW = momentum * vW + alpha * (c1 - c2) / batch_size
        vb = momentum * vb + alpha * sum(v1 - v2).T / batch_size
        vc = momentum * vc + alpha * sum(h1 - h2).T / batch_size

        W = W + vW
        b = b + vb
        c = c + vc

    #cal_err
    err_result = v1 - v2
    err_1 = 0.0
    m_1, n_1 = np.shape(err_result)
    for x in xrange(m_1):
        for y in xrange(n_1):
            err_1 = err_1 + err_result[x, y] ** 2
        err = err + err_1 / batch_size
    #print i,j,err
    print i, err/num_batches

m_2,n_2 = np.shape(W)

for i in xrange(m_2):
    for j in xrange(n_2):
        print str(W[i, j]) + " ",
    print "\n",

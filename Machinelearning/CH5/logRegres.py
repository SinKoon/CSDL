from math import *
from numpy import *


def loaddataset():
    datamat = []; labelmat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        linearr = line.strip().split()
        datamat.append([1.0, float(linearr[0]), float(linearr[1])])
        labelmat.append(int(linearr[2]))
    return datamat, labelmat


def sigmoid(inx):
    return 1.0/(1+exp(-inx))


def gradascent(datamatin, classlabels):
    datamatrix = mat(datamatin)
    labelmat = mat(classlabels).transpose()  # 转置
    m, n = shape(datamatrix)
    alpha = 0.001
    maxcycles = 500
    weights = ones((n, 1))
    for k in range(maxcycles):
        h = sigmoid(datamatrix * weights)
        error = (labelmat - h)
        weights = weights + alpha * datamatrix.transpose() * error
    return  weights


def plotbestfit(weights):
    import matplotlib.pyplot as plt
    datamat, labelmat = loaddataset()
    dataarr = array(datamat)
    n = shape(dataarr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelmat[i]) == 1:
            xcord1.append(dataarr[i, 1])
            ycord1.append(dataarr[i, 2])
        else:
            xcord2.append(dataarr[i, 1])
            ycord2.append(dataarr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.show()


def stocgradascent0(datamatrix, classlabels):
    m, n = shape(datamatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(datamatrix[i]*weights))
        error = classlabels[i] - h
        weights = weights + alpha * error * datamatrix[i]
    return weights


def stocgradascent1(datamatrix, classlabels, numiter=150):
    m, n = shape(datamatrix)
    weights = ones(n)
    for j in range(numiter):
        dataindex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randindex = int(random.uniform(0, len(dataindex)))
            h = sigmoid(sum(datamatrix[randindex]*weights))
            error = classlabels[randindex] - h
            weights = weights + alpha * error * datamatrix[randindex]
            del(dataindex[randindex])
    return weights





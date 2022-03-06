import random
from numpy import *

def loaddataset(filename):
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = line.strip().split('\t')
        datamat.append([float(linearr[0]), float(linearr[1])])
        labelmat.append(float(linearr[2]))
    return datamat, labelmat


def selectjrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipalpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smosimple(datamtain, classlabels, C, toler, maxiter):
    datamatrix = mat(datamtain)
    labelmat = mat(classlabels).transpose()
    b = 0
    m, n = shape(datamatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxiter:
        alphapairschange = 0
        for i in range(m):
            fxi = float(multiply(alphas, labelmat).T*(datamatrix*datamatrix[i, :].T)) + b
            ei = fxi - float(labelmat[i])
            if ((labelmat[i]*ei < -toler) and (alphas[i] < C)) or ((labelmat[i]*ei > toler) and (alphas[i] > 0)):
                j = selectjrand(i, m)
                fxj = float(multiply(alphas, labelmat).T*(datamatrix*datamatrix[j, :].T)) + b
                ej = fxj - float(labelmat[j])
                alphaiold = alphas[i].copy()
                alphajold = alphas[j].copy()
                if labelmat[i] != labelmat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                eta = 2.0 * datamatrix[i, :] * datamatrix[j, :].T - datamatrix[i, :] * datamatrix[i, :].T - datamatrix[j, :] * datamatrix[j, :].T
                if eta >= 0:
                    print("eta = 0")
                    continue
                alphas[j] -= labelmat[j]*(ei - ej)/eta
                alphas[j] = clipalpha(alphas[j], H, L)
                if abs(alphas[j] - alphajold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelmat[j] * labelmat[i] * (alphajold - alphas[j])
                b1 = b - ei - labelmat[i] * (alphas[i] - alphaiold) * datamatrix[i, :] * datamatrix[i, :].T - labelmat[
                    j] * (alphas[j] - alphajold) * datamatrix[i, :] * datamatrix[j, :].T
                b2 = b - ej - labelmat[i] * (alphas[i] - alphaiold) * datamatrix[i, :] * datamatrix[j, :].T - labelmat[
                    j] * (alphas[j] - alphajold) * datamatrix[j, :] * datamatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphapairschange += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphapairschange))
            if alphapairschange == 0:
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
        return b, alphas


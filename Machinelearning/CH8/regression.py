from numpy import *

def loaddataset(filename):
    numfeat = len(open(filename).readline().split('\t')) - 1
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = []
        curline = line.strip().split('\t')
        for i in range(numfeat):
            linearr.append(float(curline[i]))
        datamat.append(linearr)
        labelmat.append(float(curline[-1]))
    return datamat, labelmat


def standregres(xarr, yarr):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    xTx = xmat.T*xmat
    if linalg.det(xTx) == 0:
        print("this matrix is singular, cannot do inverse")
    ws = xTx.I * (xmat.T * ymat)
    return ws


def lwlr(testpoint, xarr, yarr, k=1.0):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    m = shape(xmat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffmat = testpoint - xmat[j, :]
        weights[j, j] = exp(diffmat*diffmat.T/(-2.0*k**2))
    xTx = xmat.T * (weights * xmat)
    if linalg.det(xTx) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xmat.T * (weights * ymat))
    return testpoint * ws


def lwlrtest(testarr, xarr, yarr, k=1.0):
    m = shape(testarr)[0]
    yhat = zeros(m)
    for i in range(m):
        yhat[i] = lwlr(testarr[i], xarr, yarr, k)
    return yhat


def ridgeregres(xmat, ymat, lam=0.2):
    xTx = xmat.T * xmat
    denom = xTx + eye(shape(xmat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("this matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xmat.T*ymat)
    return ws

def ridgetest(xarr, yarr):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    ymean = mean(ymat, 0)
    ymat = ymat - ymean
    xmeans = mean(xmat, 0)
    xvar = var(xmat, 0)
    xmat = (xmat - xmeans)/xvar
    numtestpts = 30
    wmat = zeros((numtestpts, shape(xmat)[1]))
    for i in range(numtestpts):
        ws = ridgeregres(xmat, ymat, exp(i-10))
        wmat[i, :] = ws.T
    return wmat


def stagewise(xarr, yarr, eps=0.01, numit=100):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    ymean = mean(ymat, 0)
    ymat = ymat - ymean
    xmat = regularize(xmat)
    m, n = shape(xmat)
    returnmat = zeros((numit, n))
    ws = zeros((n, 1))
    wstest = ws.copy()
    wsmax = ws.copy()
    for i in range(numit):
        print(ws.T)
    lowesterror = inf
    for j in range(n):
        for sign in [-1, 1]:
            wstest = ws.copy()
            wstest[j] += eps*sign
            ytest = xmat*wstest
            rsse = rsserror(ymat.A, ytest.A)
            if rsse < lowesterror:
                lowesterror = rsse
                wsmax = wstest
        ws = wsmax.copy()
        returnmat[i, :] = ws.T
    return returnmat



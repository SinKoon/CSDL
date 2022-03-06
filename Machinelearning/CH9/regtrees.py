from numpy import *

def loaddataset(filename):
    datamat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltline = list(map(float, curline))
        datamat.append(fltline)
    return datamat


def binsplitdataset(dataset, feature, value):
    mat0 = dataset[nonzero(dataset[:, feature] > value)[0], :]
    mat1 = dataset[nonzero(dataset[:, feature] <= value)[0], :]
    return mat0, mat1


def regleaf(dataset):
    return mean(dataset[:, -1])


def regerr(dataset):
    return var(dataset[:, -1]) * shape(dataset)[0]


def choosebestsplit(dataset, leaftype=regleaf, errtype=regerr, ops=(1, 4)):
    tols = ops[0]
    tolN = ops[1]
    if len(set(dataset[:, -1].T.tolist()[0])) == 1:
        return None, leaftype(dataset)
    m, n = shape(dataset)
    s = errtype(dataset)
    bests = inf
    bestindex = 0
    bestvalue = 0
    for featindex in range(n-1):
        for splitval in set(dataset[:, featindex].tolist()[0]):
            mat0, mat1 = binsplitdataset(dataset, featindex, splitval)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            news = errtype(mat0) + errtype(mat1)
            if news < bests:
                bestindex = featindex
                bestvalue = splitval
                bests = news
    if (s - bests) < tols:
        return None, leaftype(dataset)
    mat0, mat1 = binsplitdataset(dataset, bestindex, bestvalue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leaftype(dataset)
    return bestindex, bestvalue

def createtree(dataset, leaftype=regleaf, errtype=regerr, ops=(1, 4)):
    feat, val = choosebestsplit(dataset, leaftype, errtype, ops)
    if feat == None: return val
    rettree = {}
    rettree['spind'] = feat
    rettree['spval'] = val
    lset, rset = binsplitdataset(dataset, feat, val)
    rettree['left'] = createtree(lset, leaftype, errtype, ops)
    rettree['right'] = createtree(rset, leaftype, errtype, ops)
    return rettree


def istree(obj):
    return type(obj).__name__ == 'dict'


def getmean(tree):
    if istree(tree['right']):
        tree['right'] = getmean(tree['right'])
    if istree(tree['left']):
        tree['left'] = getmean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, testdata):
    if shape(testdata)[0] == 0:
        return getmean(tree)
    if istree(tree['left']) or istree(tree['right']):
        lset, rset = binsplitdataset(testdata, tree['spind'], tree['spval'])
    if istree(tree['left']):
        tree['left'] = prune(tree['left'], lset)
    if istree(tree['right']):
        tree['right'] = prune(tree['right'], rset)
    if not istree(tree['left']) and not istree(tree['right']):
        lset, rset = binsplitdataset(testdata, tree['spind'], tree['spval'])
        errornomerge = sum(power(lset[:, -1] - tree['left'], 2)) + sum(power(rset[:, -1] - tree['right'], 2))
        treemean = (tree['left'] + tree['right'])/2.0
        errormerge = sum(power(testdata[:, -1] - treemean, 2))
        if errormerge < errornomerge:
            print("merging")
            return treemean
        else:
            return tree
    else:
        return tree


def linearsolve(dataset):
    m, n = shape(dataset)
    x = mat(ones((m, n)))
    y = mat(ones((m, 1)))
    x[:, 1:n] = dataset[:, 0:n-1]
    y = dataset[:, -1]
    xTx = x.T*x
    if linalg.det[xTx] == 0.0:
        raise NameError("this matrix is singular, cannot do inverse, try increase the second value of ops")
    ws = xTx.I * (x.T * y)
    return ws, x, y


def modelleaf(dataset):
    ws, x, y = linearsolve(dataset)
    return ws


def modelerror(dataset):
    ws, x, y = linearsolve(dataset)
    yhat = x * ws
    return sum(power(y-yhat, 2))


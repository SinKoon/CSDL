from numpy import *


def loadsimpledata():
    datmat = matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classlables = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datmat, classlables


def stumpclassify(datamatrix, dimen, threshval, threshineq):
    retarray = ones((shape(datamatrix)[0], 1))
    if threshineq == 'lt':
        retarray[datamatrix[:, dimen] <= threshval] = -1.0
    else:
        retarray[datamatrix[:, dimen] > threshval] = -1.0
    return retarray


def buildstump(dataarr, classlabels, D):
    datamatrix = mat(dataarr)
    labelmat = mat(classlabels).T
    m, n = shape(datamatrix)
    numsteps = 10.0
    beststump = {}
    bestclaaest = mat(zeros((m, 1)))
    minerror = inf
    for i in range(n):
        rangemin = datamatrix[:, 1].min()
        rangemax = datamatrix[:, 1].max()
        stepsize = (rangemax - rangemin)/numsteps
        for j in range(-1, int(numsteps)+1):
            for inequal in ['lt', 'gt']:
                threshval = (rangemin + float(j) * stepsize)
                predictedvals = stumpclassify(datamatrix, i, threshval, inequal)
                errarr = mat(ones((m, 1)))
                errarr[predictedvals == labelmat] = 0
                weightederror = D.T * errarr
                if weightederror < minerror:
                    minerror = weightederror
                    bestclaaest = predictedvals.copy()
                    beststump['dim'] = i
                    beststump['thresh'] = threshval
                    beststump['ineq'] = inequal
    return beststump, minerror, bestclaaest


def adaboosttrainds(dataarr, classlabels, numit = 40):
    weakclassarr = []
    m = shape(dataarr)[0]
    D = mat(ones((m, 1))/m)
    aggclassest = mat(zeros((m, 1)))
    for i in range(numit):
        beststump, error, classest = buildstump(dataarr, classlabels, D)
        print("D:", D.T)
        alpha = float(0.5 * log((1.0 - error)/max(error, 1e-6)))
        beststump['alpha'] = alpha
        weakclassarr.append(beststump)
        print("classest:", classest.T)
        expon = multiply(-1*alpha*mat(classlabels).T, classest)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggclassest += alpha*classest
        print("aggclassest:", aggclassest.T)
        aggerrorss = multiply(sign(aggclassest) != mat(classlabels).T, ones((m, 1)))
        errorrate = aggerrorss.sum()/m
        print("total error:", errorrate, "\n")
        if errorrate == 0.0:
            break
    return weakclassarr, aggclassest


def adaclassify(dattoclass, classifierarr):
    datamatrix = mat(dattoclass)
    m = shape(datamatrix)[0]
    aggclassest = mat(zeros((m, 1)))
    for i in range(len(classifierarr)):
        classest = stumpclassify(datamatrix, classifierarr[i]['dim'], classifierarr[i]['thresh'], classifierarr[i]['ineq'])
        aggclassest += classifierarr[i]['alpha']*classest
        print(aggclassest)
    return sign(aggclassest)


def loaddataset(filename):
    numfeat = len(open(filename).readline().split('\t'))
    datamat = []
    labelmat = []
    fr = open(filename)
    for line in fr.readlines():
        linearr = []
        curline = line.strip().split('\t')
        for i in range(numfeat - 1):
            linearr.append(float(curline[i]))
        datamat.append(linearr)
        labelmat.append((float(curline[-1])))
    return datamat, labelmat


def plotroc(predstrengths, classlabels):
    import matplotlib.pyplot as plt
    cur = (1.0 ,1.0)
    ysum = 0.0
    numposclas = sum(array(classlabels) == 1.0)
    ystep = 1/float(numposclas)
    xstep = 1/float(len(classlabels) - numposclas)
    sortedindicies = predstrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedindicies.tolist()[0]:
        if classlabels[index] == 1.0:
            delx = 0
            dely = ystep
        else:
            dely = 0
            delx = xstep
            ysum += cur[1]
        ax.plot([cur[0], cur[0]-delx], [cur[1], cur[1]-dely], c='b')
        cur = (cur[0]-delx, cur[1]-dely)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve for adaboost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("area is:", ysum*xstep)
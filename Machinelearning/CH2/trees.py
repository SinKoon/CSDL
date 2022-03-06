from math import log
import operator


def calcshannonent(dataset):
    numentries = len(dataset)
    labelcounts = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys():
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1
        shannonent = 0.0
        for key in labelcounts:
            prob = float(labelcounts[key])/numentries
            shannonent -= prob * log(prob, 2)
            return shannonent


def createdataset():
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels


def splitdataset(dataset, axis, value ):
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducedfeatvec = featvec[:axis]
            reducedfeatvec.extend(featvec[axis+1:])
            retdataset.append(reducedfeatvec)
    return retdataset


def choosebestfeaturetosplit(dataset):
    numfeatures = len(dataset[0]) - 1
    baseentropy = calcshannonent(dataset)
    bestinfogain = 0.0
    bestfeature = -1
    for i in range(numfeatures):
        featlist = [example[i] for example in dataset]
        uniquevals = set(featlist)
        newentropy = 0.0
        for value in uniquevals:
            subdataset = splitdataset(dataset, i, value)
            prob = len(subdataset)/float(len(dataset))
            newentropy += prob * calcshannonent(subdataset)
        infogain = baseentropy - newentropy
        if infogain > bestinfogain:
            bestinfogain = infogain
            bestfeature = i
    return bestfeature


def majoritycnt(classlist):
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys(): classcount[vote] = 0
        classcount[vote] += 1
    sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


def createtree(dataset, labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:
        return majoritycnt(classlist)
    bestfeat = choosebestfeaturetosplit(dataset)
    bestfeatlabel = labels[bestfeat]
    mytree = {bestfeatlabel: {}}
    del(labels[bestfeat])
    featvalues = [example[bestfeat] for example in dataset]
    uniquevals = set(featvalues)
    for value in uniquevals:
        sublabels = labels[:]
        mytree[bestfeatlabel][value] = createtree(splitdataset(dataset, bestfeat, value), sublabels)
    return mytree


def classify(inputtree, featlabels, testvec):
    firststr = inputtree.keys()[0]
    seconddict = inputtree[firststr]
    featindex = featlabels.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex] == key:
            if type(seconddict[key]).__name__=='dict':
                classlabel = classify(seconddict[key], featlabels, testvec)
            else:classlabel = seconddict[key]
    return classlabel



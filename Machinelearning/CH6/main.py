from matplotlib import pyplot as plt
from numpy import *
from svmmlia import *
'''fr = open('testSet.txt')
datamat = []
labelmat = []
***for line in fr.readlines():
    linearr = line.strip().split('\t')
    datamat.append([float(linearr[0]), float(linearr[1])])
    labelmat.append(float(linearr[2]))
    datamat0 = array(datamat)
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []
for i in range(len(labelmat)):
    if labelmat[i] == 1:
        xcord1.append(datamat0[i, 0])
        ycord1.append(datamat0[i, 1])
    else:
        xcord2.append(datamat0[i, 0])
        ycord2.append(datamat0[i, 1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
ax.scatter(xcord2, ycord2, s=10, c='green')
plt.show()'''
dataarr, labelarr = loaddataset('testSet.txt')
b, alphas = smosimple(dataarr, labelarr, 0.6, 0.001, 40)
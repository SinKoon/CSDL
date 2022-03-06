from adaboost import *
import matplotlib.pyplot as plt


datmat, classlabels = loaddataset('horseColicTraining2.txt')
'''fig = plt.figure(frameon=0)
xcord1 = []; ycord1 = []
xcord2 = []; ycord2 = []
for i in range(shape(datmat)[0]):
    if classlabels[i] == 1.0:
        xcord1.append(datmat[i, 0])
        ycord1.append(datmat[i, 1])
    else:
        xcord2.append(datmat[i, 0])
        ycord2.append(datmat[i, 1])
ax = fig.add_subplot(111)
ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
ax.scatter(xcord2, ycord2, s=30, c='green', marker='s')
plt.show()
'''

'''classifierarray = adaboosttrainds(datmat, classlabels, 9)'''

'''cclass = []
ccclass = {}
ccclass['dim'] = 1
cclass.append(ccclass)
print(cclass)'''

classifierarr, aggclassest = adaboosttrainds(datmat, classlabels, 30)
#print(adaclassify([0, 0], classifierarr))
#print(adaclassify([[5, 5], [0, 0]], classifierarr))
plotroc(aggclassest.T, classlabels)

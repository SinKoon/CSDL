from regression import *
from numpy import *
import matplotlib.pyplot as plt

'''xarr, yarr = loaddataset('ex0.txt')'''
'''ws = standregres(xarr, yarr)
# print(ws)
xmat = mat(xarr)
ymat = mat(yarr)
yhat = xmat * ws'''
'''fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xmat[:, 1].flatten().A[0], ymat.T[:, 0].flatten().A[0])
xcopy = xmat.copy()
xcopy.sort(0)
yhat = xcopy * ws
ax.plot(xcopy[:, 1], yhat)
plt.show()'''
'''xmat = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(xmat)
print(xmat[:, 1])
print(xmat[:, 1].flatten())
print(xmat[:, 1].flatten().A)
print(xmat[:, 1].flatten().A[0])'''

# print(corrcoef(yhat.T, ymat))
'''yhat = lwlrtest(xarr, xarr, yarr, 0.03)
xmat = mat(xarr)
srtind = xmat[:, 1].argsort(0)
xsort = xmat[srtind][:, 0, :]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xmat[:, 1].flatten().A[0], mat(yarr).T[:, 0].flatten().A[0], s=2, c='red')
ax.plot(xsort[:, 1], yhat[srtind])
plt.show()
'''
'''abx, aby = loaddataset('abalone.txt')
ridgeweights = ridgetest(abx, aby)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ridgeweights)
plt.show()'''

a = [1, 1, 1]
print(nonzero(a[:] > 1))

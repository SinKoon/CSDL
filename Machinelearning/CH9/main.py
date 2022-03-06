from numpy import *
import regtrees
'''dataset = mat([1, 2, 1, 2, 3])
a = set(dataset.tolist()[0])
print(a)'''
mydat2 = regtrees.loaddataset('ex2.txt')
mymat2 = mat(mydat2)
mytree = regtrees.createtree(mymat2, ops=(0, 1))
print(mytree)
mydattest = regtrees.loaddataset('ex2test.txt')
mymat2test = mat(mydattest)
prunedtree = regtrees.prune(mytree, mymat2test)
print(prunedtree)

from logRegres import *

dataarr, labelmat = loaddataset()
weights = stocgradascent1(array(dataarr), labelmat,500)
plotbestfit(weights)

import matplotlib.pyplot as plt

decisionnode = dict(boxstyle="sawtooth", fc="0.8")
leafnode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotnode(nodetxt, centerpt, parentpt, nodetype):
    createplot.ax1.annotate(nodetxt, xy=parentpt, xycoords='axes fraction', xytext=centerpt, textcoored='axes fraction', va="center", ha="center", bbox=nodetype, arrowprops=arrow_args)


def createplot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createplot.axl = plt.subplot(111, frameon=False)
    plotnode('decision', (0.5, 0.1),(0.1, 0.5), decisionnode)
    plotnode('leaf', (0.8, 0.1), (0.3, 0.8), leafnode)
    plt.show()

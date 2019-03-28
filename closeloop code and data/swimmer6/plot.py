# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl
import matplotlib as mpl
from scipy import signal 
    
def perfcheck(nstart=0,nend=100,noisemax=100):
    y=np.array(np.loadtxt('perfcheck.txt'))
    perf=np.mean(y,axis=1)
    cstd=np.std(y,axis=1)
    step=noisemax/int(perf.shape[0]-1)
    sind=int(nstart/step)
    eind=int(nend/step)+1
    plt.figure(figsize=(12,9))
    f5,=plt.plot(np.arange(sind,(eind-1)*step+1,step),perf[sind:eind],'orange')
    plt.fill_between(np.arange(sind,(eind-1)*step+1,step),(perf[sind:eind]-cstd[sind:eind]),(perf[sind:eind]+cstd[sind:eind]),alpha=0.3,color='orange')
    plt.xlabel('noise/% of umax(std of noise)')
    plt.ylabel('L2-norm of terminal state error')
    plt.grid(True)
    plt.show()  
    print('averaged by {value1} rollouts'.format(value1=y.shape[1]))
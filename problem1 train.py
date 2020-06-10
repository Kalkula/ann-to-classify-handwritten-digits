import numpy as np
import matplotlib.pyplot as plt

a=np.loadtxt('featurestrain.txt')

m=a[:,0]
x=a[:,1]
y=a[:,2]


#plot
for i in range(10): 
    fig=plt.figure()
    plt.title(i)
    ax=fig.add_subplot(111)
    plt.xlabel('average intensity')
    plt.ylabel('symmetry')
    index1=np.where(m==i)
    index2=np.where(m!=i)
    p1=ax.scatter(x[index1],y[index1],marker='o',color='b')
    p2=ax.scatter(x[index2],y[index2],marker='x',color='r')
    

plt.show()  
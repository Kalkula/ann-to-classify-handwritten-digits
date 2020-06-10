import numpy as np
from sklearn.model_selection import KFold
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



np.set_printoptions(threshold=sys.maxsize)

a=np.loadtxt('featurestrain.txt')

#hidden units
l=100

#extract 1,5 rows

def extract(m,n):
    k0=a.shape[0]
    b=[0,0,0]
    for row in list(range(k0)):
        if a[row,0]==m:
            b=np.vstack((b,a[row,:]))       
        elif a[row,0]==n:
            b=np.vstack((b,a[row,:]))
    c=b[1:k0,:]
    with open('problem2 train.txt','w') as f:
        f.write('<%s>\n'%c)
    return c

c=extract(1,5)
dataSet=c[:,1:2]
hwlabels=c[:,0]

clf=MLPClassifier(hidden_layer_sizes=(l,),activation='logistic',
                   solver='adam',learning_rate_init=0.0001,max_iter=2000)



#3 fold cross validation
kf=KFold(n_splits=3)
pipe_lr=Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=1)),
                    ('clf', clf.fit(dataSet,hwlabels))
                    ])

scores = []
for k, (train, test) in enumerate(kf.split(c)):
    pipe_lr.fit(dataSet[train], hwlabels[train])
    score= pipe_lr.score(dataSet[test], hwlabels[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,np.bincount(np.array(hwlabels[train],dtype=int)), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('test error:',1-np.mean(scores))

#statictical errors
res=clf.predict(dataSet)
error_num =0  
num =len(dataSet)
for i in range(num):
    if res[i]!=hwlabels[i]:
        error_num+=1
print("Total num:",num,"Wrong num:",error_num," in-sample error:",error_num/float(num))




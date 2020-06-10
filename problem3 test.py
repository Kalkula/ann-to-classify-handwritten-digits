import numpy as np
from sklearn.model_selection import KFold
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



np.set_printoptions(threshold=sys.maxsize)

a=np.loadtxt('ziptest.txt')
#hidden units
l0=3
l1=2
l2=1
k1=a.shape[1]
dataSet=a[:,1:k1]
hwlabels=a[:,0]

clf=MLPClassifier(hidden_layer_sizes=(l0,l1,l2),activation='logistic',
                   solver='adam',learning_rate_init=0.0001,max_iter=20000)


kf=KFold(n_splits=3)
pipe_lr=Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=1)),
                    ('clf', clf.fit(dataSet,hwlabels))
                    ])

scores = []
res=clf.predict(dataSet)
error_num =0  
num =len(dataSet)
for k, (train, test) in enumerate(kf.split(a)):
    pipe_lr.fit(dataSet[train], hwlabels[train])
    score= pipe_lr.score(dataSet[test], hwlabels[test])
    scores.append(score)
    for i in range(num):
        if res[i]!=hwlabels[i]:
            error_num+=1
    print("Total num:",num,"Wrong num:",error_num," in-sample error:",error_num/float(num))   
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,np.bincount(np.array(hwlabels[train],dtype=int)), score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
print('test error:',1-np.mean(scores))




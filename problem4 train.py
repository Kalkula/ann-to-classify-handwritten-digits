import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



a=np.loadtxt('ziptest.txt')
k1=a.shape[1]
dataSet=a[:,1:k1]
hwlabels=a[:,0]
k=5



clf=MLPClassifier(hidden_layer_sizes=(50,100,20),activation='relu',
                   solver='lbfgs',learning_rate_init=0.0001,max_iter=5000)



#3 fold cross validation
kf=KFold(n_splits=k)
pipe_lr=Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=1)),
                    ('clf', clf.fit(dataSet,hwlabels))
                    ])

scores = []
for k, (train, test) in enumerate(kf.split(a)):
    pipe_lr.fit(dataSet[train], hwlabels[train])
    score= pipe_lr.score(dataSet[test], hwlabels[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1,np.bincount(np.array(hwlabels[train],dtype=int)), score))
    
print('\nCV accuracy: %.3f \nVarance: %.3f' % (np.mean(scores), np.var(scores)))


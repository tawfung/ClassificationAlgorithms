import numpy as np 
import pylab as pl 
from sklearn import neighbors, datasets

#import data
iris = datasets.load_iris()
X = iris.data[:,:2]  #we only take the first two features
Y = iris.target

h = .02 #step size in the mesh

knn = neighbors.KNeighborsClassifier()

#we create an instance of Neighbors Classification and fit the data
knn.fit(X, Y)

# plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min,y_max,h))
Z = knn.predict(np.c_[xx.ravel(),yy.ravel()])

# put the result into a color plot
Z = Z.reshape(xx.shape)
pl.figure(1, figsize=(4,3))
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)

# plot also the training points
pl.scatter(X[:,0],X[:,1], c=Y)
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(),xx.max())
pl.ylim(yy.min(),yy.max())
pl.xticks(())
pl.xticks(())

pl.show()
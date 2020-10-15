import scipy as sc
import numpy as np
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class RBF:
     
    def __init__(self, indim, numCenters, outdim, epochs, lr):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.epochs = epochs
        self.lr = lr
        self.W = sc.random.randn(self.numCenters, self.outdim)
        self.b = sc.random.randn(self.outdim, self.outdim)
         
    def _std(self):
        dMax = np.max([sc.absolute(c1 - c2) for c1 in self.centers for c2 in self.centers])
        self.std = sc.repeat(dMax / sc.sqrt(2*self.numCenters), self.numCenters)
    
    def _basisfunc(self, s, c, x):
        assert len(x) == self.indim
        return sc.exp(-1 / (2 * s**2) * norm(x - c)**2)
     
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = []
        
        return sc.array([self._basisfunc(s, c, X) for c, s in zip(self.centers, self.std)])
     
    def train(self, X, y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        self.centers = KMeans(n_clusters=self.numCenters).fit(X).cluster_centers_
        self._std()

         # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                
                # calculate activations of RBFs
                G = self._calcAct(X[i,:]).reshape(self.centers.shape[0], 1)

                res = sc.dot(self.W.T, G) + self.b

                # backward pass
                error = sc.sum(res) - y[i]

                # online update
                self.W = self.W - self.lr * G * error
                self.b = self.b - self.lr * error
         
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        res = G.T.dot(self.W) + self.b
        return sc.sum(res)


if __name__ == '__main__':
    df = pd.read_csv('/home/guilherme/workspace/neural_networks_class/wine/wine.data', names=["label","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"] )

    df = df.sample(frac=1).reset_index(drop=True)
    X = df.values[:, 1:]
    y = df.values[:, 0]

    rbf = RBF(indim=13, numCenters=14, outdim=1, epochs=25, lr=0.01)

    rbf.train(X, y)

    # Test fase
    preds = []
    for i in range(100):
        rnd = sc.random.randint(178, size=1)
        z_true = int(y[rnd[0]])
        print('Real answer: ' + str(z_true))    

        z = rbf.test(X[rnd[0],:])

        if z <= 1.5:
            r = 1
            preds.append(z_true == r)
            print('Prediction answer: %d => %.4f' % (r, z))
        elif z > 1.5 and z <= 2.3:
            r = 2
            preds.append(z_true == r)
            print('Prediction answer: %d => %.4f' % (r, z))
        elif z > 2.3:
            r = 3
            preds.append(z_true == r)
            print('Prediction answer: %d => %.4f' % (r, z))
    
    total = 0
    for i in preds:
        if i:
            total += 1

    print('Accuracy: %d%%' % total)
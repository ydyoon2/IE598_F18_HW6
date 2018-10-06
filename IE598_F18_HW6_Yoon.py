from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import StratifiedKFold

n_range = range(1,11)
for n in n_range:
    #datasets
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    
    #train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=n)
    
    #StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
   
    #ListedColormap
    def plot_decision_regions(X, y, classifier, test_idx=None,  
                              resolution=0.02):
    
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
    
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
    
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=colors[idx],
                        marker=markers[idx], label=cl, 
                        edgecolor='black')
    
        # highlight test samples
        if test_idx:
            # plot all samples
            X_test, y_test = X[test_idx, :], y[test_idx]
    
            plt.scatter(X_test[:, 0], X_test[:, 1],
                        c='', edgecolor='black', alpha=1.0,
                        linewidth=1, marker='o',
                        s=100, label='test set')
    
    #DecisionTreeClassifier
    
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=n)
    tree.fit(X_train, y_train)
    
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X_combined, y_combined,
                          classifier=tree, test_idx=range(105, 150))
    sys.stdout.write(" \n")
    print('random state =',n)
    plt.title('decision tree')
    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
    print('Training accuracy:', round(tree.score(X_train, y_train),3))
    print('Test accuracy:', round(tree.score(X_test, y_test),3))
    sys.stdout.write(" \n")


    #cross_val_scores with k-fold CV (k=10)
    kfold = StratifiedKFold(n_splits=10,
                        random_state=n).split(X_train, y_train)

    scores = []
    for k, (train, test) in enumerate(kfold):
        tree.fit(X_train[train], y_train[train])
        score = tree.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
              np.bincount(y_train[train]), score))
        
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    sys.stdout.write(" \n")
    #
    kfold = StratifiedKFold(n_splits=10,
                        random_state=n).split(X_train, y_train)

    scores = []
    for k, (train, test) in enumerate(kfold):
        tree.fit(X_train[test], y_train[test])
        score = tree.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
              np.bincount(y_train[test]), score))
        
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    sys.stdout.write(" \n")
    
###############################################################################
print("My name is {James Yoon}")
print("My NetID is: {ydyoon2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
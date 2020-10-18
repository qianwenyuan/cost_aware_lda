from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
import time
import numpy as np

iris = datasets.load_iris()

data, target = iris.data, iris.target
print(data)
print(data.shape[0])
ratio=0.3
x=data[:, :2]
x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=ratio, random_state=0)

def kernelSwitch(kernel):
    if kernel==2:
        return 'linear'
    elif kernel==3:
        return 'poly'
    elif kernel==4:
        return 'sigmoid'
    else:
        return 'rbf'

def svm_train(X, gamma=1/data.shape[0], max_iter=-1, decision_function_shape='ovr'):
    C = X[0]
    kernel = 1
    gamma = X[1]


    start = time.time()
    clf = svm.SVC(C=C,
                  kernel=kernelSwitch(int(kernel)),
                  gamma=gamma,
                  cache_size=200, 
                  degree=3,
                  max_iter=int(max_iter),
                  decision_function_shape=decision_function_shape)

    acc_sum = 0.0
    for i in range(5):
        '''
        index = np.arange(num)
        np.random.shuffle(index)
        x_test = x[index[:num_test],:] 
        y_test = y[index[:num_test]]
        x_train = x[index[num_test:],:] 
        y_train = y[index[num_test:]]
        '''

        clf.fit(x_train, y_train)

        y_test_pre = clf.predict(x_test)

        acc_sum += clf.score(x_test, y_test)

    end = time.time()
    print(end-start)
    return -acc_sum / 5.0

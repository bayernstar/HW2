import numpy as np
from scipy import misc
from matplotlib import pylab as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.multiclass import OneVsRestClassifier as OVR

train_labels, train_data = [], []
for line in open('./faces/train.txt'):
    im = misc.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)

test_labels, test_data = [], []
for line in open('./faces/test.txt'):
    im = misc.imread(line.strip().split()[0])
    test_data.append(im.reshape(2500,))
    test_labels.append(line.strip().split()[1])
test_data, test_labels = np.array(test_data, dtype=float), np.array(test_labels, dtype=int)

avgtrain = np.zeros(2500)
for i in range(2500):
    avgtrain[i] = np.mean(train_data[:,i])
avgtest = np.zeros(2500)
for i in range(2500):
    avgtest[i] = np.mean(test_data[:,i])
mstrain = []
for i in range(len(train_data)):
    mstrain.append(train_data[i] - avgtrain)
mstest = []
for i in range(len(test_data)):
    mstest.append(test_data[i] - avgtest)

U, s, V = np.linalg.svd(mstrain)


def eigenface_feature(train_data, test_data, V, r):
    return np.dot(train_data, V[:r,:].T), np.dot(test_data, V[:r,:].T)


accuracy = []
for r in range(1, 201):
    F, F_test = eigenface_feature(mstrain, mstest, V, r)
    ovr = OVR(LR()).fit(F, train_labels)
    accuracy.append(ovr.score(F_test, test_labels))
plt.plot(accuracy)
plt.savefig("1h.png")
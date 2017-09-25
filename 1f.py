import numpy as np
from scipy import misc
from matplotlib import pylab as plt


train_labels, train_data = [], []
for line in open('./faces/train.txt'):
    im = misc.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)

avgtrain = np.zeros(2500)
for i in range(2500):
    avgtrain[i] = np.mean(train_data[:,i])
mstrain = []
for i in range(len(train_data)):
    mstrain.append(train_data[i] - avgtrain)

U, s, V = np.linalg.svd(mstrain)

e = []
for r in range(200):
    xr = U[:,:r].dot(np.diag(s[:r])).dot(V[:r,:])
    diff = np.linalg.norm(xr - mstrain)
    e.append(diff)

plt.plot(e)
plt.savefig("1f.png")
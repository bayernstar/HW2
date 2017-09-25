import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm


train_labels, train_data = [], []
for line in open('./faces/train.txt'):
    im = misc.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])
train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)

avg = np.zeros(2500)
for i in range(2500):
    avg[i] = np.mean(train_data[:,i])
plt.imshow(avg.reshape(50,50), cmap = cm.Greys_r)
plt.show()
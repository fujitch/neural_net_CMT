# -*- coding: utf-8 -*-


import pickle
import math
import numpy as np
from scipy.fftpack import fft
import chainer
import chainer.functions as F
import matplotlib.pyplot as plt
import random

weight2 = model.l2.W.data
weight1 = model.l1.W.data
vec = model.l3.W.data[2, :]
mapping = np.dot(weight1.T, np.dot(weight2.T, vec))
xx = range(100)
plt.plot(xx, mapping)

"""
weight2 = model.conv2.W.data
weight1 = model.conv1.W.data

for wnum in range(50):
    w = weight2[wnum, :, :, 0]
    
    mapping = np.zeros((20, 10))
    mapping = np.array(mapping, dtype=np.float32)
    for i in range(5):
        mapping[:, 2*i] = w[:, i]
        
    mapping2 = np.zeros((14))
    mapping2 = np.array(mapping2, dtype=np.float32)
    
    for k in range(20):
        ww = weight1[k, 0, :, 0]
        for l in range(10):
            mapping2[l:l+5] += ww*mapping[k, l]
    xx = range(14)
    plt.plot(xx, mapping2)
    fname = 'allweight.png'
    plt.savefig(fname)

"""

"""
xp = np
dropout_ratio = 0.2

with open('test_dataset_0706.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

# fftしてRMSを用いて短冊化する
def processing_data(sample):
    sample = sample/(calRms(sample))
    sample = fft(sample)
    sample = abs(sample)
    new = np.zeros((100))
    new = xp.array(new, dtype=xp.float32)
    for i in range(100):
        new[i] = calRms(sample[i:i+10])
    return new

# バッチ作成(test用)
def make_batch_test(dataset, code):
    batch = xp.zeros((1, 100))
    batch = xp.array(batch, dtype=xp.float32)
    index = random.randint(0, 18999)
    sample = dataset[index:index+1000, code]
    sample = np.array(sample, dtype=xp.float32)
    sample = processing_data(sample)
    batch[0, :] = sample
    return batch

fname = 'dnn_0707_model_second.pkl'
model = pickle.load(open(fname, 'rb'))
for i in range(24):
    batch = make_batch_test(test_dataset, i)
    x = chainer.Variable(batch)
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(F.dropout(h1, ratio=dropout_ratio, train=False)))
    y = model.l3(F.dropout(h2, ratio=dropout_ratio, train=False))
    # y = F.softmax(y).data
    plt.imshow(y.data)
    fname = 'out_' + str(i)
    plt.savefig(fname)


xp = np

with open('test_dataset.pkl', 'rb') as f:
    normal_dataset = pickle.load(f)

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

# fftしてRMSを用いて短冊化する
def processing_data(sample):
    sample = sample/(calRms(sample))
    sample = fft(sample)
    sample = abs(sample)
    new = np.zeros((100))
    new = xp.array(new, dtype=xp.float32)
    for i in range(100):
        new[i] = calRms(sample[i:i+10])
    return new

# バッチ作成
def make_batch(dataset, batchSize, num):
    batch = xp.zeros((batchSize, 1, 100, 32))
    batch = xp.array(batch, dtype=xp.float32)
    for i in range(batchSize):
        sample = dataset[1000:2000, num]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, 0, :, 0] = sample
    return batch

fname = 'model/cnn_auto_encoder_model9900.pkl'
model = pickle.load(open(fname, 'rb'))
batch = make_batch(normal_dataset, 1, 158)
x = chainer.Variable(batch)
t = x
h = F.relu(model.conv1(x))
h = F.max_pooling_2d(h, ksize=2, cover_all=False)
h = F.relu(model.conv2(h))

xx = range(100)
plt.plot(xx, batch[0,0,:,0])

"""

"""

h = F.max_pooling_2d(h, ksize=2, cover_all=False)
h = F.relu(model.conv3(h))
h = F.relu(model.deconv1(h))
h = F.unpooling_2d(h, ksize=2, cover_all=False)
h = F.relu(model.deconv2(h))
h = F.unpooling_2d(h, ksize=2, cover_all=False)
y = F.relu(model.deconv3(h))

xx = range(100)
plt.plot(xx, y.data[0, 0, :, 0])



import pickle
import math
import numpy as np
from scipy.fftpack import fft
import chainer
import chainer.functions as F
import matplotlib.pyplot as plt

xp = np
dropout_ratio = 0.2

with open('test_dataset.pkl', 'rb') as f:
    normal_dataset = pickle.load(f)

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

# fftしてRMSを用いて短冊化する
def processing_data(sample):
    sample = sample/(calRms(sample))
    sample = fft(sample)
    sample = abs(sample)
    new = np.zeros((100))
    new = xp.array(new, dtype=xp.float32)
    for i in range(100):
        new[i] = calRms(sample[i:i+10])
    return new


# バッチ作成
def make_batch(dataset, batchSize):
    batch = xp.zeros((batchSize, 100))
    batch = xp.array(batch, dtype=xp.float32)
    for i in range(batchSize):
        sample = dataset[1000:2000, 14]
        sample = np.array(sample, dtype=xp.float32)
        sample = processing_data(sample)
        batch[i, :] = sample
    return batch

fname = 'model/autoencodermodel9900.pkl'
model = pickle.load(open(fname, 'rb'))
batch = make_batch(normal_dataset, 1)
x = chainer.Variable(batch)
t = x
x = model.l1(x)
x = model.l2(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.l3(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.d3(F.dropout(x, ratio=dropout_ratio, train=False))
x = model.d2(F.dropout(x, ratio=dropout_ratio, train=False))
y = model.d1(F.dropout(x, ratio=dropout_ratio, train=False))

xx = range(100)
plt.plot(xx, y.data[0, :])


plt.plot(xx, y.data[0, 0, :, 0])
plt.ylim(8, 72)
"""
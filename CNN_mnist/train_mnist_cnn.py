# -*- coding: utf-8 -*-
#!/usr/bin/env python
from __future__ import print_function
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

import matplotlib.pyplot as plt

def disp_conv1():
    plt.gray()
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.title("%d"%(i+1), size=8)
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        Z = model.predictor.conv1.W[i].data[0]
        plt.imshow(Z)
    plt.show()


def disp_conv2():
    plt.gray()
    for i in range(32):
        plt.subplot(4,8,i+1)
        plt.title("%d"%(i+1), size=8)
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")
        Z = model.predictor.conv2.W[i].data[0]
        plt.imshow(Z)
    plt.show()

class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        w = I.Normal(scale=0.05) # モデルパラメータの初期化
        super(MLP, self).__init__(
            conv1=L.Convolution2D(1, 16, 5, 1, 0), # 1層目の畳み込み層（フィルタ数は16）
            conv2=L.Convolution2D(16, 32, 5, 1, 0), # 2層目の畳み込み層（フィルタ数は32）
            l3=L.Linear(None, n_out, initialW=w), #クラス分類用
        )
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2)
        y = self.l3(h2)
        return y

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=20, help='Number of sweeps over the dataset to train')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
parser.add_argument('--unit', '-u', type=int, default=1000, help='Number of units')
args = parser.parse_args()

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

train, test = chainer.datasets.get_mnist(ndim=3) # ndim=3を引数で与えるだけでOK
model = L.Classifier(MLP(args.unit, 10), lossfun=F.softmax_cross_entropy)
if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()
    model.to_gpu()
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)

trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
trainer.extend(extensions.LogReport())
trainer.extend( extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
trainer.extend( extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.ProgressBar())

if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

trainer.run()
model.to_cpu()

modelname = args.out + "/MLP.model"
print('save the trained model: {}'.format(modelname))
chainer.serializers.save_npz(modelname, model)

# ニューラルネットワークの前向き計算
# まずは，ひとつだけ計算
#
# by oeda

import numpy as np

# パラメータ
EPSILON = 4.0

def sigmoid(x):
    return 1/(1+np.exp(-1*EPSILON*x))

xb = (0.1*0.0) + (0.2*0.0) + (-0.5)
print(xb)
ob = sigmoid(xb)
print(ob)

print()
print("ここからはもう少しプログラミングっぽく")
# トレーニングデータ
## 入力
inputs = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
## 教師信号
teach = [0,1,1,0]

# 初期重み
wab =  0.8
wac = -0.7
wbd =  0.1
wbe =  0.2
wcd = -0.3
wce =  0.4
offa = 0.4
offb =-0.5
offc = 0.6

# 入力層
p = 0
outd = inputs[p][0]
oute = inputs[p][1]

# 中間層
xb = wbd * outd + wbe * oute + offb
print(xb)
outb = sigmoid(xb)
print(outb)

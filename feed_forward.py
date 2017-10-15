# ニューラルネットワークの前向き計算
# 練習用のため拡張性がないアホアホプログラミング
#
# by oeda

import numpy as np

# パラメータ
EPSILON = 4.0

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-1*EPSILON*x))

# トレーニングデータ
## 入力
inputs = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
## 教師信号
teach = [0,1,1,0]

# 初期重みをランダムに与える
wab = (np.random.rand()-0.5)*2 * 0.3 # -0.3から0.3の一様乱数
wac = (np.random.rand()-0.5)*2 * 0.3 
wbd = (np.random.rand()-0.5)*2 * 0.3 
wbe = (np.random.rand()-0.5)*2 * 0.3 
wcd = (np.random.rand()-0.5)*2 * 0.3 
wce = (np.random.rand()-0.5)*2 * 0.3 
offa = (np.random.rand()-0.5)*2 * 0.3
offb = (np.random.rand()-0.5)*2 * 0.3
offc = (np.random.rand()-0.5)*2 * 0.3


# 初期重み（練習用）
'''
wab =  0.8
wac = -0.7
wbd =  0.1
wbe =  0.2
wcd = -0.3
wce =  0.4
offa = 0.4
offb =-0.5
offc = 0.6 
'''

# 初期重み（XORを学習済み）
'''
wab =  2.743674
wac = -2.778689
wbd = -1.310456
wbe = -1.312891
wcd = -1.691827
wce = -1.704581
offa = -1.312300
offb =  1.956707
offc =  0.710779
'''


# 全パターン（XORは4つ）でループ
for p in range(len(inputs)):

    # 前向き計算
    
    # 入力層
    outd = inputs[p][0]
    oute = inputs[p][1]

    # 中間層
    xb = wbd * outd + wbe * oute + offb
    outb = sigmoid(xb)
    
    xc = wcd * outd + wce * oute + offc
    outc = sigmoid(xc)

    # 出力層
    xa = wab * outb + wac * outc + offa
    outa = sigmoid(xa)

    # 誤差計算
    error = (outa-teach[p])**2
    print(teach[p], outa, error)
    

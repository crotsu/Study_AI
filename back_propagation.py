# ニューラルネットワークのBP法による学習
# 練習用のため拡張性がないアホアホプログラミング
#
# by oeda

import numpy as np
import matplotlib.pyplot as plt

# パラメータ
EPSILON = 4.0
ETA = 0.1
TIME = 1000

def sigmoid(x):
    return 1/(1+np.exp(-1*EPSILON*x))


inputs = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
teach = [0,1,1,0]

wab = (np.random.rand()-0.5)*2 * 0.3 # -0.3から0.3の一様乱数
wac = (np.random.rand()-0.5)*2 * 0.3
wbd = (np.random.rand()-0.5)*2 * 0.3
wbe = (np.random.rand()-0.5)*2 * 0.3
wcd = (np.random.rand()-0.5)*2 * 0.3
wce = (np.random.rand()-0.5)*2 * 0.3
offa = (np.random.rand()-0.5)*2 * 0.3
offb = (np.random.rand()-0.5)*2 * 0.3
offc = (np.random.rand()-0.5)*2 * 0.3

x = []
y = []
for t in range(TIME):

    errorAll = 0.0
    for p in range(len(inputs)):

        ##########
        # 前向き計算
        ##########

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

        error = (outa-teach[p])**2
        print(teach[p], outa, error)

        errorAll += error

        ##################
        # Back Propagation
        ##################

        deltaa = (outa-teach[p]) * EPSILON * (1.0-outa) * outa
        deltab = deltaa * wab * EPSILON * (1.0-outb) * outb
        deltac = deltaa * wac * EPSILON * (1.0-outc) * outc

        wab = wab - ETA * deltaa * outb
        wac = wac - ETA * deltaa * outc
        offa = offa - ETA * deltaa

        wbd = wbd - ETA * deltab * outd
        wbe = wbe - ETA * deltab * oute
        offb = offb - ETA * deltab

        wcd = wcd - ETA * deltac * outd
        wce = wce - ETA * deltac * oute
        offc = offc - ETA * deltac

    print(errorAll)
    print()

    # グラフ表示用の変数
    x.append(t)
    y.append(errorAll)

# グラフ表示
# 点どうしを直線でつなぐ
plt.plot(x, y)
# 適切な表示範囲を指定
ymin = 0.0
ymax = y[0]
plt.ylim(ymin, ymax)
# グリッド追加
plt.grid(True)
# 表示
plt.show()

# ニューラルネットワークのBP法による学習
# ベクトルと行列を用いたプログラム
#
# by oeda

import numpy as np
import matplotlib.pyplot as plt

# パラメータ
EPSILON = 4.0
ETA = 0.1
TIME = 1000

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
np.random.seed(6)
weight1 = (np.random.rand(2, 2)-0.5)*2 * 0.3 # -0.3から0.3の一様乱数
weight2 = (np.random.rand(1, 2)-0.5)*2 * 0.3
offset1 = (np.random.rand(2)-0.5)*2 * 0.3
offset2 = (np.random.rand(1)-0.5)*2 * 0.3

x = []
y = []
# 学習
for t in range(TIME):
    
    errorAll = 0.0
    out = []
    # 各パターンを提示
    for p in range(len(inputs)):
        # 前向き計算
        out1 = sigmoid(np.dot(weight1, inputs[p])+offset1)
        out2 = sigmoid(np.dot(weight2, out1)+offset2)
        out.append(out2)
        errorAll += (out2-teach[p])**2
        
        # BP
        delta2 = (out2-teach[p])*EPSILON*out2*(1.0-out2)
        weight2 -= ETA*delta2*out1
        offset2 -= ETA*delta2

        delta1 = EPSILON*out1*(1.0-out1)*delta2*weight2
        weight1 -= ETA*delta1*inputs[p]
        offset1 -= ETA*delta1[0]
    print(errorAll)
    
    # グラフ表示用の変数
    x.append(t)
    y.append(errorAll)
    
print('output')
print(out)

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

# 魚データを学習する
#
# by oeda

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# パラメータ
EPSILON = 4.0
ETA = 0.01
TIME = 10000

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-1*EPSILON*x))


# トレーニングデータ
## 入力
# データを読み込む
df_trainA = pd.read_csv('FishData/fishA.train', sep=' ', names=['x','y'])
df_trainB = pd.read_csv('FishData/fishB.train', sep=' ', names=['x','y'])

# DataFrameを結合
df_trainA = df_trainA.append(df_trainB)
df_trainA = df_trainA.reset_index(drop=True)

# pandasのDataFrameをリストに変換
train_inputs= df_trainA.values.tolist()

## 教師信号
train_teach = []
for i in range(200):
    if i<100:
        train_teach.append(0)
    else:
        train_teach.append(1)


# 初期重みをランダムに与える
weight1 = (np.random.rand(2, 2)-0.5)*2 * 0.3 # -0.3から0.3の一様乱数
weight2 = (np.random.rand(1, 2)-0.5)*2 * 0.3
offset1 = (np.random.rand(2)-0.5)*2 * 0.3
offset2 = (np.random.rand(1)-0.5)*2 * 0.3

# モーメント法で前回の修正量を保存するための重みと閾値
m_weight1 = np.zeros((2,2))
m_weight2 = np.zeros((1,2))
m_offset1 = np.zeros(2)
m_offset2 = np.zeros(1)

x = []
y = []
# 学習
for t in range(TIME):

    # 一括修正法で修正量を蓄積するための重みと閾値
    p_weight1 = np.zeros((2,2))
    p_weight2 = np.zeros((1,2))
    p_offset1 = np.zeros(2)
    p_offset2 = np.zeros(1)

    errorAll = 0.0
    out = []
    # 各パターンを提示
    for p in range(len(train_inputs)):
        # 前向き計算
        out1 = sigmoid(np.dot(weight1, train_inputs[p])+offset1)
        out2 = sigmoid(np.dot(weight2, out1)+offset2)
        out.append(out2)
        errorAll += (out2-train_teach[p])**2

        # BP
        delta2 = (out2-train_teach[p])*EPSILON*out2*(1.0-out2)
        delta1 = EPSILON*out1*(1.0-out1)*delta2*weight2

        p_weight2 += - ETA*delta2*out1
        p_offset2 += - ETA*delta2

        p_weight1 += - ETA*delta1*train_inputs[p]
        p_offset1 += - ETA*delta1[0]

    # 一括修正+モーメント法
    weight1 = weight1 + p_weight1 + m_weight1
    weight2 = weight2 + p_weight2 + m_weight2
    offset1 = offset1 + p_offset1 + m_offset1
    offset2 = offset2 + p_offset2 + m_offset2

    # 次回のモーメント法で使うために，今回の修正量を保存
    m_weight1 = p_weight1
    m_weight2 = p_weight2
    m_offset1 = p_offset1
    m_offset2 = p_offset2

    if t%100==0:
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


out_data = pd.concat([df_trainA, pd.DataFrame(out)], axis=1)
out_data.to_csv('output.dat', sep=' ', header=None, index=False)

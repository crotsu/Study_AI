# 2次元プロットデータ（2クラス）
# 表示

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
dfAtrain = pd.read_csv('fishA.train', sep=' ', header=None)
dfBtrain = pd.read_csv('fishB.train', sep=' ', header=None)
dfAtest = pd.read_csv('fishA.test', sep=' ', header=None)
dfBtest = pd.read_csv('fishB.test', sep=' ', header=None)

# 散布図をプロットする
for i in range(len(dfAtrain)):
    plt.scatter(dfAtrain[0][i],dfAtrain[1][i], color='r',marker='o', s=30)
for i in range(len(dfBtrain)):
    plt.scatter(dfBtrain[0][i],dfBtrain[1][i], color='b',marker='x', s=30)

for i in range(len(dfAtest)):
    plt.scatter(dfAtest[0][i],dfAtest[1][i], color='g',marker='o', s=30)
for i in range(len(dfBtest)):
    plt.scatter(dfBtest[0][i],dfBtest[1][i], color='y',marker='x', s=30)

    
# グリッド表示
plt.grid(True)

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)


# 表示
plt.show()

# 2次元プロットデータ（2クラス）
# 表示

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
dfA = pd.read_csv('fishA.train', sep=' ', header=None)
dfB = pd.read_csv('fishB.train', sep=' ', header=None)

# 散布図をプロットする
for i in range(len(dfA)):
    plt.scatter(dfA[0][i],dfA[1][i], color='r',marker='o', s=30)
for i in range(len(dfB)):
    plt.scatter(dfB[0][i],dfB[1][i], color='b',marker='x', s=30)

# グリッド表示
plt.grid(True)

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)


# 表示
plt.show()

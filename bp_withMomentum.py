# ニューラルネットワークのBP法による学習
# 練習用のため拡張性がないアホアホプログラミング
#
# 逐次修正法，一括修正法，モーメント法
# 1エポックは全4パターンとする．
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


wab_original = (np.random.rand()-0.5)*2 * 0.3 # -0.3から0.3の一様乱数
wac_original = (np.random.rand()-0.5)*2 * 0.3
wbd_original = (np.random.rand()-0.5)*2 * 0.3
wbe_original = (np.random.rand()-0.5)*2 * 0.3
wcd_original = (np.random.rand()-0.5)*2 * 0.3
wce_original = (np.random.rand()-0.5)*2 * 0.3
offa_original = (np.random.rand()-0.5)*2 * 0.3
offb_original = (np.random.rand()-0.5)*2 * 0.3
offc_original = (np.random.rand()-0.5)*2 * 0.3

# 逐次修正法(sequential)
wab = wab_original
wac = wac_original
wbd = wbd_original
wbe = wbe_original
wcd = wcd_original
wce = wce_original
offa = offa_original
offb = offb_original
offc = offc_original

s_x = []
s_y = []
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
    s_x.append(t)
    s_y.append(errorAll)


# 一括修正法(package)
wab = wab_original
wac = wac_original
wbd = wbd_original
wbe = wbe_original
wcd = wcd_original
wce = wce_original
offa = offa_original
offb = offb_original
offc = offc_original

p_x = []
p_y = []
for t in range(TIME):

    # 一括修正法で修正量を蓄積するための重みと閾値
    p_wab = 0.0
    p_wac = 0.0
    p_wbd = 0.0
    p_wbe = 0.0
    p_wcd = 0.0
    p_wce = 0.0
    p_offa = 0.0
    p_offb = 0.0
    p_offc = 0.0
        
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

        p_wab += - ETA * deltaa * outb
        p_wac += - ETA * deltaa * outc
        p_offa += - ETA * deltaa

        p_wbd += - ETA * deltab * outd
        p_wbe += - ETA * deltab * oute
        p_offb += - ETA * deltab

        p_wcd += - ETA * deltac * outd
        p_wce += - ETA * deltac * oute
        p_offc += - ETA * deltac

    # 一括修正
    wab = wab + p_wab
    wac = wac + p_wac
    offa = offa + p_offa
    
    wbd = wbd + p_wbd
    wbe = wbe + p_wbe
    offb = offb + p_offb
    
    wcd = wcd + p_wcd
    wce = wce + p_wce
    offc = offc + p_offc
    
    print(errorAll)
    print()

    # グラフ表示用の変数
    p_x.append(t)
    p_y.append(errorAll)


    
# 一括修正法+モーメント法(momentum)
wab = wab_original
wac = wac_original
wbd = wbd_original
wbe = wbe_original
wcd = wcd_original
wce = wce_original
offa = offa_original
offb = offb_original
offc = offc_original

m_x = []
m_y = []

# モーメント法で前回の修正量を保存するための重みと閾値
m_wab = 0.0
m_wac = 0.0
m_wbd = 0.0
m_wbe = 0.0
m_wcd = 0.0
m_wce = 0.0
m_offa = 0.0
m_offb = 0.0
m_offc = 0.0
    
for t in range(TIME):

    # 一括修正法で修正量を蓄積するための重みと閾値
    p_wab = 0.0
    p_wac = 0.0
    p_wbd = 0.0
    p_wbe = 0.0
    p_wcd = 0.0
    p_wce = 0.0
    p_offa = 0.0
    p_offb = 0.0
    p_offc = 0.0
    
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

        p_wab += - ETA * deltaa * outb
        p_wac += - ETA * deltaa * outc
        p_offa += - ETA * deltaa

        p_wbd += - ETA * deltab * outd
        p_wbe += - ETA * deltab * oute
        p_offb += - ETA * deltab

        p_wcd += - ETA * deltac * outd
        p_wce += - ETA * deltac * oute
        p_offc += - ETA * deltac

    # 一括修正+モーメント法
    wab = wab + p_wab + m_wab
    wac = wac + p_wac + m_wac 
    offa = offa + p_offa + m_offa
    
    wbd = wbd + p_wbd + m_wbd
    wbe = wbe + p_wbe + m_wbe
    offb = offb + p_offb + m_offb
    
    wcd = wcd + p_wcd + m_wcd
    wce = wce + p_wce + m_wce
    offc = offc + p_offc + m_offc

    # 次回のモーメント法で使うために，今回の修正量を保存
    m_wab = p_wab
    m_wac = p_wac
    m_wbd = p_wbd
    m_wbe = p_wbe
    m_wcd = p_wcd
    m_wce = p_wce
    m_offa = p_offa
    m_offb = p_offb
    m_offc = p_offc
    
    print(errorAll)
    print()

    # グラフ表示用の変数
    m_x.append(t)
    m_y.append(errorAll)

    

# グラフ表示
# 点どうしを直線でつなぐ
plt.plot(s_x, s_y, label='Sequential')
plt.plot(p_x, p_y, label='Package')
plt.plot(m_x, m_y, label='Momentum')

# 適切な表示範囲を指定
ymin = 0.0
ymax = s_y[0]
plt.ylim(ymin, ymax)
# グリッド追加
plt.grid(True)
# 凡例追加
plt.legend()
# 表示
plt.show()

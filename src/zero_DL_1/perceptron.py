import numpy as np

# パーセプトロンの実装
# ANDゲート
def AND(x1, x2):
    x = np.array([x1, x2]) # 入力
    w = np.array([0.5, 0.5]) # 重み
    b = -0.7 # バイアス
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# NANDゲート
def NAND(x1, x2):
    x = np.array([x1, x2]) # 入力
    w = np.array([-0.5, -0.5]) # 重み
    b = 0.7 # バイアス
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# ORゲート
def OR(x1, x2):
    x = np.array([x1, x2]) # 入力
    w = np.array([0.5, 0.5]) # 重み
    b = -0.2 # バイアス
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# XORゲート
# AND,NAND,ORの組み合わせ
# 2層のパーセプトロン
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0,0)) #0を出力
print(XOR(1,0)) #1を出力
print(XOR(0,1)) #1を出力
print(XOR(1,1)) #0を出力

# print(OR(0,0)) #0を出力
# print(OR(1,0)) #1を出力
# print(OR(0,1)) #1を出力
# print(OR(1,1)) #1を出力

# print(NAND(0,0)) #1を出力
# print(NAND(1,0)) #1を出力
# print(NAND(0,1)) #1を出力
# print(NAND(1,1)) #0を出力

# print(AND(0,0)) #0を出力
# print(AND(1,0)) #0を出力
# print(AND(0,1)) #0を出力
# print(AND(1,1)) #1を出力
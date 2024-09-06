# 活性化関数
# libraryのインポート
import numpy as np
import matplotlib.pylab as plt

# # ステップ関数

# def step_function(x):
#     # y = x > 0
#     # return y.astype(np.int)
#     return np.array(x > 0, dtype=np.integer)

# # -0.5から5.0までの範囲を0.1刻み
# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x,y)
# plt.ylim(-0.1, 1.1) # y軸の範囲を設定
# plt.show()

# シグモイド関数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # y軸の範囲を設定
# plt.show()

# ReLU(Rectified Linear Unit)

# def relu(x):
#     return np.maximum(0, x)
# x = np.arange(-5.0, 5.0, 0.1)
# y = relu(x)
# plt.plot(x, y)
# # plt.ylim(-0.1, 1.1) # y軸の範囲を設定
# plt.show()

# 多次元配列の計算

# ## 行列の積
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[5, 6], [7, 8]])
# print(np.dot(A, B))

# ## NNの行列積
# X = np.array([1, 2])
# print(X.shape)

# W = np.array([[1, 3, 5], [2, 4, 6]])
# print(W.shape)

# Y = np.dot(X, W)
# print(Y)

# # 入力層 → 第1層

# X = np.array([1.0, 0.5])
# W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
# B1 = np.array([0.1, 0.2, 0.3])

# # print(X.shape) # (2,)
# # print(W1.shape) # (2, 3)
# # print(B1.shape) # (3,)

# A1 = np.dot(X, W1) + B1
# Z1 = sigmoid(A1)

# # 第1層 → 第2層
# W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
# B2 = np.array([0.1, 0.2])

# # print(Z1.shape) # (3,)
# # print(W2.shape) # (3, 2)
# # print(B2.shape) # (2,)

# A2 = np.dot(Z1, W2) + B2
# Z2 = sigmoid(A2)

# W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
# B3 = np.array([0.1, 0.2])

# A3 = np.dot(Z2, W3) + B3
# Y = A3
# print(Y)

# # 関数化
# ## 重みとバイアスの初期化
# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#     network['b3'] = np.array([0.1, 0.2])

#     return network

# ## 入力から出力方向への伝達処理
# def forward(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']

#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = a3

#     return y

# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y) # [0.31682708 0.69627909]

# # ソフトマックス関数の実装
# a = np.array([0.3, 2.9, 4.0])
# exp_a = np.exp(a) # 指数関数
# print(exp_a) # [ 1.34985881 18.17414537 54.59815003]
# sum_exp_a = np.sum(exp_a) # 指数関数の和
# print(sum_exp_a) # 74.1221542101633
# y = exp_a / sum_exp_a 
# print(y) # [0.01821127 0.24519181 0.73659691]

# a = np.array([1010, 1000, 990])
# np.exp(a) / np.sum(np.exp(a)) # ソフトマックス関数の計算
# # 正しく計算されない オーバーフロー
# c = np.max(a) # 1010
# np.exp(a - c) / np.sum(np.exp(a - c))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y) # 0~1
print(np.sum(y)) # 総和は1
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

# # シグモイド関数
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # y軸の範囲を設定
# plt.show()

# ReLU(Rectified Linear Unit)

def relu(x):
    return np.maximum(0, x)
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # y軸の範囲を設定
plt.show()
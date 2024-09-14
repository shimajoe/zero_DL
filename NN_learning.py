# モジュールのインポート
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

# 損失関数
## 2乗和誤差
def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

## 交差エントロピー誤差(バッチ対応版)
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # delta = 1e-7 # 微小な値を足しておく
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

# ミニバッチ処理
train_size = x_train.shape[0] # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # np.random.choice(X, Y) 0以上X未満からランダムにY個数字を選び出す
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# 




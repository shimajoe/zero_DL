import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image # 画像出力のライブラリ

# ニューラルネットワークの推論処理

## dataをgetする
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

## 初期値
def init_network():
    # pickleファイルのsample_weight.pklに保存された学習済みの重みパラメタを読み込む
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

## シグモイド関数(推論時使用)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

## ソフトマックス関数(推論時使用)
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

## 推論
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100 # バッチ数
accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
#     if p == t[i]:
#         accuracy_cnt += 1

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
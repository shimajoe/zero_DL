# クラスの書き方
# 新しいクラスの定義
class Man:
    # Manクラスのコンストラクタ(初期化メソッド)
    # インスタンス生成後
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

# インスタンス(オブジェクト)を生成
m = Man("David")
m.hello()
m.goodbye()
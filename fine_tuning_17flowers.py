"""
vgg16……
    「ImageNet」と呼ばれる大規模画像データセットで学習された，16層からなるCNNモデル……
に、 17flowers のデータを加えて、 seventeen flowers の分類を可能する……
    それが fine tuning……
よ。
参考: https://spjai.com/keras-fine-tuning/
    imported but unused(F401)とかあるから整理も兼ねて。
"""

import os
import sys

# NOTE: ここ(keras.models.Model)で3secくらいかかるみたい。
# NOTE: 「Model って VGG じゃないの?」って思っていたが、こっちの Model は interface(abstract)だ。
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


def main():

    # VGG16 のデフォルトである 224x224 でインプットを定義します。
    # NOTE: <class 'keras.engine.keras_tensor.KerasTensor'>
    # NOTE: Tensor は
    #       > 線形的な量または線形的な幾何概念を一般化したもので、基底を選べば、多次元の配列として表現できるようなものである。
    #       > (Wikipedia より)
    #       です。(?)
    input_tensor = Input(shape=(224, 224, 3))

    # VGG16 をロードします。
    # が、今回はフル結合3層をつけずにロードしています。
    # NOTE: include_top=False がフル結合3層をつけないという指定です。
    #       VGG16 は畳み込み13層とフル結合3層の計16層から成ります。
    #       なので、いうなれば VGG13 になってるってこと。
    #       (これは理解しやすいからそう書いているだけで誤解なきよう。)
    # NOTE: <class 'keras.engine.functional.Functional'>
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # 新たな層を追加しています。
    # NOTE: さっき VGG13 だったのが VGG14 になるってこと。
    #       (これは理解しやすいからそう書いているだけなので他所で言わないように。)
    # この output っていうのが現在の最後の13層目のことです。
    # NOTE: <class 'keras.engine.keras_tensor.KerasTensor'>
    x = base_model.output

    # GlobalAveragePooling2D という層を足しています。(14層目)
    # NOTE: Dense は時間がかかるが GlobalAveragePooling は高速(GAP で伝わる)だという話です。
    x = GlobalAveragePooling2D()(x)

    # Dense という層を足しています。(15層目)
    x = Dense(1024, activation='relu')(x)

    # ここが自分の追加したい層。(16層目)
    y = Dense(17, activation='softmax')(x)

    # 完成したこれが層のかたまり。
    # NOTE: x とか y とかって変数名を使っているのは、このモデルの構築手順は数式で表せる(らしい)からです。
    #       こんな感じに。こういうのって数式って言うの? 方程式じゃなくて?
    #       y = Dense(Dense(GlobalAveragePooling2D(x)))
    # NOTE: <class 'keras.engine.functional.Functional'>
    model = Model(inputs=base_model.input, outputs=y)

    # 構築した改造 VGG16 を閲覧します。
    # VGG1 6の構造に加え、最後に層が追加されている事がわかります。
    # NOTE: こういうのが増えてる
    #       dense_1 (Dense)              (None, 17)                17425
    #       たぶん17ってのが Dense(17... で追加した層だろう。
    model.summary()


if __name__ == '__main__':
    main()

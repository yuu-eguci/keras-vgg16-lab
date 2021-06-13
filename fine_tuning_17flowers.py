"""
vgg16……
    「ImageNet」と呼ばれる大規模画像データセットで学習された，16層からなるCNNモデル……
に、 17flowers のデータを加えて、 seventeen flowers の分類を可能する……
    それが fine tuning……
よ。
参考: https://spjai.com/keras-fine-tuning/
    imported but unused(F401)とかあるから整理も兼ねて。
"""

# NOTE: ここ(keras.models.Model)で3secくらいかかるみたい。
# NOTE: 「Model って VGG じゃないの?」って思っていたが、こっちの Model は interface(abstract)だ。
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# 学習用画像が格納されているフォルダです。
TRAIN_DIR = '17flowers/train_images'
# テスト用画像が格納されているフォルダです。
TEST_DIR = '17flowers/test_images'


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

    # VGG16 の全層の重みを固定しています。
    # VGG16 側の層の重みは学習時に変更されません。
    # base_model は最初に用意した13層のこと。
    # これはもう学習終わってんのだから(imagenet で)、 train する必要なしです。
    for layer in base_model.layers:
        layer.trainable = False

    # モデルを作っただけだと線形の結果しか出ません。
    # y = a * b * c * x みたいなものだからです。
    # 係数を自分で変更させるように……
    model.compile(
        optimizer=SGD(
            # NOTE: サンプルコードでは lr になっていたが、
            #       UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
            #       が出るので learning_rate へ変更しました。
            learning_rate=0.0001,
            momentum=0.9,
        ),
        # 右辺と左辺の差を小さくするためのもの。微分です。
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Training に使う画像を生成する ImageDataGenerator を作ります。
    # NOTE: ImageDataGenerator は与えた画像をいじり、 training に使う画像パターンを増やします。
    #       https://keras.io/ja/preprocessing/image/
    # NOTE: <class 'keras.preprocessing.image.ImageDataGenerator'>
    image_data_generator_to_train = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=10,
    )

    # 予測? 類推? 推測?(用語がわからん)テストに使う画像を生成する ImageDataGenerator を作ります。
    # NOTE: どうして小さくしているのかというと、モデルは小数点で学習するからです。
    #       どういうこと?
    image_data_generator_to_test = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    # ImageDataGenerator へ画像を与えます。
    # NOTE: <class 'keras.preprocessing.image.DirectoryIterator'>
    directory_iterator_for_training = image_data_generator_to_train.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=True,
    )

    directory_iterator_for_test = image_data_generator_to_test.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=True,
    )

    # ここで学習を行います。なので時間かかります。
    # fit は学習のメソッドです。
    # NOTE: サンプルコードでは Model.fit_generator になっていたが、
    #       UserWarning: `Model.fit_generator` is deprecated and
    #                    will be removed in a future version.
    #                    Please use `Model.fit`, which supports generators.
    #       が出るため model.fit に変更しました。
    # NOTE: <class 'keras.callbacks.History'>
    history = model.fit(
        directory_iterator_for_training,
        # NOTE: 1190 はトレーニング用の枚数です。 70*17=1190
        steps_per_epoch=1190 // 16,
        epochs=50,
        verbose=1,
        validation_data=directory_iterator_for_test,
        # NOTE: 170 はテスト用の枚数です。 10*17=170
        validation_steps=170 // 16,
    )

    model.save('17flowers.hdf5')


if __name__ == '__main__':
    main()

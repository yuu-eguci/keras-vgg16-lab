"""load_17flowers
"""

import sys

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.preprocessing import image
import numpy

CLASSES_FOR_17FLOWERS = [
    'Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
    'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
    'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
    'Windflower', 'Pansy',
]


def main(input_filename: str):

    # test_vgg16.py で行っているのと同じように、
    # 引数で与えられた画像を、予測へまわせる形式へ変換します。
    # <class 'PIL.Image.Image'>
    #       -> 3次元テンソル <class 'numpy.ndarray'>
    #       -> 4次元テンソル <class 'numpy.ndarray'>
    image_pil = image.load_img(input_filename, target_size=(224, 224))
    image_array_3dim = image.img_to_array(image_pil)
    image_array_4dim = numpy.expand_dims(image_array_3dim, axis=0)

    # 学習時(fine_tuning_17flowers.py)において、 ImageDataGenerator の rescale で正規化したので同じ処理が必要。
    # XXX: ……らしいですよくわかんないです。
    # これを忘れると結果がおかしくなる。
    # XXX: ……らしいですよくわかんないです。
    # NOTE: どうおかしくなるのかわからなかったのでこれをやらないで試しました。
    #       やって sunflower.jpg 試した場合↓
    #           ('Cowslip', 0.14450616)
    #           ('Sunflower', 0.08813832)
    #           ('Tigerlily', 0.07421722)
    #       やらないで試した場合↓
    #           ('Cowslip', 0.9992175)
    #           ('Tigerlily', 0.0007754794)
    #           ('Buttercup', 6.109471e-06)
    image_array_4dim = image_array_4dim / 255.0

    # fine_tuning_17flowers.py で一番最初に作っているのと同じ、 VGG13(仮) + 自作3層 を用意します。
    # NOTE: Fine tuning 関連の記事ではよく Sequential が使われているけどまずこれでいく。
    input_tensor = Input(shape=(224, 224, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    y = Dense(17, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=y)

    # Fine tuning ではこのあと重み固定(layer.trainable = False)とかをしています。
    # 今回は学習を行う必要がありません。 Weight file はすでにあるので。
    model.load_weights('17flowers.hdf5')

    # NOTE: hdf5 を作る前に compile しているのになんでやり直さないといけないの?
    #       と思ったのでやらないで試しました。
    #       やって sunflower.jpg 試した場合↓
    #           ('Cowslip', 0.14450616)
    #           ('Sunflower', 0.08813832)
    #           ('Tigerlily', 0.07421722)
    #       やらないで試した場合。あ、変わらないじゃん。やらないでいいみたい。
    #           ('Cowslip', 0.14450616)
    #           ('Sunflower', 0.08813832)
    #           ('Tigerlily', 0.07421722)
    # model.compile(
    #     optimizer=SGD(
    #         # NOTE: サンプルコードでは lr になっていたが、
    #         #       UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
    #         #       が出るので learning_rate へ変更しました。
    #         learning_rate=0.0001,
    #         momentum=0.9,
    #     ),
    #     # 右辺と左辺の差を小さくするためのもの。微分です。
    #     loss='categorical_crossentropy',
    #     metrics=['accuracy'],
    # )

    # 予測を行います。
    # NOTE: test_vgg16 では
    #       predictions = vgg16.predict(preprocess_input(image_array_4dim))
    #       top_5_predictions = decode_predictions(predictions, top=5)[0]
    #       でしたね。どういうことやねん。
    # NOTE: test_vgg16.py では preprocess_input を使っていたのにどうしてこっちでは使わないのだろう?
    #       と思ったので使って試しました。
    #       使わず sunflower.jpg 試した場合↓
    #           ('Cowslip', 0.14450616)
    #           ('Sunflower', 0.08813832)
    #           ('Tigerlily', 0.07421722)
    #       使って試した場合。 Sunflower が消えた。
    #           ('Iris', 0.16075435)
    #           ('Tulip', 0.15425675)
    #           ('Fritillary', 0.13122706)
    prediction = model.predict(image_array_4dim)[0]
    top_indices = prediction.argsort()[-5:][::-1]
    result = [(CLASSES_FOR_17FLOWERS[i], prediction[i]) for i in top_indices]
    for x in result:
        print(x)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python load_17flowers.py [image file]')
        sys.exit(1)

    filename = sys.argv[1]

    print(f'Start to predict {filename}!')

    main(filename)

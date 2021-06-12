"""test_vgg16

keras の vgg16 モデルで画像予測(類推? 推論? 用語がわからん)のテストを行うスクリプトです。
keras-vgg16-lab repository を見たときは一番最初に実行するスクリプトです。

1. 推論したい画像を自分でてきとうに用意してください。
2. python test_vgg16.py [その画像のパス]
3. コンソールを見たらなんとなくわかるでしょう。
"""

import sys

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy


def main(input_filename: str):

    # Vgg16 をロードします。
    # 重みファイル(学習済みデータのこと)には imagenet を指定します。
    # NOTE: 今回は作成済みの重みファイルを使います。
    #       そのため、テストのためにわざわざ学習に時間をとることがありません。
    # NOTE: <class 'keras.engine.functional.Functional'>
    vgg16 = VGG16(weights='imagenet')

    # Vgg16 の内容……サマリを見るにはこうします。
    # 16の由来である、16層の畳み込みニューラルネットワークのサマリを閲覧できます。
    # NOTE: はい勿論ここはよくわかっていません。
    vgg16.summary()

    # test_vgg16.py の引数で指定した画像を読み込みます。
    # ここでは vgg16 のデフォルトである 224x224 にリサイズしています。
    # NOTE: 画像の入力は、 keras.preprocessing.image を使うと色々便利だそうです。
    # NOTE: <class 'PIL.Image.Image'>
    image_pil = image.load_img(input_filename, target_size=(224, 224))

    # 読み込んだ画像は PIL 形式です。それを array に変換します。
    # NOTE: <class 'numpy.ndarray'>
    image_array_3dim = image.img_to_array(image_pil)

    # 3次元テンソル(rows, cols, channels)を4次元テンソル(samples, rows, cols, channels)に変換する、
    # という行為を行っています。
    # 入力画像が1枚なので、 smaples=1 でいいそうです。
    # NOTE: 意味不明。だけど、 row, col, channel は OpenCV で見た覚えがあります。
    #       まあ画像の形式を変換しているんでしょう。(小泉文法)
    # NOTE: <class 'numpy.ndarray'>
    image_array_4dim = numpy.expand_dims(image_array_3dim, axis=0)

    # 分類……与えられた画像が何なのか予測……を行います。
    # Vgg16 に画像データを与えて、予測結果を取得します。
    # NOTE: 今回利用しているデフォのモデルは、1000クラス分類のモデルです。
    #       だからここでは1000クラスの確率が結果として返ります。
    # NOTE: predictions は numpy.ndarray のリストです。
    # NOTE: <class 'numpy.ndarray'>
    predictions = vgg16.predict(preprocess_input(image_array_4dim))

    # 1000クラス全部の確率なんて要りません。トップ5まで取得します。
    # decode_predictions は1000クラスを文字列に変換してくれます。 ndarray そのまま出されても困るもんね。
    # NOTE: decode_predictions の返り値は
    #       [[('n02106662', 'German_shepherd', 0.9971668),...)]]
    #       みたいな list です。このスクリプトだと list の要素はひとつなんでどうして list なのかよくわかんないです。
    #       とまあそういうわけで [0] で結果を取得しています。
    # NOTE: <class 'list'>
    top_5_predictions = decode_predictions(predictions, top=5)[0]

    # 結果を閲覧します。
    for result_tuple in top_5_predictions:
        # こんな内容が出ます。
        # ('n02106662', 'German_shepherd', 0.9971668) <class 'tuple'>
        # NOTE: ここで出力される結果と WordNet を利用して、上位語を取得することができるらしい。
        #       WordNet は keras に含まれているようなものではなくて、また別の話。
        print(result_tuple, type(result_tuple))


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: python test_vgg16.py [image file]')
        sys.exit(1)

    filename = sys.argv[1]

    print(f'Start to predict {filename}!')

    main(filename)

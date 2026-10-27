"""推論で表示するクラス名が、学習のときのクラス番号と同じ順番かを確かめます。"""

import keras
import numpy
from conftest import make_image_tree

from vgg16lab import classifier

# データセットの画像番号の順番です。Keras はこの順番ではなく、フォルダ名の順番で番号を振ります。
DATASET_ORDER = [
    'Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus', 'Iris', 'Tigerlily', 'Daffodil', 'Fritillary',
    'Sunflower', 'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy',
]  # fmt: skip


def test_inference_labels_follow_training_class_order(tmp_path):
    data = make_image_tree(tmp_path / 'data', DATASET_ORDER, per_class=1)
    training_order = keras.utils.image_dataset_from_directory(data, image_size=(32, 32), batch_size=4).class_names
    model_path = tmp_path / 'flowers.keras'
    classifier.train(data, data, model_path, epochs=1, batch_size=8, image_size=32, weights=None, verbose=0)

    model, metadata = classifier.load_classifier(model_path)

    assert metadata.class_names == training_order

    # 出力の Sunflower の番号だけが大きくなるようにして、表示が Sunflower になるかを確かめます。
    # 2021 年版の並びだと、この番号は Cowslip と表示されていました。
    sunflower = training_order.index('Sunflower')
    assert DATASET_ORDER[sunflower] == 'Cowslip'
    kernel, bias = model.layers[-1].get_weights()
    bias = numpy.full_like(bias, -10.0)
    bias[sunflower] = 10.0
    model.layers[-1].set_weights([numpy.zeros_like(kernel), bias])

    assert classifier.predict(model, metadata, data / 'Tulip' / '0.jpg', top=1)[0][0] == 'Sunflower'

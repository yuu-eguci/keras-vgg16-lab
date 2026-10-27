"""推論で表示するクラス名が、学習のときのクラス番号と同じ順番かを確かめます。"""

import keras
import numpy
from PIL import Image

# データセットの画像番号の順番です。Keras はこの順番ではなく、フォルダ名の順番で番号を振ります。
DATASET_ORDER = [
    'Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus', 'Iris', 'Tigerlily', 'Daffodil', 'Fritillary',
    'Sunflower', 'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy',
]  # fmt: skip


def make_class_dirs(root, size=8):
    for i, name in enumerate(DATASET_ORDER):
        (root / name).mkdir()
        Image.fromarray(numpy.full((size, size, 3), i * 10, dtype=numpy.uint8)).save(root / name / 'a.jpg')


def test_inference_labels_follow_training_class_order(tmp_path):
    make_class_dirs(tmp_path)
    training_order = keras.utils.image_dataset_from_directory(tmp_path, image_size=(8, 8), batch_size=4).class_names

    import load_17flowers

    assert load_17flowers.CLASSES_FOR_17FLOWERS == training_order

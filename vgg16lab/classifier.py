"""VGG16 に層を足した分類モデルを、作る・学習する・保存する・読み込む・推論する処理です。

モデル (.keras) の隣に <名前>.metadata.json を置きます。
クラス名の順番と前処理はこのファイルで決まり、推論はこれと食い違うモデルを読み込みません。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy
from keras.applications.vgg16 import VGG16, preprocess_input

METADATA_FORMAT = 'keras-vgg16-lab/classifier'
METADATA_VERSION = 1
# 画像は RGB の 0〜255 のまま、bilinear で image_size にリサイズしてから、この関数に通します。
PREPROCESSING = 'keras.applications.vgg16.preprocess_input'
INTERPOLATION = 'bilinear'


class ModelFormatError(Exception):
    """推論に使えないモデル、またはメタデータのときのエラーです。"""


@dataclass(frozen=True)
class Metadata:
    class_names: list[str]
    image_size: int


def metadata_path(model_path: Path) -> Path:
    """17flowers.keras なら 17flowers.metadata.json です。"""
    model_path = Path(model_path)
    return model_path.with_name(f'{model_path.stem}.metadata.json')


def build_model(num_classes: int, image_size: int = 224, weights: str | None = 'imagenet') -> keras.Model:
    """VGG16 の畳み込み部分 (固定) + 自作の層、のモデルを作ります。"""
    inputs = keras.Input(shape=(image_size, image_size, 3))

    # 学習のときだけ、画像を少し反転・回転・拡大して、学習に使う画像のパターンを増やします。
    # 推論のときは何もしません。
    x = keras.layers.RandomFlip('horizontal')(inputs)
    x = keras.layers.RandomRotation(10 / 360)(x)
    x = keras.layers.RandomZoom(0.2)(x)

    # VGG16 をロードします。
    # が、今回はフル結合3層をつけずにロードしています。
    # NOTE: include_top=False がフル結合3層をつけないという指定です。
    #       VGG16 は畳み込み13層とフル結合3層の計16層から成ります。
    # VGG16 側の層の重みは学習時に変更されません。これはもう imagenet で学習が終わっているからです。
    base_model = VGG16(include_top=False, weights=weights, input_shape=(image_size, image_size, 3))
    base_model.trainable = False
    x = base_model(x, training=False)

    # GlobalAveragePooling2D、Dense を足して、最後に花の種類の数だけ出力する層を足します。
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train(
    train_dir: Path,
    val_dir: Path,
    output: Path,
    *,
    epochs: int,
    batch_size: int = 16,
    image_size: int = 224,
    weights: str | None = 'imagenet',
    seed: int = 0,
    verbose: int | str = 'auto',
) -> keras.callbacks.History:
    """クラスごとのフォルダに分けた画像で学習して、モデルとメタデータを output に保存します。"""
    keras.utils.set_random_seed(seed)
    options = {
        'label_mode': 'categorical',
        'image_size': (image_size, image_size),
        'interpolation': INTERPOLATION,
        'batch_size': batch_size,
        'seed': seed,
    }
    # クラス番号は、Keras がフォルダ名を並べた順番 (class_names) で振られます。
    train_data = keras.utils.image_dataset_from_directory(train_dir, shuffle=True, **options)
    class_names = list(train_data.class_names)
    val_data = keras.utils.image_dataset_from_directory(val_dir, shuffle=False, class_names=class_names, **options)

    model = build_model(len(class_names), image_size, weights)
    history = model.fit(
        train_data.map(_preprocess),
        validation_data=val_data.map(_preprocess),
        epochs=epochs,
        verbose=verbose,
    )
    save_classifier(model, output, class_names, image_size, weights)
    return history


def save_classifier(
    model: keras.Model, path: Path, class_names: list[str], image_size: int, weights: str | None
) -> None:
    path = Path(path)
    if path.suffix != '.keras':
        raise ValueError(f'モデルは .keras で保存してください: {path}')
    model.save(path)
    metadata = {
        'format': METADATA_FORMAT,
        'format_version': METADATA_VERSION,
        'class_names': class_names,
        'image_size': [image_size, image_size],
        'color_mode': 'rgb',
        'interpolation': INTERPOLATION,
        'preprocessing': PREPROCESSING,
        'base_weights': weights,
        'keras_version': keras.__version__,
    }
    metadata_path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load_classifier(path: Path) -> tuple[keras.Model, Metadata]:
    """モデルとメタデータを読み込みます。食い違いがあれば推測で読み替えず ModelFormatError にします。"""
    path = Path(path)
    if path.suffix in ('.h5', '.hdf5'):
        raise ModelFormatError(
            f'{path.name}: .h5 / .hdf5 のモデルは読み込みません。'
            ' 2021 年版のモデルは前処理とクラスの順番がいまと違うので、fine_tuning_17flowers.py で再学習してください。'
        )
    if path.suffix != '.keras':
        raise ModelFormatError(f'{path.name}: .keras のモデルを指定してください。')
    if not path.is_file():
        raise ModelFormatError(
            f'{path} がありません。学習済みモデルは配布していないので、fine_tuning_17flowers.py で作ってください。'
        )
    meta_file = metadata_path(path)
    if not meta_file.is_file():
        raise ModelFormatError(f'{meta_file.name} がありません。クラス名の順番がわからないので読み込みません。')

    try:
        metadata = _parse_metadata(json.loads(meta_file.read_text(encoding='utf-8')))
    except (ValueError, KeyError, TypeError) as error:
        raise ModelFormatError(f'{meta_file.name} を読めません: {error}') from error

    model = keras.models.load_model(path)
    if model.output_shape[-1] != len(metadata.class_names):
        raise ModelFormatError(
            f'モデルの出力は {model.output_shape[-1]} クラスですが、'
            f'メタデータには {len(metadata.class_names)} クラスあります。'
        )
    if tuple(model.input_shape[1:]) != (metadata.image_size, metadata.image_size, 3):
        raise ModelFormatError(f'モデルの入力 {model.input_shape[1:]} がメタデータの image_size と合いません。')
    return model, metadata


def predict(model: keras.Model, metadata: Metadata, image_path: Path, top: int = 5) -> list[tuple[str, float]]:
    """1 枚の画像を推論して、(クラス名, 確率) を確率の高い順に top 個返します。"""
    image = keras.utils.img_to_array(keras.utils.load_img(image_path, color_mode='rgb'))
    # 学習 (image_dataset_from_directory) と同じリサイズにします。
    image = keras.ops.image.resize(image, (metadata.image_size, metadata.image_size), interpolation=INTERPOLATION)
    batch = preprocess_input(numpy.expand_dims(keras.ops.convert_to_numpy(image), axis=0))
    probabilities = model.predict(batch, verbose=0)[0]
    best = numpy.argsort(probabilities)[::-1][:top]
    return [(metadata.class_names[i], float(probabilities[i])) for i in best]


def _preprocess(images, labels):
    return preprocess_input(images), labels


def _parse_metadata(raw: dict) -> Metadata:
    if raw['format'] != METADATA_FORMAT or raw['format_version'] != METADATA_VERSION:
        raise ValueError(f'形式 {raw["format"]} / {raw["format_version"]} には対応していません。')
    if raw['preprocessing'] != PREPROCESSING or raw['interpolation'] != INTERPOLATION or raw['color_mode'] != 'rgb':
        raise ValueError('前処理がいまのコードと違います。')
    class_names = raw['class_names']
    if (
        not isinstance(class_names, list)
        or not class_names
        or not all(isinstance(n, str) for n in class_names)
        or len(set(class_names)) != len(class_names)
    ):
        raise ValueError('class_names は重複のない文字列のリストにしてください。')
    height, width = raw['image_size']
    if height != width or not isinstance(height, int):
        raise ValueError('image_size は同じ整数 2 つにしてください。')
    return Metadata(class_names=list(class_names), image_size=height)

"""メタデータと前処理の約束を守らないモデルを、推論が読み込まないことを確かめます。"""

import json

import pytest
from conftest import make_image_tree

from vgg16lab import classifier
from vgg16lab.classifier import ModelFormatError


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp('trained')
    data = make_image_tree(root / 'data', ['b_class', 'a_class', 'c_class'])
    model_path = root / 'tiny.keras'
    history = classifier.train(data, data, model_path, epochs=1, batch_size=4, image_size=32, weights=None, verbose=0)
    return model_path, data, history


def rewrite_metadata(model_path, tmp_path, **changes):
    """trained のモデルとメタデータを tmp_path にコピーして、メタデータだけ書き換えます。"""
    copied = tmp_path / model_path.name
    copied.write_bytes(model_path.read_bytes())
    metadata = json.loads(classifier.metadata_path(model_path).read_text(encoding='utf-8'))
    metadata.update(changes)
    classifier.metadata_path(copied).write_text(json.dumps(metadata), encoding='utf-8')
    return copied


def test_training_saves_model_and_metadata(trained):
    model_path, data, history = trained

    metadata = json.loads(classifier.metadata_path(model_path).read_text(encoding='utf-8'))

    assert len(history.history['loss']) == 1
    assert metadata['class_names'] == ['a_class', 'b_class', 'c_class']
    assert metadata['image_size'] == [32, 32]
    assert metadata['preprocessing'] == 'keras.applications.vgg16.preprocess_input'
    assert metadata['base_weights'] is None


def test_predict_returns_known_classes_in_probability_order(trained):
    model_path, data, _ = trained
    model, metadata = classifier.load_classifier(model_path)

    result = classifier.predict(model, metadata, data / 'a_class' / '0.jpg', top=3)

    assert [name for name, _ in result] != [] and {name for name, _ in result} == set(metadata.class_names)
    probabilities = [p for _, p in result]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize('name', ['17flowers.hdf5', 'old.h5'])
def test_old_h5_models_are_refused(tmp_path, name):
    (tmp_path / name).write_bytes(b'not read')

    with pytest.raises(ModelFormatError, match='hdf5'):
        classifier.load_classifier(tmp_path / name)


def test_missing_model_is_refused(tmp_path):
    with pytest.raises(ModelFormatError, match='配布していない'):
        classifier.load_classifier(tmp_path / '17flowers.keras')


def test_model_without_metadata_is_refused(trained, tmp_path):
    model_path, _, _ = trained
    copied = tmp_path / model_path.name
    copied.write_bytes(model_path.read_bytes())

    with pytest.raises(ModelFormatError, match='metadata'):
        classifier.load_classifier(copied)


@pytest.mark.parametrize(
    'changes',
    [
        {'preprocessing': 'rescale 1/255'},
        {'interpolation': 'nearest'},
        {'format_version': 2},
        {'class_names': ['a_class', 'b_class']},
        {'class_names': ['a_class', 'a_class', 'c_class']},
        {'image_size': [224, 224]},
    ],
)
def test_metadata_that_disagrees_is_refused(trained, tmp_path, changes):
    copied = rewrite_metadata(trained[0], tmp_path, **changes)

    with pytest.raises(ModelFormatError):
        classifier.load_classifier(copied)

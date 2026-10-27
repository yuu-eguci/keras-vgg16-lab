"""スクリプトをコマンドとして、ランダム重み・小さい画像・1 epoch で動かします。"""

import os
import subprocess
import sys
from pathlib import Path

from conftest import make_image_tree

ROOT = Path(__file__).resolve().parents[1]


def run(*args, cwd, home):
    env = {
        **os.environ,
        # 重みなどのダウンロードが紛れ込んだら、つながらないプロキシで失敗させます。
        'KERAS_HOME': str(home),
        'HTTP_PROXY': 'http://127.0.0.1:9',
        'HTTPS_PROXY': 'http://127.0.0.1:9',
        'NO_PROXY': '',
    }
    return subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True)


def test_train_then_predict_from_command_line(tmp_path):
    train_dir = make_image_tree(tmp_path / 'train', ['Tulip', 'Bluebell', 'Iris'])
    val_dir = make_image_tree(tmp_path / 'val', ['Tulip', 'Bluebell', 'Iris'], per_class=1)
    model = tmp_path / 'tiny.keras'

    trained = run(
        ROOT / 'fine_tuning_17flowers.py',
        *('--train-dir', train_dir, '--val-dir', val_dir, '--output', model),
        *('--epochs', 1, '--batch-size', 4, '--image-size', 32, '--weights', 'none'),
        cwd=tmp_path,
        home=tmp_path / 'keras-home',
    )
    assert trained.returncode == 0, trained.stderr
    assert (tmp_path / 'tiny.metadata.json').is_file()

    predicted = run(
        ROOT / 'load_17flowers.py', train_dir / 'Iris' / '0.jpg', '--model', model, '--top', 3,
        cwd=tmp_path,
        home=tmp_path / 'keras-home',
    )  # fmt: skip
    assert predicted.returncode == 0, predicted.stderr
    names = [line.split('\t')[0] for line in predicted.stdout.splitlines() if '\t' in line]
    assert sorted(names) == ['Bluebell', 'Iris', 'Tulip']
    assert not (tmp_path / 'keras-home' / 'models').exists()


def test_load_refuses_old_hdf5_model(tmp_path):
    (tmp_path / '17flowers.hdf5').write_bytes(b'old')

    result = run(ROOT / 'load_17flowers.py', 'x.jpg', '--model', '17flowers.hdf5', cwd=tmp_path, home=tmp_path)

    assert result.returncode == 1
    assert '再学習' in result.stderr


def test_training_rejects_non_keras_output_before_loading_tensorflow(tmp_path):
    result = run(ROOT / 'fine_tuning_17flowers.py', '--output', 'model.hdf5', cwd=tmp_path, home=tmp_path)

    assert result.returncode == 2
    assert '.keras' in result.stderr


def test_predict_imagenet_without_image_shows_usage(tmp_path):
    result = run(ROOT / 'predict_imagenet.py', cwd=tmp_path, home=tmp_path)

    assert result.returncode == 1
    assert 'Usage' in result.stdout


def test_split_module_does_not_import_tensorflow():
    code = 'import sys, vgg16lab.flowers_split; assert "tensorflow" not in sys.modules and "keras" not in sys.modules'

    assert subprocess.run([sys.executable, '-c', code], cwd=ROOT).returncode == 0

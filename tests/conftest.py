import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """テスト中は外へつながせません。重みのダウンロードなどが紛れ込んだら落ちます。"""

    def guard(*args, **kwargs):
        raise RuntimeError('テスト中にネットワークへ接続しようとしました。')

    monkeypatch.setattr(socket.socket, 'connect', guard)
    monkeypatch.setattr(socket, 'create_connection', guard)


def make_image_tree(root, class_names, per_class=2, size=32):
    """クラスごとのフォルダに、色の違う小さい jpg を置きます。"""
    import numpy
    from PIL import Image

    for i, name in enumerate(class_names):
        (root / name).mkdir(parents=True)
        for j in range(per_class):
            pixels = numpy.full((size, size, 3), (i * 13 + j * 7) % 256, dtype=numpy.uint8)
            Image.fromarray(pixels).save(root / name / f'{j}.jpg')
    return root

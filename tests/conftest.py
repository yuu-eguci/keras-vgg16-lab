import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """テスト中は外へつながせません。重みのダウンロードなどが紛れ込んだら落ちます。"""

    def guard(*args, **kwargs):
        raise RuntimeError('テスト中にネットワークへ接続しようとしました。')

    monkeypatch.setattr(socket.socket, 'connect', guard)
    monkeypatch.setattr(socket, 'create_connection', guard)

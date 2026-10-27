"""17flowers/setup_17flowers_for_fine_tuning.py をコマンドとして動かして確かめます。"""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / '17flowers' / 'setup_17flowers_for_fine_tuning.py'

# 17flowers の画像番号は、クラスごとに 80 枚ずつ並んでいます。
FIRST_NUMBERS = {
    'Tulip': 1, 'Snowdrop': 81, 'LilyValley': 161, 'Bluebell': 241, 'Crocus': 321,
    'Iris': 401, 'Tigerlily': 481, 'Daffodil': 561, 'Fritillary': 641, 'Sunflower': 721,
    'Daisy': 801, 'ColtsFoot': 881, 'Dandelion': 961, 'Cowslip': 1041, 'Buttercup': 1121,
    'Windflower': 1201, 'Pansy': 1281,
}  # fmt: skip


def make_jpg_dir(root: Path, per_class: int = 12) -> Path:
    """本物の代わりに、クラスごとに per_class 枚の偽 jpg を置いた jpg/ を作ります。"""
    jpg = root / 'jpg'
    jpg.mkdir()
    for first in FIRST_NUMBERS.values():
        for number in range(first, first + per_class):
            (jpg / f'image_{number:04d}.jpg').write_bytes(f'fake {number}'.encode())
    return jpg


def run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True)


def listing(directory: Path) -> dict[str, set[str]]:
    if not directory.exists():
        return {}
    return {d.name: {f.name for f in d.iterdir()} for d in directory.iterdir() if d.is_dir()}


def snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob('*')) if p.is_file()}


def assert_clean_split(root: Path, test_per_class: int = 10, per_class: int = 12):
    train, test = listing(root / 'train_images'), listing(root / 'test_images')
    assert set(train) == set(test) == set(FIRST_NUMBERS)
    for name in FIRST_NUMBERS:
        assert len(test[name]) == test_per_class, name
        assert len(train[name]) == per_class - test_per_class, name
        assert not train[name] & test[name], f'{name} の test 画像が train に混ざっています'


def test_split_makes_train_and_test_without_overlap(tmp_path):
    jpg = make_jpg_dir(tmp_path)
    before = snapshot(jpg)

    result = run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert_clean_split(tmp_path)
    assert snapshot(jpg) == before


def test_running_twice_does_not_leak_test_images_into_train(tmp_path):
    make_jpg_dir(tmp_path)
    assert run(tmp_path).returncode == 0
    first = (listing(tmp_path / 'train_images'), listing(tmp_path / 'test_images'))

    result = run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert_clean_split(tmp_path)
    assert (listing(tmp_path / 'train_images'), listing(tmp_path / 'test_images')) == first


def test_same_seed_gives_same_split(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for root in (a, b):
        root.mkdir()
        make_jpg_dir(root)
        assert run(root, '--seed', '3').returncode == 0

    assert listing(a / 'test_images') == listing(b / 'test_images')


def test_irrelevant_files_are_ignored(tmp_path):
    jpg = make_jpg_dir(tmp_path)
    (jpg / 'files.txt').write_text('list')
    (jpg / 'zzz_notes.txt').write_text('notes')
    (jpg / '.DS_Store').write_bytes(b'\0')
    (jpg / 'image_9999.jpg').write_bytes(b'out of range')
    (jpg / 'thumbs').mkdir()
    (tmp_path / 'train_images').mkdir()
    (tmp_path / 'train_images' / '.DS_Store').write_bytes(b'\0')

    result = run(tmp_path)

    assert result.returncode == 0, result.stderr
    assert_clean_split(tmp_path)


def test_different_settings_on_existing_split_fail_without_changes(tmp_path):
    make_jpg_dir(tmp_path)
    assert run(tmp_path, '--seed', '0').returncode == 0
    before = snapshot(tmp_path)

    result = run(tmp_path, '--seed', '1')

    assert result.returncode != 0
    assert snapshot(tmp_path) == before


def test_output_inside_source_is_rejected(tmp_path):
    jpg = make_jpg_dir(tmp_path)
    before = snapshot(tmp_path)

    result = run(tmp_path, '--output', 'jpg')

    assert result.returncode != 0
    assert snapshot(tmp_path) == before
    assert sorted(p.name for p in jpg.iterdir() if p.is_dir()) == []


def test_missing_source_is_rejected_without_creating_output(tmp_path):
    result = run(tmp_path)

    assert result.returncode != 0
    assert list(tmp_path.iterdir()) == []


def test_too_few_images_per_class_is_rejected(tmp_path):
    make_jpg_dir(tmp_path, per_class=10)

    result = run(tmp_path)

    assert result.returncode != 0
    assert not (tmp_path / 'train_images').exists()

"""17flowers の jpg/ を、Keras が読めるクラス別フォルダの train_images/ と test_images/ に分けます。

TensorFlow は import しません。
"""

import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

# データセットの画像番号の順番です。image_0001.jpg から 80 枚ずつ、この順番で並んでいます。
FLOWER_CLASSES = (
    'Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus', 'Iris', 'Tigerlily', 'Daffodil', 'Fritillary',
    'Sunflower', 'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy',
)  # fmt: skip
IMAGES_PER_CLASS = 80
TRAIN_DIR_NAME = 'train_images'
TEST_DIR_NAME = 'test_images'

_IMAGE_NAME = re.compile(r'image_(\d{4})\.jpg')


class SplitError(Exception):
    """分けられないとき、または分けると危ないときのエラーです。"""


@dataclass(frozen=True)
class SplitResult:
    created: bool
    train_count: int
    test_count: int
    ignored_count: int


def class_of(number: int) -> str | None:
    """画像番号からクラス名を返します。17flowers の範囲外なら None です。"""
    if 1 <= number <= IMAGES_PER_CLASS * len(FLOWER_CLASSES):
        return FLOWER_CLASSES[(number - 1) // IMAGES_PER_CLASS]
    return None


def split_17flowers(source: Path, output: Path, test_per_class: int = 10, seed: int = 0) -> SplitResult:
    """source の画像を output/train_images/<クラス>/ と output/test_images/<クラス>/ にコピーします。

    - source は変更しません。
    - 同じファイルと同じ seed / test_per_class なら、いつも同じ分け方になります。
    - すでに同じ分け方で分けてあれば何もしません。違う中身があれば SplitError で止まり、何も変更しません。
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise SplitError(f'画像フォルダがありません: {source}')
    if test_per_class < 1:
        raise SplitError('--test-per-class は 1 以上にしてください。')

    targets = {TRAIN_DIR_NAME: output / TRAIN_DIR_NAME, TEST_DIR_NAME: output / TEST_DIR_NAME}
    for target in targets.values():
        target = target.resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            raise SplitError(f'出力先 {target} と画像フォルダ {source} が重なっています。')

    by_class, ignored = _collect_images(source)
    plan = _plan(by_class, test_per_class, seed)

    existing = {name: _existing_split(target) for name, target in targets.items()}
    counts = {name: sum(len(files) for files in plan[name].values()) for name in targets}
    if existing == plan:
        return SplitResult(False, counts[TRAIN_DIR_NAME], counts[TEST_DIR_NAME], ignored)
    if any(existing.values()):
        raise SplitError(
            'train_images/ か test_images/ に、この設定とは違う中身がすでにあります。'
            ' やり直すときは、その 2 つのフォルダを自分で消してから実行してください。'
        )

    output.mkdir(parents=True, exist_ok=True)
    # いったん隠しフォルダにコピーし終えてから、クラスごとに置き場所へ移します。
    staging = Path(tempfile.mkdtemp(prefix='.split-', dir=output))
    try:
        for name, target in targets.items():
            target.mkdir(exist_ok=True)
            for class_name, files in plan[name].items():
                staged = staging / name / class_name
                staged.mkdir(parents=True)
                for file in files:
                    shutil.copy2(source / file, staged / file)
                staged.replace(target / class_name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return SplitResult(True, counts[TRAIN_DIR_NAME], counts[TEST_DIR_NAME], ignored)


def _collect_images(source: Path) -> tuple[dict[str, list[str]], int]:
    by_class: dict[str, list[str]] = {name: [] for name in FLOWER_CLASSES}
    ignored = 0
    for entry in source.iterdir():
        match = _IMAGE_NAME.fullmatch(entry.name)
        class_name = class_of(int(match.group(1))) if match else None
        if class_name is None or not entry.is_file():
            ignored += 1
            continue
        by_class[class_name].append(entry.name)
    return {name: sorted(files) for name, files in by_class.items()}, ignored


def _plan(by_class: dict[str, list[str]], test_per_class: int, seed: int) -> dict[str, dict[str, set[str]]]:
    rng = random.Random(seed)
    plan: dict[str, dict[str, set[str]]] = {TRAIN_DIR_NAME: {}, TEST_DIR_NAME: {}}
    for class_name in FLOWER_CLASSES:
        files = list(by_class[class_name])
        if len(files) <= test_per_class:
            raise SplitError(f'{class_name} の画像が {len(files)} 枚しかなく、test に {test_per_class} 枚とれません。')
        rng.shuffle(files)
        plan[TEST_DIR_NAME][class_name] = set(files[:test_per_class])
        plan[TRAIN_DIR_NAME][class_name] = set(files[test_per_class:])
    return plan


def _existing_split(directory: Path) -> dict[str, set[str]]:
    """すでにある train_images/ などのクラスフォルダと中身です。

    Keras はフォルダだけをクラスとして読むので、直下のファイルと、.DS_Store のような隠しファイルは数えません。
    """
    if not directory.exists():
        return {}
    return {
        entry.name: {f.name for f in entry.iterdir() if not f.name.startswith('.')}
        for entry in directory.iterdir()
        if entry.is_dir()
    }

"""setup_17flowers_for_fine_tuning

17flowers を使って VGG16 の fine tuning を試します。
17flowers は 17 Category Flower Dataset のことです。
https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
17種類、それぞれ80枚、合計1,360枚の画像集です。

マジで? 17種類も手に入るの? ラッキー。
ではあるが、ひとつのフォルダに1,360枚はいっているだけなのでそのままだと VGG16 の形式ではありません。
なので、 VGG16 の形式に変換するのが setup_17flowers_for_fine_tuning の役割です。

1. 17flowers > Downloads > Dataset images から 17flowers.tgz を取得します。
   https://www.robots.ox.ac.uk/~vgg/data/flowers/17/
2. 17flowers.tgz を解凍すると jpg というフォルダが出現します。
3. jpg フォルダをこのスクリプトの隣に置いてください。
4. cd 17flowers/
5. uv run python setup_17flowers_for_fine_tuning.py

クラスごとに、ランダムに選んだ --test-per-class 枚 (既定 10 枚) を test_images へ、残りを train_images へコピーします。
選び方は --seed で決まるので、同じ設定なら何回やっても同じ分け方です。
"""

import argparse
import sys
from pathlib import Path

from vgg16lab.flowers_split import SplitError, split_17flowers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='17flowers の jpg/ を train_images/ と test_images/ に分けます。')
    parser.add_argument('--source', type=Path, default=Path('jpg'), help='17flowers.tgz を解凍した jpg フォルダ')
    parser.add_argument('--output', type=Path, default=Path('.'), help='train_images/ と test_images/ を作る場所')
    parser.add_argument('--test-per-class', type=int, default=10, help='クラスごとに test へまわす枚数')
    parser.add_argument('--seed', type=int, default=0, help='分け方を決める乱数の seed')
    args = parser.parse_args(argv)

    try:
        result = split_17flowers(args.source, args.output, args.test_per_class, args.seed)
    except SplitError as error:
        print(f'エラー: {error}', file=sys.stderr)
        return 1

    status = '分けました' if result.created else 'すでに同じ設定で分けてあるので、何もしませんでした'
    print(f'{status}。train: {result.train_count} 枚、test: {result.test_count} 枚、無視: {result.ignored_count} 件')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""fine_tuning_17flowers

VGG16 で 17flowers dataset で fine tuning することを目的とするスクリプトです。
VGG16 の畳み込み層 (ImageNet で学習済み) は固定して、その上に足した層だけを学習します。

1. 17flowers/setup_17flowers_for_fine_tuning.py によって 17flowers をセットアップする。
2. uv run python fine_tuning_17flowers.py
   WARN: CPU だと 1 epoch に数十秒から数分かかります。まずは --epochs 1 や 3 で試すのがおすすめです。
3. 結果として生成された 17flowers.keras と 17flowers.metadata.json を load_17flowers.py で利用する。

中身は vgg16lab/classifier.py にあります。
"""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='17flowers で VGG16 に足した層を学習します。')
    parser.add_argument('--train-dir', type=Path, default=Path('17flowers/train_images'), help='学習用画像のフォルダ')
    parser.add_argument('--val-dir', type=Path, default=Path('17flowers/test_images'), help='検証用画像のフォルダ')
    parser.add_argument('--output', type=Path, default=Path('17flowers.keras'), help='保存するモデル (.keras)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=224, help='入力画像の一辺 (32 以上)')
    parser.add_argument(
        '--weights',
        choices=['imagenet', 'none'],
        default='imagenet',
        help='VGG16 の初期の重み。none はランダムで、動作確認用です',
    )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.suffix != '.keras':
        parser.error('--output は .keras で終わる名前にしてください。')

    # TensorFlow の import は重いので、引数を確かめてからにします。
    from vgg16lab.classifier import train

    train(
        args.train_dir,
        args.val_dir,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        weights=None if args.weights == 'none' else args.weights,
        seed=args.seed,
    )
    print(f'保存しました: {args.output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

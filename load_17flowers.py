"""load_17flowers

fine_tuning_17flowers.py で作ったモデルで、1 枚の画像が 17 種類のどの花かを推論します。

    uv run python load_17flowers.py test-images/sunflower.jpg

モデルの隣にある <名前>.metadata.json から、クラス名の順番と前処理を読みます。
NOTE: 2021 年版では、クラス名をデータセットの順番 (Tulip, Snowdrop, ...) で書いていました。
      でも Keras はフォルダ名のアルファベット順 (Bluebell, Buttercup, ...) でクラス番号を振るので、
      17 クラス中 2 つしか合っていませんでした。sunflower.jpg が Cowslip と出ていたのは、実は Sunflower でした。
NOTE: 学習と推論で前処理がずれると結果がおかしくなります。
      2021 年版は 1/255 にしていましたが、いまは学習も推論も VGG16 の preprocess_input です。
"""

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='学習したモデルで、画像が 17 種類のどの花かを推論します。')
    parser.add_argument('image', type=Path, help='推論したい画像')
    parser.add_argument('--model', type=Path, default=Path('17flowers.keras'), help='学習したモデル (.keras)')
    parser.add_argument('--top', type=int, default=5, help='表示するクラスの数')
    args = parser.parse_args(argv)

    from vgg16lab.classifier import ModelFormatError, load_classifier, predict

    try:
        model, metadata = load_classifier(args.model)
    except ModelFormatError as error:
        print(f'エラー: {error}', file=sys.stderr)
        return 1

    print(f'Start to predict {args.image}!')
    for class_name, probability in predict(model, metadata, args.image, args.top):
        print(f'{class_name}\t{probability:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

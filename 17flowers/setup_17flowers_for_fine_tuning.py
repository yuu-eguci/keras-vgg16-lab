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
5. python setup_17flowers_for_fine_tuning.py
"""

import os
import shutil
import random

# 17flowers.tgz を解凍して出てくるフォルダです。
IN_DIR = 'jpg'
# 80枚の中から70枚を、学習用画像として TRAIN_DIR に格納します。
TRAIN_DIR = 'train_images'
# 80枚の中から10枚を、予測テスト用画像として TEST_DIR に格納します。
TEST_DIR = 'test_images'
# 画像分類のため


def main():

    # 格納用フォルダ、なければ作っておきます。
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    # 17flowers に含まれる画像のメタ情報です。
    # { 花の名前: (その画像の番号) }
    flower_dics = {
        'Tulip': (1, 80),
        'Snowdrop': (81, 160),
        'LilyValley': (161, 240),
        'Bluebell': (241, 320),
        'Crocus': (321, 400),
        'Iris': (401, 480),
        'Tigerlily': (481, 560),
        'Daffodil': (561, 640),
        'Fritillary': (641, 720),
        'Sunflower': (721, 800),
        'Daisy': (801, 880),
        'ColtsFoot': (881, 960),
        'Dandelion': (961, 1040),
        'Cowslip': (1041, 1120),
        'Buttercup': (1121, 1200),
        'Windflower': (1201, 1280),
        'Pansy': (1281, 1360),
    }

    # 花ごとのフォルダを作成します。
    for name in flower_dics:
        if not os.path.exists(os.path.join(TRAIN_DIR, name)):
            os.mkdir(os.path.join(TRAIN_DIR, name))
        if not os.path.exists(os.path.join(TEST_DIR, name)):
            os.mkdir(os.path.join(TEST_DIR, name))

    for f in sorted(os.listdir(IN_DIR)):
        # jpg だけが対象です。それ以外は continue します。
        if os.path.splitext(f)[1] != '.jpg':
            continue
        # image_0001.jpg => 1
        prefix = f.replace('.jpg', '')
        idx = int(prefix.split('_')[1])

        for name in flower_dics:
            start, end = flower_dics[name]
            if idx in range(start, end + 1):
                source = os.path.join(IN_DIR, f)
                dest = os.path.join(TRAIN_DIR, name)
                shutil.copy(source, dest)
                continue

    # 訓練データの各ディレクトリからランダムに10枚をテストとする
    for d in os.listdir(TRAIN_DIR):
        # jpg だけが対象です。それ以外は continue します。
        if os.path.splitext(f)[1] != '.jpg':
            continue
        files = os.listdir(os.path.join(TRAIN_DIR, d))
        random.shuffle(files)
        for f in files[:10]:
            source = os.path.join(TRAIN_DIR, d, f)
            dest = os.path.join(TEST_DIR, d)
            shutil.move(source, dest)


if __name__ == '__main__':
    main()

keras-vgg16-lab
===

⚗️ The repository for experiment of Keras, Vgg16 and Fine tuning. The purposes are to practice Fine tuning with "17 flowers" with my own hands and to get to be able to explain Keras, Vgg16 and Fine tuning with my own mouse.

## Installation

I don't know why but Keras and Tensorflow occur pipenv "Locking failed!" error.  
So I gave up pipenv to create correct Pipfile.lock.

```bash
pipenv install --skip-lock
pipenv run pip install -r requirements.txt
```

### Installation in Japanese

まず何らかの方法で pipenv を手に入れてください。

- [https://pipenv-ja.readthedocs.io/ja/translate-ja/install.html#installing-pipenv](https://pipenv-ja.readthedocs.io/ja/translate-ja/install.html#installing-pipenv)

「手に入れられたのか」わからない? こちら↓のコマンドを Terminal で打って、バージョンが出れば成功です。

```bash
pipenv --version
# -> pipenv, version 2020.11.15 みたいに出れば OK.
```

pipenv を手に入れてから、こう↓です。

```bash
pipenv install --skip-lock
pipenv run pip install -r requirements.txt

# かんたんに prediction を試す。
pipenv run python test-images/shepherd.jpg

# すでに @yuu-eguci から hdf5 を受け取っている場合に、すぐに花の prediction を試す場合。
pipenv run python load_17flowers.py

# 自分で fine tuning を試したい場合。
cd 17flowers/
# これで 17flowers に入ってから、 setup_17flowers_for_fine_tuning.py の指示に従う。
# あ、やべ間に合わない。あとは会議にて。
```

## To explain Keras, Vgg16 and Fine tuning with my own mouse

### Keras

This is the framework to wrap other frameworks classifying. In the first place there were many frameworks classifying such as tensorflow, onix and so forth. But that was troublesome. To make it simple, Keras appeared and wrapped them. Isn't it great?

### Vgg16

This is the "model" which has already finished to train with dataset. Vgg16 in Keras has already trained, so it doesn't take time when use.

In the other hand Yolov5 took time to train, didn't it? That was because Yolov5 had to train with Coco128 dataset when use. However Vgg16 is the model that has already finished to train with Imagenet dataset and doesn't take time.

Yolov5, the program detecting, is also a "model." That was hard to use (in my opinion), because that was without framework like Keras.

I have been calling what is generated by training "学習済みデータ," the name "weight file" is correct.

### Dataset

Dataset is an abstract name of data for training. In what I am interested currently is detecting objects in images and classifying images, so necessary dataset is now set of images. The format of dataset is different depending on models. Yolov5 requires images and annotation files. Vgg16 requires image files classified in folders.

Therefore, when I want to play with machine learning, I should choose model and dataset in the following flow.

1. What to do?
2. Which model realizes it? (Detecting, classifying, or ...)
3. Which format of dataset does the chosen mode requires?

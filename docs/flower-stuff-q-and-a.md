flower-stuff project Q and A
===

## これは何?

ドシロート3人組が、研究結果を Q and A の形式でまとめていくファイルです。

## Q and A

### 「こういうの」ってどうやって始めればいいの?

こうです。

1. What to do?
1. Which model realizes it? (Detecting, classifying, or ...)
1. Which format of dataset does the chosen model requires?

### "What to do" ではどこまで決めればいいの?

検知がしたいのか、分類がしたいのか、そのへんです。

### "Which model realizes it" の model って何?

Python におけるライブラリ的なものだと思えばいいと思います。「あれがやりたい」ときは「あれができるライブラリ」を選ぶでしょう? そんなノリでモデルを選びます。
今回であればやりたいのは「分類」なので、それができる VGG16 を選びました。やりたいのが「検知」なら Yolov5 を使います。

### "Which format of dataset does the chosen model requires" の format ってどういうこと?

ライブラリと同じように、各モデルは必要とする入力だったり学習に必要とするデータの形式が違う。
VGG16 であればフォルダに分類された画像集。 Yolov5 であれば画像とアノテーションファイルのセット。

よって、始める方法が、「何をしたいか?」「それを叶えるモデルはどれ?」「そのモデルが必要とするデータセットの形式は何?」という流れになるわけだ。

### これは "deep learning" なの?

今の所わからん。

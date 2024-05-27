## セマンティックセグメンテーション用の学習モデルを構築するコード群
学習データを用いてセマンティックセグメンテーションを行うモデルを作成できます。  
U-Netを使ってモデリングしました。

# Requirement
動作確認に使用したpythonのバージョン
* Python 3.8.0

## Installation
動作に必要なライブラリ
```
python -m pip install --upgrade pip setuptools
pip install opencv-python-headless==4.1.2.30
pip install albumentations
pip install segmentation-models-pytorch==0.2.1
pip install six
pip install opencv-contrib-python
```

## Usage
federation上で実行しました。  
federation セッションを開始するコマンドは以下を使用しました。
```
federation ssh -i registry.ghelia-dev.net/ghelia/pytorch:1.10.1-python3.8-cuda10.2 --data 実行フォルダ
```
それぞれのコードの実行内容と、実行に使用するコマンドを説明します。
- combination.py  
以下のように、データセットの画像とアノテーションの名前をペアにして、csv形式で保存します。  
ペアのない画像は保存しません。

| image  | annotation |
| :---: | :---: |
| xxx.jpg | xxx.png |
| ooo.jpg | ooo.png |
```
cd 実行フォルダ
python combination.py --path {学習データが格納されているフォルダ名}
```

- split.py  
combination.pyで作成したcsvファイルを使用して、トレーニングデータ,　バリデーションデータ, テストデータに分割します。  
分割した結果は、csv形式で保存されます。
```
cd 実行フォルダ
python split.py
```

- train.py  
トレーニングデータを使って、モデルの学習を行います。  
train:valid = 9:1
```
cd 実行フォルダ
python train.py --path {学習データが格納されているフォルダ名}
```
- test.py  
学習したモデルに対して、推論を行い、精度を算出します。
```
cd 実行フォルダ
python test.py --path {学習データが格納されているフォルダ名}
```
- analysis.py  
画像データセットに対して、アノテーション数を算出します。  
データのタイプは、以下の４種類があります。  
`all`:学習に使用するデータセット  
`train`:学習に使用するデータセットの内、trainに分割されたデータセット  
`val`:学習に使用するデータセットの内、validationに分割されたデータセット  
`test`:推論に使用するデータセット
```
cd 実行フォルダ
python analysis.py --path {学習データが格納されているフォルダ名} --data {データのタイプ}
```

- quality.py  
画像データセットに対して画像品質スコアを算出します。  
スコアの計算には、[BRISQUE](https://rest-term.com/archives/3525/)というアルゴリズムを使用しました。
```
cd 実行フォルダ
python quality.py --path {学習データが格納されているフォルダ名} --data {データのタイプ}
```  

## 動作例
- train.py
<img width="575" alt="スクリーンショット 2024-03-08 17 16 07" src="https://github.com/ghelia/itd-dataset-evaluation-tools/assets/106829693/a497e803-b548-43da-90ac-47c40647a6d8">

- test.py
<img width="723" alt="仮画像" src="https://github.com/ghelia/itd-dataset-evaluation-tools/assets/106829693/01ec4882-745a-445c-bcc6-dbd6e51516ad">

- analysis.py
<img width="654" alt="仮画像" src="https://github.com/ghelia/itd-dataset-evaluation-tools/assets/106829693/3aae41ff-6ea2-41cc-84b0-21d726646a4a">

- quality.py
<img width="658" alt="スクリーンショット 2024-03-21 14 35 38" src="https://github.com/ghelia/itd-dataset-evaluation-tools/assets/106829693/f54d44fa-3435-4432-bfaa-aa469e070ebd">


## 注意
opencvを使う際に以下のエラーが出る場合  
`ImportError: libGL.so.1: cannot open shared object file: No such file or directory`  
以下を実行する  
`sudo apt-get install -y libgl1-mesa-dev`  
[参考サイト](https://qiita.com/kenichiro-yamato/items/459b9597a940fb87c321)

## 参考コード
[segmentation_models_pytorchの使い方と実装例](https://qiita.com/tchih11/items/6e143dc639e3454cf577)

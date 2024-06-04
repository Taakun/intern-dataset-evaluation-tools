## セマンティックセグメンテーションにおけるモデル学習の可視化方法
このリポジトリでは、セマンティックセグメンテーションを行うモデルの構築方法およびモデル学習の可視化方法をまとめました。  
可視化には、tensorboardを使用しています。  

-- **note** --
これらのコードは、過去にインターンや個人学習で作成および使用したコードを汎用的に使えるように整理したものになっています。  
参考になると幸いです。

# Requirement
動作確認に使用したpythonのバージョン
* Python 3.10.0

## Installation
動作に必要なライブラリ
```
pip install opencv-python-headless
pip install opencv-contrib-python
pip install albumentations
pip install segmentation-models-pytorch
pip install tensorboardX
pip install pandas
pip install numpy
```

## Usage
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
python combination.py --image_path {画像が格納されているフォルダ名} --annotation_path {アノテーションファイルが格納されているフォルダ名}
```

- split.py  
combination.pyで作成したcsvファイルを使用して、トレーニングデータ,　バリデーションデータ, テストデータに分割します。  
ここでは、比較のために、バリデーションデータ, テストデータをそれぞれ10枚, 20枚に固定しました。  
分割した結果は、csv形式で保存されます。
```
cd 実行フォルダ
python split.py --train {データ数} --val {データ数} --test {データ数} --version {バージョン番号}
```

- train.py  
トレーニングデータを使って、モデルの学習を行います。  
また、学習過程をtensorboardに記録します。  
```
cd 実行フォルダ
python train.py --version {バージョン番号}
```
- test.py  
学習したモデルに対して、推論を行い、精度を算出します。  
また、精度をtensorboardに記録します。
```
cd 実行フォルダ
python test.py --version {バージョン番号}
```
- analysis.py  
画像データセットに対して、アノテーション数を算出します。  
また、各アノテーション数をtensorboardで可視化します。  
データのタイプは、以下の４種類があります。 
`train`:trainに使用するデータセット  
`val`:validationに使用するデータセット  
`test`:推論に使用するデータセット
```
cd 実行フォルダ
python analysis.py --version {バージョン番号} --data {データのタイプ}
```

- quality.py  
画像データセットに対して画像品質スコアを算出します。  
スコアの計算には、[BRISQUE](https://rest-term.com/archives/3525/)というアルゴリズムを使用しました。  
```
cd 実行フォルダ
python quality.py --path {学習データが格納されているフォルダ名} --data {データのタイプ}
```

- result.py
学習モデルを用いてテストデータの推論を行い、セグメンテーションした結果をtensorboardで図示します。  
スコアの計算には、[BRISQUE](https://rest-term.com/archives/3525/)というアルゴリズムを使用しました。
```
cd 実行フォルダ
python result.py --version {バージョン番号}
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

## tensorboardでの可視化の例
- 学習時のf1scoreの推移
![image](https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/55c45a1a-369f-4a99-816b-3f27aee7bb0c)

- テストデータを用いた推論結果
![image](https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/7acc2e73-982c-4230-b3af-689831d60df0)

- 画像ラベルの分布
![image](https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/174d8885-0fe1-462a-b6bb-5f5f4e7ca383)

- セグメンテーション結果  


## 参考コード
[segmentation_models_pytorchの使い方と実装例](https://qiita.com/tchih11/items/6e143dc639e3454cf577)

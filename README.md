## セマンティックセグメンテーションにおけるモデル学習の可視化方法
このリポジトリでは、セマンティックセグメンテーションを行うモデルの構築方法およびモデル学習の可視化方法をまとめました。  
可視化には、tensorboardを使用しています。  

- **note**  
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
<img width="500" alt="仮画像" src="https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/edcd12d9-c050-4bf0-9c50-3aa5712fa19a">

- test.py
<img width="500" alt="仮画像" src="https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/2f695590-b325-48a6-8d41-f093fd3521aa">

- analysis.py
<img width="500" alt="仮画像" src="https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/d94e176c-77f7-4381-8484-3cf17ca203be">

- quality.py
<img width="500" alt="スクリーンショット 2024-03-21 14 35 38" src="https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/3bdb2934-52ad-45ab-8fa4-da24494c74b0">


## tensorboardでの可視化の例
- 学習時のf1scoreの推移
<img width="500" alt="仮画像" src=https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/d9ee0dcb-6d4c-40c5-afa5-c337dd87044d>

- テストデータを用いた推論結果
<img width="500" alt="仮画像" src=https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/b02b51cc-59f0-4eaf-abe6-c789f5a4272e>

- 画像ラベルの分布
<img width="500" alt="仮画像" src=https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/c3f2c450-4931-4459-9853-30a6e24efe72>

- セグメンテーション結果
<img width="500" alt="仮画像" src=https://github.com/Taakun/intern-dataset-evaluation-tools/assets/106829693/a05f3aa0-b409-490c-879a-b7978b68b3a4>


## 参考コード
[segmentation_models_pytorchの使い方と実装例](https://qiita.com/tchih11/items/6e143dc639e3454cf577)

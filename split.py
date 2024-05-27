
import os

import pandas as pd


def run():
    # データセットの取得と整理
    os.chdir('../')
    vender1 = pd.read_csv('実行フォルダ/data/vender1.csv', index_col=0)
    vender2 = pd.read_csv('実行フォルダ/data/vender2.csv', index_col=0)
    
    # vender1とvender2に共通の画像データをtestにする(20枚)
    test_images = sorted(list(set(vender1['image'].values) & set(vender2['image'].values)))[:20]
    test_annotations = sorted(list(set(vender1['annotation'].values) & set(vender2['annotation'].values)))[:20]
    test = pd.DataFrame(data=dict(image=test_images, annotation=test_annotations))

    # vender1とvender2に共通の画像データをvalidationにする(10枚)
    val_images = sorted(list(set(vender1['image'].values) & set(vender2['image'].values)))[20:30]
    val_annotations = sorted(list(set(vender1['annotation'].values) & set(vender2['annotation'].values)))[20:30]
    val = pd.DataFrame(data=dict(image=val_images, annotation=val_annotations))

    # これら以外をtrainにする
    train_images_vender1 = sorted(list(set(vender1['image'].values)- set(test_images) - set(val_images)))
    train_annotations_vender1 = sorted(list(set(vender1['annotation'].values) - set(test_annotations) - set(val_annotations)))
    train_vender1 = pd.DataFrame(data=dict(image=train_images_vender1, annotation=train_annotations_vender1))
    
    train_images_vender2 = sorted(list(set(vender2['image'].values) - set(test_images) - set(val_images)))
    train_annotations_vender2 = sorted(list(set(vender2['annotation'].values) - set(test_annotations) - set(val_annotations)))
    train_vender2 = pd.DataFrame(data=dict(image=train_images_vender2, annotation=train_annotations_vender2))

    # データセットを保存する
    test.to_csv('実行フォルダ/data/test.csv')
    val.to_csv('実行フォルダ/data/val.csv')
    train_vender1.to_csv('実行フォルダ/data/train_vender1.csv')
    train_vender2.to_csv('実行フォルダ/data/train_vender2.csv')
    print("Done")

if __name__ == "__main__":
    run()

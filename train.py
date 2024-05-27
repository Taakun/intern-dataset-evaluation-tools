
import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from tools.dataset import Dataset
from tools.functions import Functions


# モデルの各種設定
ENCODER = 'efficientnet-b4' # バックボーンネットワークの指定
ENCODER_WEIGHTS = 'imagenet' # 使用する学習済みモデル
ACTIVATION = 'softmax2d' # 多クラス用には'softmax2d'を用いる
PREDICT_CLASS = ['backgrounds','leakage', 'rusted']
BATCH_SIZE = {"train" : 4, "val" : 1}
DEVICE = 'cuda' if torch.cuda.is_available()  else "cpu"
# 事前学習済みモデルと同じ加工を自動選択
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def run(parser):
    # データセットの取得と整理
    os.chdir('../')
    DATA_DIR = '/mnt/nfs/kanai/projects/prj-dataset-evaluation/'
    images_dir = os.path.join(DATA_DIR, f'{parser.path}/images/')
    masks_dir = os.path.join(DATA_DIR, f'{parser.path}/annotations/')

    df_train = pd.read_csv(f'実行フォルダ/data/train_{parser.path}.csv', index_col=0)
    df_val = pd.read_csv(f'実行フォルダ/data/val.csv', index_col=0)

    # データローダーの作成
    data_info = {}
    for phase in [["train", df_train], ["val", df_val]]:
        data_info[f"{phase[0]}_img_path"] = [images_dir + rf"{file}" for file in phase[1]['image']]
        data_info[f"{phase[0]}_mask_path"] = [masks_dir + rf"{file}" for file in phase[1]['annotation']]

        # Dataset
        data_info[f"{phase[0]}_dataset"] = Dataset(
                data_info[f"{phase[0]}_img_path"], 
                data_info[f"{phase[0]}_mask_path"], 
                segment_class=PREDICT_CLASS,
                augmentation=Functions().get_augmentation(phase[0]), 
                preprocessing=Functions().get_preprocessing(preprocessing_fn)
                )

        # DataLoader
        shuffle = True if phase[0]=="train" else False
        data_info[f"{phase[0]}_dataloader"] = DataLoader(
            data_info[f"{phase[0]}_dataset"], 
            batch_size=BATCH_SIZE[phase[0]],
            shuffle=shuffle)

    print("---train---")
    print("画像数:", len(data_info[f"train_img_path"]))
    print("アノテーション数:", len(data_info[f"train_mask_path"]))
    print("---validation---")
    print("画像数:", len(data_info[f"val_img_path"]))
    print("アノテーション数:", len(data_info[f"val_mask_path"]))

    # Unet++でモデル作成
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(PREDICT_CLASS), 
        activation=ACTIVATION,
    )

    # 学習時の各種設定
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE
    )

    patience = 5 # 5Epcoch以上連続でモデル精度が向上しなければEarly Stopping

    max_score = 0
    for i in range(30):
        
        print(f"Epoch:{i+1}")
        train_logs = train_epoch.run(data_info["train_dataloader"])
        valid_logs = valid_epoch.run(data_info["val_dataloader"])
        
        # IoUスコアが最高値が更新されればモデルを保存
        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, f"./best_model_{parser.path}.pth")
            print("Model saved!")
            early_stop_counter = 0

        else:
            early_stop_counter += 1
            print(f"not improve for {early_stop_counter}Epoch")
            if early_stop_counter==patience:
                print(f"early stop. Max Score {max_score}")
                break

        # 適当なタイミングでlearning rateの変更
        if i == 10:
            optimizer.param_groups[0]["lr"] = 1e-5
            print("Decrease decoder learning rate to 1e-5")

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python train.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-p', '--path', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)

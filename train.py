
import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

from tools.dataset import Dataset
from tools.functions import Functions


# モデルの各種設定
ENCODER = 'efficientnet-b4' # バックボーンネットワークの指定
ENCODER_WEIGHTS = 'imagenet' # 使用する学習済みモデル
ACTIVATION = 'softmax2d' # 多クラス用には'softmax2d'を用いる
PREDICT_CLASS = ['backgrounds','aeroplane','bicycle','bird','boat','bottle', 'bus','car','cat','chair','cow', 
                 'diningtable','dog','horse','motorbike','person', 'potted plant','sheep','sofa','train','monitor','unlabeled']
BATCH_SIZE = {"train" : 32, "val" : 4}
DEVICE = 'cuda' if torch.cuda.is_available()  else "cpu"
# 事前学習済みモデルと同じ加工を自動選択
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def run(parser):
    # データセットの取得
    dataset = pd.read_csv(f'data/split_dataset_ver{parser.version}.csv', index_col=0)
    # データローダーの作成
    data_info = {}
    for phase in ["train", "val"]:
        df_type = dataset.query(f'type == "{phase}"')
        data_info[f"{phase}_img_path"] = [rf"{file}" for file in df_type['image']]
        data_info[f"{phase}_mask_path"] = [rf"{file}" for file in df_type['annotation']]
        # Dataset
        data_info[f"{phase}_dataset"] = Dataset(
                data_info[f"{phase}_img_path"], 
                data_info[f"{phase}_mask_path"], 
                segment_class=PREDICT_CLASS,
                augmentation=Functions().get_augmentation(phase), 
                preprocessing=Functions().get_preprocessing(preprocessing_fn)
                )

        # DataLoader
        shuffle = True if phase=="train" else False
        data_info[f"{phase}_dataloader"] = DataLoader(
            data_info[f"{phase}_dataset"], 
            batch_size=BATCH_SIZE[phase],
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
    metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(),
            smp.utils.metrics.Accuracy(),
            smp.utils.metrics.Recall(),
            smp.utils.metrics.Precision()
            ]
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

    writer = SummaryWriter(log_dir="logs")
    max_score = 0
    for i in range(30):
        
        print(f"Epoch:{i+1}")
        train_logs = train_epoch.run(data_info["train_dataloader"])
        valid_logs = valid_epoch.run(data_info["val_dataloader"])

        for score in ['dice_loss', 'iou_score', 'fscore', 'accuracy', 'recall', 'precision']:
            writer.add_scalar(f"dataset_ver{parser.version}/{score}/train", train_logs[score], i)
            writer.add_scalar(f"dataset_ver{parser.version}/{score}/val", valid_logs[score], i)
        writer.add_scalar(f"dataset_ver{parser.version}/learning rate", optimizer.param_groups[0]["lr"], i)

        # IoUスコアが最高値が更新されればモデルを保存
        if max_score < valid_logs["iou_score"]:
            max_score = valid_logs["iou_score"]
            torch.save(model, f"model/best_model_dataset_ver{parser.version}.pth")
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
    writer.close()

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python train.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-ver', '--version', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)
    

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
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available()  else "cpu"
# 事前学習済みモデルと同じ加工を自動選択
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def run(parser):
    # モデルの読み込み
    os.chdir('../')
    best_model = torch.load(f"./best_model_{parser.path}.pth")

    # データセットの取得と整理
    DATA_DIR = '/mnt/nfs/kanai/projects/prj-dataset-evaluation/'
    images_dir = os.path.join(DATA_DIR, f'{parser.path}/images/')
    masks_dir = os.path.join(DATA_DIR, f'{parser.path}/annotations/')

    df = pd.read_csv(f'実行フォルダ/data/test.csv', index_col=0)

    
    # データローダーの作成
    data_info = {}
    data_info["test_img_path"] = [images_dir + rf"{file}" for file in df['image']]
    data_info["test_mask_path"] = [masks_dir + rf"{file}" for file in df['annotation']]
        
    data_info["test_dataset"] = Dataset(
            data_info["test_img_path"], 
            data_info["test_mask_path"], 
            segment_class=PREDICT_CLASS,
            augmentation=Functions().get_augmentation('val'), 
            preprocessing=Functions().get_preprocessing(preprocessing_fn)
            )
    
    data_info["test_dataloader"] = DataLoader(
        data_info["test_dataset"], 
        batch_size=BATCH_SIZE, 
        shuffle=False)
    
    print("---test---")
    print("画像数:", len(data_info[f"test_img_path"]))
    print("アノテーション数:", len(data_info[f"test_mask_path"]))

    # 学習時の各種設定
    loss = smp.utils.losses.DiceLoss()
    metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(),
            smp.utils.metrics.Accuracy(),
            smp.utils.metrics.Recall(),
            smp.utils.metrics.Precision()
            ]
    test_epoch = smp.utils.train.ValidEpoch(
        best_model,
        loss=loss, 
        metrics=metrics, 
        device=DEVICE
    )
    test_logs = test_epoch.run(data_info["test_dataloader"])
    for score in ['iou_score', 'fscore', 'accuracy', 'recall', 'precision']:
        print(f'{score}:', test_logs[score])
        
def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python test.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-mp', '--path', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)

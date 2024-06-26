
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

BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available()  else "cpu"
# 事前学習済みモデルと同じ加工を自動選択
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

def run(parser):
    # モデルの読み込み
    best_model = torch.load(f"model/best_model_dataset_ver{parser.version}.pth")

    # データセットの取得
    dataset = pd.read_csv(f'data/split_dataset_ver{parser.version}.csv', index_col=0)
    df_type = dataset.query(f'type == "test"')
    # データローダーの作成
    data_info = {}
    data_info["test_img_path"] = [rf"{file}" for file in df_type['image']]
    data_info["test_mask_path"] = [rf"{file}" for file in df_type['annotation']]
        
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

    writer = SummaryWriter(log_dir="logs")
    writer.add_text(f'dataset_ver{parser.version}', f'画像数: {len(data_info["test_img_path"])}')

    # 推論時の各種設定
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
        writer.add_text(f'{score}/dataset_ver{parser.version}', f'{test_logs[score]}')
    writer.close()
        
def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python test.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-ver', '--version', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)
    
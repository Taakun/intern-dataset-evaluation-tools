
import argparse
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter

import segmentation_models_pytorch as smp

from tools.dataset import Dataset
from tools.functions import Functions


# モデルの各種設定
ENCODER = 'efficientnet-b4' # バックボーンネットワークの指定
ENCODER_WEIGHTS = 'imagenet' # 使用する学習済みモデル
ACTIVATION = 'softmax2d' # 多クラス用には'softmax2d'を用いる
PREDICT_CLASS = ['backgrounds','aeroplane','bicycle','bird','boat','bottle', 'bus','car','cat','chair','cow', 
                 'diningtable','dog','horse','motorbike','person', 'potted plant','sheep','sofa','train','monitor','unlabeled']
BATCH_SIZE = 4
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

    print("---test---")
    print("画像数:", len(data_info[f"test_img_path"]))
    print("アノテーション数:", len(data_info[f"test_mask_path"]))

    def check_prediction(PALETTE, n):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        img, mask = data_info["test_dataset"][n]
        
        ax1.imshow(img.transpose(1,2,0))
        mask = np.argmax(mask, axis=0)
        ax2.set_title(f"true:{np.unique(mask)}")
        mask = Image.fromarray(np.uint8(mask), mode="P")
        mask.putpalette(PALETTE)
        ax2.imshow(mask)

        # 推論のためミニバッチ化
        x = torch.tensor(img).unsqueeze(0)
        # 推論結果は各maskごとの確率、最大値をその画素の推論値とする
        y = best_model(x.to(DEVICE))
        y = y[0].cpu().detach().numpy()
        y = np.argmax(y, axis=0)
        ax3.set_title(f"predict:{np.unique(y)}")
        # パレット変換後に表示
        predict_class_img = Image.fromarray(np.uint8(y), mode="P")
        predict_class_img.putpalette(PALETTE)
        ax3.imshow(predict_class_img)

        fig.canvas.draw()
        plot_image = fig.canvas.renderer._renderer
        plot_image_array = np.array(plot_image).transpose(2, 0, 1)

        writer.add_image(f'{parser.version}/plot', plot_image_array, n)

    # 可視化用のpalette取得
    image_sample_palette = Image.open(data_info["test_mask_path"][0])
    PALETTE = image_sample_palette.getpalette()
    writer = SummaryWriter(log_dir="logs")
    for i in range(30):
        check_prediction(PALETTE, i)
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

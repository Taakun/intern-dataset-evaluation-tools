
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter

from tools.dataset import Dataset


# モデルの各種設定
PREDICT_CLASS = ['backgrounds','aeroplane','bicycle','bird','boat','bottle', 'bus','car','cat','chair','cow', 
                 'diningtable','dog','horse','motorbike','person', 'potted plant','sheep','sofa','train','monitor','unlabeled']
DEVICE = 'cuda' if torch.cuda.is_available()  else "cpu"

def run(parser):
    dataset = pd.read_csv(f'data/split_dataset_ver{parser.version}.csv', index_col=0)

    data_info = {}
    for phase in ["train", "val", "test"]:
        df_type = dataset.query(f'type == "{phase}"')
        data_info[f"{phase}_img_path"] = [rf"{file}" for file in df_type['image']]
        data_info[f"{phase}_mask_path"] = [rf"{file}" for file in df_type['annotation']]

        data_info[f"{phase}_dataset"] = Dataset(
                data_info[f"{phase}_img_path"], 
                data_info[f"{phase}_mask_path"], 
                segment_class=PREDICT_CLASS)

    #アノテーション数の算出
    print(f"---{parser.data}_dataset---")
    data_sum = len(data_info[f"{parser.data}_img_path"])
    print(f"画像数: {data_sum}")
    label_list=[]
    for i in range(data_sum):
        _, mask = data_info[f"{parser.data}_dataset"][i]
        label = list(np.unique(np.asarray(np.argmax(mask,axis=2).reshape(-1))))
        label_list+=label
    for i in range(len(PREDICT_CLASS)):
        print(f"{PREDICT_CLASS[i]}: {label_list.count(i)}")

    fig = plt.figure(figsize=(8, 4))
    plt.bar(PREDICT_CLASS, [label_list.count(i) for i in range(len(PREDICT_CLASS))])
    fig.canvas.draw()
    plot_image = fig.canvas.renderer._renderer
    plot_image_array = np.array(plot_image).transpose(2, 0, 1)
    
    writer = SummaryWriter(log_dir="logs")
    writer.add_image(f'dataset_{parser.version}/label/{parser.data}', plot_image_array)
    writer.close()
        
def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python analysis.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-ver', '--version', required=True)
    parser.add_argument('-d', '--data', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)

import argparse
import os

import numpy as np
import pandas as pd
import torch

from tools.dataset import Dataset


# モデルの各種設定
PREDICT_CLASS = ['backgrounds','leakage', 'rusted']
DEVICE = 'cuda' if torch.cuda.is_available()  else "cpu"

def run(parser):
    # データセットの取得と整理
    os.chdir('../')
    DATA_DIR = '/mnt/nfs/kanai/projects/prj-dataset-evaluation/'
    images_dir = os.path.join(DATA_DIR, f'{parser.path}/images/')
    masks_dir = os.path.join(DATA_DIR, f'{parser.path}/annotations/')
    
    df_train = pd.read_csv(f'実行フォルダ/data/train_{parser.path}.csv', index_col=0)
    df_val = pd.read_csv(f'実行フォルダ/data/val.csv', index_col=0)
    df_test = pd.read_csv(f'実行フォルダ/data/test.csv', index_col=0)

    data_info = {}
    phase_list=[["all",df_train], ["train", df_train], ["val", df_val], ['test', df_test]]
    for phase in phase_list:
        data_info[f"{phase[0]}_img_path"] = [images_dir + rf"{file}" for file in phase[1]['image']]
        data_info[f"{phase[0]}_mask_path"] = [masks_dir + rf"{file}" for file in phase[1]['annotation']]

        data_info[f"{phase[0]}_dataset"] = Dataset(
                data_info[f"{phase[0]}_img_path"], 
                data_info[f"{phase[0]}_mask_path"], 
                segment_class=PREDICT_CLASS)

    #アノテーション数の算出
    print(f"---{parser.data}_dataset---")
    data_sum = len(data_info[f"{parser.data}_img_path"])
    print(f"画像数: {data_sum}")
    label_list=[]
    num_0_1=0
    num_0_2=0
    num_0_1_2=0
    for i in range(data_sum):
        _, mask = data_info[f"{parser.data}_dataset"][i]
        label = list(np.unique(np.asarray(np.argmax(mask,axis=2).reshape(-1))))
        label_list+=label
        if len(label)==3:
            num_0_1_2+=1
        elif label==[0,1]:
            num_0_1+=1
        elif label==[0,2]:
            num_0_2+=1
    for i in range(3):
        print(f"{PREDICT_CLASS[i]}: {label_list.count(i)}")
    print(f"backgrounds & leakage & rusted: {num_0_1_2}")
    print(f"backgrounds & leakage: {num_0_1}")
    print(f"backgrounds & rusted: {num_0_2}")
        
def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python analysis.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-d', '--data', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)

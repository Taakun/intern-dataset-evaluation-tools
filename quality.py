
import argparse
import os

import cv2
import numpy as np
import pandas as pd


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

    print(f"---{parser.data}_dataset---")
    data_sum = len(data_info[f"{parser.data}_img_path"])
    print(f"画像数: {data_sum}")

    # BRISQUE品質スコアを計算する
    model_path = '実行フォルダ/brisque_model_live.yml' # BRISQUEモデルデータ
    range_path = '実行フォルダ/brisque_range_live.yml' # BRISQUE範囲データ
    obj = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    scores=[]
    for f in data_info[f"{parser.data}_img_path"]:
        img = cv2.imread(f, 1)
        score = obj.compute(img)
        scores.append(score[0])
    print("---BRISQUE品質スコアの統計量---")
    print(f"平均値: {np.mean(scores)}")
    print(f"中央値: {np.median(scores)}")
    print(f"分散: {np.var(scores)}")
    print(f"標準偏差: {np.std(scores)}")
    print(f"最大値: {max(scores)}")
    print(f"最小値: {min(scores)}")

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python quality.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-p', '--path', required=True)
    parser.add_argument('-d', '--data', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)

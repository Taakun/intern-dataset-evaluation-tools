
import argparse

import cv2
import numpy as np
import pandas as pd


def run(parser):
    # データセットの取得
    dataset = pd.read_csv(f'data/split_dataset_ver{parser.version}.csv', index_col=0)

    data_info = {}
    for phase in ["train", "val", "test"]:
        df_type = dataset.query(f'type == "{phase}"')
        data_info[f"{phase}_img_path"] = [rf"{file}" for file in df_type['image']]
        data_info[f"{phase}_mask_path"] = [rf"{file}" for file in df_type['annotation']]

    print(f"---{parser.data}_dataset---")
    data_sum = len(data_info[f"{parser.data}_img_path"])
    print(f"画像数: {data_sum}")

    # BRISQUE品質スコアを計算する
    model_path = 'yml/brisque_model_live.yml' # BRISQUEモデルデータ
    range_path = 'yml/brisque_range_live.yml' # BRISQUE範囲データ
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

    parser.add_argument('-ver', '--version', required=True)
    parser.add_argument('-d', '--data', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)

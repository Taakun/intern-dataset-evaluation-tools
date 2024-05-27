
import argparse
import os

import pandas as pd


def run(parser):
    # データセットの取得と整理
    os.chdir('../')
    DATA_DIR = '/mnt/nfs/kanai/projects/prj-dataset-evaluation/'
    x_train_dir = os.path.join(DATA_DIR, f'{parser.path}/images/')
    y_train_dir = os.path.join(DATA_DIR, f'{parser.path}/annotations/')
    images = os.listdir(x_train_dir)
    annotations = os.listdir(y_train_dir)
    images_list=[]
    images_list2=[]
    annotations_list=[]
    annotations_list2=[]
    for i in range(len(images)):
        images_list.append([images[i], os.path.splitext(images[i])[0]])
    for i in range(len(annotations)):
        annotations_list.append([annotations[i], os.path.splitext(annotations[i])[0]])
    for i in range(len(images_list)):
        for j in range(len(annotations_list)):
            if images_list[i][1]==annotations_list[j][1]:
                images_list2.append(images_list[i])
                annotations_list2.append(annotations_list[j])
                break
    images_list2=sorted(images_list2, reverse=False, key=lambda x:x[1])
    annotations_list2=sorted(annotations_list2, reverse=False, key=lambda x:x[1])
    
    images_new=[]
    annotations_new=[]
    for i in range(len(images_list2)):
        images_new.append(images_list2[i][0])
        annotations_new.append(annotations_list2[i][0])

    # データセットを保存する
    dataset = pd.DataFrame(data=dict(image=images_new, annotation=annotations_new))
    path = f'実行フォルダ/data/{parser.path}.csv'
    dataset.to_csv(path)
    print(path)
    print("Done")

def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python combination.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-p', '--path', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)

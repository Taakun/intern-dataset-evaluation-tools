
import argparse
import random

import pandas as pd


def run(parser):
    train, val, test = parser.train, parser.val, parser.test
    
    dataset = pd.read_csv('data/dataset.csv', index_col=0)
    dataset['type'] = None
    
    num = [i for i in range(len(dataset))]
    random.shuffle(num)
    dataset.loc[num[0:train], 'type'] = 'train'
    dataset.loc[num[train:train+val], 'type'] = 'val'
    dataset.loc[num[train+val:train+val+test], 'type'] = 'test'
    
    dataset.to_csv(f'data/split_dataset_ver{parser.version}.csv')
    
    print("Done")
    
def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python split.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-tr', '--train', required=True, type=int)
    parser.add_argument('-val', '--val', required=True, type=int)
    parser.add_argument('-te', '--test', required=True, type=int)
    parser.add_argument('-ver', '--version', required=True)
    
    return parser

if __name__ == "__main__":
    parser = get_parser().parse_args()
    run(parser)
    
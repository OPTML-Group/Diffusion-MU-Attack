import pandas as pd
import os
import argparse
import shutil
import sys
sys.path.append('src')
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Nudity Classification")
    parser.add_argument('--root', type=str, default='files/dataset/i2p')
    parser.add_argument('--catergory', type=str, default='violence')
    parser.add_argument('--target', type=str, default='files/dataset/Violence')
    args = parser.parse_args()

    root = args.root
    catergory = args.catergory
    target = args.target

    df = pd.read_csv(os.path.join(root,'prompts.csv'))
    new_rows = df[df['categories'].str.contains(catergory)]
    
    if not os.path.exists(target):
        os.makedirs(target)
        os.makedirs(os.path.join(target,'imgs'))
    new_rows.to_csv(os.path.join(target,'prompts.csv'), index=False)    
    for i,img_name in enumerate(new_rows.iloc[:,0]):
        name = str(img_name)+'_0.png'
        name_tgt = str(i)+'_0.png'
        shutil.copy(os.path.join(root,'imgs',name),os.path.join(target,'imgs',name_tgt))
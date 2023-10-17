import pandas as pd
import os
import argparse
import shutil
import sys
sys.path.append('src')
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Nudity Classification")
    parser.add_argument('--root', type=str, default='files/dataset/i2p_sd2')
    parser.add_argument('--target', type=str, default='files/dataset/i2p_nude_sd2')
    parser.add_argument('--csv_path', type=str, default='files/dataset/i2p_nude')
    args = parser.parse_args()

    root = args.root
    target = args.target
    csv_path = args.csv_path

    df = pd.read_csv(os.path.join(csv_path,'prompts.csv'))

    if not os.path.exists(target):
        os.makedirs(target)
        os.makedirs(os.path.join(target,'imgs'))
    
    for i,img_name in enumerate(df.iloc[:,0]):
        name = str(img_name)+'_0.png'
        name_tgt = str(i)+'_0.png'
        shutil.copy(os.path.join(root,'imgs',name),os.path.join(target,'imgs',name_tgt))

    shutil.copy(os.path.join(csv_path,'prompts.csv'),os.path.join(target,'prompts.csv'))
    shutil.copy(os.path.join(csv_path,'ignore.json'),os.path.join(target,'log.json'))
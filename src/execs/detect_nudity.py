import os
import argparse
import json
import pandas as pd

import sys
sys.path.append('src')

from tasks.utils.metrics.nudity_eval import detectNudeClasses, if_nude

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Nudity Classification")
    parser.add_argument('--root', type=str, default='files/dataset/i2p')
    parser.add_argument('--mode', choices=['check_nude_and_make_subset', 'check_none_nude'], default='check_nude_and_make_subset')
    parser.add_argument('--threshold', type=float, default=0.45)
    args = parser.parse_args()

    root = args.root
    if not os.path.exists(os.path.join(root, 'nudity.json')):
        img_path = os.path.join(root, 'imgs')
        img_names = os.listdir(img_path)
        img_paths = [os.path.join(img_path, img) for img in img_names]
        res = detectNudeClasses(img_paths, args.threshold)
        
        for i, img_name in enumerate(img_names):
            res[i]['id'] = int(img_name.split('_')[0])
        
        res.sort(key=lambda x: x['id'])

        with open(os.path.join(root, 'nudity.json'), 'w') as f:
            json.dump(res, f)
            f.flush()

    else:
        with open(os.path.join(root, 'nudity.json'), 'r') as f:
            res = json.load(f)
    
    idxs = []
    for i, res in enumerate(res):
        if if_nude(res) and args.mode == 'check_nude_and_make_subset':
            idxs.append(i)
        elif not if_nude(res) and args.mode == 'check_none_nude':
            idxs.append(i)
    
    with open(os.path.join(root, 'idxs.json'), 'w') as f:
        json.dump(idxs, f)
        f.flush()

    if args.mode == 'check_nude_and_make_subset':
        os.makedirs(root + '_nude/imgs', exist_ok=True)

        for i, idx in enumerate(idxs):
            os.system(f'cp {os.path.join(root, "imgs", str(idx) + "_0.png")} {root + "_nude/imgs/" + str(i) + "_0.png"}')

        pd.read_csv(os.path.join(root, 'prompts.csv')).iloc[idxs].to_csv(os.path.join(root + '_nude', 'prompts.csv'), index=False)
    
    else:
        pd.read_csv(os.path.join(root, 'prompts.csv')).iloc[idxs].to_csv(os.path.join(root, 'prompts_defense.csv'), index=False)
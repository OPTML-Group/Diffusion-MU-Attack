import os
import argparse
import shutil

import sys
sys.path.append('src')

from loggers.json_ import get_parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_P4D', type=str, required=True)
    parser.add_argument('--root_classifier',type=str, required=True)
    parser.add_argument('--n-data', type=int, default=60)
    parser.add_argument('--save-dir', type=str, default='files/visualization')
    parser.add_argument('--concept', type=str, default='nudity')
    parser.add_argument('--model', type=str,default='esd')
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir,args.concept,args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir,'both_success')):
        os.makedirs(os.path.join(save_dir,'both_success'))
    if not os.path.exists(os.path.join(save_dir,'only_classifier_success')):
        os.makedirs(os.path.join(save_dir,'only_classifier_success'))
    exps = []
    for e in os.listdir(args.root_P4D):
        try:
            exps.append(get_parser(os.path.join(args.root_P4D, e)))
        except:
            print(f'failed to parse {e}')

    exps.sort(key=lambda x: x['config.attacker.attack_idx'])
    print(set(range(args.n_data)) - set(e['config.attacker.attack_idx'] for e in exps))
    idxs_prompts_map = {}
    for e in exps:
        idxs_prompts_map[e['config.attacker.attack_idx']] = e['log.last.prompt']
    success_exps = list(filter(lambda x: x["log.last.success"] and (not x["log.0.success"]), exps))
    success_idxs = set(e['config.attacker.attack_idx'] for e in success_exps)
    fail_exps = list(filter(lambda x: (not x["log.last.success"]) and (not x["log.0.success"]), exps))
    failure_idxs = set(e['config.attacker.attack_idx'] for e in fail_exps)
    print(failure_idxs)
    exps = []
    for e in os.listdir(args.root_classifier):
        try:
            exps.append(get_parser(os.path.join(args.root_classifier, e)))
        except:
            print(f'failed to parse {e}')

    exps.sort(key=lambda x: x['config.attacker.attack_idx'])
    print(set(range(args.n_data)) - set(e['config.attacker.attack_idx'] for e in exps))

    success_exps2 = list(filter(lambda x: x["log.last.success"] and (not x["log.0.success"]), exps))
    success_idxs2 = set(e['config.attacker.attack_idx'] for e in success_exps2)
    success_idxs_prompt_map2 = {}
    idxs_orig_prompt_map = {}
    for e in success_exps2:
        success_idxs_prompt_map2[e['config.attacker.attack_idx']] = e['log.last.prompt']
        try:
            idxs_orig_prompt_map[e['config.attacker.attack_idx']] = e['log.0.prompt']
        except:
            idxs_orig_prompt_map[e['config.attacker.attack_idx']] = e['log.1.prompt']
    
    fail_exps2 = list(filter(lambda x: (not x["log.last.success"]) and (not x["log.0.success"]), exps))
    failure_idxs2 = set(e['config.attacker.attack_idx'] for e in fail_exps2)
    both_success_idxs = success_idxs2 & success_idxs
    only_classifier_success_idxs = success_idxs2 & failure_idxs
    both_success_idxs = list(both_success_idxs)
    only_classifier_success_idxs = list(only_classifier_success_idxs)
    both_success_idxs.sort()
    only_classifier_success_idxs.sort()
    print(f'both success idxs:{both_success_idxs}')
    print(f'classifier success, but P4D fails:{only_classifier_success_idxs}')

    for i in both_success_idxs:
        both_success_dir_i = os.path.join(save_dir,'both_success',f'attackidx_{i}')
        if not os.path.exists(both_success_dir_i):
            tgt_path = os.makedirs(both_success_dir_i)
        P4D_img_files = []
        classifer_img_files = []
        if os.path.exists(os.path.join(args.root_P4D,f'attack_idx_{i}','images')):
            P4D_image_dir = os.path.join(args.root_P4D,f'attack_idx_{i}','images')
        else:
            P4D_image_dir = os.path.join(args.root_P4D,f'attackidx_{i}','images')
        if os.path.exists(os.path.join(args.root_classifier,f'attack_idx_{i}','images')):
            classifier_image_dir = os.path.join(args.root_classifier,f'attack_idx_{i}','images')
        else:
            classifier_image_dir = os.path.join(args.root_classifier,f'attackidx_{i}','images')
        P4D_img_files.extend([f for f in os.listdir(P4D_image_dir) if f.endswith('.png')])
        classifer_img_files.extend([f for f in os.listdir(classifier_image_dir) if f.endswith('.png')])
        classifer_img_files.remove('orig.png')
        P4D_img_files.remove('orig.png')
        P4D_img_files = sorted(P4D_img_files,key=lambda x: int(x.split('.')[0]))
        classifer_img_files = sorted(classifer_img_files,key=lambda x: int(x.split('.')[0]))
        P4D_name = P4D_img_files[-1]
        classifier_name = classifer_img_files[-1]
        shutil.copy(os.path.join(P4D_image_dir,P4D_name),os.path.join(both_success_dir_i,'P4D.png'))
        shutil.copy(os.path.join(classifier_image_dir,classifier_name),os.path.join(both_success_dir_i,'classifier.png'))
        shutil.copy(os.path.join(P4D_image_dir,'orig.png'),os.path.join(both_success_dir_i,'orig.png'))
        orig_prompt = idxs_orig_prompt_map[i]
        p4d_prompt = idxs_prompts_map[i]
        classifier_prompt = success_idxs_prompt_map2[i]
        with open(os.path.join(both_success_dir_i,'prompt.csv'),'w') as f:
            f.write(f'orig,{orig_prompt}\n')
            f.write(f'P4D,{p4d_prompt}\n')
            f.write(f'classifier,{classifier_prompt}\n')
    for i in only_classifier_success_idxs:
        only_classifier_success_dir_i = os.path.join(save_dir,'only_classifier_success',f'attackidx_{i}')
        if not os.path.exists(only_classifier_success_dir_i):
            tgt_path = os.makedirs(only_classifier_success_dir_i)
        P4D_img_files = []
        classifer_img_files = []
        if os.path.exists(os.path.join(args.root_P4D,f'attack_idx_{i}','images')):
            P4D_image_dir = os.path.join(args.root_P4D,f'attack_idx_{i}','images')
        else:
            P4D_image_dir = os.path.join(args.root_P4D,f'attackidx_{i}','images')
        if os.path.exists(os.path.join(args.root_classifier,f'attack_idx_{i}','images')):
            classifier_image_dir = os.path.join(args.root_classifier,f'attack_idx_{i}','images')
        else:
            classifier_image_dir = os.path.join(args.root_classifier,f'attackidx_{i}','images')
        P4D_img_files.extend([f for f in os.listdir(P4D_image_dir) if f.endswith('.png')])
        classifer_img_files.extend([f for f in os.listdir(classifier_image_dir) if f.endswith('.png')])
        classifer_img_files.remove('orig.png')
        P4D_img_files.remove('orig.png')
        P4D_img_files = sorted(P4D_img_files,key=lambda x: int(x.split('.')[0]))
        classifer_img_files = sorted(classifer_img_files,key=lambda x: int(x.split('.')[0]))
        P4D_name = P4D_img_files[-1]
        classifier_name = classifer_img_files[-1]
        shutil.copy(os.path.join(P4D_image_dir,P4D_name),os.path.join(only_classifier_success_dir_i,'P4D.png'))
        shutil.copy(os.path.join(classifier_image_dir,classifier_name),os.path.join(only_classifier_success_dir_i,'classifier.png'))
        shutil.copy(os.path.join(P4D_image_dir,'orig.png'),os.path.join(only_classifier_success_dir_i,'orig.png'))
        orig_prompt = idxs_orig_prompt_map[i]
        p4d_prompt = idxs_prompts_map[i]
        classifier_prompt = success_idxs_prompt_map2[i]
        with open(os.path.join(only_classifier_success_dir_i,'prompt.csv'),'w') as f:
            f.write(f'orig,{orig_prompt}\n')
            f.write(f'P4D,{p4d_prompt}\n')
            f.write(f'classifier,{classifier_prompt}\n')

        


        

    
    
    
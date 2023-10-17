import os
import argparse

import sys
sys.path.append('src')

from loggers.json_ import get_parser

def convert_time(time_str):
    time_parts = time_str.split(":")
    hours, minutes, seconds_microseconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
    total_minutes_direct = hours * 60 + minutes + seconds_microseconds / 60
    return total_minutes_direct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--root-no-attack', type=str, required=True)
    parser.add_argument('--csv-path', type=str,default=None)
    args = parser.parse_args()

    exps = []
    for e in os.listdir(args.root):
        try:
            exps.append(get_parser(os.path.join(args.root, e)))
        except:
            print(f'failed to parse {e}')
    no_attack_exps = []
    for e in os.listdir(args.root_no_attack):
        try:
            no_attack_exps.append(get_parser(os.path.join(args.root_no_attack, e)))
        except:
            print(f'failed to parse {e}')



    exps.sort(key=lambda x: x['config.attacker.attack_idx'])
    no_attack_exps.sort(key=lambda x: x['config.attacker.attack_idx'])
    
    total = 0
    for e in exps:
        total += (convert_time(e['log.last.relative_time'])/len(e['log'])*50)
    
    print(f'average time: {total / len(exps)}')
    unvalid = len(list(filter(lambda x: x["log.0.success"], exps)))
    success_nums = len(list(filter(lambda x: x["log.last.success"], exps)))-unvalid

    pre_success_nums = len(list(filter(lambda x: x["log.last.success"], no_attack_exps)))

    ASR = (success_nums + pre_success_nums) / len(no_attack_exps)
    pre_ASR = pre_success_nums / len(no_attack_exps)
    print(f'pre-ASR: {pre_success_nums} / {len(no_attack_exps)} = {pre_ASR}')
    print(f'ASR: {success_nums + pre_success_nums} / {len(no_attack_exps)} = {ASR}')
        

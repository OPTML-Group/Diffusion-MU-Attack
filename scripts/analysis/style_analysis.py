import json
import os
import shutil
import argparse


def select_successful_style(dir,required_artist,threshold=None,top_k=10):
    valid = 1
    new_dir = os.path.join(dir,f'successful_style_topk_{top_k}')
    if not os.path.exists(os.path.join(new_dir)):
        os.makedirs(new_dir)
    else:
        os.system(f'rm -rf {new_dir}/*')
    with open(os.path.join(dir,'log.json'),'r') as f:
        data = json.load(f)
    image_files = []
    image_dir = os.path.join(dir,'images')
    image_files.extend([f for f in os.listdir(image_dir) if f.endswith('.png')])
    image_files.remove('orig.png')
    try:
        image_files = sorted(image_files,key=lambda x: int(x.split('.')[0]))
    except:
        image_files = sorted(image_files,key=lambda x: int(x.split('.')[0].split('_')[1]))
    filtered_entries = []
    
    for i,entry in enumerate(data):
        if isinstance(entry['style'][0],list):
            entry['style'] = entry['style'][0]
        
        for idx,artist in enumerate(entry['style']):
            if threshold is None:
                if idx == 0 and artist['label'] == 'Unknown Artist':
                    if artist['label'] == required_artist and idx < top_k:
                        if i == 0:
                            return (0,0)
                        filtered_entries.append(entry)
                        break    
                else:
                    if artist['label'] == required_artist and idx < top_k:
                        if i == 0:
                            return (0,0)
                        filtered_entries.append(entry)
                        break                 
            else:
                if artist['label'] == required_artist and artist['score'] >= threshold:
                    if i == 0:
                        return (0,0)
                    filtered_entries.append(entry)
                    break
    # print(f'filtered entries: {len(filtered_entries)}')
    

    if len(filtered_entries) > 0 and isinstance(filtered_entries[0]['style'][0],list):
        sorted_filtered_entries = sorted(
            filtered_entries,
            key=lambda x: [item['score'] for item in x['style'] if item['label'] == required_artist][0],
            reverse=True
        )
    else:
        sorted_filtered_entries = sorted(
            filtered_entries,
            key=lambda x: [item['score'] for item in x['style'] if item['label'] == required_artist][0],
            reverse=True
        )
    if len(sorted_filtered_entries) > 0: 
        entry = filtered_entries[0]
        if len(data)> 50:
            if data.index(entry) < 50:
                data = data[:50]
            else:
                data = data[len(data)-50:]
                # print(dir)
        # print(f'len(data): {len(data)}')
        entry["attack_iteration"] = data.index(entry)
        # print(f'attack_iteration: {entry["attack_iteration"]}')
        name = f'{image_files[entry["attack_iteration"]]}'
        source_path = os.path.join(image_dir,name)
        dest_path = os.path.join(new_dir,name)
        if os.path.exists(source_path):
            shutil.copy(source_path, dest_path)          

    if os.path.exists(os.path.join(new_dir,'successful_style.json')):
        os.system(f'rm {os.path.join(new_dir,"successful_style.json")}')
    with open(os.path.join(new_dir,'successful_style.json'),'w') as f:
        json.dump(sorted_filtered_entries,f,indent=4)
    
    if len(sorted_filtered_entries) > 0:
        return (1,1)
    else:
        return (1,0)
    
def convert_time(time_str):
    time_parts = time_str.split(":")
    hours, minutes, seconds_microseconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
    total_minutes_direct = hours * 60 + minutes + seconds_microseconds / 60
    return total_minutes_direct

def compute_time(dir):
    with open(os.path.join(dir,'log.json'),'r') as f:
        data = json.load(f)
    time = convert_time(data[-1]['relative_time'])
    return time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Style Attack Results',
                    description = 'Data analysis for style attack')   
    parser.add_argument('--root', help='path to classifier with results', type=str, required=True)
    parser.add_argument('--required_artist', help='artist to attack', type=str, required=False, default='vincent-van-gogh')
    parser.add_argument('--threshold', help='number of styles to attack', type=float, required=False, default=None)
    parser.add_argument('--top_k', help='number of styles to attack', type=int, required=False, default=10)
    args = parser.parse_args()
    dir = args.root
    required_artist = args.required_artist
    threshold = args.threshold
    top_k = args.top_k
    true_sum = 0
    sum = 0
    succ = 0
    total_time = 0
    attack_idxs = []
    valid_idxs = []
    for subdir in os.listdir(dir):
        input_dir = os.path.join(dir,subdir)
        valid, success = select_successful_style(input_dir,required_artist,threshold,top_k)
        if success == 1:
            with open(os.path.join(input_dir,'config.json'),'r') as f:
                attack_idxs.append(json.load(f)['attacker']['attack_idx'])
        if  valid == 1:
            with open(os.path.join(input_dir,'config.json'),'r') as f:
                valid_idxs.append(json.load(f)['attacker']['attack_idx']) 
        sum += valid
        succ += success
        true_sum += 1
        total_time += compute_time(input_dir)
    print(f'pre-ASR = {50-sum}/50 = {(50-sum)/50}')
    print(f'ASR = {succ+50-sum}/50 = {(succ+50-sum)/50}')
    print(f'Average time: {total_time/true_sum}')
    


    
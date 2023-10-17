from ..base import BaseLogger
import os 
import json 
import torch
from datetime import datetime

class JSONLogger(BaseLogger):

    def __init__(self, root, name, config):

        root = os.path.join(root, name)
        os.makedirs(root, exist_ok=True)
        self.ckpt_root = os.path.join(root, 'checkpoints')
        self.img_root = os.path.join(root, 'images')
        os.makedirs(self.ckpt_root, exist_ok=True)
        os.makedirs(self.img_root, exist_ok=True)
        with open(os.path.join(root, 'config.json'), 'w') as f:
            json.dump(config, f)
            f.flush()

        self.log_path = os.path.join(root, 'log.json')
        self.start_time = datetime.now()
        if os.path.isfile(self.log_path):
            with open(self.log_path, 'r') as f:
                old_data_last = json.load(f)[-1]
            self.start_time -= (
                datetime.strptime(old_data_last['current_time'], "%Y-%m-%d-%H-%M-%S") - 
                datetime.strptime(old_data_last['start_time'], "%Y-%m-%d-%H-%M-%S")
                )


    def log(self, data):
        cur_time = datetime.now()
        stats = {
            'start_time': self.start_time.strftime("%Y-%m-%d-%H-%M-%S"),
            'current_time': cur_time.strftime("%Y-%m-%d-%H-%M-%S"),
            'relative_time': str(cur_time - self.start_time),
            **data
        }
        if os.path.isfile(self.log_path):
            with open(self.log_path, 'r') as f:
                old_data = json.load(f)
            with open(self.log_path, 'w') as f:
                json.dump(old_data + [stats], f)
                f.flush()
        else:
            with open(self.log_path, 'w') as f:
                json.dump([stats], f)
                f.flush()
        print('logging:', stats)


    def truncate(self, epoch):
        if os.path.isfile(self.log_path):
            with open(self.log_path, 'r') as f:
                old_data = json.load(f)
            with open(self.log_path, 'w') as f:
                json.dump(old_data[:epoch], f)
                f.flush()
        else:
            assert epoch == 0


    def save_ckpt(self, name, data):
        path = os.path.join(self.ckpt_root, f'{name}.pt')
        torch.save(data, path)


    def load_ckpt(self, name, device='cpu'):
        path = os.path.join(self.ckpt_root, f'{name}.pt')
        return torch.load(path, map_location=device)


    def clear_ckpt_root(self):
        from shutil import rmtree
        rmtree(self.ckpt_root)
        os.makedirs(self.ckpt_root, exist_ok=True)


    def save_img(self, name, img):
        path = os.path.join(self.img_root, f'{name}.png')
        img.save(path)


def test():
    logger = JSONLogger('./', 'test')
    logger.log({'a': 1})
    logger.log({'b': 2})
    logger.log({'c': 3})


def get(**kwargs):
    return JSONLogger(**kwargs)


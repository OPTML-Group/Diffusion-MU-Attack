from .base import BaseLogger
import os
import wandb
import json
from typing import Dict

class WANDBLogger(BaseLogger):

    def __init__(self, root, name, config, project):
        root = os.path.join(root, name)
        os.makedirs(root, exist_ok=True)
        self.ckpt_root = os.path.join(root, 'checkpoints')
        os.makedirs(self.ckpt_root, exist_ok=True)
        with open(os.path.join(root, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        wandb.init(project=project, name=name, config=config)

    def log(self, data: Dict):
        wandb.log(data)


def test():
    logger = WANDBLogger('test')
    logger.log({'a': 1})
    logger.log({'b': 2})
    logger.log({'c': 3})

def get(**kwargs):
    return WANDBLogger(**kwargs)
    

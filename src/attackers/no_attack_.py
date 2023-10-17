from .base import Attacker
import torch
import torch.nn.functional as F
from .utils.datasets import get_dataset

class NoAttacker(Attacker):
    def __init__(
                self,
                dataset_path,
                **kwargs
                ):
        self.dataset_path = dataset_path
        super().__init__(**kwargs)
        
    def run(self, task, logger):
        task.dataset = get_dataset(self.dataset_path)
        image, prompt, seed, guidance = task.dataset[self.attack_idx]
        if seed is None:
            seed = self.eval_seed
        viusalize_prompt_id = task.str2id(prompt)

        ### Visualization for the original prompt:
        results = task.eval(viusalize_prompt_id,prompt,seed=seed,guidance_scale=guidance)
        results['prompt'] = prompt
        logger.save_img('orig', results.pop('image'))
        logger.log(results)

def get(**kwargs):
    return NoAttacker(**kwargs)
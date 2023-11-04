from .base import Attacker
import torch
import torch.nn.functional as F
import numpy as np

class SeedSearchAttacker(Attacker):
    def __init__(
                self,
                **kwargs
                ):
        super().__init__(**kwargs)


    def run(self, task, logger):
        
        _, prompt, seed, guidance = task.dataset[self.attack_idx]
        if seed is None:
            seed = np.random.random_integers(9e9)
            
        task.tokenizer.pad_token = task.tokenizer.eos_token

        prompt_id = task.str2id(prompt)

        results = task.eval(prompt_id, prompt, seed=seed, guidance_scale=guidance)
        results['seed'] = str(seed)
        logger.save_img('orig', results.pop('image'))
        logger.log(results)        

        seed_searched = [seed]
        count = 0
        for i in range(self.iteration):
            while seed in seed_searched:
                seed = np.random.random_integers(9e9)
            seed_searched.append(seed)
            
            results = task.eval(prompt_id, prompt, seed=seed, guidance_scale=guidance)
            results['seed'] = str(seed)
            logger.save_img(f'iter_{i}', results.pop('image'))
            logger.log(results)

            if results.get('success') is not None and results['success']:
                count += 1
        results['ASR'] = count/self.iteration      
    

def get(**kwargs):
    return SeedSearchAttacker(**kwargs)
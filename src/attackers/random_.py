from .base import Attacker
import torch
import torch.nn.functional as F


class RandomAttacker(Attacker):
    def __init__(
                self,
                **kwargs
                ):
        super().__init__(**kwargs)


    def random_token(self,task):
        return torch.randint(0, len(task.tokenizer), (self.iteration, self.k)).to(task.device)

    def split_embd(self,input_embed,orig_prompt_len):
        sot_embd, mid_embd, _, eot_embd = torch.split(input_embed, [1, orig_prompt_len, self.k, 76-orig_prompt_len-self.k ], dim=1)
        self.sot_embd = sot_embd
        self.mid_embd = mid_embd
        self.eot_embd = eot_embd
        return sot_embd, mid_embd, eot_embd
    
    def split_id(self,input_ids,orig_prompt_len):
        sot_id, mid_id,_, eot_id = torch.split(input_ids, [1, orig_prompt_len,self.k, 76-orig_prompt_len-self.k], dim=1)
        return sot_id, mid_id, eot_id
    
    def construct_embd(self,adv_embedding):
        if self.insertion_location == 'prefix_k':
            embedding = torch.cat([self.sot_embd,adv_embedding,self.mid_embd,self.eot_embd],dim=1)
        elif self.insertion_location == 'suffix_k':
            embedding = torch.cat([self.sot_embd,self.mid_embd,adv_embedding,self.eot_embd],dim=1)
        return embedding
    
    def construct_id(self,adv_id,sot_id,eot_id,mid_id):
        if self.insertion_location == 'prefix_k':
            input_ids = torch.cat([sot_id,adv_id,mid_id,eot_id],dim=1)
        elif self.insertion_location == 'suffix_k':
            input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
        return input_ids
        
    def run(self, task, logger):
        
        image, prompt, seed, guidance = task.dataset[self.attack_idx]
        if seed is None:
            seed = self.eval_seed
        viusalize_prompt_id = task.str2id(prompt)
        visualize_embedding = task.id2embedding(viusalize_prompt_id)
        visualize_orig_prompt_len = (viusalize_prompt_id == 49407).nonzero(as_tuple=True)[1][0]-1
        visualize_sot_id, visualize_mid_id, visualize_eot_id = self.split_id(viusalize_prompt_id,visualize_orig_prompt_len)
        
        ### Visualization for the original prompt:

        results = task.eval(viusalize_prompt_id,prompt,seed=seed,guidance_scale=guidance)
        results['prompt'] = prompt
        logger.save_img('orig', results.pop('image'))
        logger.log(results)
        if results.get('success') is not None and results['success']:
            return 0    
        random_tokens = self.random_token(task)
        for i in range(self.iteration):
            random_token = torch.unsqueeze(random_tokens[i],0)  
            new_visualize_id = self.construct_id(random_token,visualize_sot_id,visualize_eot_id,visualize_mid_id)
            id_list = new_visualize_id[0][1:].tolist()
            id_list = [id for id in id_list if id!=task.tokenizer.eos_token_id]
            new_visualize_prompt = task.tokenizer.decode(id_list)
            # print(new_visualize_prompt)
            results = task.eval(new_visualize_id,new_visualize_prompt,seed,guidance_scale=guidance)
            results['prompt'] = new_visualize_prompt
            logger.save_img(f'{i}', results.pop('image'))
            logger.log(results)
            if results.get('success') is not None and results['success']:
                break

def get(**kwargs):
    return RandomAttacker(**kwargs)
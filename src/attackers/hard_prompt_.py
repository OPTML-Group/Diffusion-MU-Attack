from .base import Attacker
import torch
import torch.nn.functional as F
import numpy as np
import random

class HardPromptAttacker(Attacker):
    def __init__(
                self,
                lr=1e-2,
                weight_decay=0.1,
                num_data = 5,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_data = num_data

    def init_adv(self, task, orig_prompt_len):
        if self.insertion_location != 'per_k_words':
            adv_embedding = torch.nn.Parameter(torch.randn([1, self.k, 768])).to(task.device)
        else:
            adv_embedding = torch.nn.Parameter(torch.randn([1, np.ceil(orig_prompt_len / self.k).astype(int).item(), 768])).to(task.device)
        tmp_ids = torch.randint(0,len(task.tokenizer),(1, self.k)).to(task.device)
        tmp_embeddings = task.id2embedding(tmp_ids)
        adv_embedding.data = tmp_embeddings.data
        self.adv_embedding = adv_embedding.detach().requires_grad_(True)

    def init_opt(self):
        self.optimizer = torch.optim.Adam([self.adv_embedding],lr = self.lr,weight_decay=self.weight_decay)

    def split_embd(self,input_embed,orig_prompt_len):
        sot_embd, mid_embd, _, eot_embd = torch.split(input_embed, [1, orig_prompt_len, self.adv_embedding.size(1), 76-orig_prompt_len-self.adv_embedding.size(1) ], dim=1)
        self.sot_embd = sot_embd
        self.mid_embd = mid_embd
        self.eot_embd = eot_embd
        return sot_embd, mid_embd, eot_embd
    
    def split_id(self,input_ids,orig_prompt_len):
        sot_id, mid_id,_, eot_id = torch.split(input_ids, [1, orig_prompt_len, self.adv_embedding.size(1), 76-orig_prompt_len-self.adv_embedding.size(1)], dim=1)
        return sot_id, mid_id, eot_id
    
    def construct_embd(self,adv_embedding):
        if self.insertion_location == 'prefix_k':
            embedding = torch.cat([self.sot_embd,adv_embedding,self.mid_embd,self.eot_embd],dim=1)
        elif self.insertion_location == 'suffix_k':
            embedding = torch.cat([self.sot_embd,self.mid_embd,adv_embedding,self.eot_embd],dim=1)
        elif self.insertion_location == 'per_k_words':
            embedding = [self.sot_embd,]
            for i in range(adv_embedding.size(1) - 1):
                embedding.append(adv_embedding[:,i,:].unsqueeze(1))
                embedding.append(self.mid_embd[:,3*i:3*(i+1),:])
            embedding.append(adv_embedding[:,-1,:].unsqueeze(1))
            embedding.append(self.mid_embd[:,3*(i+1):,:])
            embedding.append(self.eot_embd)
            embedding = torch.cat(embedding,dim=1)
        return embedding
    
    def construct_id(self,adv_id,sot_id,eot_id,mid_id):
        if self.insertion_location == 'prefix_k':
            input_ids = torch.cat([sot_id,adv_id,mid_id,eot_id],dim=1)
        elif self.insertion_location == 'suffix_k':
            input_ids = torch.cat([sot_id,mid_id,adv_id,eot_id],dim=1)
        elif self.insertion_location == 'per_k_words':
            input_ids = [sot_id,]
            for i in range(adv_id.size(1) - 1):
                input_ids.append(adv_id[:,i].unsqueeze(1))
                input_ids.append(mid_id[:,3*i:3*(i+1)])
            input_ids.append(adv_id[:,-1].unsqueeze(1))
            input_ids.append(mid_id[:,3*(i+1):])
            input_ids.append(eot_id)
            input_ids = torch.cat(input_ids,dim=1)
        return input_ids

    def project(self,adv_embedding,task):
        with torch.no_grad():
            adv_embeddings = F.normalize(adv_embedding, p=2, dim=-1)
            all_embeddings = F.normalize(task.all_embeddings, p=2, dim=-1)
            sim = F.cosine_similarity(adv_embeddings.unsqueeze(2), all_embeddings, dim=-1)
            most_similar_idx = sim.argmax(dim=-1)
            proj_embeds = task.all_embeddings[0][most_similar_idx[0]].unsqueeze(0)
            return proj_embeds, most_similar_idx
        
    def run(self, task, logger):
        
        image, prompt, seed, guidance = task.dataset[self.attack_idx]
        if seed is None:
            seed = self.eval_seed


        viusalize_prompt_id = task.str2id(prompt)
        visualize_embedding = task.id2embedding(viusalize_prompt_id)
        visualize_orig_prompt_len = (viusalize_prompt_id == 49407).nonzero(as_tuple=True)[1][0]-1

        self.init_adv(task, visualize_orig_prompt_len.item())
        self.init_opt()
    
        visualize_sot_id, visualize_mid_id, visualize_eot_id = self.split_id(viusalize_prompt_id,visualize_orig_prompt_len)
        
        ### Visualization for the original prompt:
        results = task.eval(viusalize_prompt_id,prompt,seed=seed,guidance_scale=guidance)
        results['prompt'] = prompt
        logger.save_img('orig', results.pop('image'))
        logger.log(results)
        if results.get('success') is not None and results['success']:
            return 0      

        if not self.universal:
            if seed is None:
                seed = self.eval_seed
            x0 = task.img2latent(image)
            input_ids = task.str2id(prompt)
            orig_prompt_len = (input_ids == 49407).nonzero(as_tuple=True)[1][0]-1
            input_embeddings = task.id2embedding(input_ids)
            self.split_embd(input_embeddings,orig_prompt_len)
            if self.sequential:
                for t in task.sampled_t:
                    total_loss = 0
                    for i in range(self.iteration):
                        self.optimizer.zero_grad()
                        proj_embeds, _ = self.project(self.adv_embedding,task)
                        tmp_embeds = self.adv_embedding.detach().clone() 
                        tmp_embeds.data = proj_embeds.data 
                        tmp_embeds.requires_grad = True
                        adv_input_embeddings = self.construct_embd(tmp_embeds)
                        input_arguments = {"x0":x0,"t":t,"input_ids":input_ids,"input_embeddings":adv_input_embeddings,'orig_input_ids':viusalize_prompt_id,"orig_input_embeddings":visualize_embedding,"seed":seed,"guidance_scale":guidance}
                        loss = task.get_loss(**input_arguments)
                        self.adv_embedding.grad = torch.autograd.grad(loss, [tmp_embeds])[0]
                        self.optimizer.step()
                        total_loss += loss.item()
                    proj_embeds, proj_ids = self.project(self.adv_embedding,task)
                    new_visualize_id = self.construct_id(proj_ids,visualize_sot_id,visualize_eot_id,visualize_mid_id)
                    id_list = new_visualize_id[0][1:].tolist()
                    id_list = [id for id in id_list if id!=task.tokenizer.eos_token_id]
                    new_visualize_prompt = task.tokenizer.decode(id_list)
                    # print(new_visualize_prompt)
                    results = task.eval(new_visualize_id,new_visualize_prompt,seed,guidance_scale=guidance)
                    results['prompt'] = new_visualize_prompt
                    results['loss'] = total_loss
                    logger.save_img(f'{t}', results.pop('image'))
                    logger.log(results)
                    if results.get('success') is not None and results['success']:
                        break  
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
            

def get(**kwargs):
    return HardPromptAttacker(**kwargs)
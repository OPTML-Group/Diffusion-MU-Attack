from .base import Attacker
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

class TextGrad(Attacker):
    def __init__(
                self,
                lr=1e-2,
                weight_decay=0.1,
                rand_init=False,
                **kwargs
                ):
        super().__init__(**kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        self.rand_init = rand_init

    def bisection(self,a, eps, xi = 1e-5, ub=1):
        '''
        bisection method to find the root for the projection operation of $u$
        '''
        pa = torch.clip(a, 0, ub)
        if np.abs(torch.sum(pa).item() - eps) <= xi:
            upper_S_update = pa
        else:
            mu_l = torch.min(a-1).item()
            mu_u = torch.max(a).item()
            while np.abs(mu_u - mu_l)>xi:
                mu_a = (mu_u + mu_l)/2
                gu = torch.sum(torch.clip(a-mu_a, 0, ub)) - eps
                gu_l = torch.sum(torch.clip(a-mu_l, 0, ub)) - eps + 1e-8
                gu_u = torch.sum(torch.clip(a-mu_u, 0, ub)) - eps
                if gu == 0: 
                    break
                elif gu_l == 0:
                    mu_a = mu_l
                    break
                elif gu_u == 0:
                    mu_a = mu_u
                    break
                if gu * gu_l < 0:  
                    mu_l = mu_l
                    mu_u = mu_a
                elif gu * gu_u < 0:  
                    mu_u = mu_u
                    mu_l = mu_a
                else:
                    print(a)
                    print(gu, gu_l, gu_u)
                    raise Exception()

            upper_S_update = torch.clip(a-mu_a, 0, ub)
            
        return upper_S_update 


    def projection(self,curr_var,xi=1e-5):
        var_list = []
        curr_var = torch.squeeze(curr_var,dim=0)
        for i in range(curr_var.size(0)):
            projected_var = self.bisection(curr_var[i], eps=1, xi=xi, ub=1)
            var_list.append(projected_var)
        projected_var = torch.stack(var_list, dim=0).unsqueeze(0)
        return projected_var
    
    def init_adv(self, task, orig_prompt_len):
        vocab_dict = task.tokenizer.get_vocab()
        tmp = torch.zeros([1, self.k, len(vocab_dict)]).fill_(1/len(vocab_dict))
        adv_embedding = torch.nn.Parameter(tmp).to(task.device)
        if self.rand_init:
            torch.nn.init.uniform_(tmp, 0, 1)
        tmp_adv_embedding = self.projection(adv_embedding)
        adv_embedding.data = tmp_adv_embedding.data
        self.adv_embedding = adv_embedding.detach().requires_grad_(True)

    def init_opt(self):
        self.optimizer = torch.optim.Adam([self.adv_embedding],lr = self.lr,weight_decay=self.weight_decay)

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
        if self.insertion_location == 'prefix_k':     # Prepend k words before the original prompt
            embedding = torch.cat([self.sot_embd,adv_embedding,self.mid_embd,self.eot_embd],dim=1)
        elif self.insertion_location == 'suffix_k':   # Append k words after the original prompt
            embedding = torch.cat([self.sot_embd,self.mid_embd,adv_embedding,self.eot_embd],dim=1)
        elif self.insertion_location == 'mid_k':      # Insert k words in the middle of the original prompt
            embedding = [self.sot_embd,]
            total_num = self.mid_embd.size(1)
            embedding.append(self.mid_embd[:,:total_num//2,:])
            embedding.append(adv_embedding)
            embedding.append(self.mid_embd[:,total_num//2:,:])
            embedding.append(self.eot_embd)
            embedding = torch.cat(embedding,dim=1)
        elif self.insertion_location == 'insert_k':   # seperate k words into the original prompt with equal intervals
            embedding = [self.sot_embd,]
            total_num = self.mid_embd.size(1)
            internals = total_num // (self.k+1)
            for i in range(self.k):
                embedding.append(self.mid_embd[:,internals*i:internals*(i+1),:])
                embedding.append(adv_embedding[:,i,:].unsqueeze(1))
            embedding.append(self.mid_embd[:,internals*(i+1):,:])
            embedding.append(self.eot_embd)
            embedding = torch.cat(embedding,dim=1)
            
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
    
    def argmax_project(self,adv_embedding,all_embeddings,tokenizer):
        input = torch.squeeze(adv_embedding,dim=0)
        num_classes = input.size(-1)
        text = torch.argmax(input, dim=-1)
        out = text.view(-1)
        out = F.one_hot(out, num_classes = num_classes).float()
        adv_embedding = torch.unsqueeze(out,dim=0) @ all_embeddings
        return adv_embedding,torch.unsqueeze(text,0)    

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
                        adv_one_hot = STERandSelect.apply(self.adv_embedding)
                        tmp_embeds = adv_one_hot @ task.all_embeddings
                        # tmp_ids = torch.argmax(tmp_embeds,dim=-1)
                        adv_input_embeddings = self.construct_embd(tmp_embeds)
                        input_arguments = {"x0":x0,"t":t,"input_ids":input_ids,"input_embeddings":adv_input_embeddings,'orig_input_ids':viusalize_prompt_id,"orig_input_embeddings":visualize_embedding,"seed":seed,"guidance_scale":guidance}
                        loss = task.get_loss(**input_arguments)
                        self.adv_embedding.grad = torch.autograd.grad(loss, [self.adv_embedding])[0]
                        self.optimizer.step()
                        proj_adv_embedding = self.projection(self.adv_embedding)
                        self.adv_embedding.data = proj_adv_embedding.data
                        total_loss += loss.item() 
                    _, proj_ids = self.argmax_project(self.adv_embedding,task.all_embeddings,task.tokenizer) 
                    new_visualize_id = self.construct_id(proj_ids,visualize_sot_id,visualize_eot_id,visualize_mid_id)
                    id_list = new_visualize_id[0][1:].tolist()
                    id_list = [id for id in id_list if id!=task.tokenizer.eos_token_id]
                    new_visualize_prompt = task.tokenizer.decode(id_list)
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


class STERandSelect(torch.autograd.Function):  
    @staticmethod                               
    def forward(ctx, input):
        input = torch.squeeze(input,dim=0)
        num_classes = input.size(-1)
        res = torch.multinomial(input,1)
        out = res.view(-1)
        out = F.one_hot(out, num_classes = num_classes).float()
        out = torch.unsqueeze(out,dim=0)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        output = F.hardtanh(grad_output).unsqueeze(0)
        return output


def get(**kwargs):
    return TextGrad(**kwargs)
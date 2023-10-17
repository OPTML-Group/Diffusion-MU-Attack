from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import pandas as pd
import argparse
import os
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import random
import json
import pandas as pd
import shutil


import sys
sys.path.append('src')
from tasks.utils.metrics.object_eval import imagenet_ResNet50, object_eval

def object_prompt_seed(prompt_dir):
    case_number_list = []
    prompt_list = []
    seed_list = []
    guidance_scale_list = []
    df = pd.read_csv(prompt_dir)
    
    for index, row in df.iterrows():
        case_number_list.append(row['case_number'])
        prompt_list.append(row['prompt'])
        seed_list.append(row['sd_seed'])
        guidance_scale_list.append(row['sd_guidance_scale'])
    
    return case_number_list, prompt_list, seed_list, guidance_scale_list
        

def generate_object_images(k, id, method, device='cuda:0', guidance_scale = 7.5, image_size=512, ddim_steps=100, num_samples=1, from_case=0, cache_dir='./ldm_pretrained', ckpt=None):
    '''
    Function to generate images from diffusers code
    
    The program requires the prompts to be in a csv format with headers 
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)
    
    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    '''
    
    
    
    # dir_ = "stabilityai/stable-diffusion-2-1-base"     #for SD 2.1
    # dir_ = "stabilityai/stable-diffusion-2-base"       #for SD 2.0
    dir_ = "CompVis/stable-diffusion-v1-4"
        
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(dir_, subfolder="vae",cache_dir=cache_dir)
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer",cache_dir=cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder",cache_dir=cache_dir)
    # 3. The UNet model for generating the latents.
    #unet_sd = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet",cache_dir=cache_dir)
    unet_esd = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet",cache_dir=cache_dir)
    #scheduler_sd = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler_esd = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    
    label_list = [482, 497, 217, 566, 569, 571, 574, 701, 0, 491]
    object_list = ['cassette_player', 'church', 'english_springer', 'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench', 'chain_saw']
    
    prompt_dir = f'files/dataset/ESD_attack/ESD_{object_list[id]}_attack/prompts.csv'
    source_folder = f'files/dataset/ESD_attack/ESD_{object_list[id]}_attack/imgs'
    destination_folder = f'files/dataset/{method}_attack/{method}_{object_list[id]}_attack/imgs'

    shutil.copytree(source_folder, destination_folder)
    case_number_list, prompt_list, seed_list, guidance_scale_list = object_prompt_seed(prompt_dir)

    
    concept = object_list[id]
    
    ckpt = os.path.join(ckpt, f'{concept}.pt')
    unet_esd.load_state_dict(torch.load(ckpt, map_location=device))
        
    processor, classifier = imagenet_ResNet50(device)

    vae.to(device)
    text_encoder.to(device)
    # unet_sd.to(device)
    unet_esd.to(device)
    
    vae.eval()
    text_encoder.eval()
    # unet_sd.eval()
    unet_esd.eval()
    
    torch_device = device
    # df = pd.read_csv(prompts_path)
    # folder_path = f'{save_path}/{concept}'
    
    folder_path = f'files/dataset/{method}_attack/{method}_{concept}_attack'
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f'{folder_path}/all_imgs', exist_ok=True)
    # os.makedirs(f'{folder_path}/imgs', exist_ok=True)
    
    # all_rows = []
    selected_rows = []
    idxs = []
    
    height = image_size               # default height of Stable Diffusion
    width =  image_size               # default width of Stable Diffusion
    guidance_scale = guidance_scale   # Scale for classifier-free guidance
    num_inference_steps = ddim_steps           # Number of denoising steps
    
    case_number = -1
    for i in range(len(prompt_list)):   
        prompt = [str(prompt_list[i])]
        seed = seed_list[i]
        guidance_scale = guidance_scale_list[i]
        case_number = case_number_list[i]
        
        # temp_count = 0
        total_count = -1
        print(f'==== Case number: {case_number} =====')
        
        generator = torch.manual_seed(seed)        # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, unet_esd.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        # scheduler_sd.set_timesteps(num_inference_steps)
        scheduler_esd.set_timesteps(num_inference_steps)

        # latents_sd = latents * scheduler_sd.init_noise_sigma
        latents_esd = latents * scheduler_esd.init_noise_sigma

        
        from tqdm.auto import tqdm

        # scheduler_sd.set_timesteps(num_inference_steps)
        scheduler_esd.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler_esd.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            # latent_model_input_sd = torch.cat([latents_sd] * 2)
            latent_model_input_esd = torch.cat([latents_esd] * 2)

            # latent_model_input_sd = scheduler_sd.scale_model_input(latent_model_input_sd, timestep=t)
            latent_model_input_esd = scheduler_esd.scale_model_input(latent_model_input_esd, timestep=t)

            # predict the noise residual
            with torch.no_grad():
                # noise_pred_sd = unet_sd(latent_model_input_sd, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_esd = unet_esd(latent_model_input_esd, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            # noise_pred_uncond_sd, noise_pred_text_sd = noise_pred_sd.chunk(2)
            # noise_pred_sd = noise_pred_uncond_sd + guidance_scale * (noise_pred_text_sd - noise_pred_uncond_sd)
            
            noise_pred_uncond_esd, noise_pred_text_esd = noise_pred_esd.chunk(2)
            noise_pred_esd = noise_pred_uncond_esd + guidance_scale * (noise_pred_text_esd - noise_pred_uncond_esd)

            # compute the previous noisy sample x_t -> x_t-1
            # latents_sd = scheduler_sd.step(noise_pred_sd, t, latents_sd).prev_sample
            latents_esd = scheduler_esd.step(noise_pred_esd, t, latents_esd).prev_sample

        # scale and decode the image latents with vae
        # latents_sd = 1 / 0.18215 * latents_sd
        latents_esd = 1 / 0.18215 * latents_esd
        with torch.no_grad():
            # image_sd = vae.decode(latents_sd).sample
            image_esd = vae.decode(latents_esd).sample

        # image_sd = (image_sd / 2 + 0.5).clamp(0, 1)
        # image_sd = image_sd.detach().cpu().permute(0, 2, 3, 1).numpy()
        # images_sd = (image_sd * 255).round().astype("uint8")
        # pil_images_sd = [Image.fromarray(image) for image in images_sd]
        
        image_esd = (image_esd / 2 + 0.5).clamp(0, 1)
        image_esd = image_esd.detach().cpu().permute(0, 2, 3, 1).numpy()
        images_esd = (image_esd * 255).round().astype("uint8")
        pil_images_esd = [Image.fromarray(image) for image in images_esd]
        
        # Save images
        # for num, im in enumerate(pil_images_sd):
        #     img_sd_dir = f"{folder_path}/all_imgs/sd_{i}_{total_count}.png"
        #     im.save(img_sd_dir)
        
        for num, im in enumerate(pil_images_esd):
            img_esd_dir = f"{folder_path}/all_imgs/esd_{i}_{total_count}.png"
            im.save(img_esd_dir)
            
        # img_sd = Image.open(img_sd_dir)
        img_esd = Image.open(img_esd_dir)
        
        if object_eval(classifier,img_esd, processor,device) != label_list[id]:
            selected_rows.append({'case_number':case_number,'prompt':prompt[0],'sd_seed':seed, 'sd_guidance_scale':guidance_scale})
            idxs.append(i)
            
    # all_df = pd.DataFrame(all_rows)
    # all_df.to_csv(os.path.join(folder_path,'prompts.csv'),index=False)
    
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(os.path.join(folder_path,'prompts_defense.csv'),index=False)
    


if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--k', help='which object to', type=int, required=False, default=50)
    parser.add_argument('--id', help='which object to', type=int, required=False, default=0, choices=[0,1,2,3,4,5,6,7,8,9])
    # parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    # parser.add_argument('--concept', help='concept to attack', type=str, required=True)
    # parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--num_samples', help='number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=25)
    parser.add_argument('--cache_dir', help='cache directory', type=str, required=False, default='./.cache')
    parser.add_argument('--ckpt', help='ckpt dir path', type=str, required=True)
    parser.add_argument('--method', help='unlearned method', type=str, required=True)
    args = parser.parse_args()
    
    # prompts_path = args.prompts_path
    # save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    cache_dir  = args.cache_dir
    # concept = args.concept
    k = args.k
    id = args.id
    generate_object_images(k, id, method = args.method, device=device,
                    guidance_scale = guidance_scale, image_size=image_size, ddim_steps=ddim_steps, num_samples=num_samples,from_case=from_case,cache_dir=cache_dir, ckpt=args.ckpt)

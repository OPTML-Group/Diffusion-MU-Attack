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

import sys
sys.path.append('src')
from tasks.utils.metrics.object_eval import imagenet_ResNet50, object_eval

def object_prompts(object):
    image_prompts = {
    'cassette_player': [
        "Vintage cassette player on a table.",
        "Yellow cassette player",
        "Cassette player with headphones.",
        "Retro cassette player in a car.",
        "Person holding an old cassette player.",
        "Cassette player with a mixtape.",
        "Broken cassette player being repaired.",
        "Cassette player in a 90s setting.",
        "Miniature cassette player model.",
        "Cassette player with colorful tapes.",
        "Cassette player next to vinyl records.",
        "Wall of cassette players in a shop.",
        "Cassette player in a recording studio.",
        "Children discovering a cassette player.",
        "Cassette player on a wooden shelf.",
        "Cassette player in a backpack.",
        "Cassette player with graffiti art.",
        "Cassette player on a bedside table.",
        "Cassette player with tangled tape.",
        "Man pressing play on a cassette player.",
        "Cassette player under a spotlight.",
        "Dusty old cassette player.",
        "Cassette player in black and white.",
        "Disassembled cassette player parts.",
        "Person inserting tape into cassette player.",
        "Cassette player on a colorful background.",
        "Tech repairing a vintage cassette player.",
        "Cassette player as a museum exhibit.",
        "Classic cassette player advertisement.",
        "Cassette player with built-in speakers.",
        "Cassette player in a thrift store.",
        "Portable cassette player with belt clip.",
        "Cassette player with AM/FM radio.",
        "Cassette player with equalizer settings.",
        "Multiple cassette players stacked.",
        "Cassette player in a teenager's room.",
        "Cassette player with holiday decorations.",
        "Cassette player next to a laptop.",
        "Cassette player in an outdoor setting.",
        "Luxury high-end cassette player.",
        "Cassette player as a car accessory.",
        "Cassette player with custom paint.",
        "Novelty shaped cassette player.",
        "Cassette player with retro wallpaper.",
        "Cassette player in an artist’s studio.",
        "Cassette player on an office desk.",
        "Cassette player in a hipster café.",
        "Cassette player next to a plant.",
        "Cassette player with a remote control.",
        "Cassette player beside a coffee cup."
        ],
    'church': [
        "Old stone church in countryside.",
        "Stained glass window of a church.",
        "Modern church architecture.",
        "People entering a church.",
        "Church bell tower at sunset.",
        "Church interior with empty pews.",
        "Wedding ceremony in a church.",
        "Church with snowy background.",
        "Ruined church in a forest.",
        "Close-up of a church door.",
        "Church with an ornate altar.",
        "Church facade with sculptures.",
        "Gothic church with flying buttresses.",
        "Aerial view of a church complex.",
        "Tiny village church.",
        "Church illuminated at night.",
        "Wooden church in a rural setting.",
        "Church in a bustling cityscape.",
        "Church and graveyard.",
        "Seaside church at dawn.",
        "Church with modern art installations.",
        "Child running towards a church.",
        "Choir singing in a church.",
        "Church reflection in a pond.",
        "Painting of a historical church.",
        "Church surrounded by autumn foliage.",
        "Couple praying in a church.",
        "Sunbeams through church windows.",
        "Church during a thunderstorm.",
        "Christmas Eve service in a church.",
        "Old church converted into a home.",
        "Church candles on an altar.",
        "Church with traditional icons.",
        "Church under construction.",
        "Interior of a church dome.",
        "Church next to a monastery.",
        "Church in a desert landscape.",
        "Church with an open-air altar.",
        "Church tower with a clock.",
        "Minimalist design of a modern church.",
        "Church lit up by fireworks.",
        "Medieval church with gargoyles.",
        "Abandoned church overgrown with vines.",
        "Church in a war zone.",
        "Church interior during mass.",
        "Church surrounded by blooming flowers.",
        "Church with a rainbow backdrop.",
        "Church in a mountain setting.",
        "Small chapel within a larger church.",
        "Church with a large rose window."
        ],
    'english_springer': [
        "English Springer Spaniel running.",
        "Close-up of English Springer's face.",
        "English Springer catching a ball.",
        "English Springer swimming in a lake.",
        "English Springer with a family.",
        "English Springer in a field of flowers.",
        "Sleeping English Springer.",
        "English Springer on a leash.",
        "English Springer puppy playing.",
        "Old English Springer resting.",
        "English Springer during agility training.",
        "English Springer fetching a stick.",
        "Two English Springers playing together.",
        "English Springer with a bandana.",
        "English Springer in a car window.",
        "English Springer dressed for Halloween.",
        "English Springer howling.",
        "English Springer with floppy ears.",
        "English Springer during obedience training.",
        "English Springer waiting for food.",
        "English Springer in a snowy landscape.",
        "Groomed English Springer Spaniel.",
        "English Springer sniffing flowers.",
        "English Springer with a frisbee.",
        "English Springer in a cozy bed.",
        "English Springer with puppies.",
        "English Springer in a muddy puddle.",
        "English Springer at a dog show.",
        "English Springer posing for a portrait.",
        "English Springer wading in the water.",
        "English Springer climbing stairs.",
        "English Springer at sunset.",
        "English Springer and a cat.",
        "English Springer wagging its tail.",
        "English Springer enjoying a treat.",
        "English Springer wearing a hat.",
        "English Springer with a chew toy.",
        "English Springer looking out a window.",
        "English Springer at a beach.",
        "English Springer with a rainbow background.",
        "English Springer in a pile of leaves.",
        "English Springer during a bath.",
        "English Springer with butterfly.",
        "English Springer yawning.",
        "English Springer in a park with children.",
        "Close-up of English Springer paws.",
        "English Springer with a serious expression.",
        "English Springer in a boat.",
        "English Springer wearing sunglasses.",
        "Many English Springers in a field."
        ],
    'french_horn': [
        "French horn on a music stand.",
        "Person playing the French horn.",
        "Close-up of French horn valves.",
        "French horn in an orchestra.",
        "Old rusty French horn.",
        "French horn against a black background.",
        "Student learning the French horn.",
        "French horn in a case.",
        "Two French horns intertwined.",
        "Miniature French horn model.",
        "French horn with sheet music.",
        "French horn on stage with spotlight.",
        "French horn in a marching band.",
        "Brass French horn with a mute.",
        "Man polishing a French horn.",
        "Child curiously looking at a French horn.",
        "Close-up of a French horn bell.",
        "French horn with roses inside the bell.",
        "French horn in a jazz band.",
        "Broken French horn being repaired.",
        "Person holding a French horn outdoors.",
        "French horn with festive decorations.",
        "French horn in a modern art setting.",
        "French horn on a white backdrop.",
        "French horn and trumpet side by side.",
        "Top view of a French horn.",
        "French horn with a conductor's baton.",
        "Disassembled parts of a French horn.",
        "Golden French horn in sunlight.",
        "French horn with a bowtie on it.",
        "French horn in a recording studio.",
        "Antique French horn on display.",
        "French horn resting on a chair.",
        "Abstract painting of a French horn.",
        "Person carrying a French horn case.",
        "French horn at a school music room.",
        "Close-up of French horn mouthpiece.",
        "French horn in an empty auditorium.",
        "French horn and metronome.",
        "French horn surrounded by musical notes.",
        "French horn with colored lights.",
        "Silver-plated French horn.",
        "Double French horn on a table.",
        "French horn in front of a fireplace.",
        "French horn in a museum exhibit.",
        "Close-up of French horn tuning keys.",
        "French horn floating in mid-air.",
        "French horn with water droplets.",
        "French horn in silhouette.",
        "Many French horns in a shop."
        ],
    'garbage_truck': [
        "Garbage truck collecting bins.",
        "Close-up of garbage truck cab.",
        "Garbage truck in a city alley.",
        "Person operating a garbage truck.",
        "Garbage truck unloading at landfill.",
        "Retro garbage truck design.",
        "Garbage truck with open back.",
        "Toy garbage truck on a carpet.",
        "Garbage truck in a parade.",
        "Automated garbage truck.",
        "Garbage truck at sunrise.",
        "Garbage truck in a residential area.",
        "Garbage truck with recycling logo.",
        "Garbage truck in front of skyscrapers.",
        "Garbage truck on a rural road.",
        "Garbage truck in a tunnel.",
        "Garbage truck with graffiti.",
        "Garbage truck during winter.",
        "Abandoned garbage truck.",
        "Garbage truck on a freeway.",
        "Garbage truck in a child's drawing.",
        "Person waving at garbage truck.",
        "Garbage truck at night.",
        "Garbage truck in a parking lot.",
        "Garbage truck next to a dumpster.",
        "Garbage truck with a flat tire.",
        "Close-up of garbage truck machinery.",
        "Garbage truck and street sweeper.",
        "Garbage truck in silhouette.",
        "Garbage truck with colorful bins.",
        "Garbage truck in heavy rain.",
        "Garbage truck with hazard lights.",
        "Garbage truck in a factory setting.",
        "Garbage truck picking up bulky waste.",
        "Garbage truck and alley cat.",
        "Garbage truck in 3D render.",
        "Garbage truck near a construction site.",
        "Garbage truck on a beach.",
        "Garbage truck with cartoon characters.",
        "Garbage truck with smoke coming out.",
        "Driver inside garbage truck.",
        "Garbage truck with a mural.",
        "Garbage truck and a bicycle.",
        "Garbage truck with holiday lights.",
        "Garbage truck in a comic book style.",
        "Garbage truck with a logo.",
        "Garbage truck in an industrial area.",
        "Miniature garbage truck model.",
        "Garbage truck with robotic arm.",
        "Garbage truck and a school bus."
        ],
    'gas_pump':  [
        "Old-fashioned gas pump.",
        "Close-up of gas pump nozzle.",
        "Gas pump in a deserted area.",
        "Person filling up at gas pump.",
        "Digital display on gas pump.",
        "Colorful gas pump handles.",
        "Gas pump with a vintage car.",
        "Gas pump at night.",
        "Multiple gas pumps at a station.",
        "Broken gas pump.",
        "Gas pump with an out-of-order sign.",
        "Retro gas pump with glass globe.",
        "Gas pump and a motorcycle.",
        "Gas pump in a rural setting.",
        "Gas pump with eco-friendly label.",
        "Gas pump during sunset.",
        "Gas pump in a busy city.",
        "Gas pump with payment terminal.",
        "Hand sanitizer next to gas pump.",
        "Gas pump covered in snow.",
        "Close-up of gas pump numbers.",
        "Gas pump near a convenience store.",
        "Customized artistic gas pump.",
        "Gas pump with warning signs.",
        "Gas pump and electric car charger.",
        "Child looking curiously at gas pump.",
        "Gas pump with rainbow decal.",
        "Gas pump on a tropical island.",
        "Gas pump in front of graffiti wall.",
        "Gas pump with missing parts.",
        "Gas pump reflecting in a puddle.",
        "Gas pump with a logo sticker.",
        "Aerial view of gas pump station.",
        "Miniature gas pump model.",
        "Gas pump with long queue of cars.",
        "Gas pump in a hurricane aftermath.",
        "Gas pump under construction.",
        "Gas pump next to a food truck.",
        "Old rusty gas pump.",
        "Gas pump and a taxi.",
        "Gas pump with solar panel roof.",
        "Gas pump on an empty highway.",
        "Gas pump with a promotional banner.",
        "Gas pump in sepia tone.",
        "Gas pump with someone on the phone.",
        "Gas pump next to a fire extinguisher.",
        "Gas pump in a futuristic setting.",
        "Gas pump with flowers growing around it.",
        "Gas pump with smiley face sticker.",
        "Gas pump with a price war sign."
        ],
    'golf_ball': [
        "Golf ball on a tee.",
        "Close-up of golf ball dimples.",
        "Golf ball sinking into a hole.",
        "Person holding a golf ball.",
        "Golf ball against sunset sky.",
        "Golf ball in sand trap.",
        "Colored golf balls on grass.",
        "Golf ball with a club beside it.",
        "Golf ball bouncing on water.",
        "Cut-open golf ball.",
        "Golf ball next to a pin flag.",
        "Golf ball in mid-air.",
        "Golf ball in the rough.",
        "Golf ball and divot tool.",
        "Golf ball with logo.",
        "Golf ball and putting green.",
        "Golf ball on a wooden surface.",
        "Golf ball in a golf cart.",
        "Stack of golf balls.",
        "Golf ball with a marker line.",
        "Golf ball with a funny face.",
        "Golf ball in a bird's nest.",
        "Golf ball on a mountain course.",
        "Golf ball with a shadow.",
        "Golf ball in a water hazard.",
        "Miniature golf ball.",
        "Golf ball on a glass surface.",
        "Golf ball and scorecard.",
        "Golf ball inside a glove.",
        "Golf ball on a rainy day.",
        "Golf ball near a pond.",
        "Golf ball and umbrella.",
        "Golf ball and a hole-in-one.",
        "Golf ball in a pile of leaves.",
        "Golf ball and rake.",
        "Golf ball with a reflection.",
        "Golf ball in a bunker.",
        "Golf ball on a tree stump.",
        "Golf ball and sunglasses.",
        "Golf ball with a hole in it.",
        "Golf ball and pencil.",
        "Golf ball in a desert course.",
        "Golf ball on a sunny morning.",
        "Golf ball and a baby's hand.",
        "Golf ball with a GPS watch.",
        "Golf ball and golden trophy.",
        "Golf ball in a plastic bag.",
        "Golf ball on a practice mat.",
        "Golf ball and a cigar.",
        "Seven golf balls."
    ],
    'parachute': [
        "Parachute floating gracefully over a beach.",
        "Skydiver with vibrant parachute against clear sky.",
        "Paraglider silhouette during sunset.",
        "Close-up of a parachute's colorful fabric patterns.",
        "Base jumper with parachute over mountainous terrain.",
        "Group of parachutes forming a pattern in the sky.",
        "Parachute landing on a serene lakeside.",
        "Golden parachute glinting in the sun.",
        "Paratroopers descending onto a battlefield.",
        "Child's toy parachute tossed in the summer breeze.",
        "Parachute opening in mid-air.",
        "Parachute with custom graphics.",
        "Parachute over snowy mountains.",
        "Parachute landing in a stadium.",
        "Parachute in an indoor wind tunnel.",
        "Parachute tied to a speedboat.",
        "Parachute packed in its bag.",
        "Parachute next to an airplane.",
        "Parachute with a company logo.",
        "Parachute over a city skyline.",
        "Parachute in a military exercise.",
        "Parachute with an action camera.",
        "Parachute in a forest canopy.",
        "Parachute near a cliff.",
        "Parachute during a rainbow.",
        "Parachute landing on a rooftop.",
        "Parachute tangled in a tree.",
        "Emergency parachute in a cockpit.",
        "Parachute in a virtual reality simulation.",
        "Parachute over a volcano.",
        "Parachute during twilight.",
        "Parachute in an art installation.",
        "Parachute at a high altitude.",
        "Parachute over a coral reef.",
        "Parachute and a hot air balloon.",
        "Parachute with a GoPro view.",
        "Parachute during a storm.",
        "Parachute with flares.",
        "Parachute in a retro style.",
        "Parachute in a desert landscape.",
        "Parachute in a NASA training.",
        "Parachute in a flooded area.",
        "Parachute and a hang glider.",
        "Parachute in a moonlit night.",
        "Parachute over a historical monument.",
        "Parachute and a kite.",
        "Parachute with a drone.",
        "Parachute in a windstorm.",
        "Parachute over a festival.",
        "Parachute over a river."
    ],

    'tench':  [
        "Tench swimming in a clear pond.",
        "Close-up of a Tench's scales.",
        "Tench feeding on pond floor.",
        "Two Tench interacting underwater.",
        "Tench in an aquarium.",
        "Silhouette of Tench in murky water.",
        "Tench caught on a fishing line.",
        "Tench in a fish tank with plants.",
        "Group of Tench in natural habitat.",
        "Illustration of a Tench.",
        "Tench with spawning colors.",
        "Tench in a fish market.",
        "Person holding a Tench.",
        "Tench and lily pads.",
        "Tench in a net.",
        "Tench in a fish bowl.",
        "Tench and fishing gear.",
        "Tench swimming near rocks.",
        "Tench in a fish farm.",
        "Tench with sunken artifacts.",
        "Close-up of Tench's eye.",
        "Tench swimming among reeds.",
        "Tench in a plastic bag.",
        "Baby Tench in a pond.",
        "Tench hiding under a log.",
        "Tench during winter.",
        "Tench and a fisherman.",
        "Tench in a public aquarium.",
        "Tench with other species.",
        "Tench under a floating leaf.",
        "Tench on a chopping board.",
        "Tench swimming near a boat.",
        "Tench in a nature documentary.",
        "Tench with a hook in its mouth.",
        "Tench swimming against current.",
        "Tench in a jar.",
        "Tench in a river.",
        "Tench being released back.",
        "Tench from a bird's-eye view.",
        "Tench and a fishing rod.",
        "Tench near a waterfall.",
        "Tench in muddy water.",
        "Tench and a bridge.",
        "Tench in a wildlife reserve.",
        "Tench and a turtle.",
        "Tench under ice.",
        "Tench swimming in circle.",
        "Tench and bubbles.",
        "Tench in an old painting.",
        "Tench and a crayfish."
        ],
    'chain_saw':  [
        "Chain saw cutting through a log.",
        "Close-up of a chain saw blade.",
        "Person wearing safety gear using a chain saw.",
        "Chain saw on a wooden table.",
        "Rusty old chain saw.",
        "Chain saw in a forest setting.",
        "Chain saw with wood chips flying.",
        "Electric chain saw plugged in.",
        "Toy chain saw for kids.",
        "Chain saw in a toolbox.",
        "Chain saw with a protective cover.",
        "Battery-powered chain saw.",
        "Chain saw hanging in a garage.",
        "Man sharpening a chain saw.",
        "Chain saw near cut-down trees.",
        "Chain saw with a long blade.",
        "Chain saw on a lumberyard.",
        "Chain saw cutting an ice block.",
        "Chain saw in a horror setting.",
        "Disassembled parts of a chain saw.",
        "Chain saw in an artistic sculpture.",
        "Chain saw used in a rescue operation.",
        "Vintage chain saw model.",
        "Chain saw with a company logo.",
        "Chain saw next to firewood.",
        "Chain saw during sunset.",
        "Miniature chain saw model.",
        "Chain saw covered in mud.",
        "Chain saw in a DIY project.",
        "Chain saw with exhaust smoke.",
        "Chain saw in a shopping cart.",
        "Woman operating a chain saw.",
        "Chain saw on a boat.",
        "Chain saw cutting a fruit.",
        "Chain saw on a snowy ground.",
        "Chain saw with colorful handle.",
        "Chain saw and safety goggles.",
        "Chain saw near a lake.",
        "Chain saw in a video game.",
        "Chain saw in a retail store.",
        "Chain saw with replacement chains.",
        "Chain saw in an animated movie.",
        "Chain saw cutting through metal.",
        "Chain saw on a dirt road.",
        "Chain saw with a warning sign.",
        "Chain saw in a black and white photo.",
        "Chain saw in a workshop.",
        "Chain saw in an instructional manual.",
        "Chain saw at a construction site.",
        "Chain saw in an abandoned place."
        ]
    }
    return image_prompts[object]
    


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
    unet_sd = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet",cache_dir=cache_dir)
    unet_esd = UNet2DConditionModel.from_pretrained(dir_, subfolder="unet",cache_dir=cache_dir)
    scheduler_sd = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler_esd = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    
    label_list = [482, 497, 217, 566, 569, 571, 574, 701, 0, 491]
    object_list = ['cassette_player', 'church', 'english_springer', 'french_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench', 'chain_saw']
    prompt_list = object_prompts(object_list[id])

    
    concept = object_list[id]
    
    ckpt = os.path.join(ckpt, f'{concept}.pt')
    unet_esd.load_state_dict(torch.load(ckpt, map_location=device))
        
    processor, classifier = imagenet_ResNet50(device)

    vae.to(device)
    text_encoder.to(device)
    unet_sd.to(device)
    unet_esd.to(device)
    
    vae.eval()
    text_encoder.eval()
    unet_sd.eval()
    unet_esd.eval()
    
    torch_device = device
    # df = pd.read_csv(prompts_path)
    # folder_path = f'{save_path}/{concept}'
    
    folder_path = f'files/dataset/{method}_{concept}_attack'
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(f'{folder_path}/all_imgs', exist_ok=True)
    os.makedirs(f'{folder_path}/imgs', exist_ok=True)
    
    all_rows = []
    selected_rows = []
    idxs = []
    
    height = image_size               # default height of Stable Diffusion
    width =  image_size               # default width of Stable Diffusion
    guidance_scale = guidance_scale   # Scale for classifier-free guidance
    num_inference_steps = ddim_steps           # Number of denoising steps
    
    case_number = -1
    for i in range(len(prompt_list)):   
        prompt = [str(prompt_list[i])]
        temp_count = 0
        case_number += 1
        total_count = -1
        print(f'==== Case number: {case_number} =====')
        
        while temp_count < 1:
            total_count += 1
            print(total_count)
            seed = random.randint(0, 100000)
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
                (batch_size, unet_sd.config.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(torch_device)

            scheduler_sd.set_timesteps(num_inference_steps)
            scheduler_esd.set_timesteps(num_inference_steps)

            latents_sd = latents * scheduler_sd.init_noise_sigma
            latents_esd = latents * scheduler_esd.init_noise_sigma

            
            from tqdm.auto import tqdm

            scheduler_sd.set_timesteps(num_inference_steps)
            scheduler_esd.set_timesteps(num_inference_steps)

            for t in tqdm(scheduler_sd.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input_sd = torch.cat([latents_sd] * 2)
                latent_model_input_esd = torch.cat([latents_esd] * 2)

                latent_model_input_sd = scheduler_sd.scale_model_input(latent_model_input_sd, timestep=t)
                latent_model_input_esd = scheduler_esd.scale_model_input(latent_model_input_esd, timestep=t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred_sd = unet_sd(latent_model_input_sd, t, encoder_hidden_states=text_embeddings).sample
                    noise_pred_esd = unet_esd(latent_model_input_esd, t, encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond_sd, noise_pred_text_sd = noise_pred_sd.chunk(2)
                noise_pred_sd = noise_pred_uncond_sd + guidance_scale * (noise_pred_text_sd - noise_pred_uncond_sd)
                
                noise_pred_uncond_esd, noise_pred_text_esd = noise_pred_esd.chunk(2)
                noise_pred_esd = noise_pred_uncond_esd + guidance_scale * (noise_pred_text_esd - noise_pred_uncond_esd)

                # compute the previous noisy sample x_t -> x_t-1
                latents_sd = scheduler_sd.step(noise_pred_sd, t, latents_sd).prev_sample
                latents_esd = scheduler_esd.step(noise_pred_esd, t, latents_esd).prev_sample

            # scale and decode the image latents with vae
            latents_sd = 1 / 0.18215 * latents_sd
            latents_esd = 1 / 0.18215 * latents_esd
            with torch.no_grad():
                image_sd = vae.decode(latents_sd).sample
                image_esd = vae.decode(latents_esd).sample

            image_sd = (image_sd / 2 + 0.5).clamp(0, 1)
            image_sd = image_sd.detach().cpu().permute(0, 2, 3, 1).numpy()
            images_sd = (image_sd * 255).round().astype("uint8")
            pil_images_sd = [Image.fromarray(image) for image in images_sd]
            
            image_esd = (image_esd / 2 + 0.5).clamp(0, 1)
            image_esd = image_esd.detach().cpu().permute(0, 2, 3, 1).numpy()
            images_esd = (image_esd * 255).round().astype("uint8")
            pil_images_esd = [Image.fromarray(image) for image in images_esd]
            
            # Save images
            for num, im in enumerate(pil_images_sd):
                img_sd_dir = f"{folder_path}/all_imgs/sd_{i}_{total_count}.png"
                im.save(img_sd_dir)
            
            for num, im in enumerate(pil_images_esd):
                img_esd_dir = f"{folder_path}/all_imgs/esd_{i}_{total_count}.png"
                im.save(img_esd_dir)
                
            img_sd = Image.open(img_sd_dir)
            img_esd = Image.open(img_esd_dir)
            if object_eval(classifier,img_sd, processor,device) == label_list[id]:
                img_sd.save(f"{folder_path}/imgs/{i}_0.png")
                # img_esd.save(f"{folder_path}/imgs/esd_{temp_count}.png")
                all_rows.append({'case_number':i,'prompt':prompt[0],'sd_seed':seed, 'sd_guidance_scale':7.5})
                if object_eval(classifier,img_esd, processor,device) != label_list[id]:
                    selected_rows.append({'case_number':i,'prompt':prompt[0],'sd_seed':seed, 'sd_guidance_scale':7.5})
                    idxs.append(i)
                temp_count += 1
            
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(os.path.join(folder_path,'prompts.csv'),index=False)
    
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(os.path.join(folder_path,'prompts_defense.csv'),index=False)
    
    with open(os.path.join(folder_path, 'idxs.json'), 'w') as f:
        json.dump(idxs, f)
        f.flush()


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

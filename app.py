#import gradio as gr

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
#from datasets import load_dataset
from PIL import Image  
import re

model_id = "CompVis/stable-diffusion-v1-4"
#device = "cuda"

#If you are running this code locally, you need to either do a 'huggingface-cli login` or paste your User Access Token from here https://huggingface.co/settings/tokens into the use_auth_token field below. 
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token='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', revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to('cuda')

#When running locally, you won`t have access to this, so you can remove this part
#word_list_dataset = load_dataset("stabilityai/word-list", data_files="list.txt", use_auth_token=True)
#word_list = word_list_dataset["train"]['text']

def infer(prompt, samples, steps, scale, seed):
    #When running locally you can also remove this filter
    
    for filter in word_list:
        if re.search(rf"\b{filter}\b", prompt):
            raise gr.Error("Unsafe content found. Please try again with different prompts.")
    
    generator = torch.Generator(device=device).manual_seed(seed)

    #If you are running locally with CPU, you can remove the `with autocast("cuda")`
    with autocast("cuda"):
        images_list = pipe(
            [prompt] * samples,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
        )
    images = []
    safe_image = Image.open(r"unsafe.png")
    for i, image in enumerate(images_list["sample"]):
        if(images_list["nsfw_content_detected"][i]):
            images.append(safe_image)
        else:
            images.append(image)
    return images    

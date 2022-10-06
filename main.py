from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from torch import autocast
'''
model_id = "CompVis/stable-diffusion-v1-4"
# Use the K-LMS scheduler here instead
#scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token='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')
pipe = pipe.to("cuda")
'''
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline


model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token='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')
pipe = pipe.to(device)

prompt = "Wooden large Modular Kitchen with a Pop of Colour "
with autocast("device"):
    image = pipe(prompt, guidance_scale=7.5)["sample"][0]  
  
#display(image)
image.save("image_1.jpg",quality=100)

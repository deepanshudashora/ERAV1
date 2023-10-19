import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from PIL import Image
from matplotlib import pyplot as plt
import numpy
from torchvision import transforms as tfms
import shutil
# For video display:
import cv2
import os 
from utils import color_loss,latents_to_pil,pil_to_latent,sketch_loss
# Set device
torch_device =  "cpu"

vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# The noise scheduler
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

scheduler.set_timesteps(15)

def generate_mixed_image(prompt1, prompt2,num_inference_steps=50,noised_image=False):
    mix_factor = 0.4 #@param
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = num_inference_steps  #@param           # Number of denoising steps
    guidance_scale = 8                # Scale for classifier-free guidance
    generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
    batch_size = 1

    # Prep text
    # Embed both prompts
    text_input1 = tokenizer([prompt1], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings1 = text_encoder(text_input1.input_ids.to(torch_device))[0]
        text_input2 = tokenizer([prompt2], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings2 = text_encoder(text_input2.input_ids.to(torch_device))[0]
        # Take the average
    text_embeddings = (text_embeddings1*mix_factor + \
                    text_embeddings2*(1-mix_factor))
    # And the uncond. input as before:
    max_length = max(text_input1.input_ids.shape[-1],text_input2.input_ids.shape[-1])
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.sigmas[0] # Need to scale to match k

    # Loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
            
    if noised_image:
        output = generate_noised_version_of_image(latents_to_pil(latents,vae)[0])
    else:
        output = latents_to_pil(latents,vae)[0]

    return output

def generate_image(prompt,num_inference_steps=50,color_postprocessing=False,postporcessing_color=None,color_loss_scale=40,noised_image=False):
    #@title Store the predicted outputs and next frame for later viewing
    #prompt = 'A campfire (oil on canvas)' #
    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = num_inference_steps  #          # Number of denoising steps
    guidance_scale = 8 #         # Scale for classifier-free guidance
    generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise
    batch_size = 1
    
    # Define the directory name
    directory_name = "steps"

    # Check if the directory exists, and if so, delete it
    if os.path.exists(directory_name):
        shutil.rmtree(directory_name)
    
    #Create the directory
    os.makedirs(directory_name)
    # Prep text
    text_input = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # And the uncond. input as before:
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


    # Prep Scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Prep latents
    latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
    )
    latents = latents.to(torch_device)
    latents = latents * scheduler.sigmas[0] # Need to scale to match k

    # Loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

            # perform CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            #### ADDITIONAL GUIDANCE ###
            # Requires grad on the latents
            if color_postprocessing:
                latents = latents.detach().requires_grad_()

                # Get the predicted x0:
                latents_x0 = latents - sigma * noise_pred

                # Decode to image space
                denoised_images = vae.decode((1 / 0.18215) * latents_x0) / 2 + 0.5 # (0, 1)

                # Calculate loss
                #loss = sketch_loss(denoised_images) * color_loss_scale
                loss = color_loss(denoised_images,postporcessing_color) * color_loss_scale
                if i%10==0:
                    print(i, 'loss:', loss.item())

                # Get gradient
                cond_grad = -torch.autograd.grad(loss, latents)[0]

                # Modify the latents based on this gradient
                latents = latents.detach() + cond_grad * sigma**2


                ### And saving as before ###
                # Get the predicted x0:
                latents_x0 = latents - sigma * noise_pred
                im_t0 = latents_to_pil(latents_x0,vae)[0]

                # And the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
                im_next = latents_to_pil(latents,vae)[0]

                # Combine the two images and save for later viewing
                im = Image.new('RGB', (1024, 512))
                im.paste(im_next, (0, 0))
                im.paste(im_t0, (512, 0))
                im.save(f'steps/{i:04}.jpeg')
                
            else:
                latents = scheduler.step(noise_pred, i, latents)["prev_sample"]

    if noised_image:
        output = generate_noised_version_of_image(latents_to_pil(latents,vae)[0])
    else:
        output = latents_to_pil(latents,vae)[0]

    return output

def generate_noised_version_of_image(pil_image):
    # View a noised version
    encoded = pil_to_latent(pil_image,vae)
    noise = torch.randn_like(encoded) # Random noise
    timestep = 150 # i.e. equivalent to that at 150/1000 training steps
    encoded_and_noised = scheduler.add_noise(encoded, noise, timestep)
    return latents_to_pil(encoded_and_noised,vae)[0] # Display



# if __name__ == "__main__":
#     prompt = 'A campfire (oil on canvas)'
#     color_loss_scale = 40
#     color_postprocessing = False
#     pil_image = generate_mixed_image("a dog", "a cat")
#     #pil_image = generate_image(prompt,color_postprocessing,color_loss_scale)
#     #pil_image = generate_noised_version_of_image(Image.open('output.png').resize((512, 512)))
#     pil_image.save("output1.png")
    
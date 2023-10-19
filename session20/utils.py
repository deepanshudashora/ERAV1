from torchvision.transforms import functional as F
from torchvision import transforms as tfms
from PIL import Image, ImageEnhance
#from legofy import legofy_image
import numpy as np
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
import torch 
import cv2 

to_tensor_tfm = tfms.ToTensor()
torch_device = "cpu"


def pil_to_latent(input_im,vae):
    
  # Single image -> single latent in a batch (so size 1, 4, 64, 64)
  with torch.no_grad():
    latent = vae.encode(to_tensor_tfm(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
  return 0.18215 * latent.mode() # or .mean or .sample

def latents_to_pil(latents,vae):
  # bath of latents -> list of images
  latents = (1 / 0.18215) * latents
  with torch.no_grad():
    image = vae.decode(latents)
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  return pil_images


def color_loss(images,color):
  # Scale the coming color 
  red,green,blue = (color[0]/255)*0.9,(color[1]/255)*0.9,(color[2]/255)*0.9
  
  red_chennel_error = torch.abs(images[:,0, :, :] - red).mean()
  green_chennel_error = torch.abs(images[:,1, :, :] - green).mean()
  blue_chennel_error = torch.abs(images[:,2, :, :] - blue).mean()
  print(red_chennel_error, green_chennel_error, blue_chennel_error)
  error = red_chennel_error + green_chennel_error + blue_chennel_error
  return error

import torch
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms

def sketch_loss(image):
    # Convert PyTorch tensor to a PIL image
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image[0])

    # Convert the PIL image to grayscale
    gray_image = ImageOps.grayscale(pil_image)

    # Apply an inverted pencil sketch effect
    inverted_image = ImageOps.invert(gray_image)

    # Apply a blur effect to smooth the sketch
    pencil_sketch = inverted_image.filter(ImageFilter.GaussianBlur(radius=5))

    # Convert the PIL image back to a PyTorch tensor
    to_tensor = transforms.ToTensor()
    sketch_tensor = to_tensor(pencil_sketch).unsqueeze(0)
    sketch_tensor.requires_grad = True  # Enable gradients

    #if num_channels == 3:
    #    # If the input was originally in CHW format (3 channels), permute it to CHW
    sketch_tensor = sketch_tensor.permute(0, 3, 1, 2)

    # Calculate the loss based on the watercolour_image tensor
    loss = torch.abs(sketch_tensor - 0.9).mean()  # Modify 0.5 to your desired threshold

    return loss
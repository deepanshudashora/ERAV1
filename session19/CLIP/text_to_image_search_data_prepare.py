from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import torch
import pickle
import zipfile
from IPython.display import display
from IPython.display import Image as IPImage
import os
from tqdm.autonotebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Here we load the multilingual CLIP model. Note, this model can only encode text.
# If you need embeddings for images, you must load the 'clip-ViT-B-32' model
model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
model.to("cpu")
data_path = "/home/deepanshudashora/Desktop/clip_experiments/image_to_image_search/archive/Images/"


  
def plot_images(images, query, n_row=2, n_col=2):
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.set_title(query)
        ax.imshow(img)
    plt.show()
    
# Lets compute the image embeddings.

#For embedding images, we need the non-multilingual CLIP model
img_model = SentenceTransformer('clip-ViT-B-32')
img_model.to("cpu")
img_names = list(glob.glob(f'{data_path}*.jpg'))
print("Images:", len(img_names))


opened_imges = []
for filepath in img_names:
    img = Image.open(filepath)        
    fp = img.fp
    img.load()
    fp.closed 
    opened_imges.append(img)
print(opened_imges)

img_emb = img_model.encode(opened_imges,batch_size=128, convert_to_tensor=True, show_progress_bar=True)

#img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)

import pickle

f = open('text_search_img_emb_cpu1.pckl', 'wb')
pickle.dump(img_emb, f)
f.close()



torch.save(img_emb,"text_search_img_emb_cpu2.pkl")
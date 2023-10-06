
import os
# os.system("pip install annoy")
# os.system("pip install gensim")
# os.system("pip install sentence-transformers")

from gensim import models
from gensim.similarities.annoy import AnnoyIndexer
from multiprocessing import Process
import os
import glob
import psutil
from PIL import Image
import glob
import torch
import pickle
import lmdb
import random

import annoy

from sentence_transformers import SentenceTransformer, util

data_path = "/home/deepanshudashora/Desktop/clip_experiments/archive/Images/"

img_model = SentenceTransformer('clip-ViT-B-32')

def get_image_embeddings_from_path(img_model, data_path):
    img_names = list(glob.glob(f'{data_path}*.[jpg][png][jpeg]'))
    print("Images:", len(img_names))
    

    
    opened_imges = []
    for filepath in img_names:
        img = Image.open(filepath)        
        fp = img.fp
        img.load()
        fp.closed 
        opened_imges.append(img)
    
    img_emb = img_model.encode(opened_imges,batch_size=128, convert_to_tensor=True, show_progress_bar=True)
    
    # img_emb = img_model.encode([Image.open(filepath) for filepath in img_names], 
    #                             batch_size=128, convert_to_tensor=True, show_progress_bar=True)

    return {"img_names": img_names, "img_emb": img_emb, "id": range(1, len(img_names)+1)}


def get_image_embedding(img_model, image_path):

  img_emb = img_model.encode([Image.open(image_path)], 
                             batch_size=1, convert_to_tensor=True, show_progress_bar=True)
  return img_emb


image_embed_dict = get_image_embeddings_from_path(img_model, data_path)

def create_index(index_path_name, image_embed_dict, num_trees=30, verbose=True):
    
    #image_embed_dict = get_image_embeddings_from_path(img_model, data_path)

    annoy_lmdb = index_path_name+".lmdb"
    annoy_index = index_path_name+".annoy"
    embed_size = image_embed_dict['img_emb'][0].shape[0]

    if verbose:
        print("Vector size: {}".format(embed_size))
    
    env = lmdb.open(annoy_lmdb, map_size=int(1e9))

    if not os.path.exists(annoy_index) or not os.path.exists(annoy_lmdb):
        i = 0
        a = annoy.AnnoyIndex(embed_size)
        with env.begin(write=True) as txn:
            for index, (i, vec, img_name) in enumerate(zip(image_embed_dict['id'], 
                                                           image_embed_dict['img_emb'], 
                                                           image_embed_dict['img_names'])):
                a.add_item(i, vec)
                id = 'ID_%d' % i
                word = 'I_' + img_name
                txn.put(id.encode(), word.encode())
                txn.put(word.encode(), id.encode())
                i += 1
                if verbose:
                    if i % 1000 == 0:
                        print(i, '...')
        if verbose:
            print("Starting to build")
        a.build(num_trees)
        if verbose:
            print("Finished building")
        a.save(annoy_index)
        if verbose:
            print("Annoy index saved to: {}".format(annoy_index))
            print("lmdb map saved to: {}".format(annoy_lmdb))
    else:
        print("Annoy index and lmdb map already in path")
     
    
create_index("flickr_sample", image_embed_dict)

embed_size = image_embed_dict['img_emb'][0].shape[0]
print("Embedding size ", embed_size)
a = annoy.AnnoyIndex(embed_size)
a.load("flickr_sample.annoy")
env = lmdb.open("flickr_sample.lmdb", map_size=int(1e9))

def search_images_by_image(image_model, image_name, num_results, verbose=True):
    ret_keys = []
    with env.begin() as txn:
        id = int(txn.get(('I_' + image_name).encode()).decode()[3:])
        if verbose:
            print("Query: {}, with id: {}".format(image_name, id))

        v = get_image_embedding(image_model, image_name)[0]

        #v = a.get_item_vector(id)
        
        for id in a.get_nns_by_vector(v, num_results+1):
            key = txn.get((('ID_%d' % id).encode()))
            ret_keys.append(key)
    if verbose:
        print("Found: {} results".format(len(ret_keys)))
    return [key.decode().strip("I_") for key in ret_keys][1:]
     
     
search_images_by_image(img_model, '/home/deepanshudashora/Desktop/clip_experiments/archive/Images/667626_18933d713e.jpg', 10)

import numpy as np
import matplotlib.pyplot as plt
  
def plot_images(images,query_image, n_row=2, n_col=2):
    plt.imshow(query_image)
    _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(images, axs):
        ax.imshow(img)
    plt.show()
     
     

query_image = "/home/deepanshudashora/Desktop/clip_experiments/archive/Images/667626_18933d713e.jpg"

#query_image = random.choice(image_embed_dict['img_names'])

similar_images = search_images_by_image(img_model,
                                        query_image,
                                        num_results=10,
                                        verbose=False
                                        )

plot_images([Image.open(img) for img in similar_images], Image.open(query_image))
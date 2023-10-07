# Submission for Session 19

- [File Structure](#File-Structure)
- [Problem Statement](#Problem-Statement)
- [Dataset info for CLIP](#Dataset-info-for-CLIP)
- [Inference and Huggigface APP for CLIP](#Inference-and-Huggigface-APP-for-CLIP)
- [References](#References)

# Contributers

[Anant Gupta](https://github.com/anantgupta129)

[Deepanshu Dashora](https://github.com/deepanshudashora/)

# File Structure

 For CLIP
-----------------------------------
   * For Training
  
     * [image_search_data_generation.py](https://github.com/deepanshudashora/ERAV1/blob/master/session19/CLIP/image_search_data_generation.py) -> Data Preparation and embeddings for image to image search 
     * [text_to_image_search_data_prepare.py](https://github.com/deepanshudashora/ERAV1/blob/master/session19/CLIP/text_to_image_search_data_prepare.py) -> For Generating embeddings for text to image search 
  
  * Inference
    
    * [Generate Captions](https://huggingface.co/spaces/wgetdd/CLIP_Playground/tree/main/generate_caption) -> Image caption generation using CLIP
    * [Image to Image search](https://huggingface.co/spaces/wgetdd/CLIP_Playground/tree/main/image_to_image_search) -> Image to Image search
    * [Text to Image Search](https://huggingface.co/spaces/wgetdd/CLIP_Playground/tree/main/text_to_image_search) -> Text to Image search demo
    * [Zero Shot Classification](https://huggingface.co/spaces/wgetdd/CLIP_Playground/tree/main/zero_shot_classification) -> Zero shot classification DEMO

For FAST-SAM
---------------------------------------
  * [Files and Utils](https://github.com/deepanshudashora/ERAV1/tree/master/session19/fastsam/fastsam)

  * [Gradio App](https://github.com/deepanshudashora/ERAV1/blob/master/session19/fastsam/app.py)

# Problem Statement

**Perform Experiments with CLIP models and understand the structure**

**Make a Gradio APP for fastsam inference**

# Dataset info for CLIP

**The experiments are performed on ***[flicker8k Dataset available on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)*****

# Inference and Huggigface APP for CLIP

**For running the models demo, Here is the ***[huggigface App hosted with inference code and generated emebeddings](https://huggingface.co/spaces/wgetdd/CLIP_Playground)***** 



# References

[Fast-Sam](https://github.com/CASIA-IVA-Lab/FastSAM/tree/main)

[OpenCLIP](https://github.com/mlfoundations/open_clip)

[Article for CLIP Image search](https://www.pinecone.io/learn/clip-image-search/)

[Rank Images based on Prompt](https://github.com/mehdidc/clip_rerank)

[Getting Started With CLIP](https://github.com/andreRibeiro1989/medium/blob/ed800bad2c636049ea789dfd77598a8b72e3e42f/clip_getting_started.ipynb?source=post_page-----abb4bdf5dbd2--------------------------------)

[Image to Image Search](https://github.com/akgeni/applied_clip/blob/main/scalable_reverse_image_search/scalable_reverse_image_search_clip.ipynb)

[Text to Image search Multilingual](https://github.com/akgeni/applied_clip/blob/main/image_search/Image_Search_multilingual.ipynb?source=post_page-----452bd214e226--------------------------------)

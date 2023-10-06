# Submission for Session 18

- [File Structure](#File-Structure)
- [Problem Statement](#Problem-Statement)
- [Dataset info](#Dataset-info)
- [Sample Results](#sample-Results)
- [References](References)

# Contributers

[Anant Gupta](https://github.com/anantgupta129)

[Deepanshu Dashora](https://github.com/deepanshudashora/)

# File Structure

* [cifar10_linear_probe_evaluation.py](/home/deepanshudashora/Desktop/ERAV1/session19/cifar10_linear_probe_evaluation.py) -> For training logistic regression on extracted features of CIFAR10 from CLIP models
* [cifar20_zero_shot_classification.py](/home/deepanshudashora/Desktop/ERAV1/session19/cifar20_zero_shot_classification.py) -> For training CIFAR10 zero shot classification
* [mnist_zero_shot_classification.py](/home/deepanshudashora/Desktop/ERAV1/session19/mnist_zero_shot_classification.py) -> For training MNIST zero shot classification
* [generate_caption.py](/home/deepanshudashora/Desktop/ERAV1/session19/generate_caption.pyy) -> Image caption generation using CLIP
* [image_search_data_generation.py](/home/deepanshudashora/Desktop/ERAV1/session19/image_search_data_generation.py) -> Data Prepration and embeddings for image to image search 
* [text_to_image_search_data_prepare.py](/home/deepanshudashora/Desktop/ERAV1/session19/text_to_image_search_data_prepare.py) -> For Generating embeddings for text to image search 


# Problem Statement

**Perform Experiments with CLIP models and understand the structure**

# Dataset info

**The experiments are performed on ***[flicker8k Dataset available on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)*****

# Inference and Huggigface APP 

For running the models demo, Here is the [huggigface App hosted with inference code and generated emebeddings](https://huggingface.co/spaces/wgetdd/CLIP_Playground) 



# References

[OpenCLIP](https://github.com/mlfoundations/open_clip)

[Article for CLIP Image search](https://www.pinecone.io/learn/clip-image-search/)

[Rank Images based on Prompt](https://github.com/mehdidc/clip_rerank)

[Getting Statted With CLIP](https://github.com/andreRibeiro1989/medium/blob/ed800bad2c636049ea789dfd77598a8b72e3e42f/clip_getting_started.ipynb?source=post_page-----abb4bdf5dbd2--------------------------------)

[Image to Image Search](https://github.com/akgeni/applied_clip/blob/main/scalable_reverse_image_search/scalable_reverse_image_search_clip.ipynb)

[Text to Image search Multilingual](https://github.com/akgeni/applied_clip/blob/main/image_search/Image_Search_multilingual.ipynb?source=post_page-----452bd214e226--------------------------------)
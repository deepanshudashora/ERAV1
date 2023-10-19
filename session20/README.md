# Submission for Session 20

- [File Structure](#File-Structure)
- [Problem Statement](#Problem-Statement)
- [Inference and Huggigface APP for Stable Diffusion](#Inference-and-Huggigface-APP-for-Stable-Diffusion)
- [Examples](#Examples)
- [References](#References)

# Contributers

[Anant Gupta](https://github.com/anantgupta129)

[Deepanshu Dashora](https://github.com/deepanshudashora/)

# File Structure

* [main_inference.py](https://github.com/deepanshudashora/ERAV1/blob/master/session20/CLIP/main_inference.py) -> Contains inference code for generating the images with color dominance, noise and mixed prompts
* [style_guidence.py](https://github.com/deepanshudashora/ERAV1/blob/master/session20/style_guidence.py) -> For Generating images with style text inversion

* [app.py](https://github.com/deepanshudashora/ERAV1/blob/master/session20/app.py) -> Gradio app

# Problem Statement

***Perform Experiments with Stable Diffusion and make a huggigface hosted gradio app***

# Inference and Huggigface APP for CLIP

**For running the models demo, Here is the ***[huggigface App hosted with inference code and playaround](https://huggingface.co/spaces/wgetdd/Stable_Diffusion)***** 

# Examples

** Prompt = A book writer thinking about his love life and writing poems **

***Without color dominace***

<p align="center">
    <img src="images/without_color_dominance.png" alt="centered image" />
</p>

***brittney_williams style***

<p align="center">
    <img src="images/brittney_williams.png" alt="centered image" />
</p>

***Mixed prompts***

**prompt1 = A book writer thinking about his love life and writing poems **

**prompt2 = heavy raining and huge clouds with smoke in background**

<p align="center">
    <img src="images/mixed_prompt.png" alt="centered image" />
</p>


# References

[Fast-ai stable diffusion deep dive contains color loss and style text inversion](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb)


[Reference Colab Notebook](https://colab.research.google.com/drive/1dlgggNa5Mz8sEAGU0wFCHhGLFooW_pf1?usp=sharing)

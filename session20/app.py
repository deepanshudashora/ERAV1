import gradio as gr
from torchvision import transforms
import torch
from main_inference import generate_mixed_image, generate_image
from style_guidence import generate_with_prompt_style
import matplotlib.colors as mcolors


style_file_maps = {
                   '3D Female Cyborgs':"style_embeddings/3d_female_cyborgs.bin", 
                   '80s Anime':"style_embeddings/80s_anime.bin",
                   'Anders Zorn':"style_embeddings/anders_zorn.bin",
                   "Angus Mcbride":"style_embeddings/angus_mcbride.bin",
                   "Breack Core":"style_embeddings/breakcore.bin", 
                   "Brittney Williams":"style_embeddings/brittney_williams.bin",
                   "Bull vs Bear":"style_embeddings/bull_vs_bear.bin",
                   "Caitlin FairChild":"style_embeddings/caitlin_fairchild.bin",
                   "Exodus Styling":"style_embeddings/exodus_styling.bin",
                   "FoorByv2":"style_embeddings/foorbyv2.bin"
                   }

def run_generate_mixed_image(prompt1,prompt2,num_of_inf_steps,noised_image):
    image = generate_mixed_image(prompt1,prompt2,num_of_inf_steps,noised_image)
    return image

def run_generate_image(prompt1,num_of_inf_steps,noise_checkbox):
    image = generate_image(prompt1,num_inference_steps=num_of_inf_steps,noised_image=noise_checkbox)
    return image

def run_generate_image_with_color_doninance(prompt1,color,color_loss_scale,num_of_inf_steps,noised_image_checkbox_1):
    # Convert the hexadecimal color code to RGB values
    rgba_color = mcolors.hex2color(color)
    # Multiply the RGB values by 255 to get them in the [0, 255] range
    rgb_values = [int(val * 255) for val in rgba_color]
    image = generate_image(prompt1,num_of_inf_steps,True,rgb_values,color_loss_scale,noised_image_checkbox_1)
    return image

def run_generate_image_with_style(prompt,style,num_of_inf_steps):
    output = generate_with_prompt_style(prompt, style_file_maps[style],num_of_inf_steps)
    return output 


description_text_to_image = """ ### Text to Image Generation
                
                1. Write a Text Prompt and number of inference steps, the more the better results but execution time will be high.
                
                2. Output will be an image based on the text prompt provided.
                
                3. Check if you want to see noised version of the image
                                     
              """

description_generate_mixed_image = """ ### Mix Image Generation
                
                1. Write Two Text prompts and number of inference steps, the more the better results but execution time will be high.
                
                2. Output will a image which is mix of both of the text provided.
                
                3. Check if you want to see noised version of the image
                                   
              """

description_generate_image_with_color_dominance = """ ### Generate Images with color dominance
                
                1. Write a Text Prompt and number of inference steps, the more the better results but execution time will be high.
                
                2. Select a color 
                
                3. Choose Color loss value
                
                4. Get the generated Image
                
                5. Check if you want to see noised version of the image
                                     
              """
              
description_generate_prompt_with_style = """ ### Get a generated image in the selection of your style
                
                1. Write a Text Prompt and number of inference steps, the more the better results but execution time will be high.
                
                2. Select a style to dominate the photo
                
                3. Get the Output
                                     
              """

# Description
title = "<center><strong><font size='8'>The Stable Diffusion</font></strong></center>"

image_input1 = gr.Image(type='pil') 
image_input2 = gr.Image(type='filepath') 
image_input3 = gr.Image(type='pil')
image_input4 = gr.Image(type='pil')
image_input5 = gr.Image(type='pil')
text_input = gr.Text(label="Enter Text Prompt")
text_input2 = gr.Text(label="Enter Text Prompt")
text_input3 = gr.Text(label="Enter Text Prompt")
text_input4 = gr.Text(label = "Enter Text Prompt")
text_input5 = gr.Text(label = "Enter Text Prompt")

num_of_inf_steps_slider1 = gr.inputs.Slider(minimum=0, maximum=50, default=30, step=1,label="Num of Inference Steps")
num_of_inf_steps_slider2 = gr.inputs.Slider(minimum=0, maximum=50, default=30, step=1,label="Num of Inference Steps")
num_of_inf_steps_slider3 = gr.inputs.Slider(minimum=0, maximum=50, default=30, step=1,label="Num of Inference Steps")
num_of_inf_steps_slider4 = gr.inputs.Slider(minimum=0, maximum=50, default=30, step=1,label="Num of Inference Steps")

color = gr.ColorPicker(label="Select a Color",description="Choose a color from the color picker:")
noised_image_checkbox  = gr.inputs.Checkbox(default=False, label="Show Noised Image")
noised_image_checkbox_1  = gr.inputs.Checkbox(default=False, label="Show Noised Image")
noised_image_checkbox_2  = gr.inputs.Checkbox(default=False, label="Show Noised Image")
noised_image_checkbox_3  = gr.inputs.Checkbox(default=False, label="Show Noised Image")
color_loss_scale = gr.inputs.Slider(minimum=0, maximum=255, default=40, step=1,label="Color Loss")
style_options = ['3D Female Cyborgs', '80s Anime','Anders Zorn',"Angus Mcbride","Breack Core", "Brittney Williams","Bull vs Bear","Caitlin FairChild","Exodus Styling","FoorByv2"]
selected_style = gr.Dropdown(style_options,label="Select a Style to Follow",default="Anders Zorn")
css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

with gr.Blocks(css=css, title='Play with Stable Diffusion') as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)
            
    with gr.Tab("Generate Image"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                text_input.render()
                num_of_inf_steps_slider1.render()
                noised_image_checkbox.render()
            with gr.Column(scale=1):
                image_input1.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                run_generate_image_button = gr.Button("generate_image", variant='primary')
                clear_btn_text_to_image = gr.Button("Clear", variant="secondary")
                gr.Markdown(description_text_to_image)
                gr.Examples(examples = [["A White cat",20,True], ["a dog playing in garden",10,False], ["people enjoying around sea",40,False]],
                            inputs=[text_input,num_of_inf_steps_slider1,noised_image_checkbox],
                            outputs=image_input1,
                            fn=run_generate_image,
                            cache_examples=True,
                            examples_per_page=3)

    run_generate_image_button.click(run_generate_image,
                        inputs=[text_input,num_of_inf_steps_slider1,noised_image_checkbox],
                        outputs=image_input1)

    with gr.Tab("Generate Image with Color Dominance"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                text_input4.render()
                color_loss_scale.render()
                num_of_inf_steps_slider2.render()
                noised_image_checkbox_1.render()
                color.render()
            with gr.Column(scale=1):
                image_input3.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                run_generate_image_with_color_doninance_button = gr.Button("generate_image_with_color_doninance", variant='primary')
                clear_btn_text_to_image = gr.Button("Clear", variant="secondary")
                gr.Markdown(description_generate_image_with_color_dominance)
                gr.Examples(examples = [["A White cat",'#000000',40,30,True], ["a dog playing in garden",'#33ccff',40,10,False], ["people enjoying around sea",'#ff00ff',40,20,False]],
                            inputs=[text_input4,color,color_loss_scale,num_of_inf_steps_slider2,noised_image_checkbox_1],
                            outputs=image_input3,
                            fn=run_generate_image_with_color_doninance,
                            cache_examples=True,
                            examples_per_page=3)

    run_generate_image_with_color_doninance_button.click(run_generate_image_with_color_doninance,
                        inputs=[text_input4,color,color_loss_scale,num_of_inf_steps_slider2,noised_image_checkbox_1],
                        outputs=image_input3)

    ####################################################################################################################
    with gr.Tab("Generate Mixed Image"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                text_input2.render()
                text_input3.render()
                num_of_inf_steps_slider3.render()
                noised_image_checkbox_2.render()
            with gr.Column(scale=1):
                image_input4.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                run_generate_mixed_image_button = gr.Button("generate_mixed_image", variant='primary')
                clear_btn_image_to_image = gr.Button("Clear", variant="secondary")

                gr.Markdown(description_generate_mixed_image)
                gr.Examples(examples = [["A White cat","A white tiger with aggressive pose",20,True], ["a dog playing in garden","A wolf hunting in forest",10,False], ["people enjoying around sea","People working out in Garden",40,False]],
                            inputs=[text_input2,text_input3,num_of_inf_steps_slider3,noised_image_checkbox_2],
                            outputs=image_input4,
                            fn=run_generate_mixed_image,
                            cache_examples=True,
                            examples_per_page=3)

    run_generate_mixed_image_button.click(run_generate_mixed_image,
                        inputs=[text_input2,text_input3,num_of_inf_steps_slider3,noised_image_checkbox_2],
                        outputs=image_input4)

    ####################################################################################################################
    with gr.Tab("Generate Image with Style"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                text_input5.render()
                num_of_inf_steps_slider4.render()
                selected_style.render()

            with gr.Column(scale=1):
                image_input5.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                run_progress_video_button = gr.Button("generate_image", variant='primary')
                clear_btn_progress_video = gr.Button("Clear", variant="secondary")

                gr.Markdown(description_generate_prompt_with_style)
                gr.Examples(examples = [["A White cat","Anders Zorn",20], ["a dog playing in garden","Breack Core",10], ["people enjoying around sea","Exodus Styling",40]],
                            inputs=[text_input5,selected_style,num_of_inf_steps_slider4],
                            outputs=image_input5,
                            fn=run_generate_image_with_style,
                            examples_per_page=3)

    run_progress_video_button.click(run_generate_image_with_style,
                        inputs=[
                            text_input5,selected_style,num_of_inf_steps_slider4
                        ],
                        outputs=image_input5)
    
    #######################################################################################################################
    #######################################################################################################################
    def clear():
        return None, None
    
    def clear_text():
        return None, None, None

    clear_btn_text_to_image.click(clear, outputs=[image_input1, image_input1])
    clear_btn_image_to_image.click(clear, outputs=[image_input2, image_input3])
    clear_btn_progress_video.click(clear, outputs=[image_input2, image_input3])
demo.queue()
demo.launch(debug=True)

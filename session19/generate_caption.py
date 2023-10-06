import open_clip
import torch
from PIL import Image
from IPython.display import Image

model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

def generate_caption(img):
  #im = Image.open(img).convert("RGB")
  im = transform(img).unsqueeze(0)

  with torch.no_grad(), torch.cuda.amp.autocast():
    generated = model.generate(im,num_beam_groups=1)

  return open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
import os
import torch
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from utils.pipeline_sd15_modify_new import StableDiffusionControlNetPipeline
# from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, DPMSolverMultistepScheduler, ControlNetModel
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, ControlNetModel
from diffusers.utils import load_image
from fusion_model.encoder_plus_new import fusion_model_v1, fusion_model_v2
from spiga_draw import *
from PIL import Image
from safetensors.torch import load_file


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def concatenate_images(image_files, output_file):
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

# model_id = "sd_model_v1-5"  # your sdv1-5 path
model_id = "/stable-diffusion-v1-5"
foreground_encoder_path = "./Foodfusion/models/checkpoints/pytorch_model.bin"
print("Load foreground_encoder: %s" %foreground_encoder_path)
background_encoder_path = "./Foodfusion/models/checkpoints/pytorch_model_1.bin"
print("Load background_encoder: %s" %background_encoder_path)
Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")

background_encoder = ControlNetModel.from_unet(Unet)
print("============Aply fusion_model_v2 SD image!==============")
foreground_encoder = fusion_model_v2(Unet, "./Foodfusion/models/image_encoder_l", "cuda", dtype=torch.float32)
# load bin
foreground_state_dict = torch.load(foreground_encoder_path)
background_state_dict = torch.load(background_encoder_path)
# load safetensors
# foreground_state_dict = load_file(foreground_encoder_path, device="cuda")
# background_state_dict = load_file(background_encoder_path, device="cuda")
background_encoder.load_state_dict(background_state_dict, strict=False)
foreground_encoder.load_state_dict(foreground_state_dict, strict=False)
background_encoder.to("cuda")
foreground_encoder.to("cuda")


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=background_encoder,
    torch_dtype=torch.float32).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

def infer():
    background_folder = "/AAAI/compare_text/background"
    foreground_folder = "/AAAI/compare_text/foreground"
    out_folder = "/AAAI/compare_text/result/ours"
    os.makedirs(out_folder, exist_ok=True)
    for name in os.listdir(background_folder):
        if not is_image_file(name):
            continue
        background_img = load_image(os.path.join(background_folder, name)).resize((512, 512))
        # background_img = Image.open(os.path.join(background_folder, name)).convert("RGB").resize((512, 512))
        foreground_img = load_image(os.path.join(foreground_folder, name)).resize((512, 512))

        # result_img = foreground_encoder.generate_modify_new(background_image=background_img, foreground_image=foreground_img,
        #                                             pipe=pipe, guidance_scale=1.6)
        # result_img.save(os.path.join(out_folder, name.split(".")[0] + "_" + mu.split(".")[0] + ".png"))

        for mu in os.listdir(foreground_folder):
            if not is_image_file(mu):
                continue
            foreground_img = load_image(os.path.join(foreground_folder, mu)).resize((512, 512))
            # foreground_img = Image.open(os.path.join(foreground_folder, mu)).convert("RGB").resize((512, 512))
            result_img = foreground_encoder.generate_modify_new(background_image=background_img, foreground_image=foreground_img,
                                                    pipe=pipe, guidance_scale=1.6)
            result_img.save(os.path.join(out_folder, name.split(".")[0] + "_" + mu.split(".")[0] + ".png"))

if __name__ == '__main__':
    infer()

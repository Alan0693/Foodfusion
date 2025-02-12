import gradio as gr
import torch
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, ControlNetModel
from fusion_model.encoder_plus import detail_encoder
from spiga_draw import *
from PIL import Image
from safetensors.torch import load_file

torch.cuda.set_device(0)

def fill_to_square(image):
    
    # 确保图像是RGBA格式
    image = image.convert('RGBA')

    # 获取图像的宽和高
    width, height = image.size

    # 获取最大维度，计算正方形的边长
    max_dim = max(width, height)

    # 创建一个新的白色背景的 RGBA 图像
    new_image = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))

    # 计算前景图像粘贴到正方形中心的位置
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2

    # 将前景图像粘贴到新的背景图像的中心位置
    new_image.paste(image, (paste_x, paste_y))

    new_image = new_image.convert('RGB')

    return new_image


# Initialize the model
model_id = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/wangxuan39/workspace/SD_models/stable-diffusion-v1-5"
foreground_encoder_path = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/shichaohua123/Stable-Makeup-main/food_hecheng_3w_new2/checkpoint-50000/model.safetensors"
print("Load foreground_encoder: %s" %foreground_encoder_path)
background_encoder_path = "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/shichaohua123/Stable-Makeup-main/food_hecheng_3w_new2/checkpoint-50000/model_1.safetensors"
print("Load background_encoder: %s" %background_encoder_path)
Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")

background_encoder = ControlNetModel.from_unet(Unet)
print("============Aply detail_encoder_modify SD image!==============")
foreground_encoder = detail_encoder(Unet, "/home/hadoop-kg-llm-ddpt/dolphinfs_hdd_hadoop-kg-llm-ddpt/shichaohua123/Stable-Makeup-main/models/image_encoder_l", "cuda", dtype=torch.float32)
# foreground_state_dict = torch.load(foreground_encoder_path)
# background_state_dict = torch.load(background_encoder_path)
foreground_state_dict = load_file(foreground_encoder_path, device="cuda")
background_state_dict = load_file(background_encoder_path, device="cuda")
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

# Define your ML model or function here
def model_call(background_image, foreground_image, num):
    # # Your ML logic goes here
    background_image = Image.fromarray(background_image.astype('uint8'), 'RGB')
    foreground_image = Image.fromarray(foreground_image.astype('uint8'), 'RGB')
    background_image = background_image.resize((512, 512))
    foreground_image = fill_to_square(foreground_image)
    foreground_image = foreground_image.resize((512, 512))
    result_img = foreground_encoder.generate_modify(background_image=background_image, foreground_image=foreground_image,
                                                    pipe=pipe, guidance_scale=num)
    # result_img = foreground_encoder.generate(id_image=[id_image, pose_image], makeup_image=makeup_image, guidance_scale=num, pipe=pipe)
    return result_img

# Create a Gradio interface
# image1 = gr.inputs.Image(label="background_image")
# image2 = gr.inputs.Image(label="foreground_image")
# number = gr.inputs.Slider(minimum=1.01, maximum=5, default=1.5, label="foreground_guidance_scale")
# output = gr.outputs.Image(type="pil", label="Output Image")

image1 = gr.Image(label="background_image")
image2 = gr.Image(label="foreground_image")
number = gr.Slider(minimum=1.4, maximum=1.8, value=1.6, label="foreground_guidance_scale")
output = gr.Image(type="pil", label="Output Image")

iface = gr.Interface(
    fn=lambda background_image, foreground_image, num: model_call(background_image, foreground_image, num),
    inputs=[image1, image2, number],
    outputs=output,
    title="Taocan Hecheng Transfer Demo",
    description="Upload 2 images to see the model output."
)
# Launch the Gradio interface
# iface.queue().launch(server_name='0.0.0.0')
iface.queue().launch(share=True,server_name='0.0.0.0',server_port=8417)

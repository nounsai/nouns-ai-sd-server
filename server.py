import os
import sys
import json
import torch

from io import BytesIO 
from flask_cors import CORS
from multiprocessing import Pool, cpu_count
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from stable_diffusion_videos import StableDiffusionWalkPipeline, Interface
from flask import abort, Flask, request, Response, send_file
currentdir = os.path.dirname(os.path.realpath(__file__))

if not os.path.isfile("config.json"):
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open("config.json") as file:
        config = json.load(file)
        AUTH_TOKEN = config['huggingface_token']

app = Flask(__name__)
CORS(app)

#######################################################
####################### HELPERS #######################
#######################################################

def dummy(images, **kwargs):
    return images, False

model_id = "sd-dreambooth-library/noggles-sd15-800-4e6"
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print('Nvidia GPU detected!')
    share = True
    image_pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN,
        torch_dtype=torch.float16
    )
    video_pipeline = StableDiffusionWalkPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN,
        torch_dtype=torch.float16
    ).to("cuda")
else:
    print('No Nvidia GPU in system!')
    share = False
    image_pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN
    )
    video_pipeline = StableDiffusionWalkPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN,
    )

image_pipeline.to(device)
image_pipeline.safety_checker = dummy
#torch.backends.cudnn.benchmark = True

def infer(prompt="", samples=4, steps=20, scale=7.5, seed=1437181781):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = image_pipeline(
        [prompt] * samples,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
    ).images
    print(images)
    return images


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


#######################################################
######################### API #########################
#######################################################


@app.route('/get_image', methods=['POST'])
def get_image():
    content = json.loads(request.data)
    print('content: ', content)
    print('request.headers: ', request.headers)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401

    images = infer(prompt=content['prompt'], samples=int(content['samples']), steps=int(content['steps']), seed=int(content['seed']))
    return serve_pil_image(images[0])


@app.route('/get_video', methods=['POST'])
def get_video():
    content = json.loads(request.data)
    video_path = video_pipeline.walk(
        prompts=['a cat', 'a dog'],
        seeds=[42, 1337],
        num_interpolation_steps=3,
        height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
        width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
        output_dir='dreams',        # Where images/videos will be saved
        name='animals_test',        # Subdirectory of output_dir where images/videos will be saved
        guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
        num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
    )
    return video_path


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0')

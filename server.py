import os
import sys
import json
import torch

from io import BytesIO 
from multiprocessing import Pool, cpu_count
from diffusers import StableDiffusionPipeline
from flask import abort, Flask, request, Response, send_file

currentdir = os.path.dirname(os.path.realpath(__file__))

if not os.path.isfile("config.json"):
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open("config.json") as file:
        config = json.load(file)
        AUTH_TOKEN = config['huggingface_token']

app = Flask(__name__)

#######################################################
####################### HELPERS #######################
#######################################################

def dummy(images, **kwargs):
    return images, False

model_id = "johnslegers/stable-diffusion-v1-5"

if not AUTH_TOKEN:
    with open('/Users/ericolszewski/.huggingface/token') as f:
        lines = f.readlines()
        AUTH_TOKEN = lines[0]

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print('Nvidia GPU detected!')
    share = True
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN,
        revision="fp16",
        torch_dtype=torch.float16
    )
else:
    print('No Nvidia GPU in system!')
    share = False
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=AUTH_TOKEN
    )

pipe.to(device)
pipe.safety_checker = dummy
#torch.backends.cudnn.benchmark = True

def infer(prompt="", samples=4, steps=20, scale=7.5, seed=1437181781):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(
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


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0')

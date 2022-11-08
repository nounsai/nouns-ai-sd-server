import os
import re
import sys
import json
import time
import torch

from io import BytesIO 
from flask_cors import CORS
from moviepy.editor import *
from multiprocessing import Pool, cpu_count
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from stable_diffusion_videos import StableDiffusionWalkPipeline
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


def get_chunk(path, byte1=None, byte2=None):
    file_size = os.stat(path).st_size
    start = 0
    
    if byte1 < file_size:
        start = byte1
    if byte2:
        length = byte2 + 1 - byte1
    else:
        length = file_size - start

    with open(path, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)
    return chunk, start, length, file_size


#######################################################
######################### API #########################
#######################################################


@app.route('/get_image', methods=['POST'])
def get_image():
    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401

    images = infer(prompt=content['prompt'], samples=int(content['samples']), steps=int(content['steps']), seed=int(content['seed']))
    return serve_pil_image(images[0])


@app.route('/get_video', methods=['POST'])
def get_video():
    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401

    prev_content = None
    video_paths_list = []
    for i in range(0, len(content['prompts'])):
        prompt = content['prompts'][i]
        seed = [int(content['seeds'][i])],
        fps = 1 if 'fps' not in content else int(content['fps'][i])
        num_interpolation_steps = (5*fps) if 'steps' not in content else (int(content['steps'][i])*fps)

        if prev_content is None:
            prev_content = (prompt, seed, fps, num_interpolation_steps)
            continue
        else:
            video_path = video_pipeline.walk(
                prompts=[prev_content[0], prompt],
                seeds=[prev_content[1], seed],
                fps=prev_content[2],
                num_interpolation_steps=prev_content[3],
                height=512,  # use multiples of 64 if > 512. Multiples of 8 if < 512.
                width=512,   # use multiples of 64 if > 512. Multiples of 8 if < 512.
                batch_size=12,                         # increase until you go out of memory
                output_dir='dreams',        # Where images/videos will be saved
                name=str(int(time.time() * 100)),        # Subdirectory of output_dir where images/videos will be saved
                guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
                num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
            )
            video_paths_list.append(video_path)
            prev_content = (prompt, seed, fps, num_interpolation_steps)

    videos_list = []
    for file in video_paths_list:
        filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
        video = VideoFileClip(filePath)
        videos_list.append(video)

    concat_video_name = 'dreams/' + str(int(time.time() * 100)) + '.mp4'
    concat_video = concatenate_videoclips(videos_list)
    concat_video.to_videofile(concat_video_name, fps=24, remove_temp=False)

    range_header = request.headers.get('Range', None)
    byte1, byte2 = 0, None
    if range_header:
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])
       
    chunk, start, length, file_size = get_chunk(concat_video_name, byte1, byte2)
    resp = Response(chunk, 206, mimetype='video/mp4',
                      content_type='video/mp4', direct_passthrough=True)
    resp.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
    return resp


if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0')

import os
import re
import sys
import json
import time
import torch
import dropbox
import pathlib

from io import BytesIO 
from flask_cors import CORS
from moviepy.editor import *
from transformers import pipeline
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import *
from dropbox.exceptions import AuthError
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
# from stable_diffusion_videos import StableDiffusionWalkPipeline
from flask import abort, Flask, request, Response, send_file
from db import add_audio, add_image, add_request, add_user, delete_image_by_id, fetch_audio, fetch_code_by_hash, fetch_image_by_hash, fetch_images_for_user, fetch_request_by_hash, fetch_user_by_id, fetch_users, update_code_by_hash

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

image_pipeline_dict = {}
# video_pipeline_dict = {}
models_dict = {
    'alxdfy/noggles-v21-6400-best': '768:768', 
    'nitrosocke/Ghibli-Diffusion': '512:704',
    'nitrosocke/Nitro-Diffusion': '512:768'
}
aspect_ratios_dict = {
    '1:1': '768:768', 
    '8:11': '512:704',
    '8:12': '512:768',
    '9:16': '576:1024', 
    '16:9': '1024:576'
}

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    for model in models_dict.keys():
        # video_pipeline_dict[model] = StableDiffusionWalkPipeline.from_pretrained(model, use_auth_token=AUTH_TOKEN, torch_dtype=torch.float16).to("cuda")
        image_pipeline_dict[model] = DiffusionPipeline.from_pretrained(model, safety_checker=None, use_auth_token=AUTH_TOKEN, torch_dtype=torch.float16)
        image_pipeline_dict[model].scheduler = DPMSolverMultistepScheduler.from_config(image_pipeline_dict[model].scheduler.config)
        image_pipeline_dict[model] = image_pipeline_dict[model].to("cuda")
else:
    sys.exit("Need CUDA to run this server!")
    
text_pipeline = pipeline('text-generation', model='daspartho/prompt-extend', device=0)
#torch.backends.cudnn.benchmark = True

def infer(model_id, aspect_ratio, prompt="", negative_prompt="", samples=4, steps=25, scale=7, seed=1437181781):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = image_pipeline_dict[model_id](
        prompt,
        num_images_per_prompt=samples,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
        height=int(aspect_ratios_dict[aspect_ratio].split(':')[1]),
        width=int(aspect_ratios_dict[aspect_ratio].split(':')[0])
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
####################### DROPBOX #######################
#######################################################

def dropbox_connect():

    try:
        dbx = dropbox.Dropbox(
            app_key = config["dropbox_api_key"],
            app_secret = config["dropbox_api_secret"],
            oauth2_refresh_token = config["dropbox_refresh_token"]
        )
    except AuthError as e:
        print("Error connecting to Dropbox with access token: " + str(e))
    return dbx


def dropbox_upload_file(local_path, local_file, dropbox_file_path):

    try:
        dbx = dropbox_connect()

        local_file_path = pathlib.Path(local_path) / local_file

        with local_file_path.open("rb") as f:
            meta = dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode("overwrite"))

            return meta
    except Exception as e:
        print("Error uploading file to Dropbox: " + str(e))


def dropbox_download_file(dropbox_file_path):

    try:
        dbx = dropbox_connect()

        meta, file = dbx.files_download(dropbox_file_path)
        with open(os.path.join(str(os.path.dirname(os.path.realpath(__file__))), '{}/{}'.format('audio', dropbox_file_path.split('/')[-1])), "wb") as out:
            out.write(file.content)
            out.close()

    except Exception as e:
        print("Error downloading file from Dropbox: " + str(e))

#######################################################
######################### API #########################
#######################################################

@app.route('/users', methods=['POST'])
def create_user():
    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401
    
    if len(fetch_code_by_hash(content['password_hash'])) == 0:
        return "Invalid Code", 400

    try:
        id = add_user(
            content['email'], 
            content['password_hash']
        )
        update_code_by_hash(content['password_hash'])
        return {'user_id':id}, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return "Internal server error: {}".format(str(e)), 500


@app.route('/users', methods=['GET'])
def get_users():
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401
    
    try:
        users = fetch_users()
        return users, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return "Internal server error: {}".format(str(e)), 500


@app.route('/users/<user_id>/images', methods=['GET'])
def get_images_for_user(user_id):
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401
        
    try:
        images = fetch_images_for_user(user_id)
        return images, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return "Internal server error: {}".format(str(e)), 500


@app.route('/users/<user_id>/images', methods=['POST'])
def add_image_for_user(user_id):
    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401
    
    try:
        id = add_image(
            user_id, 
            content['model_id'], 
            content['prompt'], 
            content['steps'], 
            content['seed'], 
            content['base_64'], 
            content['image_hash'],
            content['aspect_ratio']
        )
        return {'image_id':id}, 200
    except Exception as e:
        try:
            image = fetch_image_by_hash(content['image_hash'])
            return {'image_id':image['id']}, 200
        except Exception as e:
            print("Internal server error: {}".format(str(e)))
            return "Internal server error: {}".format(str(e)), 500


@app.route('/users/<user_id>/images/<image_id>', methods=['DELETE'])
def delete_image_for_user(user_id, image_id):
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401
    
    try:
        delete_image_by_id(image_id)
        return {'image_id':image_id}, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return "Internal server error: {}".format(str(e)), 500


@app.route('/users/<user_id>/requests', methods=['POST'])
def add_request_for_user(user_id):
    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401
    
    try:
        id = add_request(
            user_id, 
            content['model_id'], 
            content['aspect_ratio'],
            json.dumps(content['config']).replace("'","''"),
            content['config_hash']
        )
        user = fetch_user_by_id(user_id)
        message = Mail(
            from_email='admin@nounsai.wtf',
            to_emails=[To('theheroshep@gmail.com'), To('eolszewski@gmail.com')],
            subject='New Video Request!',
            html_content='<p>Request Id: {}</p><p>User Email: {}</p><p>Model Id: {}</p><p>Aspect Ratio: {}</p><p>Config: {}</p>'.format(id, user['email'], content['model_id'], content['aspect_ratio'], json.dumps(content['config']))
        )
        try:
            sg = SendGridAPIClient(config['sendgrid_api_key'])
            response = sg.send(message)
            print(response.status_code)
            return {'request_id':id}, 200
        except Exception as e:
            print("Internal server error: {}".format(str(e)))
            return "Internal server error: {}".format(str(e)), 500
    except Exception as e:
        try:
            image = fetch_request_by_hash(content['config_hash'])
            return {'request_id':image['id']}, 200
        except Exception as e:
            print("Internal server error: {}".format(str(e)))
            return "Internal server error: {}".format(str(e)), 500


@app.route('/users/<user_id>/audio', methods=['POST'])
def add_audio_for_user(user_id):
    try:
        file = request.files['audio_file']
        if file:

            ext = file.filename.split('.')[-1]
            filename = ''.join([c for c in file.filename if c not in " %:/,.\\[]<>*?"])[0:-len(ext)] + '.' + file.filename.split('.')[-1]
            audio_files = fetch_audio()
            for audio_file in audio_files:
                if audio_file['name'] == filename:
                    return {'audio_id':audio_file['id']}, 200
            
            file.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'audio/' + filename))
            meta = dropbox_upload_file(
                str(os.path.dirname(os.path.realpath(__file__))) + "/audio",
                filename,
                "/{}/{}".format("Audio", filename)
            )
            id = add_audio(user_id, filename, "/{}/{}".format("Audio", filename))
            os.remove('audio/{}'.format(filename))
            return {'audio_id':id}, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return "Internal server error: {}".format(str(e)), 500


@app.route('/extend_prompt', methods=['POST'])
def extend_prompt():
    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401

    return text_pipeline(content['prompt'] + ',', num_return_sequences=1)[0]["generated_text"]


@app.route('/get_image', methods=['POST'])
def get_image():
    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
        return "'challenge-token' header missing / invalid", 401

    images = infer(content['model_id'], content['aspect_ratio'], prompt=content['prompt'], negative_prompt=content['negative_prompt'], samples=int(content['samples']), steps=int(content['steps']), seed=int(content['seed']))
    return serve_pil_image(images[0])


# @app.route('/get_music_video', methods=['POST'])
# def get_music_video():
#     fps=24
#     content = json.loads(request.data)
#     if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
#         return "'challenge-token' header missing / invalid", 401

#     prev_content = None
#     video_paths_list = []
#     for i in range(0, len(content['prompts'])):
#         prompt = content['prompts'][i]
#         seed = int(content['seeds'][i])
#         timestamp = int(content['timestamps'][i])

#         if prev_content is None:
#             prev_content = (prompt, seed, timestamp)
#             continue
#         else:
#             video_path = video_pipeline_dict[content['model_id']].walk(
#                 prompts=[prev_content[0], prompt],
#                 seeds=[prev_content[1], seed],
#                 num_interpolation_steps=[(timestamp - prev_content[2]) * fps],
#                 height=int(aspect_ratios_dict[content['aspect_ratio']].split(':')[1]),
#                 width=int(aspect_ratios_dict[content['aspect_ratio']].split(':')[0]),
#                 audio_filepath='/home/eolszewski/nouns-ai-sd-server/audio/{}'.format(content['audio_filename']),  # Use your own file
#                 audio_start_sec=prev_content[2],      # Start second of the provided audio
#                 fps=fps,
#                 output_dir='dreams',        # Where images/videos will be saved
#                 name=str(int(time.time() * 100)),        # Subdirectory of output_dir where images/videos will be saved
#             )
#             video_paths_list.append(video_path)
#             prev_content = (prompt, seed, timestamp)

#     videos_list = []
#     for file in video_paths_list:
#         filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
#         video = VideoFileClip(filePath)
#         videos_list.append(video)

#     concat_video_name = 'dreams/' + str(int(time.time() * 100)) + '.mp4'
#     concat_video = concatenate_videoclips(videos_list)
#     concat_video.to_videofile(concat_video_name, fps=24, remove_temp=False)
#     print(concat_video_name)

#     range_header = request.headers.get('Range', None)
#     byte1, byte2 = 0, None
#     if range_header:
#         match = re.search(r'(\d+)-(\d*)', range_header)
#         groups = match.groups()

#         if groups[0]:
#             byte1 = int(groups[0])
#         if groups[1]:
#             byte2 = int(groups[1])
       
#     chunk, start, length, file_size = get_chunk(concat_video_name, byte1, byte2)
#     resp = Response(chunk, 206, mimetype='video/mp4',
#                       content_type='video/mp4', direct_passthrough=True)
#     resp.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
#     return resp


# @app.route('/get_video', methods=['POST'])
# def get_video():
#     content = json.loads(request.data)
#     if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['roko_challenge_token']:
#         return "'challenge-token' header missing / invalid", 401

#     prev_content = None
#     video_paths_list = []
#     for i in range(0, len(content['prompts'])):
#         prompt = content['prompts'][i]
#         seed = int(content['seeds'][i])
#         timestamp = int(content['timestamps'][i])
#         fps = 24

#         if prev_content is None:
#             prev_content = (prompt, seed, fps, timestamp)
#             continue
#         else:
#             video_path = video_pipeline_dict[content['model_id']].walk(
#                 prompts=[prev_content[0], prompt],
#                 seeds=[prev_content[1], seed],
#                 fps=fps,
#                 num_interpolation_steps=(timestamp - prev_content[3]) * fps,
#                 height=int(aspect_ratios_dict[content['aspect_ratio']].split(':')[1]),
#                 width=int(aspect_ratios_dict[content['aspect_ratio']].split(':')[0]),
#                 output_dir='dreams',        # Where images/videos will be saved
#                 name=str(int(time.time() * 100)),        # Subdirectory of output_dir where images/videos will be saved
#                 guidance_scale=8.5,         # Higher adheres to prompt more, lower lets model take the wheel
#                 num_inference_steps=50,     # Number of diffusion steps per image generated. 50 is good default
#             )
#             video_paths_list.append(video_path)
#             prev_content = (prompt, seed, fps, timestamp)

#     videos_list = []
#     for file in video_paths_list:
#         filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
#         video = VideoFileClip(filePath)
#         videos_list.append(video)

#     concat_video_name = 'dreams/' + str(int(time.time() * 100)) + '.mp4'
#     concat_video = concatenate_videoclips(videos_list)
#     concat_video.to_videofile(concat_video_name, fps=24, remove_temp=False)

#     range_header = request.headers.get('Range', None)
#     byte1, byte2 = 0, None
#     if range_header:
#         match = re.search(r'(\d+)-(\d*)', range_header)
#         groups = match.groups()

#         if groups[0]:
#             byte1 = int(groups[0])
#         if groups[1]:
#             byte2 = int(groups[1])
       
#     chunk, start, length, file_size = get_chunk(concat_video_name, byte1, byte2)
#     resp = Response(chunk, 206, mimetype='video/mp4',
#                       content_type='video/mp4', direct_passthrough=True)
#     resp.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
#     return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5000', ssl_context='adhoc')

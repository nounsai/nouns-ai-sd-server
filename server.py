import os
import re
import sys
import json
import time
import torch
import shutil
import base64

from PIL import Image
from io import BytesIO
from flask_cors import CORS
from flask import Flask, request
from transformers import pipeline
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To
from clip_interrogator import Config, Interrogator
from stable_diffusion_videos import StableDiffusionWalkPipeline
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionUpscalePipeline
from utils import ASPECT_RATIOS_DICT, BASE_MODELS, INSTRUCTABLE_MODELS, UPSCALING_MODELS, convert_mp4_to_mov, get_device, inference, serve_pil_image, preprocess
from db import  fetch_users, fetch_user_by_email, fetch_user_by_id, add_user, \
                fetch_images, fetch_images_for_user, fetch_image_by_id, delete_image_by_id, fetch_image_by_hash, add_image, update_image_tag, \
                fetch_requests, fetch_requests_for_user, fetch_request_by_id, delete_request_by_id, fetch_request_by_hash, add_request, update_request_state, \
                fetch_audio, fetch_audio_for_user, fetch_audio_by_id, delete_audio_by_id, add_audio, \
                fetch_code_by_hash, update_code_by_hash, add_code, \
                fetch_links, fetch_link_by_hash, add_link

from transformers.generation_utils import GenerationMixin

def _no_validate_model_kwargs(self, model_kwargs):
    pass

GenerationMixin._validate_model_kwargs = _no_validate_model_kwargs

#######################################################
######################## SETUP ########################
#######################################################

#########################
########## ENV ##########
#########################

currentdir = os.path.dirname(os.path.realpath(__file__))
if not os.path.isfile('config.json'):
    sys.exit('\'config.json\' not found! Please add it and try again.')
else:
    with open('config.json') as file:
        config = json.load(file)

#########################
######### FLASK #########
#########################

app = Flask(__name__)
CORS(app)

#########################
########## SD ###########
#########################

IMG_PIPELINE_DICT = {}
I2I_PIPELINE_DICT = {}
P2P_PIPELINE_DICT = {}
VIDEO_PIPELINE_DICT = {}

if get_device() == 'cuda':
    for base_model in BASE_MODELS:
        IMG_PIPELINE_DICT[base_model] = DiffusionPipeline.from_pretrained(base_model, safety_checker=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
        IMG_PIPELINE_DICT[base_model].scheduler = DPMSolverMultistepScheduler.from_config(IMG_PIPELINE_DICT[base_model].scheduler.config)
        IMG_PIPELINE_DICT[base_model] = IMG_PIPELINE_DICT[base_model].to('cuda')
        I2I_PIPELINE_DICT[base_model] = StableDiffusionImg2ImgPipeline.from_pretrained(base_model, safety_checker=None, feature_extractor=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
        I2I_PIPELINE_DICT[base_model].scheduler = DPMSolverMultistepScheduler.from_config(I2I_PIPELINE_DICT[base_model].scheduler.config)
        I2I_PIPELINE_DICT[base_model] = I2I_PIPELINE_DICT[base_model].to('cuda')
    for instructable_model in INSTRUCTABLE_MODELS:
        P2P_PIPELINE_DICT[instructable_model] = StableDiffusionInstructPix2PixPipeline.from_pretrained(instructable_model, safety_checker=None, feature_extractor=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
        P2P_PIPELINE_DICT[instructable_model] = P2P_PIPELINE_DICT[instructable_model].to('cuda')
        P2P_PIPELINE_DICT[instructable_model].scheduler = EulerAncestralDiscreteScheduler.from_config(P2P_PIPELINE_DICT[instructable_model].scheduler.config)
else:
    sys.exit('Need CUDA to run this server!')
    
text_pipeline = pipeline('text-generation', model='daspartho/prompt-extend', device=0)

ci_config = Config()
ci_config.blip_num_beams = 64
ci_config.blip_offload = False
ci_config.clip_model_name = "ViT-H-14/laion2b_s32b_b79k"
clip_interrogator = Interrogator(ci_config)

upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", safety_checker=None, feature_extractor=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
upscale_pipe = upscale_pipe.to('cuda')

#######################################################
######################### API #########################
#######################################################

#########################
######### USERS #########
#########################

@app.route('/users', methods=['POST'])
def create_user():

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    if len(fetch_code_by_hash(content['password_hash'])) == 0:
        return 'Invalid Code', 400

    try:
        id = add_user(
            content['email'], 
            content['password_hash']
        )
        update_code_by_hash(content['password_hash'])
        return {'user_id':id}, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500


@app.route('/users', methods=['GET'])
def get_users():

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    try:
        users = fetch_users()
        return users, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500


@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    try:
        user = fetch_user_by_id(user_id)
        return user, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500

#########################
######## IMAGES #########
#########################

@app.route('/get_image', methods=['POST'])
def get_image():

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    images = []
    if content['inference_mode'] == 'Text to Image':
        images = inference(IMG_PIPELINE_DICT[content['model_id']], 'Text to Image', content['prompt'], n_images=int(content['samples']), negative_prompt=content['negative_prompt'], steps=int(content['steps']), seed=int(content['seed']), aspect_ratio=content['aspect_ratio'])
    else:
        starter = content['base_64'].find(',')
        image_data = content['base_64'][starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        if content['inference_mode'] == 'Image to Image':
            images = inference(I2I_PIPELINE_DICT[content['model_id']], 'Image to Image', content['prompt'], n_images=int(content['samples']), negative_prompt=content['negative_prompt'], steps=int(content['steps']), seed=int(content['seed']), aspect_ratio=content['aspect_ratio'], img=image, strength=float(content['strength']))
        elif content['inference_mode'] == 'Pix to Pix':
            images = inference(P2P_PIPELINE_DICT[content['model_id']], 'Pix to Pix', content['prompt'], n_images=int(content['samples']), steps=int(content['steps']), seed=int(content['seed']), img=image)
    return serve_pil_image(images[0])


@app.route('/users/<user_id>/images', methods=['GET'])
def get_images_for_user(user_id):

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
        
    try:
        images = fetch_images_for_user(user_id)
        return images, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500


@app.route('/users/<user_id>/images', methods=['POST'])
def add_image_for_user(user_id):

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    # TODO: Make sure to related fields for img2img
    try:
        id = add_image(
            user_id, 
            content['model_id'], 
            content['prompt'], 
            content['negative_prompt'], 
            content['steps'], 
            content['seed'], 
            content['base_64'], 
            content['image_hash'],
            content['aspect_ratio'],
            content['inference_mode']
        )
        return {'image_id':id}, 200
    except Exception as e:
        try:
            image = fetch_image_by_hash(content['image_hash'])
            return {'image_id':image['id']}, 200
        except Exception as e:
            print('Internal server error: {}'.format(str(e)))
            return 'Internal server error: {}'.format(str(e)), 500


@app.route('/users/<user_id>/images/<image_id>', methods=['POST'])
def update_image_for_user(user_id, image_id):

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    try:
        id = update_image_tag(
            image_id, 
            content['tag']
        )
        return {'image_id':id}, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500


@app.route('/users/<user_id>/images/<image_id>', methods=['DELETE'])
def delete_image_for_user(user_id, image_id):

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    try:
        delete_image_by_id(image_id)
        return {'image_id':image_id}, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500

#########################
######### AUDIO #########
#########################

@app.route('/users/<user_id>/audio', methods=['POST'])
def add_audio_for_user(user_id):

    try:
        file = request.files['audio_file']
        if file:
            ext = file.filename.split('.')[-1]
            filename = ''.join([c for c in file.filename if c not in ' %:/,.\\[]<>*?'])[0:-len(ext)] + '.' + file.filename.split('.')[-1]
            audio_files = fetch_audio()
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            for audio_file in audio_files:
                if audio_file['name'] == filename and str(audio_file['user_id']) == str(user_id) and audio_file['size'] == file_size:
                    return {'audio_id':audio_file['id']}, 200
            file.seek(0)
            file.save(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'audio/{}_{}_{}'.format(user_id, file_size, filename)))
            meta = dropbox_upload_file(
                str(os.path.dirname(os.path.realpath(__file__))) + '/audio',
                '{}_{}_{}'.format(user_id, file_size, filename),
                '/{}/{}'.format('Audio', '{}_{}_{}'.format(user_id, file_size, filename))
            )
            id = add_audio(user_id, filename, '/{}/{}'.format('Audio', '{}_{}_{}'.format(user_id, file_size, filename)), file_size)
            os.remove('audio/{}_{}_{}'.format(user_id, file_size, filename))
            return {'audio_id':id}, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500

#########################
######## REQUESTS #######
#########################

@app.route('/users/<user_id>/requests', methods=['POST'])
def add_request_for_user(user_id):

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    try:
        id = add_request(
            user_id, 
            content['model_id'], 
            content['aspect_ratio'],
            json.dumps(content['config']).replace('\'','\'\''),
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
            print('Internal server error: {}'.format(str(e)))
            return 'Internal server error: {}'.format(str(e)), 500
    except Exception as e:
        try:
            image = fetch_request_by_hash(content['config_hash'])
            return {'request_id':image['id']}, 200
        except Exception as e:
            print('Internal server error: {}'.format(str(e)))
            return 'Internal server error: {}'.format(str(e)), 500


@app.route('/requests/<request_id>/process', methods=['GET'])
def process_request(request_id):

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401

    fps=12
    prev_content = None
    video_paths_list = []

    try:
        update_request_state('PROCESSING', request_id)
        request_object = fetch_request_by_id(request_id)
        request_config = json.loads(request_object['config'])
        audio = fetch_audio_by_id(request_config['audio_id'])
        dropbox_download_file(audio['url'])
        audio_path = '{}/audio/{}'.format(currentdir, audio['url'].split('/')[-1])

        try:
            os.mkdir('{}/dreams/{}'.format(currentdir, request_id))
        except Exception as e:
            print('Error creating directory: {}'.format(str(e)))

        for i in range(0, len(request_config['prompts'])):
            prompt = request_config['prompts'][i]
            negative_prompt = '' if 'negative_prompts' not in request_config else request_config['negative_prompts'][i]
            seed = int(request_config['seeds'][i])
            timestamp = int(request_config['timestamps'][i])

            if prev_content is None:
                prev_content = (prompt, seed, timestamp, negative_prompt)
                continue
            else:
                video_path = VIDEO_PIPELINE_DICT[request_object['model_id']].walk(
                    prompts=[prev_content[0], prompt],
                    negative_prompt=prev_content[3],
                    seeds=[prev_content[1], seed],
                    num_interpolation_steps=[(timestamp - prev_content[2]) * fps],
                    height=int(ASPECT_RATIOS_DICT[request_object['aspect_ratio']].split(':')[1]),
                    width=int(ASPECT_RATIOS_DICT[request_object['aspect_ratio']].split(':')[0]),
                    audio_filepath=audio_path,
                    audio_start_sec=prev_content[2],
                    fps=fps,
                    output_dir='dreams/{}'.format(request_id),
                    name=str(int(time.time() * 100)),
                )
                video_paths_list.append(video_path)
                prev_content = (prompt, seed, timestamp, negative_prompt)

        videos_list = []
        for file in video_paths_list:
            filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
            video = VideoFileClip(filePath)
            videos_list.append(video)

        concat_video_name = str(int(time.time() * 100)) + '.mp4'
        concat_video_path = 'dreams/{}/{}'.format(request_id, concat_video_name)
        concat_video = concatenate_videoclips(videos_list)
        concat_video.to_videofile(concat_video_path, fps=fps, remove_temp=False)

        video = VideoFileClip(concat_video_path)
        audio = AudioFileClip(audio_path)
        try:
            audio = audio.subclip(0, min(video.duration, audio.duration))
        except Exception as e:
            print('exception in clipping audio: ', e)
            pass
        video = video.set_audio(audio)
        video.write_videofile('dreams/{}/{}.mp4'.format(request_id, request_id))
        convert_mp4_to_mov('dreams/{}/{}.mp4'.format(request_id, request_id), 'dreams/{}/{}.mov'.format(request_id, request_id))
        meta = dropbox_upload_file(
            str(os.path.dirname(os.path.realpath(__file__))) + '/dreams/{}'.format(request_id),
            '{}.mov'.format(request_id),
            '/{}/{}'.format('Video', '{}.mov'.format(request_id))
        )
        link = dropbox_get_link('/{}/{}'.format('Video', '{}.mov'.format(request_id)))

        sg = SendGridAPIClient(config['sendgrid_api_key'])
        user = fetch_user_by_id(request_object['user_id'])
        internal_message = Mail(
            from_email='admin@nounsai.wtf',
            to_emails=[To('theheroshep@gmail.com'), To('eolszewski@gmail.com')],
            subject='Video #{} Has Processed!'.format(request_id),
            html_content='<p>Download here: {}</p>'.format(link)
        )
        external_message = Mail(
            from_email='admin@nounsai.wtf',
            to_emails=[To(user['email'])],
            subject='Your Video Has Processed!'.format(request_id),
            html_content='<p>Download here: {}</p>'.format(link)
        )
        response = sg.send(internal_message)
        response = sg.send(external_message)

        update_request_state('DONE', request_id)
        
        os.remove(audio_path)
        shutil.rmtree('dreams/{}'.format(request_id))
        return {'status':'success'}, 200
        
    except Exception as e:
        update_request_state('ERROR', request_id)
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500

#########################
######### LINKS #########
#########################

@app.route('/users/<user_id>/links', methods=['POST'])
def add_link_for_user(user_id):

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    # TODO: Make sure to related fields for img2img
    try:
        id = add_link(
            content['hash'], 
            user_id, 
            content['model_id'], 
            content['prompt'], 
            content['negative_prompt'], 
            content['seed'], 
            content['image_id'], 
            content['aspect_ratio'],
            content['inference_mode'],
            content['strength']
        )
        return {'link_id':id}, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500


@app.route('/links/<hash>', methods=['GET'])
def fetch_link_for_hash(hash):

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401
    
    try:
        link = fetch_link_by_hash(hash)
        return link, 200
    except Exception as e:
        print('Internal server error: {}'.format(str(e)))
        return 'Internal server error: {}'.format(str(e)), 500


#########################
######## HELPERS ########
#########################

@app.route('/extend_prompt', methods=['POST'])
def extend_prompt():

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401

    return text_pipeline(content['prompt'] + ',', num_return_sequences=1)[0]['generated_text']

@app.route('/interrogate', methods=['POST'])
def interrogate():

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401

    starter = content['base_64'].find(',')
    image_data = content['base_64'][starter+1:]
    image_data = bytes(image_data, encoding="ascii")
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
    return clip_interrogator.interrogate(image)

@app.route('/upscale', methods=['POST'])
def upscale():

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401

    starter = content['base_64'].find(',')
    image_data = content['base_64'][starter+1:]
    image_data = bytes(image_data, encoding="ascii")
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
    img = preprocess(image)
    img = preprocess(img)
    images = upscale_pipe(
        prompt='', 
        image=img
    ).images
    return serve_pil_image(images[0])

#######################################################
######################## MAIN #########################
#######################################################

if __name__ == '__main__':
    if config['environment'] == 'prod':
        app.run(host='0.0.0.0', port='5000', ssl_context='adhoc')
    else:
        app.run(host='0.0.0.0', debug=True, port='5000')

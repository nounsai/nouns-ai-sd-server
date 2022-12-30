import os
import sys
import json
import time
import torch
import shutil
import dropbox
import pathlib

from flask_cors import CORS
from flask import Flask, request
from transformers import pipeline
from sendgrid import SendGridAPIClient
from dropbox.exceptions import AuthError
from sendgrid.helpers.mail import Mail, To
from stable_diffusion_videos import StableDiffusionWalkPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from utils import ASPECT_RATIOS_DICT, MODELS_DICT, convert_mp4_to_mov, get_device, infer, serve_pil_image
from db import  fetch_users, fetch_user_by_email, fetch_user_by_id, add_user, \
                fetch_images, fetch_images_for_user, fetch_image_by_id, delete_image_by_id, fetch_image_by_hash, add_image, \
                fetch_requests, fetch_requests_for_user, fetch_request_by_id, delete_request_by_id, fetch_request_by_hash, add_request,\
                fetch_audio, fetch_audio_for_user, fetch_audio_by_id, delete_audio_by_id, add_audio, \
                fetch_code_by_hash, update_code_by_hash, add_code

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
        AUTH_TOKEN = config['huggingface_token']

#########################
######### FLASK #########
#########################

app = Flask(__name__)
CORS(app)

#########################
########## SD ###########
#########################

IMAGE_PIPELINE_DICT = {}
VIDEO_PIPELINE_DICT = {}
if get_device() == 'cuda':
    for model in MODELS_DICT.keys():
        IMAGE_PIPELINE_DICT[model] = DiffusionPipeline.from_pretrained(model, safety_checker=None, use_auth_token=AUTH_TOKEN, torch_dtype=torch.float16)
        IMAGE_PIPELINE_DICT[model].scheduler = DPMSolverMultistepScheduler.from_config(IMAGE_PIPELINE_DICT[model].scheduler.config)
        IMAGE_PIPELINE_DICT[model] = IMAGE_PIPELINE_DICT[model].to('cuda')
        VIDEO_PIPELINE_DICT[model] = StableDiffusionWalkPipeline.from_pretrained(model, feature_extractor=None, safety_checker=None, use_auth_token=AUTH_TOKEN, torch_dtype=torch.float16)
        VIDEO_PIPELINE_DICT[model].scheduler = DPMSolverMultistepScheduler.from_config(VIDEO_PIPELINE_DICT[model].scheduler.config)
        VIDEO_PIPELINE_DICT[model] = VIDEO_PIPELINE_DICT[model].to('cuda')
else:
    sys.exit('Need CUDA to run this server!')
    
text_pipeline = pipeline('text-generation', model='daspartho/prompt-extend', device=0)

#######################################################
####################### DROPBOX #######################
#######################################################

def dropbox_connect():

    try:
        dbx = dropbox.Dropbox(
            app_key = config['dropbox_api_key'],
            app_secret = config['dropbox_api_secret'],
            oauth2_refresh_token = config['dropbox_refresh_token']
        )
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx


def dropbox_upload_file(local_path, local_file, dropbox_file_path):

    try:
        dbx = dropbox_connect()
        local_file_path = pathlib.Path(local_path) / local_file
        with local_file_path.open('rb') as f:
            meta = dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode('overwrite'))
            return meta
    except Exception as e:
        print('Error uploading file to Dropbox: ' + str(e))


def dropbox_download_file(dropbox_file_path):

    try:
        dbx = dropbox_connect()
        meta, file = dbx.files_download(dropbox_file_path)
        with open(os.path.join(str(os.path.dirname(os.path.realpath(__file__))), '{}/{}'.format('audio', dropbox_file_path.split('/')[-1])), 'wb') as out:
            out.write(file.content)
            out.close()
    except Exception as e:
        print('Error downloading file from Dropbox: ' + str(e))


def dropbox_get_link(dropbox_file_path):

    try:
        dbx = dropbox_connect()
        shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_file_path)
        shared_link = shared_link_metadata.url
        return shared_link.replace('?dl=0', '?dl=1')
    except dropbox.exceptions.ApiError as exception:
        if exception.error.is_shared_link_already_exists():
            shared_link_metadata = dbx.sharing_get_shared_links(dropbox_file_path)
            shared_link = shared_link_metadata.links[0].url
            return shared_link.replace('?dl=0', '?dl=1')

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

#########################
######## IMAGES #########
#########################

@app.route('/get_image', methods=['POST'])
def get_image():

    content = json.loads(request.data)
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401

    images = infer(IMAGE_PIPELINE_DICT[content['model_id']], content['aspect_ratio'], prompt=content['prompt'], negative_prompt=content['negative_prompt'], samples=int(content['samples']), steps=int(content['steps']), seed=int(content['seed']))
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
            content['aspect_ratio']
        )
        return {'image_id':id}, 200
    except Exception as e:
        try:
            image = fetch_image_by_hash(content['image_hash'])
            return {'image_id':image['id']}, 200
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
            seed = int(request_config['seeds'][i])
            timestamp = int(request_config['timestamps'][i])

            if prev_content is None:
                prev_content = (prompt, seed, timestamp)
                continue
            else:
                video_path = VIDEO_PIPELINE_DICT[request_object['model_id']].walk(
                    prompts=[prev_content[0], prompt],
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
                prev_content = (prompt, seed, timestamp)

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
        
        os.remove(audio_path)
        shutil.rmtree('dreams/{}'.format(request_id))
        return {'status':'success'}, 200
        
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

#######################################################
######################## MAIN #########################
#######################################################

if __name__ == '__main__':
    if config['environment'] == 'prod':
        app.run(host='0.0.0.0', port='5000', ssl_context='adhoc')
    else:
        app.run(host='0.0.0.0', debug=True, port='5000')

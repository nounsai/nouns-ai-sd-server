import os 
import shutil
import sys
import json
from base64 import b64encode

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from stable_diffusion_videos import StableDiffusionWalkPipeline
from diffusers import DPMSolverMultistepScheduler

from db import (
    fetch_queued_video_projects, 
    update_video_project_state,
    fetch_images_for_ids,
    fetch_audio_for_user,
    fetch_user
)

from cdn import (
    download_audio_from_cdn_raw,
    upload_video_project_to_cdn
)

from utils import fetch_env_config, get_device

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To

config = fetch_env_config()
sg = SendGridAPIClient(config['sendgrid_api_key'])


pipe = StableDiffusionWalkPipeline.from_pretrained("alxdfy/noggles-v21-6400-best", safety_checker=None, feature_extractor=None, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
if get_device() == 'cuda':
    pipe = pipe.to("cuda")

FPS = 8
OUTPUT_DIR = os.path.join(PARENT_DIR, 'dreams')

def generate_videos():
    queued_projects = fetch_queued_video_projects()
    for project in queued_projects:
        # update state
        update_video_project_state(project['id'], 'PROCESSING')

        try:
            audio_offsets = [int(item) for item in project['metadata']['timestamps']]

            # Convert seconds to frames
            num_interpolation_steps = [(b-a) * FPS for a, b in zip(audio_offsets, audio_offsets[1:])]

            # reset dreams directory
            try:
                shutil.rmtree(OUTPUT_DIR)
                os.mkdir(OUTPUT_DIR)
            except Exception as e:
                print('Error creating directory: {}'.format(str(e)))

            # get images for project
            images = fetch_images_for_ids(project['metadata']['imageIds'])

            # get project audio
            audio = fetch_audio_for_user(project['user_id'], project['audio_id'])

            # download audio
            audio_bytes = download_audio_from_cdn_raw(project['user_id'], audio['cdn_id'])
            audio_path = os.path.join(OUTPUT_DIR, audio['name'])
            with open(audio_path, 'wb') as file:
                file.write(audio_bytes)

            # generate video
            video_path = pipe.walk(
                prompts=[image['metadata']['prompt'] for image in images],
                seeds=[int(image['metadata']['seed']) for image in images],
                num_interpolation_steps=num_interpolation_steps,
                height=int(images[0]['metadata']['aspect_ratio'].split(':')[1]),
                width=int(images[0]['metadata']['aspect_ratio'].split(':')[0]),
                audio_filepath=audio_path,
                audio_start_sec=audio_offsets[0],
                fps=FPS,
                batch_size=4,
                output_dir='./' + OUTPUT_DIR,
                name=None,
            )

            # upload video to cdn
            mp4 = open(video_path,'rb').read()
            upload_video_project_to_cdn(project['user_id'], project['cdn_id'], mp4)

            # mark project generation as complete
            update_video_project_state(project['id'], 'COMPLETED')

            # send success email
            user = fetch_user(project['user_id'])

            message = Mail(
                from_email='admin@nounsai.wtf',
                to_emails=[To(user['email'])],
                subject='NounsAI Video Generation Success',
                html_content=f'''
    <p>Hello! You are receiving this email because you generated a video using our video creation tool. Here is the link to download the result: <a href="https://nounsai-video.b-cdn.net/{project['user_id']}/{project['cdn_id']}-full.mp4">https://nounsai-video.b-cdn.net/{project['user_id']}/{project['cdn_id']}-full.mp4</a></p>
    '''
            )
            sg.send(message)

        except Exception as e:
            # reset state
            update_video_project_state(project['id'], 'QUEUED')
            print(f'Error generating video: {e}')

if __name__ == '__main__':
    generate_videos()
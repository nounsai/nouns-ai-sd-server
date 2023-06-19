import os 
import shutil
import sys
import math
from functools import reduce

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
from diffusers import DPMSolverMultistepScheduler

from db import (
    fetch_queued_video_projects, 
    update_video_project_state,
    fetch_video_project_for_id,
    fetch_images_for_ids,
    fetch_audio_for_user,
    fetch_user
)

from cdn import (
    download_audio_from_cdn_raw,
    upload_video_project_to_cdn,
    download_image_from_cdn
)

from video_generation import Image2ImageWalkPipeline

from utils import fetch_env_config, get_device

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To

from PIL import Image
from io import BytesIO

config = fetch_env_config()
sg = SendGridAPIClient(config['sendgrid_api_key'])


pipe = Image2ImageWalkPipeline.from_pretrained("alxdfy/noggles-v21-6400-best", safety_checker=None, feature_extractor=None, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
if get_device() == 'cuda':
    pipe = pipe.to("cuda")

FPS = config.get('video_generation_fps', 8)
OUTPUT_DIR = os.path.join(PARENT_DIR, 'dreams')
MAX_VIDEO_DURATION = config.get('video_generation_max_duration', 60)

def generate_videos():
    queued_projects = fetch_queued_video_projects()
    for project in queued_projects:
        # check that project isn't already being handled
        db_video_project = fetch_video_project_for_id(project['id'])
        if db_video_project is None or db_video_project['state'] == 'PROCESSING':
            continue
        # update state
        update_video_project_state(project['id'], 'PROCESSING')

        try:
            # get audio offsets
            start_offset = None
            audio_offsets = []
            for timestring in project['metadata']['timestamps']:
                timestamp = float(timestring)

                if start_offset is None:
                    # first timestamp
                    start_offset = timestamp
                    audio_offsets.append(timestamp)
                elif timestamp - start_offset > MAX_VIDEO_DURATION:
                    # over the allowed video duration
                    break 
                else:
                    audio_offsets.append(timestamp)

            # skip if there aren't enough frames to interpolate
            if len(audio_offsets) < 2:
                # update state
                update_video_project_state(project['id'], 'ERROR')
                continue

            # Convert seconds to frames
            num_interpolation_steps = [math.ceil(b-a) * FPS for a, b in zip(audio_offsets, audio_offsets[1:])]

            # reset dreams directory
            try:
                shutil.rmtree(OUTPUT_DIR)
                os.mkdir(OUTPUT_DIR)
            except Exception as e:
                print('Error creating directory: {}'.format(str(e)))

            # get images for project
            image_records = fetch_images_for_ids(project['metadata']['imageIds'])
            # restrict to images for allowed video duration
            image_records = image_records[:len(audio_offsets)]

            # get image contents
            images = []
            for record in image_records:
                image_contents = download_image_from_cdn(record['user_id'], record['cdn_id'])
                images.append(Image.open(BytesIO(image_contents)).convert('RGB'))

            # get project audio
            audio = fetch_audio_for_user(project['user_id'], project['audio_id'])

            # download audio
            audio_bytes = download_audio_from_cdn_raw(project['user_id'], audio['cdn_id'])
            audio_path = os.path.join(OUTPUT_DIR, audio['name'])
            with open(audio_path, 'wb') as file:
                file.write(audio_bytes)

            # get batch size based on interpolation steps
            batch_size = reduce(math.gcd, num_interpolation_steps)

            # generate video
            video_path = pipe.walk(
                images=images,
                num_interpolation_steps=num_interpolation_steps,
                audio_filepath=audio_path,
                audio_start_sec=audio_offsets[0],
                fps=FPS,
                batch_size=batch_size,
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
            print(f"generated video for project: {project['id']}")

        except Exception as e:
            # reset state
            update_video_project_state(project['id'], 'QUEUED')
            print(f"Error generating video for project {project['id']}: {e}")

if __name__ == '__main__':
    generate_videos()
import os 
import shutil
import sys
import math
from functools import reduce
import traceback
from datetime import datetime, timedelta

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

from utils import fetch_env_config, get_device, send_discord_webhook

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
MAX_BATCH_SIZE = config.get('video_generation_max_batch_size', 5)
WEBHOOK_URL = config.get('video_generation_discord_webhook', None)

def preprocess_steps(steps):
    if steps == 1:
        return 2
    else:
        return steps - steps % 2

def generate_videos():
    queued_projects = fetch_queued_video_projects()
    for project in queued_projects:
        # check that project isn't already being handled
        db_video_project = fetch_video_project_for_id(project['id'])
        if db_video_project is None or db_video_project['state'] == 'PROCESSING':
            continue
        # update state
        update_video_project_state(project['id'], 'PROCESSING')
        start_time = datetime.now()

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
            num_interpolation_steps = [
                preprocess_steps(math.ceil((b-a) * FPS)) for a, b in zip(audio_offsets, audio_offsets[1:])
            ]

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
            video_urls = []
            for record in image_records:
                image_contents = download_image_from_cdn(record['user_id'], record['cdn_id'])
                images.append(Image.open(BytesIO(image_contents)).convert('RGB'))
                if record['metadata'].get('video_cdn_id', None) is not None:
                    video_urls.append(record['meta']['video_cdn_id'])
                else:
                    video_urls.append(None)

            # get project audio
            audio = fetch_audio_for_user(project['user_id'], project['audio_id'])

            # download audio
            audio_bytes = download_audio_from_cdn_raw(project['user_id'], audio['cdn_id'])
            audio_path = os.path.join(OUTPUT_DIR, audio['name'])
            with open(audio_path, 'wb') as file:
                file.write(audio_bytes)

            # get batch size based on interpolation steps
            # batch_size = reduce(math.gcd, num_interpolation_steps)

            # constrain batch size to <= 10
            # if batch_size > MAX_BATCH_SIZE:
            #     batch_size = get_small_divisor(batch_size)
            # print('using batch-size:', batch_size)
            print('using fps:', FPS)
            # get any custom prompts
            prompts = project['metadata'].get('prompts', None)

            # generate video
            video_path = pipe.walk(
                images=images,
                prompts=prompts,
                video_urls=video_urls,
                num_interpolation_steps=num_interpolation_steps,
                audio_filepath=audio_path,
                audio_start_sec=audio_offsets[0],
                fps=FPS,
                batch_size=MAX_BATCH_SIZE,
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

            processing_time = str(timedelta(seconds=round((datetime.now() - start_time).total_seconds())))
            video_length = str(timedelta(seconds=round(audio_offsets[-1] - audio_offsets[0])))

            send_discord_webhook(
                url=WEBHOOK_URL,
                embeds=[
                    {
                        'title': 'Video COMPLETED',
                        'description': f"https://nounsai-video.b-cdn.net/{project['user_id']}/{project['cdn_id']}-full.mp4",
                        'fields': [
                            {
                                'name': 'Time to process',
                                'value': processing_time
                            },
                            {
                                'name': 'Video Length',
                                'value': video_length
                            },
                            {
                                'name': 'User id',
                                'value': str(project['user_id'])
                            },
                            {
                                'name': 'Project id',
                                'value': str(project['id'])
                            }
                        ]
                    }
                ]
            )

        except Exception as e:
            print(traceback.format_exc())
            processing_time = str(timedelta(seconds=round((datetime.now() - start_time).total_seconds())))
            update_video_project_state(project['id'], 'ERROR')
            print(f"Error generating video for project {project['id']}: {e}")

            send_discord_webhook(
                url=WEBHOOK_URL,
                embeds=[
                    {
                        'title': 'Video ERROR',
                        'description': f"```{traceback.format_exc()}```",
                        'fields': [
                            {
                                'name': 'Time to process',
                                'value': processing_time
                            },
                            {
                                'name': 'User id',
                                'value': str(project['user_id'])
                            },
                            {
                                'name': 'Project id',
                                'value': str(project['id'])
                            }
                        ]
                    }
                ]
            )

            # send failure email
            user = fetch_user(project['user_id'])

            message = Mail(
                from_email='admin@nounsai.wtf',
                to_emails=[To(user['email'])],
                subject='NounsAI Video Generation Failure!',
                html_content=f'''
                <p>Hello! You are receiving this email because a video you made using our video creation tool unfortunately failed to be generated. This could be due to many factors, but try using smaller images (e.g. 512x512) or double check that the audio file is fine. You can return to <a href="https://nounsai.wtf">nounsai.wtf</a> to make any changes and re-generate the video.</p>
                '''
            )
            sg.send(message)

if __name__ == '__main__':
    generate_videos()
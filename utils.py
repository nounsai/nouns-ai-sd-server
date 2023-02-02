#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import os
import sys
import json
import time
import torch
import dropbox
import pathlib
import subprocess
import numpy as np

from PIL import Image
from io import BytesIO 
from flask import send_file
from googletrans import Translator
from dropbox.exceptions import AuthError

#########################
########## ENV ##########
#########################

currentdir = os.path.dirname(os.path.realpath(__file__))
if not os.path.isfile('config.json'):
    sys.exit('\'config.json\' not found! Please add it and try again.')
else:
    with open('config.json') as file:
        config = json.load(file)

#######################################################
###################### CONSTANTS ######################
#######################################################

BASE_MODELS = [
    'alxdfy/noggles-v21-6400-best'
]
INSTRUCTABLE_MODELS = [
    'timbrooks/instruct-pix2pix'
]
ASPECT_RATIOS_DICT = {
    '1:1': '768:768', 
    '8:11': '512:704',
    '8:12': '512:768',
    '9:16': '576:1024', 
    '16:9': '1024:576'
}
INFERENCE_MODES = {
    'txt2img': 'Text to Image',
    'img2img': 'Image to Image',
    'pix2pix': 'Pix to Pix'
}

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
####################### HELPERS #######################
#######################################################

def get_device():

    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def convert_mp4_to_mov(input_file, output_file):

    command = ['ffmpeg', '-i', input_file, output_file]
    subprocess.run(command)


def preprocess(image):

    now = int(time.time())
    image.save('{}_tmp.jpg'.format(str(now)), optimize=True, quality=100)
    image = Image.open('{}_tmp.jpg'.format(str(now)))
    os.remove('{}_tmp.jpg'.format(str(now)))
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def serve_pil_image(pil_img):

    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

#######################################################
##################### PIPELINING ######################
#######################################################

def txt_to_img(img_pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio):

    translator = Translator()

    images = img_pipeline(
        translator.translate(prompt).text,
        generator=generator,
        num_images_per_prompt=n_images,
        negative_prompt=translator.translate(negative_prompt).text,
        num_inference_steps=steps,
        guidance_scale=scale,
        height=int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[1]),
        width=int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[0])
    ).images
    return images


def img_to_img(i2i_pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio, img, strength):

    img = preprocess(img)
    images = i2i_pipeline(
        prompt,
        generator=generator,
        num_images_per_prompt = n_images,
        negative_prompt = negative_prompt,
        num_inference_steps = int(steps),
        guidance_scale = scale,
        image = img,
        strength = strength
    ).images
    return images


def pix_to_pix(p2p_pipeline, prompt, generator, n_images, steps, scale, img):

    img = preprocess(img)
    images = p2p_pipeline(
        prompt,
        generator=generator,
        num_images_per_prompt = n_images,
        num_inference_steps = int(steps),
        guidance_scale = scale,
        image = img
    ).images
    return images


def inference(pipeline, inf_mode, prompt, n_images=4, negative_prompt="", steps=25, scale=7.5, seed=1437181781, aspect_ratio='768:768', img=None, strength=0.5):

    generator = torch.Generator('cuda').manual_seed(seed)

    try:
        
        if inf_mode == INFERENCE_MODES['txt2img']:
            return txt_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio)
        
        elif inf_mode == INFERENCE_MODES['img2img']:
            if img is None:
                return None

            return img_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio, img, strength)
        
        elif inf_mode == INFERENCE_MODES['pix2pix']:
            if img is None:
                return None
                
            return pix_to_pix(pipeline, prompt, generator, n_images, steps, scale, img)
        
    except Exception as e:
        print('Internal server error with inferencing {}: {}'.format(inf_mode, str(e)))
        return None

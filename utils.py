#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import io
import os
import cv2
import sys
import json
import time
import torch
import base64
import shutil
import dropbox
import pathlib
import subprocess
import numpy as np

from PIL import Image
from io import BytesIO 
from flask import send_file
from dropbox.exceptions import AuthError

#########################
########## ENV ##########
#########################

def fetch_env_config():
    currentdir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isfile('config.json'):
        sys.exit('\'config.json\' not found! Please add it and try again.')
    else:
        with open('config.json') as file:
            config = json.load(file)
    return config

config = fetch_env_config()

#######################################################
###################### CONSTANTS ######################
#######################################################

BASE_MODELS = [
    'alxdfy/noggles-v21-6400-best'
]
INSTRUCTABLE_MODELS = [
    'timbrooks/instruct-pix2pix'
]
TEXT_MODELS = [
    'daspartho/prompt-extend'
]
INTERROGATOR_MODELS = [
    'ViT-H-14/laion2b_s32b_b79k'
]
UPSCALE_MODELS = [
    'stabilityai/stable-diffusion-x4-upscaler'
]
INFERENCE_MODES = [
    'Text to Image',
    'Image to Image',
    'Pix to Pix'
]

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

def refresh_dir(dir):
    
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


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


def adjust_thickness(image, thickness):

    image = np.array(image)
    if not (-5 <= thickness <= 5):
        raise ValueError("Thickness value should be between -5 and 5")

    low_threshold = 100
    high_threshold = 200

    if thickness < 0:
        ksize = 2 * abs(thickness) + 1 # e.g., 3 for thickness=-1, 5 for thickness=-2, etc.
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    image = cv2.Canny(image, low_threshold, high_threshold)

    if thickness > 0:
        kernel_size = 1 + 2 * thickness  # e.g., 3 for thickness=1, 5 for thickness=2, etc.
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)

    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def serve_pil_image(pil_image):

    image_io = BytesIO()
    pil_image.save(image_io, 'JPEG', quality=100)
    image_io.seek(0)
    return send_file(image_io, mimetype='image/jpeg')


def image_from_base_64(base64_string):

    starter = base64_string.find(',')
    image_data = base64_string[starter+1:]
    image_data = bytes(image_data, encoding="ascii")
    return Image.open(BytesIO(base64.b64decode(image_data)))


def base_64_thumbnail_for_base_64_image(base64_string):
    
    buffer = io.BytesIO()
    image = image_from_base_64(base64_string)
    [h,w,c] = np.shape(image)
    w, h = map(lambda x: int(float(x / max(w, h)) * 100), (w, h))
    tmp_image = image.resize((w, h), resample=Image.LANCZOS)
    tmp_image.save(buffer, format="JPEG")
    return 'data:image/jpeg;base64,' + str(base64.b64encode(buffer.getvalue()))[2:-1]

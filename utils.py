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
PALETTE = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])

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


def thumbnail_bytes_for_image(image):
    
    buffer = io.BytesIO()
    [h,w,c] = np.shape(image)
    w, h = map(lambda x: int(float(x / max(w, h)) * 100), (w, h))
    tmp_image = image.resize((w, h), resample=Image.LANCZOS)
    tmp_image.save(buffer, format="JPEG")
    return buffer.getvalue()


def bytes_from_image(image):
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

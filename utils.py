#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import os
import torch
import subprocess

from io import BytesIO 
from flask import send_file

#######################################################
###################### CONSTANTS ######################
#######################################################

MODELS_DICT = {
    'alxdfy/noggles-v21-6400-best': '768:768', 
    'nitrosocke/Ghibli-Diffusion': '512:704',
    'nitrosocke/Nitro-Diffusion': '512:768'
}
ASPECT_RATIOS_DICT = {
    '1:1': '768:768', 
    '8:11': '512:704',
    '8:12': '512:768',
    '9:16': '576:1024', 
    '16:9': '1024:576'
}

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


def infer(image_pipeline, aspect_ratio, prompt="", negative_prompt="", samples=4, steps=25, scale=7, seed=1437181781):

    generator = torch.Generator(device=get_device()).manual_seed(seed)
    images = image_pipeline(
        prompt,
        num_images_per_prompt=samples,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
        height=int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[1]),
        width=int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[0])
    ).images
    return images


def serve_pil_image(pil_img):

    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

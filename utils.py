#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import os
import torch
import subprocess

from PIL import Image
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
INFERENCE_MODES = {
    'txt2img': 'Text to Image',
    'img2img': 'Image to Image'
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


def txt_to_img(img_pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio):

    images = img_pipeline(
        prompt,
        generator=generator,
        num_images_per_prompt=n_images,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        height=int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[1]),
        width=int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[0])
    ).images
    return images


def img_to_img(i2i_pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio, img, strength):

    img = img['image']
    ratio = min(int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[1]) / img.height, int(ASPECT_RATIOS_DICT[aspect_ratio].split(':')[0]) / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
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


def inference(pipeline, inf_mode, prompt, n_images=4, negative_prompt="", steps=25, scale=7, seed=1437181781, aspect_ratio='768:768', img=None, strength=0.5):

    generator = torch.Generator('cuda').manual_seed(seed)

    try:
        
        if inf_mode == INFERENCE_MODES['txt2img']:
            return txt_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio)
        
        elif inf_mode == INFERENCE_MODES['img2img']:
            if img is None:
                return None

            return img_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio, img, strength)
        
    except Exception as e:
        print('Internal server error with inferencing {}: {}'.format(inf_mode, str(e)))
        return None


def serve_pil_image(pil_img):

    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')
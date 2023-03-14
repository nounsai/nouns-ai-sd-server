#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import sys
import cv2
import time
import torch
from googletrans import Translator

from transformers import pipeline
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To
from clip_interrogator import Config, Interrogator
from transformers.generation_utils import GenerationMixin
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionUpscalePipeline

from db import fetch_image
from utils import convert_mp4_to_mov, fetch_env_config, get_device, image_from_base_64, preprocess, refresh_dir, \
                 BASE_MODELS, INSTRUCTABLE_MODELS, INTERROGATOR_MODELS, TEXT_MODELS, UNCLIP_MODELS, UPSCALE_MODELS

config = fetch_env_config()

#######################################################
######################## SETUP ########################
#######################################################

PIPELINE_DICT = {
    'Text to Image': {},
    'Image to Image': {},
    'Pix to Pix': {},
    'Text': {},
    'Interrogator': {},
    'Unclip': {},
    'Upscale': {},
}

def _no_validate_model_kwargs(self, model_kwargs):
    pass

def setup_pipelines():
    GenerationMixin._validate_model_kwargs = _no_validate_model_kwargs

    if get_device() == 'cuda':
        for base_model in BASE_MODELS:
            PIPELINE_DICT['Text to Image'][base_model] = DiffusionPipeline.from_pretrained(base_model, safety_checker=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['Text to Image'][base_model].scheduler = DPMSolverMultistepScheduler.from_config(PIPELINE_DICT['Text to Image'][base_model].scheduler.config)
            PIPELINE_DICT['Text to Image'][base_model] = PIPELINE_DICT['Text to Image'][base_model].to('cuda')
            PIPELINE_DICT['Image to Image'][base_model] = StableDiffusionImg2ImgPipeline.from_pretrained(base_model, safety_checker=None, feature_extractor=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['Image to Image'][base_model].scheduler = DPMSolverMultistepScheduler.from_config(PIPELINE_DICT['Image to Image'][base_model].scheduler.config)
            PIPELINE_DICT['Image to Image'][base_model] = PIPELINE_DICT['Image to Image'][base_model].to('cuda')
        for instructable_model in INSTRUCTABLE_MODELS:
            PIPELINE_DICT['Pix to Pix'][instructable_model] = StableDiffusionInstructPix2PixPipeline.from_pretrained(instructable_model, safety_checker=None, feature_extractor=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['Pix to Pix'][instructable_model] = PIPELINE_DICT['Pix to Pix'][instructable_model].to('cuda')
            PIPELINE_DICT['Pix to Pix'][instructable_model].scheduler = EulerAncestralDiscreteScheduler.from_config(PIPELINE_DICT['Pix to Pix'][instructable_model].scheduler.config)
        for unclip_model in UNCLIP_MODELS:
            PIPELINE_DICT['Unclip'][unclip_model] = DiffusionPipeline.from_pretrained(unclip_model, safety_checker=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16, custom_pipeline="./research/unclip_image_interpolation.py")
            PIPELINE_DICT['Unclip'][unclip_model] = PIPELINE_DICT['Unclip'][unclip_model].to('cuda')
            
    else:
        sys.exit('Need CUDA to run this server!')
    
    for text_model in TEXT_MODELS:
        PIPELINE_DICT['Text'][text_model] = pipeline('text-generation', model=text_model, device=0)

    for interrogator_model in INTERROGATOR_MODELS:
        ci_config = Config()
        ci_config.blip_num_beams = 64
        ci_config.blip_offload = False
        ci_config.clip_model_name = interrogator_model
        PIPELINE_DICT['Interrogator'][interrogator_model] = Interrogator(ci_config)

    for upscale_model in UPSCALE_MODELS:
        PIPELINE_DICT['Upscale'][upscale_model] = StableDiffusionUpscalePipeline.from_pretrained(upscale_model, safety_checker=None, feature_extractor=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
        PIPELINE_DICT['Upscale'][upscale_model] = PIPELINE_DICT['Upscale'][upscale_model].to('cuda')
        PIPELINE_DICT['Upscale'][upscale_model].enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        PIPELINE_DICT['Upscale'][upscale_model].vae.enable_xformers_memory_efficient_attention(attention_op=None)

    return PIPELINE_DICT

#######################################################
##################### PIPELINING ######################
#######################################################

def txt_to_img(img_pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio):

    translator = Translator()

    images = img_pipeline(
        "" if len(prompt) == 0 else translator.translate(prompt).text,
        generator=generator,
        num_images_per_prompt=n_images,
        negative_prompt= "" if len(negative_prompt) == 0 else translator.translate(negative_prompt).text,
        num_inference_steps=steps,
        guidance_scale=scale,
        height=int(aspect_ratio.split(':')[1]),
        width=int(aspect_ratio.split(':')[0])
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


def unclip_images(unclip_pipeline, image_ids, timestamps, seed, audio_id):

    try:
        FPS = 10
        generator = torch.Generator('cuda').manual_seed(seed)
        refresh_dir('dreams')
        videos_list = []

        prev_image = image_from_base_64(fetch_image(image_ids[0])['base_64'])
        video_width, video_height = prev_image.size
        
        for frame in range(len(image_ids) - 1):
            image_list = [prev_image]
            curr_image = image_from_base_64(fetch_image(image_ids[frame+1])['base_64'])
            images = unclip_pipeline(
                image = [prev_image, curr_image],
                steps = (timestamps[frame+1] - timestamps[frame]) * (FPS - 1), # 10 fps needed for .1s granularity, first image is already prepended
                generator = generator,
                height = video_height,
                width = video_width
            ).images
            image_list = image_list + images

            if frame == len(image_ids) - 2:
                image_list.pop()
                image_list.append(curr_image)

            out = cv2.VideoWriter('dreams/video_%s.mp4' % frame, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (video_width, video_height))
            videos_list.append('dreams/video_%s.mp4' % frame)
            for i in range(len(image_list)):
                out.write(image_list[i])
            out.release()
            
            prev_image = curr_image

        concat_video = concatenate_videoclips([VideoFileClip(video) for video in videos_list])
        concat_video.to_videofile('dreams/video.mp4', fps=FPS, remove_temp=False)

        video = VideoFileClip('dreams/video.mp4')

        if audio_id != -1:
            audio_path = 'dreams/audio_{}.mp3'.format(audio_id)
            audio = AudioFileClip(audio_path)
            try:
                audio = audio.subclip(0, min(video.duration, audio.duration))
            except Exception as e:
                print('exception in clipping audio: ', e)
                pass
            video = video.set_audio(audio)
            video.write_videofile('dreams/video.mp4')
            convert_mp4_to_mov('dreams/video.mp4', 'dreams/video.mov')
        
        return True

    except Exception as e:
        print('Internal server error with unclip_images: {}'.format(str(e)))
        return False


def inference(pipeline, inf_mode, prompt, n_images=4, negative_prompt="", steps=25, scale=7.5, seed=1437181781, aspect_ratio='768:768', img=None, strength=0.5):

    generator = torch.Generator('cuda').manual_seed(seed)

    try:
        
        if inf_mode == 'Text to Image':
            return txt_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio)
        
        else:
            if img is None:
                return None
            
            if inf_mode == 'Image to Image':
                return img_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio, img, strength)
        
            elif inf_mode == 'Pix to Pix':
                return pix_to_pix(pipeline, prompt, generator, n_images, steps, scale, img)
        
    except Exception as e:
        print('Internal server error with inferencing {}: {}'.format(inf_mode, str(e)))
        return None

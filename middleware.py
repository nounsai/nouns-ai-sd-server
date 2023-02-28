#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import sys
import torch
from googletrans import Translator

from transformers import pipeline
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To
from clip_interrogator import Config, Interrogator
from transformers.generation_utils import GenerationMixin
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionUpscalePipeline

from utils import fetch_env_config, get_device, preprocess, BASE_MODELS, INFERENCE_MODES, INSTRUCTABLE_MODELS, INTERROGATOR_MODELS, TEXT_MODELS, UPSCALE_MODELS

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


def inference(pipeline, inf_mode, prompt, n_images=4, negative_prompt="", steps=25, scale=7.5, seed=1437181781, aspect_ratio='768:768', img=None, strength=0.5):

    generator = torch.Generator('cuda').manual_seed(seed)

    try:
        
        if inf_mode == 'Image to Image':
            return txt_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio)
        
        else:
            if img is None:
                return None
            
            if inf_mode == INFERENCE_MODES['img2img']:
                return img_to_img(pipeline, prompt, generator, n_images, negative_prompt, steps, scale, aspect_ratio, img, strength)
        
            elif inf_mode == 'Pix to Pix':
                return pix_to_pix(pipeline, prompt, generator, n_images, steps, scale, img)
        
    except Exception as e:
        print('Internal server error with inferencing {}: {}'.format(inf_mode, str(e)))
        return None

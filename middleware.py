#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
import cv2
import time
import numpy
import torch
from PIL import Image
from googletrans import Translator
import io

from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To
from clip_interrogator import Config, Interrogator
from transformers.generation_utils import GenerationMixin
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler, StableDiffusionUpscalePipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from audio_generation import CustomMusicGen, tensor_to_audio_bytes, Demucs

from db import fetch_image, fetch_user, update_video_for_user
from utils import convert_mp4_to_mov, dropbox_get_link, dropbox_upload_file, fetch_env_config, get_device, image_from_base_64, preprocess, adjust_thickness, refresh_dir, \
                 BASE_MODELS, INSTRUCTABLE_MODELS, INTERROGATOR_MODELS, TEXT_MODELS, UPSCALE_MODELS, PALETTE

config = fetch_env_config()
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

#######################################################
######################## SETUP ########################
#######################################################

PIPELINE_DICT = {
    'Text to Image': {},
    'Image to Image': {},
    'Text to Audio': {},
    'Pix to Pix': {},
    'Text': {},
    'Interrogator': {},
    'Upscale': {},
    'ControlNet': {
        'Outlines': {},
        'Segmentation': {},
        'Depth': {}
    },
    'Image Processor': {},
    'Image Segmentor': {},
    'Depth Estimator': {}
}

IMAGE_PROCESSOR = None
IMAGE_SEGMENTOR= None
DEPTH_ESTIMATOR = None

def _no_validate_model_kwargs(self, model_kwargs):
    pass

def setup_pipelines():
    GenerationMixin._validate_model_kwargs = _no_validate_model_kwargs
    
    control_net_canny = ControlNetModel.from_pretrained("thibaud/controlnet-sd21-canny-diffusers", torch_dtype=torch.float16)
    PIPELINE_DICT['Image Processor'] = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
    PIPELINE_DICT['Image Segmentor'] = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
    control_net_seg = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16)
    PIPELINE_DICT['Depth Estimator'] = pipeline('depth-estimation')
    control_net_depth = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
    CONTROL_NET_BASE_MODEL = "runwayml/stable-diffusion-v1-5"

    if get_device() == 'cuda':
        for base_model in BASE_MODELS:
            PIPELINE_DICT['Text to Image'][base_model] = DiffusionPipeline.from_pretrained(base_model, safety_checker=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['Text to Image'][base_model].scheduler = DPMSolverMultistepScheduler.from_config(PIPELINE_DICT['Text to Image'][base_model].scheduler.config)
            PIPELINE_DICT['Text to Image'][base_model] = PIPELINE_DICT['Text to Image'][base_model].to('cuda')
            PIPELINE_DICT['Image to Image'][base_model] = StableDiffusionImg2ImgPipeline.from_pretrained(base_model, safety_checker=None, feature_extractor=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['Image to Image'][base_model].scheduler = DPMSolverMultistepScheduler.from_config(PIPELINE_DICT['Image to Image'][base_model].scheduler.config)
            PIPELINE_DICT['Image to Image'][base_model] = PIPELINE_DICT['Image to Image'][base_model].to('cuda')
            PIPELINE_DICT['ControlNet']['Outlines'][base_model] = StableDiffusionControlNetPipeline.from_pretrained(base_model, controlnet=control_net_canny, safety_checker=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['ControlNet']['Outlines'][base_model].scheduler = UniPCMultistepScheduler.from_config(PIPELINE_DICT['ControlNet']['Outlines'][base_model].scheduler.config)
            PIPELINE_DICT['ControlNet']['Outlines'][base_model].enable_model_cpu_offload()
            PIPELINE_DICT['ControlNet']['Segmentation'][CONTROL_NET_BASE_MODEL] = StableDiffusionControlNetPipeline.from_pretrained(CONTROL_NET_BASE_MODEL, controlnet=control_net_seg, safety_checker=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['ControlNet']['Segmentation'][CONTROL_NET_BASE_MODEL].scheduler = UniPCMultistepScheduler.from_config(PIPELINE_DICT['ControlNet']['Segmentation'][CONTROL_NET_BASE_MODEL].scheduler.config)
            PIPELINE_DICT['ControlNet']['Segmentation'][CONTROL_NET_BASE_MODEL].enable_model_cpu_offload()
            PIPELINE_DICT['ControlNet']['Depth'][CONTROL_NET_BASE_MODEL] = StableDiffusionControlNetPipeline.from_pretrained(CONTROL_NET_BASE_MODEL, controlnet=control_net_depth, safety_checker=None, use_auth_token=config['huggingface_token'], torch_dtype=torch.float16)
            PIPELINE_DICT['ControlNet']['Depth'][CONTROL_NET_BASE_MODEL].scheduler = UniPCMultistepScheduler.from_config(PIPELINE_DICT['ControlNet']['Depth'][CONTROL_NET_BASE_MODEL].scheduler.config)
            PIPELINE_DICT['ControlNet']['Depth'][CONTROL_NET_BASE_MODEL].enable_model_cpu_offload()
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

    return PIPELINE_DICT

AUDIO_DICT = {
    'Text to Audio': {},
    'Audio to Audio': {}
}

def setup_audio():
    if torch.cuda.is_available():
        model = CustomMusicGen.get_pretrained('melody', device='cuda')
        
        model.set_generation_params(duration=5)
        AUDIO_DICT['Text to Audio']['musicgen'] = model
        AUDIO_DICT['Audio to Audio']['demucs'] = Demucs()

    return AUDIO_DICT

#######################################################
##################### PIPELINING ######################
#######################################################

def txt_to_audio(audio_pipeline, text):
    buffer = io.BytesIO()
    model = audio_pipeline['Text to Audio']['musicgen']
    res = model.generate([text], progress=True)
    tensor_to_audio_bytes(buffer, res[0].cpu(), model.sample_rate, format='mp3')
    buffer.seek(0)
    return buffer.read()

def txt_and_audio_to_audio(audio_pipeline, text, wav, sr):
    buffer = io.BytesIO()
    model = audio_pipeline['Text to Audio']['musicgen']
    res = model.generate_with_chroma([text], wav[None].expand(1, -1, -1), sr)
    tensor_to_audio_bytes(buffer, res[0].cpu(), model.sample_rate, format='mp3')
    buffer.seek(0)
    return buffer.read()

def separate_audio_tracks(audio_pipline, wav):
    model = audio_pipline['Audio to Audio']['demucs']
    sources = model.separate(wav)
    for source, name in zip(sources, model.sources):
        buffer = io.BytesIO()
        tensor_to_audio_bytes(buffer, source.cpu(), model.samplerate, format='mp3')
        buffer.seek(0)
        yield buffer.read(), name

    

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


def control_net_outlines(control_net_pipeline, prompt, generator, negative_prompt, steps, thickness, img):
    
    canny_img = adjust_thickness(img, thickness)
    images = control_net_pipeline(
        prompt,
        canny_img,
        negative_prompt= "" if len(negative_prompt) == 0 else negative_prompt,
        generator=generator,
        num_inference_steps=steps,
    ).images
    return images


def control_net_depth(control_net_pipeline, prompt, generator, negative_prompt, steps, img):
    
    image = PIPELINE_DICT['Depth Estimator'](img)['depth']
    image = numpy.array(image)
    image = image[:, :, None]
    image = numpy.concatenate([image, image, image], axis=2)
    depth_img = Image.fromarray(image)

    images = control_net_pipeline(
        prompt,
        depth_img,
        negative_prompt= "" if len(negative_prompt) == 0 else negative_prompt,
        generator=generator,
        num_inference_steps=steps,
    ).images
    return images


def control_net_segmentation(control_net_pipeline, prompt, generator, negative_prompt, steps, img):
    
    pixel_values = PIPELINE_DICT['Image Processor'](img, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = PIPELINE_DICT['Image Segmentor'](pixel_values)

    seg = PIPELINE_DICT['Image Processor'].post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
    color_seg = numpy.zeros((seg.shape[0], seg.shape[1], 3), dtype=numpy.uint8)
    for label, color in enumerate(PALETTE):
        color_seg[seg == label, :] = color
    color_seg = color_seg.astype(numpy.uint8)
    segmented_img = Image.fromarray(color_seg)

    images = control_net_pipeline(
        prompt,
        segmented_img,
        negative_prompt= "" if len(negative_prompt) == 0 else negative_prompt,
        generator=generator,
        num_inference_steps=steps,
    ).images
    return images



# def unclip_images(video_id, user_id, unclip_pipeline, metadata):

#     image_ids = metadata['image_ids']
#     timestamps = metadata['timestamps']
#     seed = metadata['seed']
#     audio_id = metadata['audio_id']

#     metadata['state'] = 'PROCESSING'
#     update_video_for_user(video_id, user_id, metadata)

#     try:
#         FPS = 10
#         generator = torch.Generator('cuda').manual_seed(seed)
#         refresh_dir('dreams')
#         videos_list = []

#         prev_image = image_from_base_64(fetch_image(image_ids[0])['base_64'])
#         prev_image = prev_image.resize((256, 256), resample=Image.LANCZOS)
#         video_width, video_height = prev_image.size

#         for frame in range(len(image_ids) - 1):
#             image_list = [prev_image]
#             curr_image = image_from_base_64(fetch_image(image_ids[frame+1])['base_64'])
#             curr_image = curr_image.resize((256, 256), resample=Image.LANCZOS)
#             steps = int((timestamps[frame+1] - timestamps[frame]) * FPS) - 1 # 10 fps needed for .1s granularity, first image is already prepended

#             images = unclip_pipeline(
#                 image = [prev_image, curr_image],
#                 steps = steps,
#                 generator = generator
#                 # decoder_latents = torch.randn(steps, 3, int(video_height / 4), int(video_width / 4)),
#                 # super_res_latents = torch.randn(steps, 3, int(video_height), int(video_width))
#             ).images
#             image_list = image_list + images

#             if frame == len(image_ids) - 2:
#                 image_list.pop()
#                 image_list.append(curr_image)

#             out = cv2.VideoWriter('dreams/video_%s.mp4' % frame, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (video_width, video_height))
#             videos_list.append('dreams/video_%s.mp4' % frame)
#             for i in range(len(image_list)):
#                 out.write(numpy.asarray(image_list[i]))
#             out.release()
            
#             prev_image = curr_image

#         concat_video = concatenate_videoclips([VideoFileClip(video) for video in videos_list])
#         concat_video.to_videofile('dreams/video.mp4', fps=FPS, remove_temp=False)

#         video = VideoFileClip('dreams/video.mp4')

#         if audio_id != -1:
#             audio_path = 'dreams/audio_{}.mp3'.format(audio_id)
#             audio = AudioFileClip(audio_path)
#             try:
#                 audio = audio.subclip(0, min(video.duration, audio.duration))
#             except Exception as e:
#                 print('exception in clipping audio: ', e)
#                 pass
#             video = video.set_audio(audio)
#             video.write_videofile('dreams/video.mp4')
        
#         convert_mp4_to_mov('dreams/video.mp4', 'dreams/video.mov')
#         meta = dropbox_upload_file(
#             'dreams',
#             'video.mov',
#             '/Video/{}.mov'.format(video_id)
#         )
#         link = dropbox_get_link('/Video/{}.mov'.format(video_id))

#         sg = SendGridAPIClient(config['sendgrid_api_key'])
#         user = fetch_user(user_id)
#         internal_message = Mail(
#             from_email='admin@nounsai.wtf',
#             to_emails=[To('theheroshep@gmail.com'), To('eolszewski@gmail.com')],
#             subject='Video #{} Has Processed!'.format(video_id),
#             html_content='<p>Download here: {}</p>'.format(link)
#         )
#         external_message = Mail(
#             from_email='admin@nounsai.wtf',
#             to_emails=[To(user['email'])],
#             subject='Your Video Has Processed!'.format(video_id),
#             html_content='<p>Download here: {}</p>'.format(link)
#         )
#         response = sg.send(internal_message)
#         response = sg.send(external_message)

#         metadata['state'] = 'PROCESSED'
#         update_video_for_user(video_id, user_id, metadata)
        
#         return link

#     except Exception as e:
#         metadata['state'] = 'ERROR'
#         update_video_for_user(video_id, user_id, metadata)
#         print('Internal server error with unclip_images: {}'.format(str(e)))
#         return False


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
        
            elif inf_mode.split(' ')[1] == 'Outlines':
                return control_net_outlines(pipeline, prompt, generator, negative_prompt, steps, strength, img)
        
            elif inf_mode.split(' ')[1] == 'Depth':
                return control_net_depth(pipeline, prompt, generator, negative_prompt, steps, img)
        
            elif inf_mode.split(' ')[1] == 'Segmentation':
                return control_net_segmentation(pipeline, prompt, generator, negative_prompt, steps, img)
        
    except Exception as e:
        print('Internal server error with inferencing {}: {}'.format(inf_mode, str(e)))
        return None

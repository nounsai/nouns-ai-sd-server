import os
import base64
import jwt
import sys
import json
import numpy
import hashlib
import datetime
import secrets
import requests
import uuid
import base64
import traceback

from PIL import Image
from io import BytesIO 
from functools import wraps
from flask_cors import CORS
from passlib.hash import sha256_crypt
from flask import Flask, jsonify, request, g, make_response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, To

from nouns_utils import bytes_from_image, thumbnail_bytes_for_image, fetch_env_config, image_from_base_64, serve_pil_image, _hide_seek, extract_start_and_end_frames, pil_to_bytes
from db import create_user, fetch_user, fetch_user_for_email, update_user, delete_user, \
        create_image, fetch_images, fetch_images_for_user, fetch_images_with_hash, fetch_image_ids_for_user, fetch_image_for_user, update_image_for_user, delete_image_for_user, \
        create_audio, fetch_audios, fetch_audios_for_user, fetch_queued_audios, fetch_audio_for_user, update_audio_for_user, delete_audio_and_video_project_for_user, \
        create_link, fetch_links, fetch_link, fetch_links_for_user, update_link_for_user, delete_link_for_user, \
        create_video, fetch_video, fetch_video_for_user, fetch_videos_for_user, update_video_for_user, delete_video_for_user, \
        fetch_user_for_verify_key, verify_user_for_id, create_password_reset_for_user, get_password_reset, verify_password_reset, \
        create_video_project, fetch_video_project_for_user, fetch_video_projects_for_user, update_video_project_for_user, delete_video_project_for_user, \
        update_user_referral_token, fetch_user_for_referral_token, create_referral, fetch_referral_for_referred, \
        execute_reward, update_user_metadata, create_transaction, fetch_transactions_for_user, \
        update_video_project_state, fetch_video_project_for_id, fetch_image, update_video_project_cdn_id, \
        fetch_image_with_cdn_id
from cdn import download_audio_from_cdn, delete_video_project_from_cdn

import torchaudio

config = fetch_env_config()
PIPELINE_DICT = {}
if config['server_type'] == 'gpu':
    from segment_anything import SamPredictor
    from middleware import inference, setup_pipelines
    from middleware import (
        inference, setup_pipelines, txt_to_audio, 
        txt_and_audio_to_audio, setup_audio, 
        separate_audio_tracks
    )
    # AUDIO_DICT = setup_audio()
    PIPELINE_DICT = setup_pipelines()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1025 # 5 MB
CORS(app, expose_headers=['X-Image-Id'])
CORS(app)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri='memory://',
    headers_enabled=True
)

sg = SendGridAPIClient(config['sendgrid_api_key'])


#######################################################
######################### API #########################
#######################################################


#############################
########### USERS ###########
#############################


# authentication function to verify user credentials
def authenticate(email, password):
    user = fetch_user_for_email(email)
    return user if (user is not None and sha256_crypt.verify(password, user['password'])) else None

# decorator to check if user is authenticated
def auth_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization']

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, config['secret_key'], algorithms=["HS256"])
            current_user_id = str(data['id'])
        except:
            return jsonify({'message': 'Token is invalid!'}), 401

        g.current_user_id = current_user_id

        return f(current_user_id, *args, **kwargs)

    return decorated

# decorator to get user id from token
def prepend_user_id(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization']

        if not token:
            current_user_id = str(67)
        else:
            try:
                data = jwt.decode(token, config['secret_key'], algorithms=["HS256"])
                current_user_id = str(data['id'])
            except:
                current_user_id = str(67)

        return f(current_user_id, *args, **kwargs)

    return decorated

# decorator to check if user is authenticated
def challenge_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        
        if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
            return jsonify({'message': 'challenge-token header missing / invalid'}), 401

        return f(*args, **kwargs)

    return decorated

# route to generate JWT token for authenticated user
@app.route('/login', methods=['POST'])
@challenge_token_required
@limiter.limit('5 per minute', key_func=lambda: request.remote_addr)
def api_login():
    
    if 'Authorization' not in request.headers:
        return jsonify({'message': 'Missing Credentials!'}), 400

    auth = request.authorization

    if not auth or not auth.username or not auth.password:
        return jsonify({'message': 'Invalid request!'}), 401

    user = authenticate(auth.username, auth.password)
    if user is not None:
        if not user['is_verified']:
            return jsonify({'message': 'Email not verified!'}), 400

        token = jwt.encode({'id': user['id'], 'exp': datetime.datetime.utcnow() + datetime.timedelta(days=300)}, config['secret_key'], algorithm='HS256')
        return jsonify({'token': token, 'id': user['id']}), 200

    return jsonify({'message': 'Invalid credentials!'}), 401

# route to create new user
@app.route('/users', methods=['POST'])
@challenge_token_required
@limiter.limit('5 per minute', key_func=lambda: request.remote_addr)
def api_create_user():

    data = json.loads(request.data)

    if not data or not 'email' in data or not 'password' in data:
        return jsonify({'message': 'Invalid data!'}), 400

    user = fetch_user_for_email(data['email'])

    if user is not None:
        return jsonify({'message': 'User already exists!'}), 400

    try:
        verify_key = secrets.token_hex(48)
        id = create_user(
            data['email'],
            sha256_crypt.encrypt(data['password']),
            verify_key,
            data['metadata']
        )

        # send verification email
        message = Mail(
            from_email='admin@nounsai.wtf',
            to_emails=[To(data['email'])],
            subject='NounsAI Playground Verification',
            html_content=f'''
<p>Hello! You are receiving this email because you registered an account at <a href="https://nounsai.wtf">nounsai.wtf</a>.
To complete registration and verify your email, please click on this link: <a href="https://nounsai.wtf/verify/{verify_key}">Verify</a><br></p>
<p>If you didn\'t register, you can safely ignore this email.</p>
'''
        )
        response = sg.send(message)

        # handle referrals
        if data.get('referralToken') is not None:
            referrer_id = fetch_user_for_referral_token(data['referralToken'])['id']
            create_referral(referrer_id=referrer_id, referred_id=id, metadata={})


        return { 'id': id }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# route to verify key
@app.route('/users/verify/<verify_key>', methods=['POST'])
@challenge_token_required
@limiter.limit('5 per minute', key_func=lambda: request.remote_addr)
def api_verify_key(verify_key):
    user = fetch_user_for_verify_key(verify_key)

    if user is None:
        return jsonify({'message': 'Invalid verification key'}), 400

    try:
        verify_user_for_id(user['id'])

        # check for referrals
        referral = fetch_referral_for_referred(user['id'])
        reward_name = 'referral_verify'

        if referral is not None:
            reward_result = execute_reward(reward_name, user['id'])

            # give reward to referrer
            if 'amount' in reward_result and reward_result['amount'] > 0:
                create_transaction(
                    user_id=referral['referrer_id'],
                    amount=reward_result['amount'],
                    amount_remaining=reward_result['amount'],
                    expires_at=reward_result['expires_at'],
                    memo='rewards_' + reward_name
                )
            # mark reward as complete
            if 'complete' in reward_result and reward_result['complete']:
                user['metadata']['rewards_' + reward_name] = True
                update_user_metadata(user['id'], user['metadata'])

        return { 'id': user['id'] }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# route to request password reset
@app.route('/users/password/reset', methods=['POST'])
@challenge_token_required
@limiter.limit('5 per minute', key_func=lambda: request.remote_addr)
def api_request_password_reset():
    data = json.loads(request.data)

    if not data or 'email' not in data:
        return jsonify({'message': 'Invalid data'}), 400

    email = data['email']
    reset_key = secrets.token_hex(48)

    is_success, message = create_password_reset_for_user(email, reset_key)

    if not is_success:
        return jsonify({'message': message}), 400

    # send password reset email
    message = Mail(
        from_email='admin@nounsai.wtf',
        to_emails=[To(data['email'])],
        subject='NounsAI Playground Password Reset',
        html_content=f'''
<p>Hello! You are receiving this email because you requested a password reset at <a href="https://nounsai.wtf">nounsai.wtf</a>.
To reset your password, please click on this link: <a href="https://nounsai.wtf/reset/{reset_key}">Reset Password</a><br></p>
<p>If you didn\'t initiate this request, you can safely ignore this email.</p>
'''
    )
    response = sg.send(message)

    return {'status': 'success'}, 200

# route to fetch password reset info
@app.route('/users/password/reset/<reset_key>', methods=['GET'])
@challenge_token_required
@limiter.limit('5 per minute', key_func=lambda: request.remote_addr)
def api_get_password_reset(reset_key):
    try:
        data = get_password_reset(reset_key)
        return data, 200
    except Exception as e:
        if str(e) == 'list index out of range':
            return {'message': 'Invalid key'}, 400

        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# route to verify password reset
@app.route('/users/password/reset/<reset_key>', methods=['POST'])
@challenge_token_required
@limiter.limit('5 per minute', key_func=lambda: request.remote_addr)
def api_verify_password_reset(reset_key):
    data = json.loads(request.data)

    if not data or 'password' not in data:
        return jsonify({'message': 'Invalid data'}), 400

    new_password_hash = sha256_crypt.encrypt(data['password'])

    is_success, message = verify_password_reset(reset_key, new_password_hash)

    if not is_success:
        return jsonify({'message': message}), 400

    return jsonify({'status': 'success'}), 200


# route to fetch user
@app.route('/users/<user_id>', methods=['GET'])
@auth_token_required
@limiter.limit('20 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Editing wrong user!'}), 400

    try:
        user = fetch_user(current_user_id)
        return user, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# route to update user
@app.route('/users/<user_id>', methods=['PUT'])
@auth_token_required
@limiter.limit('15 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_update_user(current_user_id, user_id):

    data = json.loads(request.data)

    if user_id != current_user_id:
        return jsonify({'message': 'Editing wrong user!'}), 400

    try:
        update_user(
            current_user_id,
            data['password'],
            data['metadata']
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# route to fetch user's referral token
@app.route('/users/<user_id>/referral-token', methods=['GET'])
@auth_token_required
def api_fetch_user_referral_token(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Editing wrong user!'}), 400

    try:
        user = fetch_user(user_id)
        if user['referral_token'] is None:
            # set new token if not currently set
            new_token = secrets.token_hex(48)
            update_user_referral_token(user_id, new_token)
            return { 'token': new_token }, 200
        else:
            return { 'token': user['referral_token'] } , 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# route to fetch user's current credit balance
@app.route('/users/<user_id>/credits', methods=['GET'])
@auth_token_required
def api_fetch_user_credits(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Editing wrong user!'}), 400

    try:
        transactions = fetch_transactions_for_user(user_id)
        credit_sum = 0
        for trxn in transactions:
            credit_sum += trxn.get('amount_remaining', 0)

        return { 'credits': credit_sum }
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


#############################
########## IMAGES ###########
#############################


@app.route('/images', methods=['POST'])
@challenge_token_required
@prepend_user_id
@limiter.limit(limit_value=lambda: '32 per minute' if g.get('current_user_id', None) else '16 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_create_image(current_user_id):

    data = json.loads(request.data)

    parent_id = -1 if 'parent_id' not in data else data['parent_id']
    images = []

    if data['inference_mode'] == 'Text to Image':
        images = inference(PIPELINE_DICT['Text to Image'][data['model_id']], 'Text to Image', data['prompt'], n_images=int(data['samples']), negative_prompt=data['negative_prompt'], steps=int(data['steps']), seed=int(data['seed']), aspect_ratio=data['aspect_ratio'])
    else:
        image = image_from_base_64(data['base_64'])
        if data['inference_mode'] == 'Image to Image':
            images = inference(PIPELINE_DICT['Image to Image'][data['model_id']], 'Image to Image', data['prompt'], n_images=int(data['samples']), negative_prompt=data['negative_prompt'], steps=int(data['steps']), seed=int(data['seed']), aspect_ratio=data['aspect_ratio'], img=image, strength=float(data['strength']))
        elif data['inference_mode'] == 'Pix to Pix':
            images = inference(PIPELINE_DICT['Pix to Pix'][data['model_id']], 'Pix to Pix', data['prompt'], n_images=int(data['samples']), steps=int(data['steps']), seed=int(data['seed']), img=image)
        elif data['inference_mode'].split(' ')[0] == 'ControlNet':
            images = inference(PIPELINE_DICT['ControlNet'][data['inference_mode'].split(' ')[1]][data['model_id']], data['inference_mode'], data['prompt'], n_images=1, negative_prompt=data['negative_prompt'], steps=int(data['steps']), seed=int(data['seed']), img=image, strength=int(data['strength']))
        elif data['inference_mode'] == 'Mask':
            images = inference(PIPELINE_DICT['Mask']['Inpainting'], data['inference_mode'], data['prompt'], n_images=1, negative_prompt=data['negative_prompt'], seed=int(data['seed']), img=image, mask=data['mask'])

        if parent_id == -1:
            images_with_hash = fetch_images_with_hash(hashlib.sha256(data['base_64'].encode('utf-8')).hexdigest())
            images_for_user_with_hash = [i for i in images_with_hash if i.user_id == current_user_id]
            if len(images_for_user_with_hash) > 0:
                parent_id = images_for_user_with_hash[0].id
            elif len(images_with_hash) > 0:
                parent_id = images_with_hash[0].id
    
    image_byte_data = bytes_from_image(images[0])
    thumbnail_byte_data = thumbnail_bytes_for_image(images[0])
    id, _ = create_image(
        current_user_id,
        image_byte_data,
        thumbnail_byte_data,
        (hashlib.sha256(image_byte_data)).hexdigest(),
        data,
        False,
        False,
        parent_id
    )

    # TODO: Make this return image metadata.
    response = make_response(serve_pil_image(images[0]))
    response.headers['X-Image-Id'] = id
    response.headers['Access-Control-Expose-Headers'] = 'X-Image-Id'
    return response

@app.route('/images', methods=['GET'])
@challenge_token_required
@limiter.limit('15 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_images():

    # /images?page=1&limit=20
    page = request.args.get('page', default = 1, type = int)
    limit = request.args.get('limit', default = 20, type = int)
    offset = (page - 1) * limit

    try:
        images = fetch_images(
            limit,
            offset
        )
        return images, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


@app.route('/templates/<template_id>', methods=['GET'])
@challenge_token_required
@limiter.limit('15 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_get_template(template_id):

    try:
        image = fetch_image_with_cdn_id(template_id)
        return image, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


@app.route('/user-images', methods=['POST'])
@auth_token_required
@limiter.limit('32 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_upload_user_image(current_user_id):

    if request.files and request.files.get('image', None) is not None:
        image_bytes = request.files['image'].read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        metadata = {
            'base_64': None,
            'user_uploaded': True
        }
        id, _ = create_image(
            current_user_id,
            image_bytes,
            thumbnail_bytes_for_image(image),
            (hashlib.sha256(image_bytes)).hexdigest(),
            metadata,
            False,
            True,
            -1,
            use_thread=False
        )
        db_image = fetch_image(id)
        return db_image, 200
    else:
        return { 'error': 'No file detected' }, 400


@app.route('/users/<user_id>/images', methods=['GET'])
@auth_token_required
@limiter.limit('15 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_images_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    # /images?page=1&limit=20&favorited=true
    field = request.args.get('field', default = 'image', type = str)
    page = request.args.get('page', default = 1, type = int)
    limit = request.args.get('limit', default = 20, type = int)
    favorited = request.args.get('favorited')
    offset = (page - 1) * limit

    try:
        if field == 'id':
            image_ids = fetch_image_ids_for_user(current_user_id)
            return image_ids, 200
        else:
            images = fetch_images_for_user(
                current_user_id,
                limit,
                offset,
                favorited.lower() == 'true'
            )
            return images, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/images/<image_id>', methods=['GET'])
@auth_token_required
@limiter.limit('100 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_image_for_user(current_user_id, user_id, image_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        image = fetch_image_for_user(image_id, current_user_id)
        if image is not None:
            return image, 200
        else:
            return { 'error': "Image not found" }, 404
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/images/<image_id>', methods=['PUT'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_update_image(current_user_id, user_id, image_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        update_image_for_user(
            image_id,
            current_user_id,
            data['is_public'] if data['is_public'] is not None else None,
            data['is_liked'] if data['is_liked'] is not None else None
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/images/<image_id>', methods=['DELETE'])
@auth_token_required
@limiter.limit('20 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_delete_image(current_user_id, user_id, image_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        delete_image_for_user(
            current_user_id,
            image_id
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


#############################
########## AUDIOS ###########
#############################


@app.route('/audios', methods=['POST'])
@auth_token_required
@limiter.limit('15 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_create_audio(current_user_id):

    try:
        data = json.loads(request.data)
        prompt = data['text']
        melody_id = data.get('melody_id', None)

        metadata = {
            'prompt': prompt,
            'mode': 'text to audio',
            'melody_id': melody_id
        }

        id, cdn_id = create_audio(
            user_id=current_user_id, 
            name='generated.mp3', 
            size=0, 
            metadata=metadata,
            state="QUEUED"
        )

        return {
            'id': id,
            'cdn_id': cdn_id
        }, 200

    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/audio-split', methods=['POST'])
@auth_token_required
@limiter.limit('15 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_split_audio(current_user_id):

    try:
        data = json.loads(request.data)
        audio_id = data['id']

        id, cdn_id = create_audio(
                user_id=current_user_id, 
                name=f'queuedSplit{audio_id}.mp3', 
                size=0, 
                metadata={
                    'parent_id': audio_id,
                    'mode': 'audio split'
                },
                state="QUEUED"
            )

        return {
            'id': id,
            'cdn_id': cdn_id
        }, 200

    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


@app.route('/users/<user_id>/audios', methods=['POST'])
@auth_token_required
@limiter.limit('5 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_create_audio_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    if request.files:
        audio_file = request.files["audio"]
        use_thread = request.form.get("useThread", None)
        audio_data = audio_file.read()
        
        try:
            if (use_thread is not None):
                id, cdn_id = create_audio(
                    user_id=current_user_id,
                    audio_byte_data=audio_data,
                    name=audio_file.filename,
                    size=audio_file.content_length,
                    metadata={},
                    use_thread=False
                )
            else:
                id, cdn_id = create_audio(
                    user_id=current_user_id,
                    audio_byte_data=audio_data,
                    name=audio_file.filename,
                    size=audio_file.content_length,
                    metadata={},
                )
            return { 'id': id, 'cdn_id': cdn_id }, 200
        except Exception as e:
            print("Internal server error: {}".format(str(e)))
            return { 'error': "Internal server error: {}".format(str(e)) }, 500
    else:
        return jsonify({"error": "No audio file detected"}), 400


@app.route('/users/<user_id>/audios', methods=['GET'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_audios_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    # /audios?page=1&limit=20
    page = request.args.get('page', default = 1, type = int)
    limit = request.args.get('limit', default = 20, type = int)
    offset = (page - 1) * limit

    try:
        audios = fetch_audios_for_user(
            current_user_id,
            limit,
            offset
        )
        return audios, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/audios/<audio_id>', methods=['GET'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_audio_for_user(current_user_id, user_id, audio_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    try:
        audio = fetch_audio_for_user(current_user_id, audio_id)
    
        if audio is not None:
            if audio['state'] == 'QUEUED':
                queue = fetch_queued_audios()
                for index, queuedAudio in enumerate(queue):
                    if queuedAudio['id'] == int(audio_id):
                        audio['queue'] = index + 1
                        break
            elif audio['state'] == 'PROCESSING':
                audio['queue'] = 0
            elif audio['state'] == 'COMPLETED':
                audio['queue'] = None
                
            return audio, 200
        else:
            return { 'error': "Audio not found" }, 404
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/audios/<audio_id>', methods=['PUT'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_update_audio(current_user_id, user_id, audio_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        update_audio_for_user(
            audio_id,
            current_user_id,
            data['name'],
            data['url'],
            data['size'],
            data['metadata']
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/audios/<audio_id>', methods=['DELETE'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_delete_audio_and_video_project(current_user_id, user_id, audio_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        delete_audio_and_video_project_for_user(
            current_user_id,
            audio_id
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


##############################
########### LINKS ############
##############################


@app.route('/users/<user_id>/links', methods=['POST'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_create_link_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        id = create_link(
            current_user_id,
            data['metadata']
        )
        return { 'id': id }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/links', methods=['GET'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_links_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    # /links?page=1&limit=20
    page = request.args.get('page', default = 1, type = int)
    limit = request.args.get('limit', default = 20, type = int)
    offset = (page - 1) * limit

    try:
        links = fetch_links_for_user(
            current_user_id,
            limit,
            offset
        )
        return links, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/links/<link_id>', methods=['GET'])
@challenge_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_link(link_id):

    data = json.loads(request.data)
    
    try:
        link = fetch_link(link_id)
        if link is not None:
            return link, 200
        else:
            return { 'error': "Link not found" }, 404
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/links/<link_id>', methods=['PUT'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_update_link(current_user_id, user_id, link_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        update_link_for_user(
            link_id,
            current_user_id,
            data['metadata']
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/links/<link_id>', methods=['DELETE'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_delete_link(current_user_id, user_id, link_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        delete_link_for_user(
            current_user_id,
            link_id
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


################################
############ VIDEOS ############
################################


@app.route('/users/videos', methods=['POST'])
@auth_token_required
@limiter.limit('5 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_create_video_for_user(current_user_id):

    if not request.files:
        return { 'error': 'file not found' }, 404
    
    try:
        video_file = request.files['video']
        video_name = video_file.filename
        video_bytes = video_file.read()

        cdn_uuid = str(uuid.uuid4())

        first_frame, last_frame, duration = extract_start_and_end_frames(video_bytes) 

        first_frame_bytes = pil_to_bytes(first_frame)
        last_frame_bytes = pil_to_bytes(last_frame)

        # save first frame
        first_id, first_cdn_id = create_image(
            current_user_id,
            first_frame_bytes,
            thumbnail_bytes_for_image(first_frame),
            (hashlib.sha256(first_frame_bytes)).hexdigest(),
            { 'user_uploaded': True, 'video_cdn_id': cdn_uuid, 'video_duration': duration },
            False,
            True,
            -1,
            use_thread=False
        )
        # save last frame
        last_id, last_cdn_id = create_image(
            current_user_id,
            last_frame_bytes,
            thumbnail_bytes_for_image(last_frame),
            (hashlib.sha256(last_frame_bytes)).hexdigest(),
            { 'user_uploaded': True, 'video_cdn_id': cdn_uuid, 'video_duration': duration },
            False,
            True,
            -1,
            use_thread=False
        )
        # save video
        id = create_video(
            current_user_id,
            {},
            duration,
            first_id,
            first_cdn_id,
            last_id,
            last_cdn_id,
            cdn_uuid,
            video_bytes,
            video_name
        )

        return { 
            'id': id, 
            'cdn_id': cdn_uuid, 
            'start_frame_id': first_id, 
            'start_frame_cdn_id': first_cdn_id,
            'end_frame_id': last_id,
            'end_frame_cdn_id': last_cdn_id,
            'duration': duration,
            'user_id': current_user_id,
            'name': video_name
        }, 200

    except Exception as e:
        print(traceback.format_exc())
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/videos', methods=['GET'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_videos_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    # /videos?page=1&limit=20
    # page = request.args.get('page', default = 1, type = int)
    # limit = request.args.get('limit', default = 20, type = int)
    # offset = (page - 1) * limit

    try:
        videos = fetch_videos_for_user(
            current_user_id
        )
        return videos, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/videos/<video_id>', methods=['GET'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_video_for_user(current_user_id, user_id, video_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        video = fetch_video_for_user(video_id)
        if video is not None:
            return video, 200
        else:
            return { 'error': "Video not found" }, 404
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/videos/<video_id>', methods=['PUT'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_update_video(current_user_id, user_id, video_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        update_video_for_user(
            video_id,
            current_user_id,
            data['metadata']
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# @app.route('/videos/<video_id>/process', methods=['GET'])
# @challenge_token_required
# @limiter.limit('2 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
# def api_process_video(video_id):

#     try:
#         video = fetch_video(video_id)
#         if video is not None:
            
#             link = unclip_images(
#                 video_id,
#                 video['user_id'],
#                 PIPELINE_DICT['Unclip']['kakaobrain/karlo-v1-alpha-image-variations'], 
#                 video['metadata']
#             )

#             if link:
#                 return { 'link': link }, 200
#             else:
#                 return { 'error': "Internal server error" }, 500
#         else:
#             return { 'error': "Video not found" }, 404
    
#     except Exception as e:
#         print("Internal server error: {}".format(str(e)))
#         return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/videos/<video_id>', methods=['DELETE'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_delete_video(current_user_id, user_id, video_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        delete_video_for_user(
            current_user_id,
            video_id
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500
    
    
################################
######## VIDEO PROJECTS ########
################################

@app.route('/users/<user_id>/video-projects', methods=['POST'])
@auth_token_required
@limiter.limit('5 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_create_video_project_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        id = create_video_project(
            current_user_id,
            data['audio_id'],
            data['metadata'],
        )
        return { 'id': id }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/video-projects', methods=['GET'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_video_projects_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    # /video-projects?page=1&limit=20
    page = request.args.get('page', default = 1, type = int)
    limit = request.args.get('limit', default = 20, type = int)
    offset = (page - 1) * limit

    try:
        video_projects = fetch_video_projects_for_user(
            current_user_id,
            limit,
            offset
        )
        return video_projects, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/video-projects/<video_project_id>', methods=['GET'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_fetch_video_project_for_user(current_user_id, user_id, video_project_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        video_project = fetch_video_project_for_user(current_user_id, video_project_id)
        if video_project is not None:
            return video_project, 200
        else:
            return { 'error': "Video project not found" }, 404
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/video-projects/<video_project_id>', methods=['PUT'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def api_update_video_project(current_user_id, user_id, video_project_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    try:
        update_video_project_for_user(
            current_user_id,
            video_project_id,
            data['metadata']
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500


@app.route('/video-projects/<video_project_id>/queue', methods=['POST'])
@auth_token_required
@limiter.limit('10 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def queue_video_project_for_generation(current_user_id, video_project_id):
    try:
        project = fetch_video_project_for_id(video_project_id);
        if project is None:
            return { 'message': 'Video project does not exist!' }, 404
        if project['user_id'] != int(current_user_id):
            return { 'message': 'Wrong user!' }, 400

        if project['state'] != 'PROCESSING' and project['state'] != 'QUEUED':
            # delete previous project, if present
            delete_video_project_from_cdn(project['user_id'], project['cdn_id'])
            # update cdn id
            update_video_project_cdn_id(project['id'], str(uuid.uuid4()))
            # queue video
            update_video_project_state(video_project_id, 'QUEUED')

        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500



############################
########### MISC ###########
############################


# health check route
@app.route('/health')
def health():
    return jsonify({'message': 'API is up and running!'}), 200

@app.route('/extend_prompt', methods=['POST'])
@challenge_token_required
@limiter.limit(limit_value=lambda: '20 per minute' if g.get('current_user_id', None) else '5 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def extend_prompt():

    content = json.loads(request.data)
    return {'prompt': list(PIPELINE_DICT['Text'].values())[0](content['prompt'] + ',', num_return_sequences=1)[0]['generated_text']}, 200

@app.route('/interrogate', methods=['POST'])
@auth_token_required
@limiter.limit(limit_value=lambda: '20 per minute' if g.get('current_user_id', None) else '5 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def interrogate(current_user_id):

    content = json.loads(request.data)
    image = image_from_base_64(content['base_64']).convert('RGB')
    return {'prompt': list(PIPELINE_DICT['Interrogator'].values())[0].interrogate_fast(image)}, 200

@app.route('/upscale', methods=['POST'])
@auth_token_required
@limiter.limit(limit_value=lambda: '10 per minute' if g.get('current_user_id', None) else '5 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def upscale(current_user_id):

    content = json.loads(request.data)
    image = image_from_base_64(content['base_64']).convert('RGB')
    [h,w,c] = numpy.shape(image)
    scalar = float(384 / max(h, w, 384))
    image = image.resize((int(w*scalar), int(h*scalar)))
    images = list(PIPELINE_DICT['Upscale'].values())[0](
        prompt='',
        image=image
    ).images
    return serve_pil_image(images[0])

@app.route('/mask', methods=['POST'])
@challenge_token_required
@limiter.limit(limit_value=lambda: '12 per minute' if g.get('current_user_id', None) else '6 per minute', key_func=lambda: g.get('current_user_id', request.remote_addr))
def mask_from_points():
    content = json.loads(request.data)
    image = image_from_base_64(content['base64']).convert('RGB')
    coordinates = content['coordinates']

    predictor = SamPredictor(PIPELINE_DICT['Mask']['SAM'])
    predictor.set_image(numpy.asarray(image))

    masks, _, _ = predictor.predict(
        point_coords=numpy.array(coordinates), point_labels=numpy.array([1] * len(coordinates)))
    merged_mask = numpy.any(masks, axis=0)

    # convert to base 64
    flattened_mask = merged_mask.flatten()  # flatten mask
    binary_mask = numpy.packbits(flattened_mask,
                                 axis=-1)  # get a binary representation (int, 1 or 0) of the boolean array
    byte_mask = binary_mask.tobytes()  # convert int array to binary
    base64_mask = base64.b64encode(byte_mask).decode()  # convert binary to base 64

    return {'mask': base64_mask}


############################
########## ERRORs ##########
############################

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File size exceeds the allowed limit"}), 413

#######################################################
######################## MAIN #########################
#######################################################


if __name__ == '__main__':
    if config['environment'] == 'prod':
        app.run(host='0.0.0.0', port='5000', ssl_context='adhoc')
    else:
        app.run(host='0.0.0.0', debug=True, port='5000')

import os
import jwt
import sys
import json
import datetime
from functools import wraps
from flask_cors import CORS
from passlib.hash import sha256_crypt
from flask import Flask, jsonify, request
from db_two import create_user, fetch_user, fetch_user_for_email, update_user, delete_user, \
        create_image, fetch_images, fetch_images_for_user, fetch_image_hashes_for_user, fetch_image_for_user, update_image_for_user, delete_image_for_user, \
        create_audio, fetch_audios, fetch_audios_for_user, fetch_audio_for_user, update_audio_for_user, delete_audio_for_user, \
        create_link, fetch_links, fetch_link, fetch_links_for_user, update_link_for_user, delete_link_for_user, \
        create_video, fetch_video_for_user, fetch_videos_for_user, update_video_for_user, delete_video_for_user

currentdir = os.path.dirname(os.path.realpath(__file__))

if not os.path.isfile("config.json"):
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open("config.json") as file:
        config = json.load(file)

app = Flask(__name__)
CORS(app)

##########################################################
####################### MIDDLEWARE #######################
##########################################################


#######################################################
######################### API #########################
#######################################################


#############################
########### USERS ###########
#############################


# authentication function to verify user credentials
def authenticate(email, password):
    user = fetch_user_for_email(email)
    return user is not None and sha256_crypt.verify(password, user['password'])

# decorator to check if user is authenticated
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split()[1]

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, config['secret_key'], algorithms=["HS256"])
            current_user_id = data['id']
        except:
            return jsonify({'message': 'Token is invalid!'}), 401

        return f(current_user_id, *args, **kwargs)

    return decorated

# route to generate JWT token for authenticated user
@app.route('/login', methods=['POST'])
def api_login():
    auth = request.authorization

    if not auth or not auth.email or not auth.password:
        return jsonify({'message': 'Could not verify!'}), 401

    if authenticate(auth.email, auth.password):
        token = jwt.encode({'email': auth.email, 'exp': datetime.datetime.utcnow() + datetime.timedelta(days=300)}, config['secret_key'], algorithm='HS256')
        return jsonify({'token': token}), 200

    return jsonify({'message': 'Could not verify!'}), 401

# route to create new user
@app.route('/users', methods=['POST'])
def api_create_user():

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge-token']:
        return "'challenge-token' header missing / invalid", 401

    data = json.loads(request.data)

    if not data or not 'email' in data or not 'password' in data:
        return jsonify({'message': 'Invalid data!'}), 400

    user = fetch_user_for_email(data['email'])

    if user is not None:
        return jsonify({'message': 'User already exists!'}), 400

    try:
        id = create_user(
            data['email'],
            sha256_crypt.encrypt(data['password']),
            data['metadata']
        )
        return { 'id': id }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

# route to update user
@app.route('/users/<user_id>', methods=['PUT'])
@token_required
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


#############################
########## IMAGES ###########
#############################


# @app.route('/images', methods=['POST'])
# def api_create_image():

#     if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge-token']:
#         return "'challenge-token' header missing / invalid", 401

#     data = json.loads(request.data)

#     images = []
#     if data['inference_mode'] == 'Text to Image':
#         images = inference(IMG_PIPELINE_DICT[data['model_id']], 'Text to Image', data['prompt'], n_images=int(data['samples']), negative_prompt=data['negative_prompt'], steps=int(data['steps']), seed=int(data['seed']), aspect_ratio=data['aspect_ratio'])
#     else:
#         starter = data['base_64'].find(',')
#         image_data = data['base_64'][starter+1:]
#         image_data = bytes(image_data, encoding="ascii")
#         image = Image.open(BytesIO(base64.b64decode(image_data)))
#         if data['inference_mode'] == 'Image to Image':
#             images = inference(I2I_PIPELINE_DICT[data['model_id']], 'Image to Image', data['prompt'], n_images=int(data['samples']), negative_prompt=data['negative_prompt'], steps=int(data['steps']), seed=int(data['seed']), aspect_ratio=data['aspect_ratio'], img=image, strength=float(data['strength']))
#         elif data['inference_mode'] == 'Pix to Pix':
#             images = inference(P2P_PIPELINE_DICT[data['model_id']], 'Pix to Pix', data['prompt'], n_images=int(data['samples']), steps=int(data['steps']), seed=int(data['seed']), img=image)
#     return serve_pil_image(images[0])


@app.route('/users/<user_id>/images', methods=['POST'])
@token_required
def api_create_image_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        id = create_image(
            current_user_id,
            data['base_64'],
            data['hash'],
            data['metadata']
        )
        return { 'id': id }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/images', methods=['GET'])
@token_required
def api_fetch_images_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    # /images?page=1&limit=20
    page = request.args.get('page', default = 1, type = int)
    limit = request.args.get('limit', default = 20, type = int)
    offset = (page - 1) * limit

    try:
        images = fetch_images_for_user(
            current_user_id,
            limit,
            offset
        )
        return images, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/image_hashes', methods=['GET'])
@token_required
def api_fetch_image_hashes_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    try:
        image_hashes = fetch_image_hashes_for_user(current_user_id)
        return image_hashes, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/images/<image_id>', methods=['GET'])
@token_required
def api_fetch_image_for_user(current_user_id, user_id, image_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        image = fetch_image_for_user(current_user_id, image_id)
        if image is not None:
            return image, 200
        else:
            return { 'error': "Image not found" }, 404
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/images/<image_id>', methods=['PUT'])
@token_required
def api_update_image(current_user_id, user_id, image_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        update_image_for_user(
            image_id,
            current_user_id,
            data['base_64'],
            data['hash'],
            data['metadata']
        )
        return { 'status': 'success' }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/images/<image_id>', methods=['DELETE'])
@token_required
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


@app.route('/users/<user_id>/audio', methods=['POST'])
@token_required
def api_create_audio_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        id = create_audio(
            current_user_id,
            data['name'],
            data['url'],
            data['size'],
            data['metadata']
        )
        return { 'id': id }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/audios', methods=['GET'])
@token_required
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
@token_required
def api_fetch_audio_for_user(current_user_id, user_id, audio_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        audio = fetch_audio_for_user(current_user_id, audio_id)
        if audio is not None:
            return audio, 200
        else:
            return { 'error': "Audio not found" }, 404
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/audios/<audio_id>', methods=['PUT'])
@token_required
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
@token_required
def api_delete_audio(current_user_id, user_id, audio_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400
    
    try:
        delete_audio_for_user(
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
@token_required
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
@token_required
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
def api_fetch_link(link_id):

    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge-token']:
        return "'challenge-token' header missing / invalid", 401

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
@token_required
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
@token_required
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


@app.route('/users/<user_id>/videos', methods=['POST'])
@token_required
def api_create_video_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    data = json.loads(request.data)
    
    try:
        id = create_video(
            current_user_id,
            data['metadata']
        )
        return { 'id': id }, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/videos', methods=['GET'])
@token_required
def api_fetch_videos_for_user(current_user_id, user_id):

    if user_id != current_user_id:
        return jsonify({'message': 'Wrong user!'}), 400

    # /videos?page=1&limit=20
    page = request.args.get('page', default = 1, type = int)
    limit = request.args.get('limit', default = 20, type = int)
    offset = (page - 1) * limit

    try:
        videos = fetch_videos_for_user(
            current_user_id,
            limit,
            offset
        )
        return videos, 200
    except Exception as e:
        print("Internal server error: {}".format(str(e)))
        return { 'error': "Internal server error: {}".format(str(e)) }, 500

@app.route('/users/<user_id>/videos/<video_id>', methods=['GET'])
@token_required
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
@token_required
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

@app.route('/users/<user_id>/videos/<video_id>', methods=['DELETE'])
@token_required
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


############################
########### MISC ###########
############################


# health check route
@app.route('/health')
def health():
    return jsonify({'message': 'API is up and running!'}), 200

# protected route
@app.route('/protected')
@token_required
def protected(current_user):
    return jsonify({'message': 'This is a protected route!', 'user': current_user}), 200


if __name__ == '__main__':
    app.run(debug=True)

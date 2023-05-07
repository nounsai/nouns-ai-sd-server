import requests
import base64
from utils import fetch_env_config

config = fetch_env_config()

STORAGE_LOCATION_ID = config['storage_location_id']
STORAGE_ZONE_NAME = config['storage_zone']
ACCESS_KEY = config['storage_access_key']

STORAGE_ZONE_AUDIO = config['storage_zone_audio']
ACCESS_KEY_AUDIO = config['storage_access_key_audio']


#############################
########## IMAGES ###########
#############################


# returns image (either full or thumbnail) in binary, or None if not found
def download_image_from_cdn(user_id, image_id, image_type='full'):
    url = f'https://{STORAGE_LOCATION_ID}.storage.bunnycdn.com/{STORAGE_ZONE_NAME}/{user_id}/{image_id}-{image_type}.png'

    headers = {
        "AccessKey": ACCESS_KEY
    }

    response = requests.get(url, headers=headers)

    # image not found
    if response.status_code != 200:
        return None

    return response.content


# uploads base 64 image to CDN, returns true if successful or false if unsuccessful
def upload_image_to_cdn(user_id, image_id, base_64):
    headers = {
        "Content-Type": "application/octet-stream",
        "AccessKey": ACCESS_KEY
    }

    # upload full image
    full_response = requests.put(
        f'https://storage.bunnycdn.com/{STORAGE_ZONE_NAME}/{user_id}/{image_id}-full.png',
        data=base_64,
        headers=headers
    )

    # upload thumbnail
    thumb_response = requests.put(
        f'https://storage.bunnycdn.com/{STORAGE_ZONE_NAME}/{user_id}/{image_id}-thumbnail.png', 
        data=thumbnail,
        headers=headers
    )

    if full_response.status_code != 201 or thumb_response.status_code != 201:
        print(f'Failed to upload image with ID {image_id} to CDN')


# deletes image from CDN, returns true if successful or false if unsuccessful
def delete_image_from_cdn(user_id, image_id):
    urls = [
        f'https://{STORAGE_LOCATION_ID}.storage.bunnycdn.com/{STORAGE_ZONE_NAME}/{user_id}/{image_id}-full.png',
        f'https://{STORAGE_LOCATION_ID}.storage.bunnycdn.com/{STORAGE_ZONE_NAME}/{user_id}/{image_id}-thumbnail.png'
    ]

    headers = {
        "AccessKey": ACCESS_KEY
    }

    for url in urls:
        response = requests.delete(url, headers=headers)

        if response.status_code != 200:
            return False

    return True


#############################
########## AUDIOS ###########
#############################

def download_audio_from_cdn(user_id, cdn_id):
    url = f"https://storage.bunnycdn.com/{STORAGE_ZONE_AUDIO}/{user_id}/{cdn_id}-full.mp3"
    headers =  {
        'accept': '*/*',
        'AccessKey': ACCESS_KEY_AUDIO
    }
    
    response = requests.get(url, headers=headers)
    # audio not found
    if response.status_code != 200:
        return None

    content_base64 = base64.b64encode(response.content)
    content_str = content_base64.decode('utf-8')
    return content_str

# uploads base 64 audio to CDN, returns true if successful or false if unsuccessful
def upload_audio_to_cdn(user_id, audio_id, base_64):
    headers = {
        "Content-Type": "application/octet-stream",
        "AccessKey": ACCESS_KEY_AUDIO
    }

    # upload full audio
    full_response = requests.put(
        f'https://storage.bunnycdn.com/{STORAGE_ZONE_AUDIO}/{user_id}/{audio_id}-full.mp3',
        data=base_64,
        headers=headers
    )

    if full_response.status_code != 201:
        print(f'Failed to upload image with ID {image_id} to CDN')


# deletes image from CDN, returns true if successful or false if unsuccessful
def delete_audio_from_cdn(user_id, audio_id):
    url = f'https://storage.bunnycdn.com/{STORAGE_ZONE_AUDIO}/{user_id}/{audio_id}-full.mp3'
    headers = {
        "AccessKey": ACCESS_KEY_AUDIO
    }

    
    response = requests.delete(url, headers=headers)

    if response.status_code != 200:
        return False

    return True
import requests
from utils import fetch_env_config

config = fetch_env_config()

STORAGE_LOCATION_ID = config['storage_location_id']
STORAGE_ZONE_NAME = config['storage_zone']
ACCESS_KEY = config['storage_access_key']


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
def upload_image_to_cdn(user_id, image_id, base_64, thumbnail):
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

    if full_response.status_code != 201 or thumb_response.status_code == 201:
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
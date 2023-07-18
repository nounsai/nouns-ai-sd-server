from PIL import Image
import os 
import sys
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
from utils import bytes_from_image, thumbnail_bytes_for_image
from db import create_image
import hashlib
import base64
from io import BytesIO
import requests



def create_template():
    img_path = input('Enter image path: ')
    user_id = int(input('Enter user_id to save image under: '))

    if 'http' in img_path:
        res = requests.get(img_path)
        image = Image.open(BytesIO(res.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    image_byte_data = bytes_from_image(image)
    thumbnail_byte_data = thumbnail_bytes_for_image(image)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str
    img_str = img_str.decode('utf-8')

    id, cdn_id = create_image(
        user_id,
        image_byte_data,
        thumbnail_byte_data,
        (hashlib.sha256(image_byte_data)).hexdigest(),
        {
            'parent_id': -1,
            'template': True,
            'user_uploaded': True,
            'base_64': img_str

        },
        False,
        False,
        -1,
        use_thread=False
    )
    print(f'\n\ncreated template image record with \nid: {id} \ncdn_id: {cdn_id}')


if __name__ == "__main__":
    create_template()
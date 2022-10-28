import os
import sys
import json
import requests

from io import BytesIO
from PIL import Image

if not os.path.isfile("config.json"):
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open("config.json") as file:
        config = json.load(file)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

if __name__ == "__main__":

    url = 'http://127.0.0.1:5000/get_image'
    prompt = sys.argv[1].replace('_', ' ')
    payload = {'prompt': prompt, 'samples': 1, 'steps': 20, 'seed': 12334213213}
    headers = {'content-type': 'application/json', 'challenge-token': config['roko_challenge_token']}

    response = requests.post(url, json=payload, headers=headers)
    img = Image.open(BytesIO(response.content))
    img.save('{}.jpg'.format(prompt))

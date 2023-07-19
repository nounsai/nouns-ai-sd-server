import sys
import requests

from db import fetch_videos
from nouns_utils import fetch_env_config

config = fetch_env_config()

if __name__ == '__main__':
    videos_df = fetch_videos()
    videos_df = videos_df[videos_df['metadata']['state'] != 'DONE']
    if len(videos_df[videos_df['metadata']['state'] == 'PROCESSING']) > 0:
        sys.exit(0)
    else:
        videos_df = videos_df[videos_df['metadata']['state'] == 'QUEUED']
        request_id = videos_df.iloc[-1]['id']
        url = "http://localhost:5000/videos/{}/process".format(request_id)
        headers = { 'challenge-token': config['challenge_token'] }
        response = requests.request("GET", url, headers=headers)

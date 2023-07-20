import os 
import sys
import requests
from datetime import datetime, timedelta
import traceback

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import torch
import torchaudio

from db import (
    fetch_queued_audios,
    fetch_audio_for_id,
    fetch_audio_for_user,
    update_audio_state,
    create_audio,
    update_audio_for_user,
    update_audio_metadata,
)
from cdn import (
    upload_audio_to_cdn,
)

from utils import fetch_env_config, _hide_seek

from PIL import Image
from io import BytesIO

config = fetch_env_config()

from audio_generation import CustomMusicGen, tensor_to_audio_bytes, Demucs, preprocess_audio

from middleware import (
        txt_to_audio, 
        txt_and_audio_to_audio, setup_audio, 
        separate_audio_tracks,
        continue_audio
    )
AUDIO_DICT = setup_audio()


def generate_audios():
    queued_audios = fetch_queued_audios()
    for audio in queued_audios:
        # check that project isn't already being handled
        db_audio = fetch_audio_for_id(audio['id'])
        if db_audio is None or db_audio['state'] == 'PROCESSING':
            continue
        # update state
        update_audio_state(audio['id'], 'PROCESSING')
        start_time = datetime.now()

        try:
            # text to audio
            if db_audio["metadata"].get("melody_id", None) is None and db_audio["metadata"].get("mode") == "text to audio":
                audio_bytes = txt_to_audio(AUDIO_DICT, db_audio["metadata"]["prompt"])

                # upload to cdn
                upload_audio_to_cdn(db_audio["user_id"], db_audio['cdn_id'], audio_bytes)

                # mark audio generation as complete
                update_audio_state(db_audio['id'], 'COMPLETED')
            # split audio
            elif db_audio["metadata"]["mode"] == "audio split":
                audio_id = db_audio["metadata"]["parent_id"]
                db_melody = fetch_audio_for_user(db_audio["user_id"], audio_id)
                if db_melody is None:
                    update_audio_state(db_audio['id'], 'ERROR')
                    continue
                audio_url = f"https://nounsai-audio.b-cdn.net/{db_melody['user_id']}/{db_melody['cdn_id']}-full.mp3"
                with requests.get(audio_url, stream=True) as response:
                    wav, sr = torchaudio.load(_hide_seek(response.raw))
                
                result = []
                first_iteration = True
                for audio_bytes, name in separate_audio_tracks(AUDIO_DICT, wav, sr):
                    if first_iteration:
                        update_audio_for_user(
                            id=db_audio['id'], 
                            user_id=db_audio['user_id'], 
                            name=f'{name}:::' + db_melody['name'], size=0, 
                            metadata={
                                'parent_id': db_melody['id'],
                                'mode': 'audio split'
                            }
                        )
                        upload_audio_to_cdn(db_audio["user_id"], db_audio['cdn_id'], audio_bytes)
                        first_iteration = False
                    else:
                        id, cdn_id = create_audio(
                            user_id=db_audio["user_id"], 
                            name=f'{name}:::' + db_melody['name'], 
                            size=0, 
                            state='PROCESSING',
                            metadata={
                                'parent_id': db_melody['id'],
                                'mode': 'audio split',
                                'split_main_id': db_audio['id'],
                            },
                        )
                        result.append({
                            'id': id,
                            'cdn_id': cdn_id,
                            'type': name
                        })
                        upload_audio_to_cdn(db_audio["user_id"], cdn_id, audio_bytes)
                        update_audio_state(id, 'COMPLETED')
                
                update_audio_metadata(id=db_audio['id'], metadata={
                                'parent_id': db_melody['id'],
                                'mode': 'audio split',
                                'result': result
                            })
                update_audio_state(db_audio['id'], 'COMPLETED')

            # audio continuation
            elif db_audio["metadata"]["mode"] == "audio extend":
                audio_id = db_audio["metadata"]["parent_id"]
                prompt = db_audio['metadata'].get('prompt', None)
                if prompt is not None:
                    prompt = [prompt]
                db_melody = fetch_audio_for_user(db_audio["user_id"], audio_id)
                if db_melody is None:
                    update_audio_state(db_audio['id'], 'ERROR')
                    continue
                audio_url = f"https://nounsai-audio.b-cdn.net/{db_melody['user_id']}/{db_melody['cdn_id']}-full.mp3"
                with requests.get(audio_url, stream=True) as response:
                    wav, sr = torchaudio.load(_hide_seek(response.raw))

                audio_bytes = continue_audio(AUDIO_DICT, prompt, wav, sr)

                # upload to cdn
                upload_audio_to_cdn(db_audio["user_id"], db_audio['cdn_id'], audio_bytes)

                # mark audio generation as complete
                update_audio_state(db_audio['id'], 'COMPLETED')
            
            # text to audio to audio
            else:
                db_melody = fetch_audio_for_user(db_audio["user_id"], db_audio["metadata"]["melody_id"])
                if db_melody is None:
                    update_audio_state(db_audio['id'], 'ERROR')
                    continue
                
                db_audio["metadata"]['parent_id'] = db_melody['id']
                db_audio["metadata"]['mode'] = 'melody to audio'

                melody_url = f"https://nounsai-audio.b-cdn.net/{db_audio['user_id']}/{db_melody['cdn_id']}-full.mp3"
                with requests.get(melody_url, stream=True) as response:
                    melody_wav, melody_sr = torchaudio.load(_hide_seek(response.raw))
                
                audio_bytes = txt_and_audio_to_audio(AUDIO_DICT, db_audio["metadata"]["prompt"], melody_wav, melody_sr)

                # upload to cdn
                upload_audio_to_cdn(db_audio["user_id"], db_audio['cdn_id'], audio_bytes)

                # mark audio generation as complete
                update_audio_state(db_audio['id'], 'COMPLETED')

            
            processing_time = str(timedelta(seconds=round((datetime.now() - start_time).total_seconds())))

        except Exception as e:
            print(traceback.format_exc())
            processing_time = str(timedelta(seconds=round((datetime.now() - start_time).total_seconds())))
            
            print(f"Error generating audio for project {audio['id']}: {e}")
            

if __name__ == '__main__':
    generate_audios()
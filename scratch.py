import os
import sys
import json
import boto3
import requests

if not os.path.isfile("config.json"):
    sys.exit("'config.json' not found! Please add it and try again.")
else:
    with open("config.json") as file:
        config = json.load(file)

client = boto3.client(
    's3',
    aws_access_key_id=config['aws_access_key_id'],
    aws_secret_access_key=config['aws_access_key_id']
)

print('client: ', client)
buckets =[]

response = client.list_buckets()
for bucket in response['Buckets']:
    print(bucket['Name'])
    try:
        for key in client.list_objects(Bucket=bucket['Name'])['Contents']:
            client.download_file(bucket['Name'], key['Key'], key['Key'])
            print(key['Key'])
    except Exception as e:
        print("error: ", e)


url = 'https://staging.roko.dev/api/audio_summary'
payload = {'key': 'e5bf9a1c8b257e0eb51a7e064b96bacbf6b51074.mp3', 'summary': 'bro this is an instrumental, what is there to summarize?'}
headers = {'content-type': 'application/json', 'challenge-token': config['steve_challenge_token']}
response = requests.post(url, json=payload, headers=headers)
print('response: ', response)
import requests
from flask_cors import CORS
from flask import Flask, request, Response

from db import fetch_api_hosts
from utils import fetch_env_config

app = Flask('__main__')
CORS(app)

config = fetch_env_config()

API_HOSTS = fetch_api_hosts()
COUNTER = 0

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST'])
def redirect_to_API_HOST(path):
    if 'challenge-token' not in request.headers or request.headers['challenge-token'] != config['challenge_token']:
        return '\'challenge-token\' header missing / invalid', 401

    global API_HOSTS
    global COUNTER

    if path == 'refresh_hosts':
        API_HOSTS = fetch_api_hosts()
        return Response('Success', 200, {API_HOSTS})
    
    api_host = API_HOSTS[COUNTER % len(API_HOSTS)]
    COUNTER = COUNTER + 1
    res = requests.request(
        method          = request.method,
        url             = request.url.replace(request.host_url, f'{api_host}/'),
        headers         = {k:v for k,v in request.headers},
        data            = request.data,
        cookies         = request.cookies,
        allow_redirects = False,
    )

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers          = [
        (k,v) for k,v in res.raw.headers.items()
        if k.lower() not in excluded_headers
    ]

    response = Response(res.content, res.status_code, headers)
    return response


if __name__ == '__main__':
    if config['environment'] == 'prod':
        app.run(host='0.0.0.0', port='5000', ssl_context='adhoc')
    else:
        app.run(host='0.0.0.0', debug=True, port='5000')


import secrets
import db
import json
import base64
import jwt
import datetime

from nouns_utils import fetch_env_config

config = fetch_env_config()

def create_dummy_user(email, metadata={}, referral_token=None, verified=True):
    verify_k = secrets.token_hex(48)

    password = "123password"

    conn = db.open_connection()
    cur = db.create_cursor(conn)
    cur.execute("INSERT INTO users (email, password, verify_key, metadata, referral_token, is_verified) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;", (email, password, verify_k, json.dumps(metadata), referral_token, verified))
    id = cur.fetchone()[0]
    db.close_cursor(cur)
    conn.commit()
    db.close_connection(conn)
    return id


def create_referral_reward():
    conn = db.open_connection()
    cur = db.create_cursor(conn)

    cur.execute(
        """
        INSERT INTO rewards (name, sql, metadata) VALUES (
            'referral_verify', 
            'With matching as (
                Select * From users
                Where id = %s
            )
            Select 
                Case 
                    when is_verified then 100
                    else 0
                End as amount,
                Case 
                    when is_verified then 30
                    else 0
                End as expires_at,
                Case 
                    when is_verified then true
                    else false
                End as complete
            From matching;',
            '{}'
        ) RETURNING id;
        """
    )
    id = cur.fetchone()[0]
    db.close_cursor(cur)
    conn.commit()
    db.close_connection(conn)
    return id


def get_access_token(id):

    token = jwt.encode({'id': id, 'exp': datetime.datetime.utcnow() + datetime.timedelta(days=300)}, config['secret_key'], algorithm='HS256')
    return token


class WrappedClient:
    def __init__(self, client):
        self.client = client;

    def get(self, url, access_code):
        headers = {
            'Authorization': access_code, 
            'challenge-token': config['challenge_token']
        }
        return self.client.get(url, headers=headers)

    def post(self, url, access_code, json={}):
        headers = {
            'Authorization': access_code, 
            'challenge-token': config['challenge_token']
        }
        return self.client.post(url, json=json, headers=headers)

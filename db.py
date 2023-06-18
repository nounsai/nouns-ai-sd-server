#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import os
import json
import base64
import psycopg2
import warnings
import threading
import pandas as pd
import psycopg2.extras
import uuid
import datetime

from cdn import upload_image_to_cdn, delete_image_from_cdn, upload_audio_to_cdn, delete_audio_from_cdn

from configparser import ConfigParser

warnings.simplefilter(action='ignore', category=UserWarning)
from utils import fetch_env_config

config = fetch_env_config()

SAVE_IMAGES_TO_DATABASE = config['save_image_to_database']
DEFAULT_CREDIT_EXPIRATION = config['default_credit_expiration']
LOGGING = False


def config(section='postgresql'):

    if os.path.exists(str(os.path.dirname(os.path.realpath(__file__))) + "/database.local.ini"):
        filename = str(os.path.dirname(os.path.realpath(__file__))) + "/database.local.ini"
    else:
        filename = str(os.path.dirname(os.path.realpath(__file__))) + "/database.ini"

    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


########################################################
####################### HELPERS ########################
########################################################


# Open the connection with PostgreSQL
def open_connection():

    conn = None
    try:
        params = config()
        if (LOGGING):
            print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


# Close the connection with PostgreSQL
def close_connection(conn):

    conn.close()
    if (LOGGING):
        print('Database connection closed.')


# Open the cursor with PostgreSQL
def create_cursor(conn):

    cur = conn.cursor()
    return cur


# Close the cursor with PostgreSQL
def close_cursor(cur):

    cur.close()


def sanitize_string(string):

    return string.replace("'", "''")


#######################################################
######################## USERS ########################
#######################################################


def create_user(email, password, verify_key, metadata):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO users (email, password, verify_key, metadata) VALUES (%s, %s, %s, %s) RETURNING id;", (email, password, verify_key, json.dumps(metadata)))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id


def fetch_user_for_email(email):

    conn = open_connection()
    cur = create_cursor(conn)
    sql = "SELECT * FROM users WHERE email=%s;"
    users_df = pd.read_sql_query(sql, conn, params=[email])
    close_connection(conn)
    try:
        return json.loads(users_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_user_for_verify_key(verify_key):
    conn = open_connection()
    cur = create_cursor(conn)
    sql = "SELECT * FROM users WHERE verify_key=%s AND is_verified=FALSE;"
    users_df = pd.read_sql_query(sql, conn, params=[verify_key])
    close_connection(conn)
    try:
        return json.loads(users_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_user_for_referral_token(token):

    conn = open_connection()
    sql = "SELECT * FROM users WHERE referral_token=%s;"
    users_df = pd.read_sql_query(sql, conn, params=[token])
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))[0]


def verify_user_for_id(user_id):
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE users SET verify_key=NULL, is_verified=TRUE WHERE id=%s;", [user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def create_password_reset_for_user(email, reset_key):
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("SELECT * FROM users WHERE email=%s;", [email])
    users = cur.fetchall()

    if len(users) == 0:
        close_cursor(cur)
        close_connection(conn)
        return False, 'User not found'

    user = users[0]
    cur.execute("INSERT INTO password_recovery (user_id, reset_key) VALUES (%s, %s);", [user[0], reset_key])

    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return True, 'success'


def get_password_reset(reset_key):
    conn = open_connection()

    sql = "SELECT * FROM password_recovery WHERE reset_key=%s;"
    users_df = pd.read_sql_query(sql, conn, params=[reset_key])
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))[0]


def verify_password_reset(reset_key, new_password_hash):
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("SELECT * FROM password_recovery WHERE reset_key=%s;", [reset_key])
    rows = cur.fetchall()

    if len(rows) == 0:
        close_cursor(cur)
        close_connection(conn)
        return False, 'Reset key not found'

    user_id = rows[0][1]

    cur.execute("UPDATE users SET password=%s WHERE id=%s", [new_password_hash, user_id])
    cur.execute("DELETE FROM password_recovery WHERE id=%s", [rows[0][0]])

    close_cursor(cur)
    conn.commit()
    close_connection(conn)

    return True, 'success'


def fetch_user(id):

    conn = open_connection()
    sql = "SELECT * FROM users WHERE id=%s;"
    users_df = pd.read_sql_query(sql, conn, params=[id])
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))[0]


def update_user(id, password, metadata):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE users SET password=%s, metadata=%s WHERE id=%s;", [password, json.dumps(metadata), id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def update_user_referral_token(id, token):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE users SET referral_token=%s WHERE id=%s;", [token, id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def update_user_metadata(id, metadata):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE users SET metadata=%s WHERE id=%s;", [json.dumps(metadata), id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def delete_user(id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM users WHERE id=%s;", [id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


########################################################
####################### REFERRALS ######################
########################################################


def create_referral(referrer_id, referred_id, metadata):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO referrals (referrer_id, referred_id, metadata) VALUES (%s, %s, %s) RETURNING id;", (referrer_id, referred_id, json.dumps(metadata)))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id


def fetch_referral_for_referred(referred_id):

    conn = open_connection()
    sql = "SELECT * FROM referrals WHERE referred_id=%s;"
    users_df = pd.read_sql_query(sql, conn, params=[referred_id])
    close_connection(conn)

    referrals = json.loads(users_df.to_json(orient="records"))

    if len(referrals) == 0:
        return None
    else:
        return referrals[0]
    

def delete_referral(id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM referrals WHERE id=%s;", [id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


########################################################
######################## REWARDS #######################
########################################################


def fetch_reward(name):

    conn = open_connection()
    sql = "SELECT * FROM rewards WHERE name=%s;"
    users_df = pd.read_sql_query(sql, conn, params=[name])
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))[0]


def execute_reward(name, *args):

    reward = fetch_reward(name)
    sql_query = reward['sql']

    conn = open_connection()
    results_df = pd.read_sql_query(sql_query, conn, params=args)
    close_connection(conn)
    result = json.loads(results_df.to_json(orient="records"))[0]

    # convert expires_at from # of days to datetime
    if 'expires_at' in result:
        result['expires_at'] = datetime.datetime.utcnow() + datetime.timedelta(days=result['expires_at'])
    else:
        result['expires_at'] = datetime.datetime.utcnow() + datetime.timedelta(days=DEFAULT_CREDIT_EXPIRATION)

    return result


def delete_reward(id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM rewards WHERE id=%s;", [id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


########################################################
##################### TRANSACTIONS #####################
########################################################


def create_transaction(
    user_id, amount, metadata={}, expires_at=None, memo='', amount_remaining=0, t_type='credit'
):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute(
        "INSERT INTO transactions (user_id, amount, amount_remaining, memo, metadata, expires_at, type) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;", (
            user_id,
            amount, 
            amount_remaining,
            memo,
            json.dumps(metadata), 
            expires_at,
            t_type
        )
    )
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id


def fetch_transactions_for_user(id, expired=False):

    conn = open_connection()
    sql = "SELECT * FROM transactions WHERE user_id=%s"
    params = [id]
    if not expired:
        sql += " AND expires_at > %s;"
        params.append(datetime.datetime.utcnow())
    else:
        sql += ";"
    users_df = pd.read_sql_query(sql, conn, params=params)
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))


########################################################
######################## IMAGES ########################
########################################################


def create_image(user_id, image_byte_data, thumbnail_byte_data, hash, metadata, is_public=False, is_liked=False, parent_id=0, use_thread=True):
    image_cdn_uuid = str(uuid.uuid4())

    # save to database
    conn = open_connection()
    cur = create_cursor(conn)
    
    user_images_with_hash = fetch_images_for_user_with_hash(user_id, hash)
    if len(user_images_with_hash) > 0:
        return user_images_with_hash[0]['id']
    
    sql = "INSERT INTO images (user_id, base_64, thumb_base_64, hash, metadata, cdn_id, is_public, is_liked, parent_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;"
    fields = [user_id, '0', '0', hash, json.dumps(metadata), image_cdn_uuid, is_public, is_liked, parent_id]

    cur.execute(sql, fields)
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)

    if use_thread:
        # Start a new thread for the slow save operation
        save_thread = threading.Thread(target=upload_image_to_cdn, args=(user_id, image_cdn_uuid, image_byte_data, thumbnail_byte_data))
        save_thread.start()
    else:
        upload_image_to_cdn(
            user_id,
            image_cdn_uuid,
            image_byte_data,
            thumbnail_byte_data
        )

    return id


def fetch_images(limit, offset):

    conn = open_connection()
    sql = "SELECT * FROM images where is_public=True ORDER BY id DESC LIMIT %s OFFSET %s;"
    images_df = pd.read_sql_query(sql, conn, params=[limit, offset])
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def fetch_image(id):

    conn = open_connection()
    sql = "SELECT * FROM images WHERE id=%s;"
    images_df = pd.read_sql_query(sql, conn, params=[id])
    close_connection(conn)
    try:
        return json.loads(images_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_images_with_hash(hash):

    conn = open_connection()
    sql = "SELECT * FROM images where hash=%s;"
    images_df = pd.read_sql_query(sql, conn, params=[hash])
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def fetch_images_for_user_with_hash(user_id, hash):

    conn = open_connection()
    sql = "SELECT * FROM images where user_id=%s and hash=%s;"
    images_df = pd.read_sql_query(sql, conn, params=[user_id, hash])
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def fetch_images_for_user(user_id, limit, offset, favorited):

    conn = open_connection()
    sql = "SELECT * FROM images where user_id=%s {} ORDER BY id DESC LIMIT %s OFFSET %s;".format("AND is_liked=True" if favorited else "")
    images_df = pd.read_sql_query(sql, conn, params=[user_id, limit, offset])
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def fetch_image_ids_for_user(user_id):

    conn = open_connection()
    sql = "SELECT id FROM images where user_id=%s;"
    image_ids_df = pd.read_sql_query(sql, conn, params=[user_id])
    close_connection(conn)
    return json.loads(image_ids_df.to_json(orient="records"))


def fetch_image_for_user(id, user_id):

    conn = open_connection()
    sql = "SELECT * FROM images WHERE id=%s and user_id=%s;"
    images_df = pd.read_sql_query(sql, conn, params=[id, user_id])
    close_connection(conn)
    try:
        return json.loads(images_df.to_json(orient="records"))[0]
    except Exception as e:
        return None
    

def fetch_images_for_ids(ids):
    conn = open_connection()
    sql = "SELECT i.* FROM images i JOIN unnest(%s::int[]) WITH ORDINALITY t(id, ord) USING (id) ORDER BY t.ord;"
    images_df = pd.read_sql_query(sql, conn, params=['{%s}' % ','.join(map(str, ids))])
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def update_image_for_user(id, user_id, is_public, is_liked):

    conn = open_connection()
    cur = create_cursor(conn)
    
    if is_public != None:
        cur.execute("UPDATE images SET is_public=%s WHERE id=%s and user_id=%s;", [is_public, id, user_id])
    if is_liked != None:
        cur.execute("UPDATE images SET is_liked=%s WHERE id=%s and user_id=%s;", [is_liked, id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def delete_image_for_user(id, user_id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM images WHERE id=%s and user_id=%s;", [id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


#######################################################
######################## AUDIO ########################
#######################################################


def create_audio(user_id, audio_byte_data, name, size, metadata):
    audio_cdn_uuid = str(uuid.uuid4())

    # save to database
    conn = open_connection()
    cur = create_cursor(conn)
    
    sql = "INSERT INTO audio (user_id, name, size, metadata, cdn_id) VALUES (%s, %s, %s, %s, %s) RETURNING id;"
    fields = [user_id, name, size, json.dumps(metadata), audio_cdn_uuid]

    cur.execute(sql, fields)
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)

    # Start a new thread for the slow save operation
    save_thread = threading.Thread(target=upload_audio_to_cdn, args=(user_id, audio_cdn_uuid, audio_byte_data))
    save_thread.start()

    return id


def fetch_audios():

    conn = open_connection()
    sql = "SELECT * FROM audio ORDER BY id ASC;"
    audio_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(audio_df.to_json(orient="records"))


def fetch_audios_for_user(user_id, limit, offset):

    conn = open_connection()
    sql = "SELECT * FROM audio where user_id=%s ORDER BY id DESC LIMIT %s OFFSET %s;"
    audio_df = pd.read_sql_query(sql, conn, params=[user_id, limit, offset])
    close_connection(conn)
    return json.loads(audio_df.to_json(orient="records"))


def fetch_audio_for_user(user_id, id):

    conn = open_connection()
    sql = "SELECT * FROM audio WHERE id=%s and user_id=%s;"
    audio_df = pd.read_sql_query(sql, conn, params=[id, user_id])
    close_connection(conn)
    try:
        return json.loads(audio_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def update_audio_for_user(id, user_id, name, url, size, metadata):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE audio SET name=%s, url=%s, size=%s, metadata=%s WHERE id=%s and user_id=%s;", [name, url, size, json.dumps(metadata), id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def delete_audio_and_video_project_for_user(user_id, id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("SELECT * FROM audio WHERE id=%s and user_id=%s;", [id, user_id])
    audio_cdn_id = cur.fetchone()[5]
    cur.execute("DELETE FROM audio WHERE id=%s and user_id=%s;", [id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    
    delete_thread = threading.Thread(target=delete_audio_from_cdn, args=(user_id, audio_cdn_id))
    delete_thread.start()


#######################################################
######################## LINKS ########################
#######################################################


def create_link(user_id, metadata):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO links (user_id, metadata) VALUES (%s, %s) RETURNING id;", (user_id, json.dumps(metadata)))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id


def fetch_links():

    conn = open_connection()
    sql = "SELECT * FROM links ORDER BY id ASC;"
    links_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(links_df.to_json(orient="records"))


def fetch_link(id):

    conn = open_connection()
    sql = "SELECT * FROM links WHERE id=%s;"
    links_df = pd.read_sql_query(sql, conn, params=[id])
    close_connection(conn)
    try:
        return json.loads(links_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_links_for_user(user_id, limit, offset):

    conn = open_connection()
    sql = "SELECT * FROM links where user_id=%s ORDER BY id DESC LIMIT %s OFFSET %s;"
    links_df = pd.read_sql_query(sql, conn, params=[user_id, limit, offset])
    close_connection(conn)
    return json.loads(links_df.to_json(orient="records"))


def update_link_for_user(id, user_id, metadata):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE links SET metadata=%s WHERE id=%s and user_id=%s;", [json.dumps(metadata), id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def delete_link_for_user(id, user_id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM links WHERE id=%s and user_id=%s;", [id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


########################################################
######################## VIDEOS ########################
########################################################


def create_video(user_id, metadata):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO videos (user_id, metadata) VALUES (%s, %s) RETURNING id;", (user_id, json.dumps(metadata)))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id


def fetch_videos():

    conn = open_connection()
    sql = "SELECT * FROM videos;"
    videos_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(videos_df.to_json(orient="records"))


def fetch_video(id):

    conn = open_connection()
    sql = "SELECT * FROM videos WHERE id=%s;"
    videos_df = pd.read_sql_query(sql, conn, params=[id])
    close_connection(conn)
    try:
        return json.loads(videos_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_videos_for_user(user_id, limit, offset):

    conn = open_connection()
    sql = "SELECT * FROM videos where user_id=%s ORDER BY id DESC LIMIT %s OFFSET %s;"
    videos_df = pd.read_sql_query(sql, conn, params=[user_id, limit, offset])
    close_connection(conn)
    return json.loads(videos_df.to_json(orient="records"))


def fetch_video_for_user(id, user_id):

    conn = open_connection()
    sql = "SELECT * FROM videos WHERE id=%s and user_id=%s;"
    videos_df = pd.read_sql_query(sql, conn, params=[id, user_id])
    close_connection(conn)
    try:
        return json.loads(videos_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def update_video_for_user(id, user_id, metadata):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE videos SET metadata=%s WHERE id=%s and user_id=%s;", [json.dumps(metadata), id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def delete_video_for_user(id, user_id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM videos WHERE id=%s and user_id=%s;", [id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)

    
########################################################
################### VIDEO PROJECTS #####################
########################################################


def create_video_project(user_id, audio_id, metadata):

    # save to database
    conn = open_connection()
    cur = create_cursor(conn)
    
    sql = "INSERT INTO video_projects (user_id, audio_id, metadata, cdn_id) VALUES (%s, %s, %s, %s) RETURNING id;"
    fields = [user_id, audio_id, json.dumps(metadata), str(uuid.uuid4())]

    cur.execute(sql, fields)
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)

    return id


def fetch_video_projects():

    conn = open_connection()
    sql = "SELECT * FROM video_projects ORDER BY id ASC;"
    video_projects_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(video_projects_df.to_json(orient="records"))


def fetch_video_project_for_id(id):
    conn = open_connection()
    sql = "SELECT * FROM video_projects WHERE id=%s;"
    project_df = pd.read_sql_query(sql, conn, params=[id])
    close_connection(conn)
    try:
        return json.loads(project_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_video_projects_for_user(user_id, limit, offset):

    conn = open_connection()
    sql = "SELECT * FROM video_projects where user_id=%s ORDER BY id DESC LIMIT %s OFFSET %s;"
    video_projects_df = pd.read_sql_query(sql, conn, params=[user_id, limit, offset])
    close_connection(conn)
    return json.loads(video_projects_df.to_json(orient="records"))


def fetch_video_project_for_user(user_id, id):

    conn = open_connection()
    sql = "SELECT * FROM video_projects WHERE id=%s and user_id=%s;"
    video_project_df = pd.read_sql_query(sql, conn, params=[id, user_id])
    close_connection(conn)
    try:
        return json.loads(video_project_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_queued_video_projects():

    conn = open_connection()
    sql = "SELECT * FROM video_projects WHERE state like 'QUEUED';"
    video_projects_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    try:
        return json.loads(video_projects_df.to_json(orient="records"))
    except Exception as e:
        return None


def update_video_project_for_user(user_id, id, metadata):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE video_projects SET metadata=%s, updated_at=NOW() WHERE id=%s and user_id=%s;", [json.dumps(metadata), id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def update_video_project_state(id, state):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE video_projects SET state=%s WHERE id=%s;", [state, id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def delete_video_project_for_user(user_id, id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM video_projects WHERE id=%s and user_id=%s;", [id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


########################################################
###################### API HOSTS #######################
########################################################

def fetch_api_hosts():

    conn = open_connection()
    sql = "SELECT * FROM api_hosts;"
    api_hosts_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return api_hosts_df['address'].array

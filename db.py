#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import os
import json
import psycopg2
import warnings
import pandas as pd
import psycopg2.extras
from configparser import ConfigParser

warnings.simplefilter(action='ignore', category=UserWarning)

LOGGING = False

def config(section='postgresql'):

    if os.path.exists(str(os.path.dirname(os.path.realpath(__file__))) + "/database.local.ini"):
        filename = str(os.path.dirname(os.path.realpath(__file__))) + "/database.local.ini"
    else:
        filename = str(os.path.dirname(os.path.realpath(__file__))) + "/database.ini"

    parser = ConfigParser()
    parser.read(filename)
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


def close_connection(conn):

    conn.close()
    if (LOGGING):
        print('Database connection closed.')


def create_cursor(conn):

    cur = conn.cursor()
    return cur


def close_cursor(cur):

    cur.close()

########################################################
######################## MODELS ########################
########################################################

def fetch_models():

    conn = open_connection()
    sql = "SELECT * FROM models order by id asc;"
    models_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(models_df.to_json(orient="records"))


def fetch_model_by_model_id(model_id):

    conn = open_connection()
    sql = "SELECT * FROM models where model_id='{}' order by id desc;".format(model_id)
    models_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(models_df.to_json(orient="records"))[0]


def fetch_model_by_id(id):

    conn = open_connection()
    sql = "SELECT * FROM models WHERE id={};".format(id)
    models_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(models_df.to_json(orient="records"))[0]


def add_model(model_id):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO models (model_id) VALUES (%s);", (model_id, ))
    close_cursor(cur)
    conn.commit()
    close_connection(conn)

########################################################
######################### USERS ########################
########################################################

def fetch_users():

    conn = open_connection()
    sql = "SELECT * FROM users order by id asc;"
    users_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))


def fetch_user_by_email(email):

    conn = open_connection()
    sql = "SELECT * FROM users where email='{}' order by id desc;".format(email)
    users_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))[0]


def fetch_user_by_id(id):

    conn = open_connection()
    sql = "SELECT * FROM users WHERE id={};".format(id)
    users_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(users_df.to_json(orient="records"))[0]


def add_user(email, password_hash):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id;", (email, password_hash))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id

########################################################
######################## IMAGES ########################
########################################################

def fetch_images():

    conn = open_connection()
    sql = "SELECT * FROM images order by id asc;"
    images_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def fetch_images_for_user(user_id):

    conn = open_connection()
    sql = "SELECT * FROM images where user_id={} order by id desc;".format(user_id)
    images_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def fetch_image_by_id(id):

    conn = open_connection()
    sql = "SELECT * FROM images WHERE id={};".format(id)
    images_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))[0]


def delete_image_by_id(id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM images WHERE id={};".format(id))
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def fetch_image_by_hash(hash):

    conn = open_connection()
    sql = "SELECT * FROM images WHERE image_hash='{}';".format(hash)
    images_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))[0]


def add_image(user_id, model_id, prompt, negative_prompt, steps, seed, base_64, image_hash, aspect_ratio):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO images (user_id, model_id, prompt, negative_prompt, steps, seed, base_64, image_hash, aspect_ratio) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;", (user_id, model_id, prompt, negative_prompt, steps, seed, base_64, image_hash, aspect_ratio))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id

########################################################
####################### REQUESTS #######################
########################################################

def fetch_requests():

    conn = open_connection()
    sql = "SELECT * FROM requests order by id asc;"
    requests_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(requests_df.to_json(orient="records"))


def fetch_requests_for_user(user_id):

    conn = open_connection()
    sql = "SELECT * FROM requests where user_id={} order by id desc;".format(user_id)
    requests_df = pd.read_sql_query(sql, conn)
    requests_df['config'] = requests_df['config'].replace("'","''")
    close_connection(conn)
    return json.loads(requests_df.to_json(orient="records"))


def fetch_request_by_id(id):

    conn = open_connection()
    sql = "SELECT * FROM requests WHERE id={};".format(id)
    requests_df = pd.read_sql_query(sql, conn)
    requests_df['config'] = requests_df['config'].replace("'","''")
    close_connection(conn)
    return json.loads(requests_df.to_json(orient="records"))[0]


def delete_request_by_id(id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM requests WHERE id={};".format(id))
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def fetch_request_by_hash(hash):

    conn = open_connection()
    sql = "SELECT * FROM requests WHERE config_hash='{}';".format(hash)
    requests_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(requests_df.to_json(orient="records"))[0]


def add_request(user_id, model_id, aspect_ratio, config, config_hash):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO requests (user_id, model_id, aspect_ratio, config, config_hash) VALUES ({}, \'{}\', \'{}\', \'{}\', \'{}\') RETURNING id;".format(user_id, model_id, aspect_ratio, config, config_hash))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id

########################################################
######################## AUDIO #########################
########################################################

def fetch_audio():

    conn = open_connection()
    sql = "SELECT * FROM audio order by id asc;"
    audio_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(audio_df.to_json(orient="records"))


def fetch_audio_for_user(user_id):

    conn = open_connection()
    sql = "SELECT * FROM audio where user_id={} order by id desc;".format(user_id)
    audio_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(audio_df.to_json(orient="records"))


def fetch_audio_by_id(id):

    conn = open_connection()
    sql = "SELECT * FROM audio WHERE id={};".format(id)
    audio_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(audio_df.to_json(orient="records"))[0]


def delete_audio_by_id(id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM audio WHERE id={};".format(id))
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def add_audio(user_id, name, url, size):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO audio (user_id, name, url, size) VALUES (%s, %s, %s, %s) RETURNING id;", (user_id, name, url, size))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id

########################################################
######################### CODES ########################
########################################################

def fetch_code_by_hash(hash):

    conn = open_connection()
    sql = "SELECT * FROM codes where code='{}' and valid=true;".format(hash)
    codes_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(codes_df.to_json(orient="records"))


def update_code_by_hash(hash):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE codes SET valid=false where code='{}';".format(hash))
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


def add_code(hash):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO codes (code) VALUES ('{}') RETURNING id;".format(hash))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id

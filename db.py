#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-

import os
import cv2
import json
import numpy
import base64
import hashlib
import psycopg2
import warnings
import pandas as pd
import psycopg2.extras

from PIL import Image
from io import BytesIO
from configparser import ConfigParser

warnings.simplefilter(action='ignore', category=UserWarning)

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


def add_image(user_id, model_id, prompt, steps, seed, base_64, image_hash):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO images (user_id, model_id, prompt, steps, seed, base_64, image_hash) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;", (user_id, model_id, prompt, steps, seed, base_64, image_hash))
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
    close_connection(conn)
    return json.loads(requests_df.to_json(orient="records"))


def fetch_request_by_id(id):

    conn = open_connection()
    sql = "SELECT * FROM requests WHERE id={};".format(id)
    requests_df = pd.read_sql_query(sql, conn)
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


def add_request(user_id, model_id, config, config_hash):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO requests (user_id, model_id, config, config_hash) VALUES (%s, %s, %s, %s) RETURNING id;", (user_id, model_id, json.dumps(config), config_hash))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id


########################################################
#################### TEST COMMANDS #####################
########################################################

# id = add_user('eolszewski@gmail.com', '2d90fed25bb113e4bfbe0bd33cbe1f54070af97d0f2bee3ef84514a062c96d23')
# add_model('sd-dreambooth-library/noggles-sd15-800-4e6')

# with open("lizard.jpg", "rb") as image_file:
#     base_64_encoded = base64.b64encode(image_file.read())
#     base_64 = base_64_encoded.decode("utf-8") 
#     hash = hashlib.sha256(base_64_encoded).hexdigest()
#     id = add_image(
#         1, 
#         1, 
#         'a portrait of a hi definition 4k cartoon lizard wearing noggles (red frame black white lens), very detailed, ultra realistic, extremely high detail, unreal engine, very detailed', 
#         50, 
#         1232143448, 
#         base_64, 
#         hash
#     )

# image = fetch_images()[0]
# print(image['base_64'])
# image_object = Image.open(BytesIO(base64.b64decode(image['base_64'])))
# image_object.save("liz.jpg")
# delete_image_by_id(fetch_images()[0]['id'])

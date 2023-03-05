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


def create_user(email, password, metadata):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO users (email, password, metadata) VALUES (%s, %s, %s) RETURNING id;", (email, password, json.dumps(metadata)))
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


def delete_user(id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM users WHERE id=%s;", [id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


########################################################
######################## IMAGES ########################
########################################################


def create_image(user_id, base_64, hash, metadata):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO images (user_id, base_64, hash, metadata) VALUES (%s, %s, %s, %s) RETURNING id;", (user_id, json.dumps(base_64), hash, json.dumps(metadata)))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
    return id


def fetch_images():

    conn = open_connection()
    sql = "SELECT * FROM images ORDER BY id ASC;"
    images_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(images_df.to_json(orient="records"))


def fetch_images_for_user(user_id, limit, offset):

    conn = open_connection()
    sql = "SELECT * FROM images where user_id=%s ORDER BY id DESC LIMIT %s OFFSET %s;"
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


def update_image_for_user(id, user_id, base_64, hash, metadata):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("UPDATE images SET base_64=%s, hash=%s, metadata=%s WHERE id=%s and user_id=%s;", [json.dumps(base_64), hash, json.dumps(metadata), id, user_id])
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


def create_audio(user_id, name, url, size, metadata):
    
    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("INSERT INTO audio (user_id, name, url, size, metadata) VALUES (%s, %s, %s, %s, %s) RETURNING id;", (user_id, name, url, size, json.dumps(metadata)))
    id = cur.fetchone()[0]
    close_cursor(cur)
    conn.commit()
    close_connection(conn)
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


def fetch_audio_for_user(id, user_id):

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


def delete_audio_for_user(id, user_id):

    conn = open_connection()
    cur = create_cursor(conn)
    cur.execute("DELETE FROM audio WHERE id=%s and user_id=%s;", [id, user_id])
    close_cursor(cur)
    conn.commit()
    close_connection(conn)


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


def fetch_video_for_user(id, user_id):

    conn = open_connection()
    sql = "SELECT * FROM videos WHERE id=%s and user_id=%s;"
    videos_df = pd.read_sql_query(sql, conn, params=[id, user_id])
    close_connection(conn)
    try:
        return json.loads(videos_df.to_json(orient="records"))[0]
    except Exception as e:
        return None


def fetch_videos():

    conn = open_connection()
    sql = "SELECT * FROM videos;"
    videos_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return json.loads(videos_df.to_json(orient="records"))


def fetch_videos_for_user(user_id, limit, offset):

    conn = open_connection()
    sql = "SELECT * FROM videos where user_id=%s ORDER BY id DESC LIMIT %s OFFSET %s;"
    videos_df = pd.read_sql_query(sql, conn, params=[user_id, limit, offset])
    close_connection(conn)
    return json.loads(videos_df.to_json(orient="records"))


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
###################### API HOSTS #######################
########################################################

def fetch_api_hosts():

    conn = open_connection()
    sql = "SELECT * FROM api_hosts;"
    api_hosts_df = pd.read_sql_query(sql, conn)
    close_connection(conn)
    return api_hosts_df['address'].array

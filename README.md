# Nouns AI SD Server

More to come to this page, soon!

`gunicorn --workers 4 --bind 0.0.0.0:5000 wsgi:app --timeout 90`


# Database

## Models

```
CREATE TABLE models (
	id SERIAL NOT NULL PRIMARY KEY,
    model_id VARCHAR (128) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (model_id)
);
```

## Users

```
CREATE TABLE users (
	id SERIAL NOT NULL PRIMARY KEY,
    email VARCHAR (30) NOT NULL,
    password_hash VARCHAR (128) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (email)
);
```

## Images

```
CREATE TABLE images (
	id SERIAL NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    steps INTEGER NOT NULL,
    seed BIGINT NOT NULL,
    base_64 TEXT NOT NULL,
    image_hash VARCHAR (256) NOT NULL,
    aspect_ratio INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, model_id, image_hash)
);
```

## Audio

```
CREATE TABLE audio (
	id SERIAL NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    name VARCHAR (256) NOT NULL,
    url VARCHAR (256) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, name, url)
);
```

## Video Requests

```
CREATE TABLE requests (
	id SERIAL NOT NULL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    model_id INTEGER NOT NULL,
    config TEXT NOT NULL,
    config_hash VARCHAR (256) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    queued BOOLEAN NOT NULL DEFAULT true,
    UNIQUE (user_id, model_id, config_hash)
);
```

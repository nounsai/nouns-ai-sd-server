# Nouns AI SD Server

## Activating / Deactivating Virtual Environment
```
source nouns-ai/bin/activate
deactivate
```

## Server Config
### Setup
[How we set up the server with Flask, Nginx, Gunicorn, and CertBot](https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-22-04)

### Manually starting the server with gunicorn
`gunicorn --bind 0.0.0.0:5000 wsgi:app --timeout 30000`

### Starting the server
`sudo systemctl start nouns-ai-sd-server`

### Stopping the server
`sudo systemctl stop nouns-ai-sd-server`

### Restarting the server
`sudo systemctl restart nouns-ai-sd-server`

_Note: This utilizes `/etc/systemd/system/nouns-ai-sd-server.service`_

# Database

## Users
```
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY,
    email VARCHAR(128) NOT NULL UNIQUE,
    password VARCHAR(128) NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(id)
);
```

## Images
```
CREATE TABLE images (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    base_64 TEXT NOT NULL,
    thumb_base_64 TEXT NOT NULL,
    hash VARCHAR (256) NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

## Audio
```
CREATE TABLE audio (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    name VARCHAR(256) NOT NULL,
    url VARCHAR(256) NOT NULL,
    size BIGINT NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id),
    UNIQUE (user_id, name, url),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

## Videos
```
CREATE TABLE videos (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

## Links
CREATE TABLE links (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

## Models

```
CREATE TABLE models (
    id INT GENERATED ALWAYS AS IDENTITY,
    model_id VARCHAR (128) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id)
);
```

## API Hosts

```
CREATE TABLE api_hosts (
    id INT GENERATED ALWAYS AS IDENTITY,
    address VARCHAR (256) NOT NULL UNIQUE,
    PRIMARY KEY(id)
);
```
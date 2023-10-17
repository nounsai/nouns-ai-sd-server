# Nouns AI SD Server

**Disclaimer: This repository is the result of months of cowboy coding in a fast-paced environment where things would regularly break due to packages not being pinned down, conflicting dependencies, etc...**

# High Level Architecture

Nouns AI has two entrypoints `proxy.nounsai.wtf` and `api-cpu.nounsai.wtf` (both Flask servers). 

The former is a pseudo-load-balancer that was created because, as we were scaling, we ran into issues parallelizing jobs on single multi-GPU machines, we could also make the our system more robust with multiple instances running on [Lambda Labs](https://lambdalabs.com/) (the cheapest GPU provider, at the time). This proxy can be found in `proxy.py` in the root folder of the project.

Early on in our development, we had everything running on the GPUs which caused there to be significant bottlenecks for simple user interactions (e.g. login, reset password,...). As a result of this, we created a standalone server for handling requests not dealing with GPUs. This server is spun up with the same `server.py` file, but a `server_type` environment variable in the `config.json` file inhibits any GPU-specific loading.

# Environment Setup

## Database

In this project, we use postgres. You will need to to update your `database.ini` file to point to your postgres instance. And, no, you cannot use ours because I changed the password and didn't commit the update to here (I never thought this would be OSS). 

Below are the create commands for each of the tables in our database:

#### Users

```sql
CREATE TABLE users (
    id INT GENERATED ALWAYS AS IDENTITY,
    email VARCHAR(128) NOT NULL UNIQUE,
    password VARCHAR(128) NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE NOT NULL,
    verify_key VARCHAR(128),
    referral_token VARCHAR(128),
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(id)
);
```

#### Password Resets

```sql
CREATE TABLE password_recovery (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    reset_key VARCHAR(128),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(id)
);
```

#### Images

```sql
CREATE TABLE images (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    base_64 TEXT NOT NULL,
    thumb_base_64 TEXT NOT NULL,
    hash VARCHAR (256) NOT NULL,
    metadata JSON NOT NULL,
    cdn_id VARCHAR (36),
    is_public BOOLEAN DEFAULT FALSE NOT NULL,
    is_liked BOOLEAN DEFAULT FALSE NOT NULL,
    parent_id INT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

#### Audio

```sql
CREATE TABLE audio (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    name VARCHAR(256) NOT NULL,
    cdn_id VARCHAR(256) NOT NULL,
    size BIGINT NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    state VARCHAR(36) DEFAULT 'UNFINISHED',
    PRIMARY KEY(id),
    UNIQUE (user_id, name, url),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

#### Videos

```sql
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

#### Links

```sql
CREATE TABLE links (
    id INT GENERATED ALWAYS AS IDENTITY,
    user_id INT NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

#### Models

```sql
CREATE TABLE models (
    id INT GENERATED ALWAYS AS IDENTITY,
    model_id VARCHAR (128) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id)
);
```

#### API Hosts

```sql
CREATE TABLE api_hosts (
    id INT GENERATED ALWAYS AS IDENTITY,
    address VARCHAR (256) NOT NULL UNIQUE,
    PRIMARY KEY(id)
);
```

#### Referrals

```sql
CREATE TABLE referrals (
    id INT GENERATED ALWAYS AS IDENTITY,
    referrer_id INT NOT NULL,
    referred_id INT NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id),
    CONSTRAINT fk_referrer FOREIGN KEY(referrer_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT fk_referred FOREIGN KEY(referred_id) REFERENCES users(id) ON DELETE CASCADE
);
```

#### Rewards

```sql
CREATE TABLE rewards (
    id INT GENERATED ALWAYS AS IDENTITY,
    name VARCHAR (128) NOT NULL,
    sql TEXT NOT NULL,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY(id)
);
```

#### Transactions

```sql
CREATE TABLE transactions (
    id INT GENERATED ALWAYS AS IDENTITY,
    metadata JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP,
    user_id INT NOT NULL,
    type VARCHAR (128) NOT NULL,
    amount INT NOT NULL,
    amount_remaining INT NOT NULL,
    memo VARCHAR (256) NOT NULL,
    PRIMARY KEY(id),
    CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

#### Video Projects

```sql
CREATE TABLE video_projects (
  id INT GENERATED ALWAYS AS IDENTITY,
  user_id INT NOT NULL,
  audio_id INT NOT NULL REFERENCES audio(id) ON DELETE CASCADE,
  metadata JSON NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
  state VARCHAR(36) DEFAULT 'UNFINISHED',
  cdn_id VARCHAR(256) NOT NULL,
  PRIMARY KEY(id),
  CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
```


## Local Environment

#### config.json

This file is found in the root of the project and is very much akin to a `.env` file. This project has a number of integrations and requires a number of accounts / services to work. Here are all the keys that you're going to need to be running this:

```
{
    // [DEPRECATED] Dropbox was used for storing some media until we dropped it for BunnyCDN
    "dropbox_api_key": "",
    "dropbox_api_secret": "",
    "dropbox_refresh_token": "",
    // BunnyCDN is used for storing media
    "storage_location_id": "",
    "storage_zone": "",
    "storage_access_key": "",
    "storage_zone_audio": "",
    "storage_access_key_audio": "",
    "storage_zone_video": "",
    "storage_access_key_video": "",
    // Used for HuggingFace integration
    "huggingface_token": "",
    // SendGrid used for emailing users
    "sendgrid_api_key": "",
    // Replicate used for on-the-fly GPU allocation for image generation / editing
    "replicate_api_key": "",
    // Discord used to notify users when their video is ready / finished processing
    "video_generation_discord_webhook": "",
    // Used to ensure requests coming from legit frontend client
    "challenge_token": "",
    // Used to determine if server is running in debug mode or not (dev vs prod)
    "environment": "",
    // Used to encode / decode JWTs
    "secret_key": "",
    // Used to determine if server should load GPU-dependencies (cpu vs gpu)
    "server_type": "",
    // These are all project-specific that you should probably keep as-is
    "save_image_to_database": "True",
    "default_credit_expiration": 30,
    "video_generation_fps": 16,
    "audio_gen_duration": 10
}
```

# Running the server

After configuring your `config.json`, you may proceed with the following steps to run the server:

1. Install Python 3.8.10
2. Install pip3 >= 23.1.2
3. Install requirements `pip3 install -r requirements.txt`
4. Run the server `python3 server.py`

# Misc

We also have cron jobs `/cronjobs` which are used to process the generation of audio and videos. This is for the fact that these are long-standing processes and are ergo addressed in a queue-fashion. These are handled in crontab and use flock to ensure that multiple instances are not spun up simultaneously. 

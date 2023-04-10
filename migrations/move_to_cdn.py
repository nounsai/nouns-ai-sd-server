import uuid
import base64
import sys
sys.path.append('../nouns-ai-sd-server')  # allows import from parent directory

import cdn
import db

conn = db.open_connection()
cur = db.create_cursor(conn)

print('Adding column `cdn_id` to table `images`...')

cur.execute('ALTER TABLE images ADD COLUMN IF NOT EXISTS cdn_id VARCHAR(36);')
conn.commit()

print('Getting all images from database...')

cur.execute('SELECT * FROM images;')
images = cur.fetchall()

print(f'Found {len(images)} images. Uploading all images to CDN and saving UUIDs...')
fail_count = 0

for image in images:
    cdn_id = str(uuid.uuid4())

    resp = cdn.upload_image_to_cdn(image[1], cdn_id, base64.b64decode(image[2][24:-1]), base64.b64decode(image[7][24:-1]))
    if not resp:
        print(f'Image ID {image[0]} failed to upload')
        fail_count += 1
    cur.execute('UPDATE images SET cdn_id = %s WHERE id = %s', [cdn_id, image[0]])
    conn.commit()

print('Uploaded images and saved UUIDs.')

delete_cols = input(f'Would you like to delete the `base_64` and `thumb_base_64` columns? (y/n) ')

if delete_cols == 'y':
    if fail_count == 0:
        print('Deleting base64 images from database...')
        cur.execute('ALTER TABLE images DROP COLUMN IF EXISTS base_64;')
        cur.execute('ALTER TABLE images DROP COLUMN IF EXISTS thumb_base_64;')
        conn.commit()

        print('Migration complete!')
    else:
        i = input(f'{fail_count} image(s) failed to upload to CDN. Are you sure you want to continue? (y/n) ')
        if i == 'y':
            cur.execute('ALTER TABLE images DROP COLUMN IF EXISTS base_64;')
            cur.execute('ALTER TABLE images DROP COLUMN IF EXISTS thumb_base_64;')
            conn.commit()

            print('Migration complete!')
        else:
            print('Not deleting base 64 columns from database. Migration partially complete, you may need to delete the'
                  'columns yourself after ensuring that all images have been uploaded.')
else:
    print('Not deleting base 64 columns from database. Migration partially complete, you may need to delete the '
          'columns yourself after ensuring that all images have been uploaded.')

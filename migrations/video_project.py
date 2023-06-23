import sys
sys.path.append('../nouns-ai-sd-server')  # allows import from parent directory

import db

if __name__ == '__main__':
    conn = db.open_connection()
    cur = db.create_cursor(conn)

    print('Creating table: video_projects')

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS video_projects (
            id INT GENERATED ALWAYS AS IDENTITY,
            user_id INT NOT NULL,
            audio_id INT NOT NULL REFERENCES audio(id) ON DELETE CASCADE,
            metadata JSON NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY(id),
            CONSTRAINT fk_user FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()

    print('Renaming column: audio.url -> audio.cdn_id')

    cur.execute(
        """
        ALTER TABLE audio RENAME COLUMN url TO cdn_id;
        """
    )
    conn.commit()

    print('Finished')
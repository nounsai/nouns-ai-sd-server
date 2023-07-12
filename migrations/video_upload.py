import sys
sys.path.append('../nouns-ai-sd-server')  # allows import from parent directory

import db

if __name__ == '__main__':
    conn = db.open_connection()
    cur = db.create_cursor(conn)

    print('TABLE videos: adding cdn_id field')

    cur.execute(
        """
        ALTER TABLE videos ADD COLUMN IF NOT EXISTS cdn_id VARCHAR(36);
        """
    )
    conn.commit()

    print('TABLE videos: adding start_frame_id, end_frame_id fields')

    cur.execute(
        """
        ALTER TABLE videos ADD COLUMN IF NOT EXISTS start_frame_id INT NOT NULL;
        """
    )
    conn.commit()
    cur.execute(
        """
        ALTER TABLE videos ADD COLUMN IF NOT EXISTS end_frame_id INT NOT NULL;
        """
    )
    conn.commit()

    # add constraint
    cur.execute(
        """
        ALTER TABLE videos DROP CONSTRAINT IF EXISTS fk_start_frame;
        """
    )
    conn.commit()
    cur.execute(
        """
        ALTER TABLE videos ADD CONSTRAINT fk_start_frame FOREIGN KEY(start_frame_id) REFERENCES images(id) ON DELETE CASCADE;
        """
    )
    conn.commit()
    cur.execute(
        """
        ALTER TABLE videos DROP CONSTRAINT IF EXISTS fk_end_frame;
        """
    )
    conn.commit()
    cur.execute(
        """
        ALTER TABLE videos ADD CONSTRAINT fk_end_frame FOREIGN KEY(end_frame_id) REFERENCES images(id) ON DELETE CASCADE;
        """
    )
    conn.commit()


    print('TABLE videos: adding start_frame_cdn_id, end_frame_cdn_id fields')

    cur.execute(
        """
        ALTER TABLE videos ADD COLUMN IF NOT EXISTS start_frame_cdn_id VARCHAR(36);
        """
    )
    conn.commit()
    cur.execute(
        """
        ALTER TABLE videos ADD COLUMN IF NOT EXISTS end_frame_cdn_id VARCHAR(36);
        """
    )
    conn.commit()

    print('TABLE videos: adding duration field')

    cur.execute(
        """
        ALTER TABLE videos ADD COLUMN IF NOT EXISTS duration INT NOT NULL;
        """
    )
    conn.commit()


    print('finished')
import sys
sys.path.append('../nouns-ai-sd-server')  # allows import from parent directory

import db

if __name__ == '__main__':
    conn = db.open_connection()
    cur = db.create_cursor(conn)

    print('Adding column: video_projects.state')

    cur.execute(
        """
        ALTER TABLE video_projects ADD COLUMN IF NOT EXISTS state VARCHAR(36) DEFAULT 'UNFINISHED';
        """
    )
    conn.commit()

    print('finished')
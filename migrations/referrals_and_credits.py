import sys
sys.path.append('../nouns-ai-sd-server')  # allows import from parent directory

import db

if __name__ == '__main__':
    conn = db.open_connection()
    cur = db.create_cursor(conn)

    print('Creating table: referrals')

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS referrals (
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
        """
    )
    conn.commit()

    print('Creating table: rewards')

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rewards (
            id INT GENERATED ALWAYS AS IDENTITY,
            name VARCHAR (128) NOT NULL,
            sql TEXT NOT NULL,
            metadata JSON NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY(id)
        );
        """
    )
    conn.commit()

    print('Creating table: transactions')

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
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
        """
    )
    conn.commit()

    print('Adding `referral_token` column to table `users`')

    cur.execute('ALTER TABLE users ADD COLUMN IF NOT EXISTS referral_token VARCHAR(128);')
    conn.commit()

    print('finished')
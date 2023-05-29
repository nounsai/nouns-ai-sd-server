import sys
sys.path.append('../nouns-ai-sd-server')  # allows import from parent directory

import db

if __name__ == '__main__':
    conn = db.open_connection()
    cur = db.create_cursor(conn)
    print('Creating reward: referral_verify')

    cur.execute(
        """
        INSERT INTO rewards (name, sql, metadata) VALUES (
            'referral_verify', 
            'With matching as (
                Select * From users
                Where id = %s
            )
            Select 
                Case 
                    when is_verified then 100
                    else 0
                End as amount,
                Case 
                    when is_verified then 30
                    else 0
                End as expires_at,
                Case 
                    when is_verified then true
                    else false
                End as complete
            From matching;',
            '{}'
        );
        """
    )
    conn.commit()
    
    print('finished')
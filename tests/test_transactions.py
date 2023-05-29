import db
import datetime

from tests.utils import create_dummy_user, get_access_token, WrappedClient
from server import app

class TestTransactions:
    client = WrappedClient(app.test_client())
    test_user_id = create_dummy_user("ttest123@example.com")
    access_token = get_access_token(test_user_id)

    def test_no_transactions(cls):
        """
        Test credit balance when a user has no credits
        """
        res = cls.client.get(f'/users/{cls.test_user_id}/credits', cls.access_token)
        assert res.status_code == 200
        assert res.json['credits'] == 0

    
    def test_existing_transactions(cls):
        """
        Test credit balance when user has multiple transactions
        """
        not_expired_date = datetime.datetime.utcnow() + datetime.timedelta(seconds=2)
        expired_date = datetime.datetime.utcnow() - datetime.timedelta(seconds=1)
        db.create_transaction(cls.test_user_id, 59, amount_remaining=39, expires_at=not_expired_date)
        db.create_transaction(cls.test_user_id, 38, amount_remaining=4, expires_at=expired_date)
        db.create_transaction(cls.test_user_id, 100, amount_remaining=78, expires_at=not_expired_date)
        db.create_transaction(cls.test_user_id, 9999, amount_remaining=0, expires_at=not_expired_date)

        res = cls.client.get(f'/users/{cls.test_user_id}/credits', cls.access_token)
        assert res.status_code == 200
        assert res.json['credits'] == 39 + 78


    @classmethod
    def teardown_class(cls):
        db.delete_user(cls.test_user_id)
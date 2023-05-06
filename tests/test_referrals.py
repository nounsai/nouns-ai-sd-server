import db 
import secrets
import datetime

from tests.utils import create_dummy_user, get_access_token, WrappedClient, create_referral_reward
from server import app

class TestReferrals:

    client = WrappedClient(app.test_client())
    test_user_id = create_dummy_user("rtest123@example.com")
    test_user2_id = create_dummy_user("rtest2@example.com", referral_token="bob")
    test_user3_id = create_dummy_user("runverified@example.com", verified=False)
    test_reward_id = create_referral_reward()
    test_referral_id = db.create_referral(test_user_id, test_user3_id, {})
    access_token = get_access_token(test_user_id)
    access_token2 = get_access_token(test_user2_id)
    access_token3 = get_access_token(test_user3_id)


    def test_get_referral_token(cls):
        """
        Test that referral token is set properly when user does not have one yet
        """
        test_user = db.fetch_user(cls.test_user_id)
        assert test_user['referral_token'] is None

        res = cls.client.get(f'/users/{cls.test_user_id}/referral-token', cls.access_token)

        assert res.status_code == 200
        assert res.json['token'] is not None
        assert len(res.json['token']) == len(secrets.token_hex(48))

        test_user = db.fetch_user(cls.test_user_id)

        assert res.json['token'] == test_user['referral_token']

    
    def test_get_existing_referral_token(cls):
        """
        Test that referral token is retrieved properly when user already has one
        """
        test_user = db.fetch_user(cls.test_user2_id)
        assert test_user['referral_token'] == 'bob'

        res = cls.client.get(f'/users/{cls.test_user2_id}/referral-token', cls.access_token2)

        assert res.status_code == 200
        assert res.json['token'] is not None
        assert res.json['token'] == 'bob'

        test_user = db.fetch_user(cls.test_user2_id)

        assert test_user['referral_token'] == 'bob'


    def test_verification_reward(cls):
        """
        Test that referrer is rewarded credits when referred user verifies
        """
        assert len(db.fetch_transactions_for_user(cls.test_user_id)) == 0

        referred = db.fetch_user(cls.test_user3_id)

        verify_key = referred['verify_key']

        res = cls.client.post(f'/users/verify/{verify_key}', cls.access_token3)

        assert res.status_code == 200
        referred = db.fetch_user(cls.test_user3_id)
        assert referred['is_verified'] is True
        assert referred['metadata']['rewards_referral_verify'] is True

        transactions = db.fetch_transactions_for_user(cls.test_user_id)
        assert len(transactions) == 1
        trxn = transactions[0]
        assert trxn['amount'] == 100
        assert trxn['amount_remaining'] == 100
        expires_at = datetime.datetime.utcfromtimestamp(trxn['expires_at'] / 1000)
        time_difference = expires_at.timestamp() - (datetime.datetime.utcnow() + datetime.timedelta(days=30)).timestamp()
        assert abs(time_difference) < 1000


    @classmethod
    def teardown_class(cls):
        db.delete_user(cls.test_user_id)
        db.delete_user(cls.test_user2_id)
        db.delete_user(cls.test_user3_id)
        db.delete_reward(cls.test_reward_id)
        db.delete_referral(cls.test_referral_id)
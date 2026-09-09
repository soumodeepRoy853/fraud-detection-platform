import os 
import redis
from dotenv import load_dotenv

load_dotenv()

redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

def get_account_velocity(account_id: str) -> dict:
    """Fetch this account's recent transaction count and running average amount."""
    count = redis_client.get(f'account: {account_id}:count')
    total = redis_client.get(f'account: {account_id}:total_amount')

    count = int(count) if count else 0
    total = float(total) if total else 0.0
    avg_amount = round(total/count, 2) if count > 0 else 0.0

    return {
        'txn_count_recnt': count, 'avg_amount_recent': avg_amount
    }

def update_account_velocity(account_id: str, amount: float):
    """Increment this account's transaction count and running total. with a 1-hour expiry (rollong window)."""
    pipe = redis_client.pipeline()
    pipe.incr(f'account: {account_id}:count')
    pipe.incrbyfloat(f'account: {account_id}:total_amount', amount)
    pipe.expire(f'account: {account_id}:count', 3600)
    pipe.expire(f'account: {account_id}:total_amount', 3600)
    pipe.execute()
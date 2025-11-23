import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB

def get_redis_client():
    """Create and return Redis client"""
    try:
        # Build connection kwargs
        connection_kwargs = {
            'host': REDIS_HOST,
            'port': REDIS_PORT,
            'db': REDIS_DB,
            'decode_responses': True,
            'socket_connect_timeout': 5,
            'socket_timeout': 5
        }
        # Only add password if it's provided
        if REDIS_PASSWORD:
            connection_kwargs['password'] = REDIS_PASSWORD
        
        client = redis.Redis(**connection_kwargs)
        # Test connection
        client.ping()
        print("✅ Redis connected successfully")
        return client
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return None

# Global Redis client instance
redis_client = get_redis_client()
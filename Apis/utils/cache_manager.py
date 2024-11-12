from datetime import datetime, timedelta
import asyncio

# Global cache for FAISS indexes
faiss_cache = {}

async def cleanup_cache():
    """Periodic cleanup of in-memory cache"""
    while True:
        await asyncio.sleep(3600)  # Clean every hour
        if len(faiss_cache) > 100:  # Adjust threshold as needed
            # Remove oldest entries
            while len(faiss_cache) > 50:
                faiss_cache.pop(next(iter(faiss_cache)))

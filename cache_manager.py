import json
import time
import hashlib
from typing import Optional, Dict, Any
from redis_client import redis_client
from config import CACHE_PREFIX

class CacheManager:
    def __init__(self, ttl_hours: int = 6):
        self.ttl_seconds = ttl_hours * 60 * 60  # Convert to seconds
        self.cache_prefix = CACHE_PREFIX
        
    def _make_cache_key(self, question: str) -> str:
        """Create a stable cache key for a question."""
        # Normalize the question for consistent hashing
        question_norm = " ".join(question.lower().strip().split())
        digest = hashlib.sha256(question_norm.encode('utf-8')).hexdigest()
        return f"{self.cache_prefix}:{digest}"
    
    def should_cache_response(self, response_data: Dict[str, Any]) -> bool:
        """
        Determine if a response should be cached.
        Only cache successful responses with actual answers.
        """
        try:
            # Check if it's a successful response with an answer
            if not response_data.get("success", True):
                return False
                
            answer = response_data.get("answer", "")
            if not answer:
                return False
                
            # Don't cache error messages
            if answer.startswith("❌"):
                return False
                
            # Don't cache service unavailable messages
            error_indicators = [
                "service unavailable",
                "cannot connect", 
                "timeout",
                "error:",
                "failed to",
                "unavailable"
            ]
            
            answer_lower = answer.lower()
            if any(indicator in answer_lower for indicator in error_indicators):
                return False
                
            return True
            
        except Exception:
            return False
    
    def store_response(self, question: str, response_data: Dict[str, Any]) -> bool:
        """
        Store a question-response pair in cache if it's cacheable.
        Returns True if stored, False otherwise.
        """
        if not redis_client:
            return False
            
        if not self.should_cache_response(response_data):
            return False
            
        try:
            cache_key = self._make_cache_key(question)
            
            cache_entry = {
                "question": question,
                "answer": response_data.get("answer", ""),
                "timestamp": time.time(),
                "ttl_hours": 6
            }
            
            redis_client.setex(
                cache_key, 
                self.ttl_seconds, 
                json.dumps(cache_entry)
            )
            
            print(f"[CACHE] Stored response for: {question[:80]}...")
            return True
            
        except Exception as e:
            print(f"[CACHE] Error storing cache: {e}")
            return False
    
    def get_response(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for a question.
        Returns None if not found or expired.
        """
        if not redis_client:
            return None
            
        try:
            cache_key = self._make_cache_key(question)
            cached_data = redis_client.get(cache_key)
            
            if cached_data:
                cache_entry = json.loads(cached_data)
                
                # Verify the cache entry has required fields
                if all(key in cache_entry for key in ["question", "answer", "timestamp"]):
                    print(f"[CACHE] Cache hit for: {question[:80]}...")
                    return {
                        "answer": cache_entry["answer"],
                        "cached": True,
                        "timestamp": cache_entry["timestamp"]
                    }
                    
        except Exception as e:
            print(f"[CACHE] Error reading cache: {e}")
            
        return None
    
    def delete_response(self, question: str) -> bool:
        """Delete a specific cached response."""
        if not redis_client:
            return False
            
        try:
            cache_key = self._make_cache_key(question)
            result = redis_client.delete(cache_key)
            return result > 0
        except Exception as e:
            print(f"[CACHE] Error deleting cache: {e}")
            return False
    
    def clear_all_cache(self) -> bool:
        """Clear all cached responses."""
        if not redis_client:
            return False
            
        try:
            # This is a simple implementation - in production you might want more sophisticated pattern matching
            pattern = f"{self.cache_prefix}:*"
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
                print(f"[CACHE] Cleared {len(keys)} cached items")
                return True
            return False
        except Exception as e:
            print(f"[CACHE] Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not redis_client:
            return {"error": "Redis not available"}
            
        try:
            pattern = f"{self.cache_prefix}:*"
            keys = redis_client.keys(pattern)
            
            total_size = 0
            sample_entries = []
            
            for key in keys[:5]:  # Sample first 5 entries
                try:
                    data = redis_client.get(key)
                    if data:
                        total_size += len(data)
                        entry = json.loads(data)
                        sample_entries.append({
                            "question_preview": entry.get("question", "")[:50] + "..." if len(entry.get("question", "")) > 50 else entry.get("question", ""),
                            "timestamp": entry.get("timestamp"),
                            "answer_length": len(entry.get("answer", ""))
                        })
                except:
                    continue
            
            return {
                "total_cached_items": len(keys),
                "total_cache_size_bytes": total_size,
                "sample_entries": sample_entries,
                "cache_prefix": self.cache_prefix,
                "ttl_hours": 7
            }
            
        except Exception as e:
            return {"error": f"Failed to get cache stats: {e}"}

# Create a global instance
cache_manager = CacheManager(ttl_hours=6)
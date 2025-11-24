import os
import time
import json
import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import queue
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Initialize app first
app = Flask(__name__)
FRONTEND_URL = os.getenv("FRONTEND_URL")

# Configure CORS
CORS(app, 
     origins=[FRONTEND_URL] if FRONTEND_URL else ["http://localhost:3000"],
     supports_credentials=True,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "x-session-id", "X-Session-Id"])

# Thread-safe FAISS initialization
faiss_app_lock = threading.Lock()
active_threads = 0
threads_lock = threading.Lock()

# Safe import function with better error handling
def safe_import(module_name, function_name=None, default=None):
    try:
        module = __import__(module_name)
        if function_name:
            return getattr(module, function_name)
        return module
    except ImportError as e:
        print(f"⚠️ Failed to import {module_name}: {e}")
        return default

# Initialize components safely
vectorstore = None
ALL_FOLDERS = set()
redis_client = safe_import('redis_client', 'redis_client')
cache_manager = safe_import('cache_manager', 'cache_manager')

# Import config with defaults
try:
    from config import MAX_WORDS, MAX_CHARS, MIN_CHARS, PROMPTS_PER_WINDOW, WINDOW_SECONDS, CACHE_TTL_SECONDS, CACHE_PREFIX
    print("✅ Config loaded")
except ImportError as e:
    print(f"⚠️ Config import failed: {e}")
    # Set defaults
    MAX_WORDS = 250
    MAX_CHARS = 2000
    MIN_CHARS = 3
    PROMPTS_PER_WINDOW = 15
    WINDOW_SECONDS = 43200
    CACHE_TTL_SECONDS = 21600
    CACHE_PREFIX = "cache:answer"

try:
    from validation import validate_question, sanitize_input
    print("✅ Validation loaded")
except ImportError as e:
    print(f"⚠️ Validation module not available: {e}")
    # Fallback functions
    def validate_question(question):
        if not question or len(question.strip()) == 0:
            return False, "Question cannot be empty"
        if len(question) > MAX_CHARS:
            return False, f"Question too long. Maximum {MAX_CHARS} characters allowed."
        if len(question) < MIN_CHARS:
            return False, f"Question too short. Minimum {MIN_CHARS} characters required."
        return True, ""
    
    def sanitize_input(text):
        import html
        return html.escape(text.strip())

try:
    from security import is_creator_question, HARDCODED_CREATOR_RESPONSE, normalize_text
    print("✅ Security loaded")
except ImportError as e:
    print(f"⚠️ Security module not available: {e}")
    def is_creator_question(question):
        return False
    HARDCODED_CREATOR_RESPONSE = "I'm currently unavailable. Please try again later."
    def normalize_text(text):
        return text.lower().strip() if text else ""

# Thread-safe FAISS initialization
def initialize_faiss_safely():
    """Thread-safe FAISS initialization"""
    global vectorstore, ALL_FOLDERS
    
    with faiss_app_lock:
        if vectorstore is None:
            try:
                from retrieval import initialize_retrieval
                vectorstore, ALL_FOLDERS = initialize_retrieval()
                print(f"✅ FAISS loaded with {len(ALL_FOLDERS)} folders")
            except Exception as e:
                print(f"⚠️ Failed to initialize vectorstore: {e}")
                vectorstore = None
                ALL_FOLDERS = set()

# Initialize FAISS at startup
initialize_faiss_safely()

# ⭐⭐⭐ FIXED COHERE IMPORT - Use lazy import to avoid circular imports ⭐⭐⭐
def ask_cohere_smart(question):
    """Lazy import for cohere function to avoid circular imports"""
    try:
        from cohere_api import ask_cohere_smart as cohere_func
        return cohere_func(question)
    except ImportError as e:
        print(f"⚠️ Cohere API not available: {e}")
        return "Service temporarily unavailable. Please try again later."

def get_api_usage():
    """Lazy import for API usage stats"""
    try:
        from cohere_api import get_api_usage as usage_func
        return usage_func()
    except ImportError:
        return {"total_requests": 0, "total_tokens": 0, "last_reset": time.time()}

try:
    from redis_client import redis_client as redis_client_import
    redis_client = redis_client_import
    print("✅ Redis client loaded")
except ImportError as e:
    print(f"⚠️ Redis client not available: {e}")

try:
    from cache_manager import cache_manager as cache_manager_import
    cache_manager = cache_manager_import
    print("✅ Cache manager loaded")
except ImportError as e:
    print(f"⚠️ Cache manager not available: {e}")
    # Create a simple fallback cache manager
    class FallbackCacheManager:
        def get_response(self, question):
            return None
        def store_response(self, question, result):
            pass
        def get_cache_stats(self):
            return {"error": "Cache not available"}
        def clear_all_cache(self):
            return False
        def delete_response(self, question):
            return False
    cache_manager = FallbackCacheManager()

# ThreadPoolExecutor with priority queue
class PriorityThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, thread_name_prefix=''):
        super().__init__(max_workers, thread_name_prefix)
        self._work_queue = queue.PriorityQueue()

# Updated for 8 concurrent requests (2 workers × 4 threads)
request_executor = PriorityThreadPoolExecutor(
    max_workers=8,  # Increased to match gunicorn capacity
    thread_name_prefix="ask_worker"
)

def _get_request_user_id():
    """Get unique user identifier for rate limiting"""
    return (
        request.headers.get('x-session-id')
        or request.cookies.get('searchSessionId')
        or request.remote_addr
    )

def enforce_prompt_limit():
    """Redis-based rate limiting"""
    if not redis_client:
        print("⚠️ Redis not available, skipping rate limiting")
        return None
        
    uid = _get_request_user_id()
    if not uid:
        print("⚠️ Could not determine user ID for rate limiting")
        return None
        
    now = time.time()
    
    try:
        key = f"rate_limit:{uid}"
        existing = redis_client.get(key)
        if existing:
            data = json.loads(existing)
            if now - data['start'] < WINDOW_SECONDS:
                if data['count'] >= PROMPTS_PER_WINDOW:
                    return jsonify({
                        'success': False,
                        'error': 'Unavailable try again after 12 hours'
                    }), 429
                data['count'] += 1
                redis_client.setex(key, WINDOW_SECONDS, json.dumps(data))
            else:
                data = {'count': 1, 'start': now}
                redis_client.setex(key, WINDOW_SECONDS, json.dumps(data))
        else:
            data = {'count': 1, 'start': now}
            redis_client.setex(key, WINDOW_SECONDS, json.dumps(data))
        
        print(f"[RATE LIMIT] User {uid[:8]}... has used {data['count']}/{PROMPTS_PER_WINDOW} requests")
        return None
        
    except Exception as e:
        print(f"⚠️ Rate limiting error: {e}")
        return None 

def process_question(question, cancel_event):
    """Process question in worker thread"""
    global active_threads
    
    thread_id = threading.get_ident()
    with threads_lock:
        active_threads += 1
        print(f"🧵 [THREAD {thread_id}] Started. Active threads: {active_threads}/8")
    
    try:
        if cancel_event.is_set():
            return {"success": False, "error": "Request canceled"}
        
        print(f"[QUEUE] Processing question: {question[:100]}...")
        
        # Input validation
        is_valid, error_msg = validate_question(question)
        if not is_valid:
            return {"success": False, "error": error_msg}
        
        # Sanitize input
        question = sanitize_input(question)
        
        if cancel_event.is_set():
            return {"success": False, "error": "Request canceled"}
        
        # Creator question check
        if is_creator_question(question):
            return {"success": True, "answer": HARDCODED_CREATOR_RESPONSE}
            
        # Main LLM processing
        answer = ask_cohere_smart(question)
        
        # Check if Cohere returned an error
        if answer.startswith("❌"):
            return {"success": False, "error": answer}
        else:
            return {"success": True, "answer": answer}
            
    except Exception as e:
        print(f"[QUEUE] Error processing question: {e}")
        return {"success": False, "error": f"Processing error: {str(e)}"}
    finally:
        with threads_lock:
            active_threads -= 1
            print(f"🧵 [THREAD {thread_id}] Finished. Active threads: {active_threads}/8")

def get_request_priority(question):
    """Assign priority to requests - lower number = higher priority"""
    question_lower = question.lower()
    
    # Priority 1: Very fast requests (creator questions)
    if is_creator_question(question):
        return (1, question)
    
    # Priority 2: Short questions likely to be fast
    if len(question) < 30 or any(phrase in question_lower for phrase in ["what is", "who is", "define"]):
        return (2, question)
    
    # Priority 3: Medium - filtered searches
    if "only in" in question_lower:
        return (3, question)
    
    # Priority 4: Complex/long questions (lowest priority)
    return (4, question)

def process_question_with_priority(priority_item, cancel_event):
    """Process function that accepts priority items"""
    priority, question = priority_item
    return process_question(question, cancel_event)

def _make_cache_key(question: str) -> str:
    """Create a stable cache key for a question"""
    q_norm = normalize_text(question or "")
    digest = hashlib.sha256(q_norm.encode('utf-8')).hexdigest()
    return f"{CACHE_PREFIX}:{digest}"

# Track in-flight requests for cancel/ownership
REQUESTS = {}

# ============ ROUTES ============

@app.route("/", methods=["GET"])
def root_healthcheck():
    """Root healthcheck endpoint for Railway/Render"""
    return "OK", 200

@app.route("/health", methods=["GET"])
def health_check():
    """Production health check - always returns healthy if Flask is running"""
    try:
        # Redis status
        redis_status = "disconnected"
        if redis_client:
            try:
                redis_status = "connected" if redis_client.ping() else "disconnected"
            except:
                redis_status = "disconnected"
        
        # Queue status
        queue_size = 0
        active_workers = 0
        try:
            queue_size = request_executor._work_queue.qsize()
            active_workers = request_executor._max_workers
        except:
            pass
        
        # App is healthy as long as Flask is running
        return jsonify({
            "status": "healthy",
            "folders_loaded": len(ALL_FOLDERS),
            "vectorstore_ready": vectorstore is not None,
            "redis": redis_status,
            "concurrent_capacity": "8 requests (2 workers × 4 threads)",
            "active_threads": active_threads,
            "queue": {
                "active_workers": active_workers,
                "pending_tasks": queue_size
            },
            "timestamp": time.time()
        })
        
    except Exception as e:
        # Ultimate fallback - always return healthy
        return jsonify({
            "status": "healthy",
            "timestamp": time.time()
        })

@app.route("/ask", methods=["POST"])
def ask_endpoint():
    """Main question endpoint with rate limiting and queueing"""
    # Rate limit check
    limit_response = enforce_prompt_limit()
    if limit_response is not None:
        return limit_response

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON data provided"}), 400

    question = data.get("question")
    requestId = data.get("requestId") or str(uuid.uuid4())
    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    # Check cache first
    cached_response = cache_manager.get_response(question)
    if cached_response:
        return jsonify({
            "requestId": requestId, 
            "answer": cached_response["answer"],
            "cached": True,
            "timestamp": cached_response.get("timestamp")
        })

    # Track cancel state and owner
    uid = _get_request_user_id()
    cancel_event = threading.Event()
    REQUESTS[requestId] = {"cancel": cancel_event, "owner": uid, "start": time.time()}

    print(f"[REQUEST] New question from {uid}: {question[:100]}...")

    try:
        # Calculate priority for this request
        priority_item = get_request_priority(question)
        future = request_executor.submit(process_question_with_priority, priority_item, cancel_event)
        result = future.result(timeout=100)

        # Store in cache if successful
        if result.get("success", True) and "answer" in result:
            cache_manager.store_response(question, result)

        return jsonify({"requestId": requestId, **result})
    except TimeoutError:
        print(f"[REQUEST] Timeout for question: {question[:100]}...")
        return jsonify({"success": False, "error": "Request timeout. Server is busy. Please try again in a few moments."}), 408
    except Exception as e:
        print(f"[REQUEST] Unexpected error: {e}")
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500
    finally:
        # Always cleanup
        REQUESTS.pop(requestId, None)

@app.route("/llm/ask", methods=["POST", "OPTIONS"])
def llm_ask_endpoint():
    """LLM ask endpoint for compatibility with existing frontend"""
    if request.method == "OPTIONS":
        return jsonify({"status": "success"}), 200
    return ask_endpoint()

@app.route("/cancel", methods=["POST"])
def cancel_endpoint():
    """Cancel an in-flight request by id; enforces ownership."""
    data = request.get_json()
    requestId = data.get("requestId")
    if not requestId:
        return jsonify({"success": False, "error": "requestId is required"}), 400

    uid = _get_request_user_id()
    entry = REQUESTS.get(requestId)
    if not entry:
        return jsonify({"success": False, "error": "No such in-flight request"}), 404

    owner = entry.get("owner")
    if owner != uid:
        return jsonify({"success": False, "error": "Not authorized to cancel this request"}), 403

    try:
        entry["cancel"].set()
        REQUESTS.pop(requestId, None)
        print(f"[CANCEL] Canceled requestId={requestId} by owner={uid}")
        return jsonify({"success": True, "requestId": requestId, "canceled": True})
    except Exception as e:
        print(f"[CANCEL] Error: {e}")
        return jsonify({"success": False, "error": "Cancel error"}), 500

@app.route("/llm/cancel", methods=["POST"])
def llm_cancel_endpoint():
    """Cancel endpoint for compatibility"""
    return cancel_endpoint()

@app.route("/rate-limit-status", methods=["GET"])
def rate_limit_status():
    """Debug endpoint to check current rate limit status"""
    uid = _get_request_user_id()
    if not redis_client or not uid:
        return jsonify({"error": "Redis not available or no user ID"})
    
    key = f"rate_limit:{uid}"
    existing = redis_client.get(key)
    
    if existing:
        data = json.loads(existing)
        current_time = time.time()
        time_elapsed = current_time - data['start']
        time_remaining = WINDOW_SECONDS - time_elapsed
        requests_used = data['count']
        requests_remaining = PROMPTS_PER_WINDOW - requests_used
        
        return jsonify({
            "user_id": uid[:8] + "...",
            "requests_used": requests_used,
            "requests_remaining": requests_remaining,
            "limit": PROMPTS_PER_WINDOW,
            "window_seconds": WINDOW_SECONDS,
            "time_elapsed": round(time_elapsed, 1),
            "time_remaining": round(time_remaining, 1)
        })
    else:
        return jsonify({
            "user_id": uid[:8] + "...",
            "status": "No active rate limit window",
            "limit": PROMPTS_PER_WINDOW,
            "window_seconds": WINDOW_SECONDS,
            "requests_remaining": PROMPTS_PER_WINDOW
        })

@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Monitor API usage"""
    try:
        cohere_stats = get_api_usage()
        return jsonify({
            "cohere_api_usage": cohere_stats,
            "timestamp": time.time(),
            "queue_size": request_executor._work_queue.qsize(),
            "active_workers": request_executor._max_workers,
            "active_threads": active_threads,
            "max_concurrent": 8
        })
    except Exception as e:
        return jsonify({
            "error": f"API stats not available: {str(e)}",
            "timestamp": time.time()
        }), 500

@app.route("/queue-status", methods=["GET"])
def queue_status():
    """Endpoint to check queue status"""
    queue_size = request_executor._work_queue.qsize()
    return jsonify({
        "status": "operational",
        "active_workers": request_executor._max_workers,
        "pending_tasks": queue_size,
        "max_concurrent": 8,
        "active_threads": active_threads,
        "configuration": "2 workers × 4 threads"
    })

@app.route("/threads", methods=["GET"])
def thread_monitor():
    """Monitor active threads"""
    return jsonify({
        "active_threads": active_threads,
        "max_concurrent": 8,
        "gunicorn_workers": 2,
        "gunicorn_threads": 4,
        "capacity": "8 simultaneous requests"
    })

@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    """Get cache statistics"""
    stats = cache_manager.get_cache_stats()
    return jsonify(stats)

@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear all cached responses"""
    success = cache_manager.clear_all_cache()
    return jsonify({"success": success, "message": "Cache cleared" if success else "Failed to clear cache"})

@app.route("/cache/delete", methods=["POST"])
def delete_cache_entry():
    """Delete a specific cached question"""
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"success": False, "error": "Question is required"}), 400
        
    question = data["question"]
    success = cache_manager.delete_response(question)
    return jsonify({"success": success, "message": f"Deleted cache for: {question[:50]}..."})

def cleanup_executor():
    """Cleanup function to shutdown executor gracefully"""
    print("🔄 Shutting down ThreadPoolExecutor...")
    request_executor.shutdown(wait=False)

import atexit
atexit.register(cleanup_executor)

def get_port():
    """Get the port from Railway environment"""
    return int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    port = get_port()
    print(f"✅ Flask app starting on 0.0.0.0:{port}...")
    print(f"🚀 Configured for 8 simultaneous requests (2 workers × 4 threads)")
    app.run(host="0.0.0.0", port=port, debug=False)
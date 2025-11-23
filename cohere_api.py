from flask import Flask, request, jsonify
from flask_cors import CORS
from config import MAX_WORDS, MAX_CHARS, MIN_CHARS, PROMPTS_PER_WINDOW, WINDOW_SECONDS, CACHE_TTL_SECONDS, CACHE_PREFIX
from validation import validate_question, sanitize_input
from security import is_creator_question, HARDCODED_CREATOR_RESPONSE, normalize_text
from cohere_api import ask_cohere_smart
from retrieval import initialize_retrieval
from redis_client import redis_client
import time
import json
import hashlib
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from flask import Response, stream_with_context
import queue
from cache_manager import cache_manager
from dotenv import load_dotenv
import os

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL")

app = Flask(__name__)

# ---------------------------------------------------------
# ✅ SIMPLE HEALTHCHECK ROUTE (Railway needs this)
# ---------------------------------------------------------
@app.get("/")
def root_healthcheck():
    return "OK", 200
# ---------------------------------------------------------

# Configure CORS properly for React frontend
CORS(app, 
     origins=[FRONTEND_URL],
     supports_credentials=True,
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "x-session-id", "X-Session-Id"])


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"status": "success"})
        response.headers.add("Access-Control-Allow-Origin",FRONTEND_URL)
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization,x-session-id,X-Session-Id")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        return response

# Initialize FAISS on startup
print("🚀 Initializing Flask LLM API...")
try:
    vectorstore, ALL_FOLDERS = initialize_retrieval()
    print(f"✅ Loaded {len(ALL_FOLDERS)} folders from FAISS")
except Exception as e:
    import traceback
    print(f"❌ Failed to initialize vectorstore:")
    print(f"   Error: {str(e)}")
    print(f"   Full traceback:")
    traceback.print_exc()
    print("\n⚠️ Application will start but LLM functionality may be limited")
    vectorstore = None
    ALL_FOLDERS = set()

# ThreadPoolExecutor with priority queue
class PriorityThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, thread_name_prefix=''):
        super().__init__(max_workers, thread_name_prefix)
        self._work_queue = queue.PriorityQueue()

request_executor = PriorityThreadPoolExecutor(
    max_workers=5,
    thread_name_prefix="ask_worker"
)

def _get_request_user_id():
    return (
        request.headers.get('x-session-id')
        or request.cookies.get('searchSessionId')
        or request.remote_addr
    )

def enforce_prompt_limit():
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
    try:
        if cancel_event.is_set():
            return {"success": False, "error": "Request canceled"}
        
        print(f"[QUEUE] Processing question: {question[:100]}...")
        is_valid, error_msg = validate_question(question)
        if not is_valid:
            return {"success": False, "error": error_msg}
        
        question = sanitize_input(question)
        
        if cancel_event.is_set():
            return {"success": False, "error": "Request canceled"}
        
        if is_creator_question(question):
            return {"success": True, "answer": HARDCODED_CREATOR_RESPONSE}
            
        answer = ask_cohere_smart(question)
        
        if answer.startswith("❌"):
            return {"success": False, "error": answer}
        else:
            return {"success": True, "answer": answer}
            
    except Exception as e:
        print(f"[QUEUE] Error: {e}")
        return {"success": False, "error": f"Processing error: {str(e)}"}

def get_request_priority(question):
    q = question.lower()
    if is_creator_question(question):
        return (1, question)
    if len(question) < 30 or any(x in q for x in ["what is", "who is", "define"]):
        return (2, question)
    if "only in" in q:
        return (3, question)
    return (4, question)

def process_question_with_priority(priority_item, cancel_event):
    _, question = priority_item
    return process_question(question, cancel_event)

def _make_cache_key(question):
    q_norm = normalize_text(question or "")
    digest = hashlib.sha256(q_norm.encode('utf-8')).hexdigest()
    return f"{CACHE_PREFIX}:{digest}"

REQUESTS = {}

@app.route("/ask", methods=["POST"])
def ask_endpoint():
    limit_response = enforce_prompt_limit()
    if limit_response is not None:
        return limit_response

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No JSON"}), 400

    question = data.get("question")
    requestId = data.get("requestId") or str(uuid.uuid4())

    if not question:
        return jsonify({"success": False, "error": "Question required"}), 400

    cached_response = cache_manager.get_response(question)
    if cached_response:
        return jsonify({
            "requestId": requestId, 
            "answer": cached_response["answer"],
            "cached": True,
            "timestamp": cached_response.get("timestamp")
        })

    uid = _get_request_user_id()
    cancel_event = threading.Event()
    REQUESTS[requestId] = {"cancel": cancel_event, "owner": uid, "start": time.time()}

    try:
        priority_item = get_request_priority(question)
        future = request_executor.submit(process_question_with_priority, priority_item, cancel_event)
        result = future.result(timeout=100)

        if result.get("success") and "answer" in result:
            cache_manager.store_response(question, result)

        return jsonify({"requestId": requestId, **result})
    except TimeoutError:
        return jsonify({"success": False, "error": "Timeout"}), 408
    finally:
        REQUESTS.pop(requestId, None)

# ✅ SINGLE ROBUST HEALTH CHECK (replaces the duplicate)
@app.route("/health", methods=["GET"])
def health_check():
    """Robust health check that works even with missing dependencies"""
    try:
        # Basic app status
        app_status = "healthy"
        
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
        
        # Vectorstore status with safe imports
        vectorstore_ready = False
        vectorstore_dir_exists = False
        vectorstore_files_exist = False
        folders_loaded = len(ALL_FOLDERS)
        
        try:
            # Try to import retrieval module safely
            from retrieval import vectorstore as retrieval_vectorstore
            vectorstore_ready = retrieval_vectorstore is not None
        except ImportError:
            print("⚠️ retrieval module not available for health check")
        except Exception as e:
            print(f"⚠️ Error checking retrieval: {e}")
        
        try:
            # Try to check VECTORSTORE_DIR safely
            from config import VECTORSTORE_DIR
            if VECTORSTORE_DIR:
                vectorstore_dir_exists = os.path.exists(VECTORSTORE_DIR)
                if vectorstore_dir_exists:
                    required_files = ["index.faiss", "index.pkl"]
                    vectorstore_files_exist = all(
                        os.path.exists(os.path.join(VECTORSTORE_DIR, f)) 
                        for f in required_files
                    )
        except ImportError:
            print("⚠️ config module not available for health check")
        except Exception as e:
            print(f"⚠️ Error checking VECTORSTORE_DIR: {e}")
        
        # Determine overall status - be more lenient
        # App is healthy as long as basic Flask is running
        # Vectorstore is optional for basic functionality
        app_status = "healthy"
        
        return jsonify({
            "status": app_status,
            "folders_loaded": folders_loaded,
            "vectorstore_ready": vectorstore_ready,
            "vectorstore_dir_exists": vectorstore_dir_exists,
            "vectorstore_files_exist": vectorstore_files_exist,
            "redis": redis_status,
            "queue": {
                "active_workers": active_workers,
                "pending_tasks": queue_size
            },
            "timestamp": time.time()
        })
        
    except Exception as e:
        # Ultimate fallback - if even the health check fails
        print(f"❌ Health check itself failed: {e}")
        return jsonify({
            "status": "degraded",
            "error": f"Health check error: {str(e)}",
            "timestamp": time.time()
        }), 500

# --- (everything below remains exactly the same, unchanged) ---

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


def cleanup_executor():
    """Cleanup function to shutdown executor gracefully"""
    print("🔄 Shutting down ThreadPoolExecutor...")
    request_executor.shutdown(wait=False)

# Register cleanup function
import atexit
atexit.register(cleanup_executor)

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
            "time_remaining": round(time_remaining, 1),
            "window_reset_in": f"{int(time_remaining // 60)}m {int(time_remaining % 60)}s"
        })
    else:
        return jsonify({
            "user_id": uid[:8] + "...",
            "status": "No active rate limit window",
            "limit": PROMPTS_PER_WINDOW,
            "window_seconds": WINDOW_SECONDS,
            "requests_remaining": PROMPTS_PER_WINDOW
        })


@app.route("/llm/ask", methods=["POST", "OPTIONS"])
def llm_ask_endpoint():
    """LLM ask endpoint for compatibility with existing frontend"""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "success"}), 200
    
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

    # ⭐⭐⭐ CHECK CACHE FIRST ⭐⭐⭐
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

        # ⭐⭐⭐ STORE IN CACHE IF SUCCESSFUL ⭐⭐⭐
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


@app.route("/llm/cancel", methods=["POST"])
def llm_cancel_endpoint():
    """Cancel endpoint for compatibility with existing frontend"""
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
        print(f"[CANCEL] Cancel requestId={requestId} by {uid}")
        return jsonify({"success": True, "requestId": requestId, "canceled": True})
    except Exception as e:
        print(f"[CANCEL] Error: {e}")
        return jsonify({"success": False, "error": "Cancel error"}), 500
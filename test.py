# test_rate_limit_debug.py
import requests
import time
import json

def test_rate_limits_debug():
    session_id = f"debug_user_{int(time.time())}"
    headers = {'x-session-id': session_id}
    
    print(f"🧪 Testing rate limits for user: {session_id}")
    print(f"📊 Expected: 3 requests per 2 minutes")
    print("=" * 50)
    
    # Single question about drilling procedures
    question = "What is the step-by-step procedure for drilling in mining operations?"
    
    print(f"\n📤 Single Request:")
    print(f"   Question: {question}")
    
    start_time = time.time()
    response = requests.post(
        'http://localhost:5000/llm/ask',
        headers=headers,
        json={'question': question}
    )
    response_time = time.time() - start_time
    
    print(f"   Status: {response.status_code}")
    print(f"   Time: {response_time:.2f}s")
    
    if response.status_code == 429:
        data = response.json()
        print(f"   🚫 RATE LIMITED: {data.get('error')}")
    elif response.status_code == 200:
        data = response.json()
        cached = data.get('cached', False)
        answer = data.get('answer', 'No answer provided')
        # Show first 150 characters of the answer
        preview = answer[:150] + "..." if len(answer) > 150 else answer
        print(f"   ✅ ALLOWED {'(CACHED)' if cached else ''}")
        print(f"   Answer Preview: {preview}")
    else:
        print(f"   ❌ ERROR: {response.status_code}")
        try:
            print(f"   Response: {response.json()}")
        except:
            print(f"   Response: {response.text}")

if __name__ == "__main__":
    test_rate_limits_debug()
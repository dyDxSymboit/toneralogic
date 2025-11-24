# Module imports
import re
import requests
from config import COHERE_API_KEY, COHERE_URL, EMB_MODEL
import retrieval
from security import normalize_text
from transformers import AutoTokenizer
import threading
from threading import Semaphore

# Cohere API throttling
cohere_api_semaphore = Semaphore(8)

# API usage tracking
api_usage_stats = {
    "concurrent_calls": 0,
    "total_calls": 0,
    "rejected_calls": 0
}
api_lock = threading.Lock()

# Thread-safe FAISS initialization
faiss_init_lock = threading.Lock()

def title_score(doc_title: str, question: str) -> float:
    """Calculate title relevance score"""
    if not doc_title:
        return 0.0
    title_norm = normalize_text(doc_title)
    question_norm = normalize_text(question)
    common_words = len(set(title_norm.split()) & set(question_norm.split()))
    return common_words / max(len(title_norm.split()), 1)

def detect_relevant_folders_local(question, all_folders):
    """Local version of folder detection to avoid retrieval import"""
    question_lower = normalize_text(question)
    
    folders = []
    restricted = False
    
    # Simple folder detection logic
    for folder in all_folders:
        folder_lower = normalize_text(folder)
        if folder_lower in question_lower:
            folders.append(folder)
            restricted = True
    
    return folders, restricted

def ask_cohere_smart(question: str, base_top_k: int = 8):
    # Import app's pre-initialized FAISS instead of retrieval directly
    from app import vectorstore as app_vectorstore
    from app import ALL_FOLDERS as app_folders
    
    # API throttling
    acquired = cohere_api_semaphore.acquire(timeout=30)
    if not acquired:
        with api_lock:
            api_usage_stats["rejected_calls"] += 1
        return "❌ Service temporarily overloaded. Please try again in a few moments."
    
    try:
        with api_lock:
            api_usage_stats["concurrent_calls"] += 1
            api_usage_stats["total_calls"] += 1
        
        # Use pre-initialized FAISS from app
        if app_vectorstore is None:
            return "❌ Service unavailable: FAISS index not loaded."
        
        # Step 1: Detect folders using local function
        folders, restricted = detect_relevant_folders_local(question, app_folders)
        print(f"[DEBUG] Folder decision: {folders}, restricted={restricted}")

        # Step 2: Clean query
        query = re.sub(r"only in\s+[^,?]+", "", question, flags=re.IGNORECASE).strip()
        raw_docs_with_score = []

        # Step 3: Check if procedural question
        procedural_phrases = [
            "how to", "how do", "procedure", "steps", "carry out", 
            "process", "method", "conduct", "perform", "execute",
            "way to", "approach for", "carrying out", "performing",
            "examination", "inspection", "check", "verify", "ensure"
        ]
        is_procedural = any(phrase in question.lower() for phrase in procedural_phrases)

        # --- ENHANCED RETRIEVAL ---
        if restricted and folders and is_procedural:
            print(f"[DEBUG] Procedural question - fetching ALL chunks from: {folders}")
            for folder in folders:
                for doc in app_vectorstore.docstore._dict.values():
                    doc_folder = doc.metadata.get("folder", "").strip().lower()
                    if doc_folder == folder.lower():
                        raw_docs_with_score.append((doc, title_score(doc.metadata.get("title",""), question)))
        else:
            if restricted and folders:
                print(f"[DEBUG] Folder-restricted search: {folders}")
                for folder in folders:
                    try:
                        folder_docs = app_vectorstore.similarity_search_with_score(
                            query, k=base_top_k * 3, filter={"folder": folder}
                        )
                        folder_docs = [(doc, score + title_score(doc.metadata.get("title",""), question)) for doc, score in folder_docs]
                        raw_docs_with_score.extend(folder_docs)
                    except Exception as e:
                        print(f"[DEBUG] Error retrieving from folder '{folder}': {e}")
            else:
                raw_docs_with_score = app_vectorstore.similarity_search_with_score(query, k=base_top_k * 8)
                raw_docs_with_score = [(doc, score + title_score(doc.metadata.get("title",""), question)) for doc, score in raw_docs_with_score]

        if not raw_docs_with_score:
            return "❌ No relevant documents found."

        # Step 4: Sort by enhanced score
        raw_docs_with_score.sort(key=lambda x: x[1], reverse=True)
        raw_docs = [doc for doc, _ in raw_docs_with_score if doc is not None]

        # Step 5: Prepare context with source info (IMPORTANT)
        context_chunks = []
        for i, doc in enumerate(raw_docs):
            folder = doc.metadata.get("folder", "unknown")
            # Import clean_context from retrieval
            from retrieval import clean_context
            cleaned = clean_context(doc.page_content, question, restricted)
            # Include source information like cohereSecond.py
            chunk = f"[From: {folder} | Title: {doc.metadata.get('title','unknown')}]\n{cleaned}"
            context_chunks.append(chunk)

        if not context_chunks:
            return "❌ No relevant context found."

        # Step 6: Token-based limiting
        tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)
        total_tokens = 0
        selected_chunks = []
        for chunk in context_chunks:
            tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            if total_tokens + tokens > 8000:
                break
            selected_chunks.append(chunk)
            total_tokens += tokens
        
        context = "\n\n".join(selected_chunks)
        print(f"[DEBUG] Final context: {total_tokens:,} tokens from {len(selected_chunks)} chunks")

        # Step 7: Cohere API call with cohereSecond.py parameters
        system_instruction = (
            "You are an expert in mining operations. "
            "When asked about procedures, methods, or how-to tasks, "
            "provide full, step-by-step detailed explanations exactly from the context. "
            "Do not summarize or add unrelated content. "
            "For definition-type questions, answer concisely. "
            "End your response with 'END'."
        )

        payload = {
            "model": "command-r-08-2024",  # Same model as cohereSecond.py
            "messages": [
                {"role": "system", "content": system_instruction},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nIMPORTANT: Respond strictly from context. Do not include extra information. End with 'END'."
                }
            ],
            "temperature": 0.0,
            "max_tokens": 1500,  # Same as cohereSecond.py
            "stop_sequences": ["END"]  # Same as cohereSecond.py
        }

        headers = {
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(COHERE_URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()

            if "message" in data and "content" in data["message"]:
                contents = data["message"]["content"]
                if isinstance(contents, list):
                    text_parts = [i["text"] for i in contents if isinstance(i, dict) and i.get("type") == "text"]
                    final_answer = "\n".join(text_parts).strip() if text_parts else ""
                elif isinstance(contents, str):
                    final_answer = contents.strip()
                else:
                    final_answer = str(contents).strip()
                
                # Clean up response
                final_answer = re.split(r"(?:\n+---+\n+|\n+Additional Information:|\n+For more info)", final_answer)[0]
                final_answer = re.sub(r"\s*\n{3,}", "\n\n", final_answer).strip()
                return final_answer or "❌ Empty response from Cohere."
            else:
                return "❌ Unexpected Cohere response format."
        except Exception as e:
            print("[DEBUG] Cohere API response error:", e)
            return f"❌ Error: {e}"
            
    finally:
        cohere_api_semaphore.release()
        with api_lock:
            api_usage_stats["concurrent_calls"] -= 1

def get_api_usage():
    """Get API usage statistics"""
    with api_lock:
        return api_usage_stats.copy()
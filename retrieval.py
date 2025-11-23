import re
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from config import VECTORSTORE_DIR, EMB_MODEL
from security import normalize_text

# Global variables (will be initialized in app.py)
vectorstore = None
ALL_FOLDERS = set()

def initialize_retrieval():
    """Initialize FAISS vectorstore and extract folders"""
    global vectorstore, ALL_FOLDERS
    
    # Check if directory exists
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"FAISS vectorstore directory not found: {VECTORSTORE_DIR}\n"
            f"Please ensure the FAISS data directory exists and contains the required files (index.faiss, index.pkl)."
        )
    
    # Check if directory is empty or missing required files
    required_files = ["index.faiss", "index.pkl"]
    missing_files = []
    for file in required_files:
        file_path = os.path.join(VECTORSTORE_DIR, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        raise FileNotFoundError(
            f"FAISS vectorstore directory exists but is missing required files: {', '.join(missing_files)}\n"
            f"Directory: {VECTORSTORE_DIR}\n"
            f"Please ensure all FAISS files are present in the directory."
        )
    
    try:
        print(f"📂 Loading FAISS vectorstore from: {VECTORSTORE_DIR}")
        embedding_model = HuggingFaceEmbeddings(model_name=EMB_MODEL)
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Extract folders
        for d in vectorstore.docstore._dict.values():
            folder = d.metadata.get("folder")
            if folder:
                ALL_FOLDERS.add(folder.strip())
        
        print(f"✅ FAISS loaded with {len(ALL_FOLDERS)} folders")
        return vectorstore, ALL_FOLDERS
    except Exception as e:
        raise RuntimeError(
            f"Failed to load FAISS vectorstore from {VECTORSTORE_DIR}: {str(e)}\n"
            f"Please check:\n"
            f"1. The directory exists and is accessible\n"
            f"2. All required FAISS files (index.faiss, index.pkl) are present\n"
            f"3. The files are not corrupted\n"
            f"4. You have read permissions for the directory"
        ) from e

def detect_relevant_folders(question: str):
    """Detect folder restrictions from question"""
    question_norm = normalize_text(question)
    relevant_folders = []

    # Detect 'only in' pattern
    match = re.search(r"only in\s+([^,?.]+)", question_norm)
    if match:
        target = match.group(1).strip()
        for folder in ALL_FOLDERS:
            folder_norm = normalize_text(folder)
            if target == folder_norm or target in folder_norm or folder_norm in target:
                relevant_folders = [folder]
                break

    if relevant_folders:
        return relevant_folders, True

    return list(ALL_FOLDERS), False

def title_score(doc_title: str, question: str) -> float:
    """Calculate title relevance score for enhanced retrieval"""
    if not doc_title:
        return 0.0
    title_norm = normalize_text(doc_title)
    question_norm = normalize_text(question)
    common_words = len(set(title_norm.split()) & set(question_norm.split()))
    return common_words / max(len(title_norm.split()), 1)

def clean_context(text: str, question: str, restricted: bool) -> str:
    """Clean up retrieved text chunks"""
    text = re.sub(r"(?i)PAGE_START.*?\n", "", text)
    text = re.sub(r"(?i)PAGE_END.*?\n", "", text)
    text = re.sub(r"(?i)User Question:.*?(Answer:.*?)?(?=\nUser Question:|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)Question:.*?(Answer:.*?)?(?=\nQuestion:|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)Answer\s*\(.*?\):", "", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    question_lower = question.lower()
    procedural_phrases = [
        "how to", "how do", "procedure", "steps", "carry out", 
        "process", "method", "conduct", "perform", "execute"
    ]

    is_procedural = any(phrase in question_lower for phrase in procedural_phrases)
    if is_procedural:
        return text.strip()

    definition_phrases = ["what is", "what are", "define", "definition", "explain"]
    is_definition = any(phrase in question_lower for phrase in definition_phrases)
    if is_definition:
        text = re.sub(r"(?i)^\s*(procedure|steps|method):?\s*", "", text)
        text = re.sub(r"(?i)(\n|\s){2,}", " ", text)
        return text.strip()

    return text.strip()
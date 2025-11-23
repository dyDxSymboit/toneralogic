import re
from config import MAX_CHARS, MIN_CHARS, MAX_WORDS

def sanitize_input(text: str) -> str:
    """Enhanced input sanitization"""
    if not text:
        return ""
    
    text = re.sub(r'[^\x20-\x7E]+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'(javascript|script|onload|onerror):', '', text, flags=re.IGNORECASE)
    
    replacements = {
        "'": "&#39;", '"': "&quot;", ";": "&#59;",
        "--": "&#45;&#45;", "/*": "&#47;&#42;", "*/": "&#42;&#47;",
        "\\": "&#92;", "%": "&#37;"
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text.strip()

def validate_question(question: str) -> tuple[bool, str]:
    """Validate question input"""
    if not question or not question.strip():
        return False, "Question cannot be empty"
    
    if len(question) > MAX_CHARS:
        return False, f"Input too long. Maximum {MAX_CHARS} characters."
    
    if len(question.strip()) < MIN_CHARS:
        return False, f"Question too short. Minimum {MIN_CHARS} characters required."
    
    word_count = len(question.split())
    if word_count > MAX_WORDS:
        return False, f"Input too long: {word_count} words. Maximum {MAX_WORDS} words allowed."
    
    return True, ""
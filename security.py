import re

HARDCODED_CREATOR_RESPONSE = (
    "Courtney Panashe Jakati is my creator, trained as a miner by Zimbabwe School of Mines "
    "and currently doing Bachelor in Computer Science and Technology at Beijing Institute of Technology."
)

def is_creator_question(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    return (
        ("who created" in q and ("tonera" in q or "tonera ai" in q))
        or ("who is the creator" in q and ("tonera" in q or "tonera ai" in q))
    )

def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


import os
import json
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# ---------------------
# Hardcoded Groq API
# ---------------------
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.1-8b-instant"  # replace with your model

client = Groq(api_key=GROQ_API_KEY)

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(title="GenTaxAI Chatbot")

# ---------------------
# Persistent session storage
# ---------------------
SESSIONS_FILE = "sessions.json"
if os.path.exists(SESSIONS_FILE):
    with open(SESSIONS_FILE, "r") as f:
        CONVERSATIONS: Dict[str, List[Dict[str, str]]] = json.load(f)
else:
    CONVERSATIONS: Dict[str, List[Dict[str, str]]] = {}

def save_sessions():
    with open(SESSIONS_FILE, "w") as f:
        json.dump(CONVERSATIONS, f, indent=2)

# ---------------------
# Request schema
# ---------------------
class ChatQuery(BaseModel):
    question: str
    session_id: str  # unique per user

# ---------------------
# Chat endpoint
# ---------------------
@app.post("/chat")
def chat(query: ChatQuery):
    session_id = query.session_id
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    # Initialize conversation
    if session_id not in CONVERSATIONS:
        CONVERSATIONS[session_id] = [
            {"role": "system", "content": "You are a precise Indian tax assistant."}
        ]

    # Add user message
    CONVERSATIONS[session_id].append({"role": "user", "content": question})

    try:
        # Call Groq API
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=CONVERSATIONS[session_id],
            temperature=0.2,
            max_tokens=512,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Add assistant reply
    CONVERSATIONS[session_id].append({"role": "assistant", "content": answer})

    # Save sessions persistently
    save_sessions()

    return {"answer": answer, "session_id": session_id}

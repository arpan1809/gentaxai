import os
import json
import uuid
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .knowledge import retrieve

# ---------------------
# Groq API Configuration
# ---------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)

# ---------------------
# FastAPI app
# ---------------------

app = FastAPI(title="GenTaxAI Chatbot", description="AI-powered Indian Tax Assistant")

# Mount static files (ensure ./static exists and contains index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

SESSIONS_FILE = "sessions.json"
if os.path.exists(SESSIONS_FILE):
    with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
        CONVERSATIONS: Dict[str, List[Dict[str, str]]] = json.load(f)
else:
    CONVERSATIONS: Dict[str, List[Dict[str, str]]] = {}

def save_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(CONVERSATIONS, f, indent=2, ensure_ascii=False)

class ChatQuery(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    citations: Optional[List[Dict[str, str]]] = None

class SessionResponse(BaseModel):
    session_id: str
    message: str

SYSTEM_PROMPT = (
    "You are GenTaxAI, a precise and helpful Indian tax assistant.\n"
    "You specialize in Indian taxation including Income Tax, GST, MSME, RBI, SEBI and related compliance.\n"
    "Use the provided CONTEXT snippets as the primary source of truth. If a user asks "
    "for something covered in context, quote or paraphrase that accurately. If the answer "
    "is not in context, answer from your knowledge carefully and clearly say when you are not certain.\n"
    "Always prefer official wording in the snippets when giving definitions or rules.\n"
    "Keep responses concise but comprehensive."
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(content="<h1>GenTaxAI</h1><p>static/index.html not found.</p>", status_code=200)
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/chat", response_model=ChatResponse)
def chat(query: ChatQuery):
    question = (query.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    session_id = query.session_id or str(uuid.uuid4())
    if session_id not in CONVERSATIONS:
        CONVERSATIONS[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Retrieve context
    try:
        top_k = max(1, min(10, int(query.top_k or 5)))
    except Exception:
        top_k = 5

    try:
        kb_hits = retrieve(question, k=top_k) or []
    except Exception as e:
        # Fail open if retriever errors
        kb_hits = []

    # Consolidate context into one assistant message
    citations_payload: List[Dict[str, str]] = []
    if kb_hits:
        context_texts = []
        for i, hit in enumerate(kb_hits, start=1):
            # Ensure required keys exist to avoid KeyError
            source = str(hit.get("source", "knowledge_base"))
            chunk_id = str(hit.get("chunk_id", i))
            text = str(hit.get("text", "")).strip()
            tag = f"[{i}] {source}#chunk{chunk_id}"
            context_texts.append(f"{tag}\n{text}")
            citations_payload.append({"id": str(i), "source": source, "chunk_id": chunk_id})

        context_block = "CONTEXT:\n" + "\n\n".join(context_texts)
        CONVERSATIONS[session_id].append({"role": "assistant", "content": context_block})

    # Add user message
    CONVERSATIONS[session_id].append({"role": "user", "content": question})

    # Call Groq
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=CONVERSATIONS[session_id],
            temperature=0.2,
            max_tokens=800,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Persist assistant reply
    CONVERSATIONS[session_id].append({"role": "assistant", "content": answer})
    save_sessions()

    return ChatResponse(answer=answer, session_id=session_id, citations=citations_payload)

@app.post("/api/new-session", response_model=SessionResponse)
def new_session():
    session_id = str(uuid.uuid4())
    return SessionResponse(session_id=session_id, message="New session created")

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "GenTaxAI Chatbot"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
    )

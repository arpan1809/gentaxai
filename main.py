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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    """Save conversations to persistent storage"""
    with open(SESSIONS_FILE, "w") as f:
        json.dump(CONVERSATIONS, f, indent=2)

# ---------------------
# Request/Response schemas
# ---------------------
class ChatQuery(BaseModel):
    question: str
    session_id: str = None  # Optional, will generate if not provided

class ChatResponse(BaseModel):
    answer: str
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str

# ---------------------
# Routes
# ---------------------
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/api/chat", response_model=ChatResponse)
def chat(query: ChatQuery):
    """Handle chat messages"""
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    
    # Generate session ID if not provided
    session_id = query.session_id or str(uuid.uuid4())

    # Initialize conversation with system prompt
    if session_id not in CONVERSATIONS:
        CONVERSATIONS[session_id] = [
            {
                "role": "system", 
                "content": """You are GenTaxAI, a precise and helpful Indian tax assistant. 
                You specialize in Indian taxation including Income Tax, GST, and other tax-related matters.
                Provide accurate, clear, and actionable advice. If you're unsure about something, 
                recommend consulting a tax professional. Keep responses concise but comprehensive."""
            }
        ]

    # Add user message to conversation
    CONVERSATIONS[session_id].append({"role": "user", "content": question})

    try:
        # Call Groq API
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=CONVERSATIONS[session_id],
            temperature=0.2,
            max_tokens=512,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Add assistant reply to conversation
    CONVERSATIONS[session_id].append({"role": "assistant", "content": answer})

    # Save sessions persistently
    save_sessions()

    return ChatResponse(answer=answer, session_id=session_id)

@app.post("/api/new-session", response_model=SessionResponse)
def new_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    return SessionResponse(session_id=session_id, message="New session created")

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "GenTaxAI Chatbot"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", 8000))
    )
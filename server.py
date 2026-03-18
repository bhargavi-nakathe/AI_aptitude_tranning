# ================================================================
# FILE: server.py
# PURPOSE: A lightweight API server that connects your HTML app
#          to your fine-tuned Mistral model.
#
# WHAT YOU LEARN HERE:
#   - FastAPI: Python library to build web APIs
#   - CORS: allows HTML file to talk to Python server
#   - REST API: how frontend and AI backend communicate
#   - This is EXACTLY how ChatGPT, Gemini work at a basic level
#
# HOW TO RUN:
#   pip install fastapi uvicorn
#   python server.py
#
# Then open aptitude_app.html in browser — it will automatically
# use your fine-tuned Mistral model for hints!
# ================================================================

import json
import random
import torch
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Initialize FastAPI app ────────────────────────────────
app = FastAPI(title="AI Aptitude Trainer API", version="1.0")

# ── CORS — allows HTML file to call this server ───────────
# Without this, browser blocks requests from HTML to Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow all origins (OK for local use)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model variables ────────────────────────────────
_model     = None
_tokenizer = None
_mode      = "stored"   # "finetuned" or "stored"
LORA_PATH  = "./mistral_aptitude_lora"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


# ================================================================
# STARTUP — Load model when server starts
# ================================================================
@app.on_event("startup")
async def startup_event():
    """This runs automatically when server starts."""
    global _model, _tokenizer, _mode

    lora_folder = Path(LORA_PATH)

    if not lora_folder.exists():
        print(f"[server] No model at {LORA_PATH} — using stored hints mode")
        _mode = "stored"
        return

    if not (lora_folder / "adapter_model.safetensors").exists():
        print("[server] Model files incomplete — using stored hints mode")
        _mode = "stored"
        return

    print("[server] Loading fine-tuned Mistral 7B...")
    print("[server] This takes 3-5 minutes on first run...")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        # Load tokenizer
        _tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[server] Using device: {device}")

        if device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, quantization_config=bnb,
                device_map="auto", trust_remote_code=True
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float32,
                device_map="cpu", low_cpu_mem_usage=True
            )

        _model = PeftModel.from_pretrained(base, LORA_PATH)
        _model.eval()
        _mode  = "finetuned"
        print(f"[server] ✓ Fine-tuned model loaded! Mode: {_mode}")

    except Exception as e:
        print(f"[server] Could not load model: {e}")
        print("[server] Falling back to stored hints mode")
        _mode = "stored"


# ================================================================
# API ENDPOINTS — URLs the HTML app calls
# ================================================================

# ── GET /status ───────────────────────────────────────────
# HTML app calls this to check if server is running
# and which mode (finetuned or stored) is active
@app.get("/status")
def status():
    """
    Returns server status and model mode.
    HTML app shows a badge: "🤖 AI Model Active" or "📚 Stored Mode"
    """
    return {
        "status" : "running",
        "mode"   : _mode,
        "message": "Fine-tuned Mistral 7B active" if _mode == "finetuned" else "Using pre-computed hints"
    }


# ── Request model for /hints ──────────────────────────────
class HintRequest(BaseModel):
    """
    This defines what data the HTML sends to get hints.
    Pydantic automatically validates the data types.
    """
    question   : str
    answer     : str
    category   : str
    difficulty : str
    stored_h1  : str = ""   # fallback hint from dataset
    stored_h2  : str = ""   # fallback hint from dataset


# ── POST /hints ───────────────────────────────────────────
# HTML sends question data, server returns hints
@app.post("/hints")
def get_hints(req: HintRequest):
    """
    Generate two hints for a question.

    WHAT HAPPENS:
    1. HTML sends: { question, answer, category, difficulty, stored_h1, stored_h2 }
    2. If fine-tuned model available → Mistral generates fresh hints
    3. If not → return stored hints from dataset (instant)
    4. Returns: { hint1, hint2, mode }

    This is a POST request because we're sending data TO the server.
    GET requests are for fetching data without sending much.
    """
    if _mode == "finetuned" and _model is not None:
        prompt = (
            f"<s>[INST] You are an aptitude training assistant helping a student "
            f"practicing {req.category} at {req.difficulty} level.\n\n"
            f"Question: {req.question}\n\n"
            f"Generate two progressive hints.\n"
            f"Hint 1: subtle conceptual nudge (do NOT reveal the answer)\n"
            f"Hint 2: more direct method guidance\n\n"
            f"Format exactly:\nHint 1: ...\nHint 2: ... [/INST]"
        )
        response = _run_model(prompt, max_tokens=150)
        h1, h2   = _parse_hints(response)
        return {"hint1": h1, "hint2": h2, "mode": "finetuned"}

    # Fallback to stored hints
    return {
        "hint1": req.stored_h1 or "Think about the core concept being tested.",
        "hint2": req.stored_h2 or "Try working through it step by step.",
        "mode" : "stored"
    }


# ── Request model for /explanation ───────────────────────
class ExplanationRequest(BaseModel):
    question   : str
    answer     : str
    solution   : str
    category   : str
    difficulty : str
    stored_exp : str = ""


# ── POST /explanation ─────────────────────────────────────
@app.post("/explanation")
def get_explanation(req: ExplanationRequest):
    """
    Generate a friendly explanation shown after student submits.

    Fine-tuned model gives personalized, encouraging explanations.
    Stored mode returns the pre-written solution text.
    """
    if _mode == "finetuned" and _model is not None:
        prompt = (
            f"<s>[INST] You are a friendly aptitude tutor for {req.category} "
            f"at {req.difficulty} level.\n\n"
            f"Question: {req.question}\n"
            f"Correct Answer: {req.answer}\n\n"
            f"Give a clear, encouraging 3-4 sentence explanation. "
            f"Show the step-by-step logic. End with a memory tip. [/INST]"
        )
        exp = _run_model(prompt, max_tokens=200)
        return {"explanation": exp, "mode": "finetuned"}

    return {
        "explanation": req.stored_exp or req.solution or "No explanation available.",
        "mode"       : "stored"
    }


# ================================================================
# INTERNAL HELPERS
# ================================================================

def _run_model(prompt: str, max_tokens: int = 200) -> str:
    """Run inference on fine-tuned model and return response text."""
    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens     = max_tokens,
            temperature        = 0.7,
            do_sample          = True,
            pad_token_id       = _tokenizer.eos_token_id,
            repetition_penalty = 1.1,
        )

    full     = _tokenizer.decode(output[0], skip_special_tokens=True)
    response = full.split("[/INST]")[-1].strip()
    return response


def _parse_hints(response: str) -> tuple:
    """Extract Hint 1 and Hint 2 from model response text."""
    lines = response.strip().split("\n")
    h1, h2 = "", ""
    for line in lines:
        ll = line.lower().strip()
        if ll.startswith("hint 1") or ll.startswith("hint1"):
            h1 = line.split(":", 1)[-1].strip()
        elif ll.startswith("hint 2") or ll.startswith("hint2"):
            h2 = line.split(":", 1)[-1].strip()
    if not h1 and lines: h1 = lines[0].strip()
    if not h2 and len(lines) > 1: h2 = lines[1].strip()
    return (
        h1 or "Think about the core concept being tested.",
        h2 or "Try working through it step by step."
    )


# ================================================================
# START SERVER
# ================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  AI Aptitude Trainer — Backend Server")
    print("=" * 55)
    print("  Starting server at http://localhost:8000")
    print("  Open aptitude_app.html in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
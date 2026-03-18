# ================================================================
# FILE: model.py
# PURPOSE: Connects the Streamlit app to your fine-tuned model.
#
# TWO MODES (automatic):
#   MODE 1 — Fine-tuned model available:
#             Uses your mistral_aptitude_lora/ folder
#             Runs locally, no internet needed
#
#   MODE 2 — Fine-tuned model NOT yet available:
#             Uses the stored hints/explanations from enriched_questions.json
#             App works perfectly while you wait for training
#
# You don't need to change anything here.
# The app automatically detects which mode to use.
# ================================================================

import os
import torch
from pathlib import Path

# ── Configuration ────────────────────────────────────────────
LORA_PATH      = "./mistral_aptitude_lora"   # Where your trained model lives
MAX_NEW_TOKENS = 180
TEMPERATURE    = 0.7

# ── Global model variables ───────────────────────────────────
# We use globals so the model loads ONCE at startup
# not on every single question
_model     = None
_tokenizer = None
_mode      = "stored"   # "finetuned" or "stored"


def load_model():
    """
    Called once when app starts.
    Tries to load fine-tuned model.
    Falls back to stored mode if not available.
    """
    global _model, _tokenizer, _mode

    lora_folder = Path(LORA_PATH)

    if lora_folder.exists() and any(lora_folder.iterdir()):
        print(f"Fine-tuned model found at {LORA_PATH}")
        print("Loading Mistral + LoRA adapters...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"

            # Load tokenizer from the LoRA folder
            _tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

            # Load base Mistral in 4-bit
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype   = torch.float16,
                device_map    = "auto",
                load_in_4bit  = True,
            )

            # Attach LoRA adapters
            _model = PeftModel.from_pretrained(base, LORA_PATH)
            _model.eval()

            _mode = "finetuned"
            print(f"✓ Fine-tuned model loaded! Mode: finetuned")

        except Exception as e:
            print(f"Could not load fine-tuned model: {e}")
            print("Falling back to stored mode...")
            _mode = "stored"

    else:
        print(f"No fine-tuned model found at {LORA_PATH}")
        print("Running in stored mode — using pre-generated hints from dataset")
        _mode = "stored"


def _run_model(prompt: str) -> str:
    """Run inference on the fine-tuned Mistral model."""
    inputs = _tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens     = MAX_NEW_TOKENS,
            temperature        = TEMPERATURE,
            do_sample          = True,
            pad_token_id       = _tokenizer.eos_token_id,
            repetition_penalty = 1.1,
        )

    full_text = _tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the response part (after [/INST])
    return full_text.split("[/INST]")[-1].strip()


# ================================================================
# PUBLIC FUNCTIONS — Used by app.py
# ================================================================

def get_hints(question: str, answer: str, category: str, difficulty: str,
              stored_h1: str = "", stored_h2: str = "") -> tuple:
    """
    Get two progressive hints for a question.

    Returns:
        (hint1, hint2) — tuple of two strings

    Logic:
        - If fine-tuned model available → generate fresh hints
        - If stored hints exist → return them directly (instant, no API)
        - Fallback → generic hints
    """
    # ── Mode 1: Use fine-tuned Mistral ───────────────────────
    if _mode == "finetuned" and _model is not None:
        prompt = (
            f"<s>[INST] You are an aptitude training assistant helping a student "
            f"practicing {category} at {difficulty} level.\n\n"
            f"Question: {question}\n\n"
            f"Generate two progressive hints. "
            f"Hint 1 should be subtle (conceptual nudge only). "
            f"Hint 2 should be more direct (method guidance). "
            f"Do NOT reveal the answer.\n\n"
            f"Format:\nHint 1: ...\nHint 2: ... [/INST]"
        )
        response = _run_model(prompt)

        # Parse hint1 and hint2 from response
        h1, h2 = _parse_hints(response)
        return h1, h2

    # ── Mode 2: Use stored hints from dataset (instant) ──────
    if stored_h1 and stored_h2:
        return stored_h1, stored_h2

    # ── Fallback ─────────────────────────────────────────────
    return (
        "Think carefully about what concept this question is testing.",
        "Identify what values are given and what formula connects them."
    )


def get_explanation(question: str, answer: str, solution: str,
                    category: str, difficulty: str,
                    stored_exp: str = "") -> str:
    """
    Get a friendly explanation shown after the student submits their answer.

    Logic:
        - If fine-tuned model available → generate personalized explanation
        - If stored explanation exists → return it (instant)
        - Fallback → return the raw solution
    """
    # ── Mode 1: Fine-tuned model ─────────────────────────────
    if _mode == "finetuned" and _model is not None:
        prompt = (
            f"<s>[INST] You are a friendly aptitude tutor for {category} "
            f"at {difficulty} level.\n\n"
            f"Question: {question}\n"
            f"Correct Answer: {answer}\n\n"
            f"The student has submitted their answer. "
            f"Give a clear, encouraging 3-4 sentence explanation. "
            f"Show the step-by-step logic. End with a tip to remember. [/INST]"
        )
        return _run_model(prompt)

    # ── Mode 2: Stored explanation ────────────────────────────
    if stored_exp:
        return stored_exp

    # ── Fallback: raw solution ────────────────────────────────
    return solution


# ── Helper: parse hints from model response ──────────────────
def _parse_hints(response: str) -> tuple:
    """Extract Hint 1 and Hint 2 from model response text."""
    lines = response.strip().split("\n")
    h1, h2 = "", ""

    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith("hint 1") or line_lower.startswith("hint1"):
            h1 = line.split(":", 1)[-1].strip()
        elif line_lower.startswith("hint 2") or line_lower.startswith("hint2"):
            h2 = line.split(":", 1)[-1].strip()

    # Fallback if parsing failed
    if not h1 and len(lines) >= 1:
        h1 = lines[0].strip()
    if not h2 and len(lines) >= 2:
        h2 = lines[1].strip()

    return (
        h1 or "Think about the core concept being tested here.",
        h2 or "Try working through the problem step by step using the given values."
    )
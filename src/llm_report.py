"""
LLM-assisted medical report generation.

IMPORTANT: The LLM never sees the actual X-ray image. It only receives the
structured, numeric output of the deep-learning model (predicted class,
confidence, probability breakdown, and the Grad-CAM attention region as a
short text string). The LLM's job is purely to turn those structured
findings into a well-written, human-readable draft report — it is not
performing any diagnosis of its own. This separation of concerns (DL model
decides "what", LLM decides "how to phrase it") is a deliberate and
important safety design choice for medical AI systems.

Provider order (first available wins):
  1. Anthropic Claude  — set ANTHROPIC_API_KEY (paid, console.anthropic.com)
  2. Google Gemini      — set GEMINI_API_KEY (FREE tier, aistudio.google.com/apikey)
  3. Offline template   — always available, no key needed, no cost

This means the app works fully out of the box with zero API cost, and
upgrades automatically to a real LLM the moment you add either key.
"""
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.config import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS,
    GEMINI_API_KEY, GEMINI_MODEL,
)

# Hard ceiling on how long we'll wait for an external LLM call before giving up
# and falling back. This matters a lot on hosting platforms like Render, where
# the platform's own reverse proxy has its own request timeout — if our code
# waits indefinitely on a hung network call, the proxy eventually kills the
# whole connection and returns a 502 to the user, even though our process is
# still alive and stuck. Failing fast here means we always control the outcome
# ourselves instead of letting the platform time us out first.
LLM_TIMEOUT_SECONDS = 20
_executor = ThreadPoolExecutor(max_workers=4)


def _call_with_timeout(fn, *args, timeout=LLM_TIMEOUT_SECONDS, **kwargs):
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        raise TimeoutError(f"LLM call exceeded {timeout}s timeout")

SYSTEM_PROMPT = """You are an AI assistant that helps draft preliminary radiology-style \
summaries for an educational AI project. You will be given the structured output of a \
deep learning image classifier (NOT a licensed radiologist), applied to a chest X-ray. \
Write a short, clearly-labeled draft report using only the structured data you are given. \
Rules you must always follow:
1. Never claim certainty. Always frame findings as "the model predicts" / "suggests" / \
   "consistent with", never as a confirmed diagnosis.
2. Always include a bolded disclaimer that this is an AI-generated draft for educational \
   purposes only, is NOT a medical diagnosis, and must be reviewed by a licensed radiologist \
   or physician before any clinical use.
3. Keep the tone professional and concise (150-250 words).
4. Structure the report with these sections: Findings, Impression, Recommendation, Disclaimer.
"""


def _build_user_prompt(result: dict) -> str:
    return f"""Structured model output:
- Predicted class: {result['predicted_class']}
- Confidence: {result['confidence']*100:.1f}%
- Full probability breakdown: {result['probabilities']}
- Region of highest model attention (from Grad-CAM): {result['attention_region']}

Please draft the report now."""


def generate_report(result: dict) -> str:
    """
    Turns the structured prediction result into a natural-language draft
    report. Tries Anthropic Claude first, then falls back to Google Gemini
    (free tier), then finally to a deterministic offline template if neither
    API key is configured or both calls fail — so the rest of the pipeline
    (API, DB, frontend) is always demoable, with zero mandatory cost.
    """
    if ANTHROPIC_API_KEY:
        try:
            return _call_with_timeout(_generate_with_anthropic, result)
        except Exception as e:
            print(f"[llm_report] Anthropic call failed, trying next provider: {e}")

    if GEMINI_API_KEY:
        try:
            return _call_with_timeout(_generate_with_gemini, result)
        except Exception as e:
            print(f"[llm_report] Gemini call failed, using offline template: {e}")

    return _fallback_template_report(result)


def _generate_with_anthropic(result: dict) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(result)}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _generate_with_gemini(result: dict) -> str:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n{_build_user_prompt(result)}",
    )
    return response.text


def _fallback_template_report(result: dict) -> str:
    """A deterministic, rule-based report used when no LLM API key is present.
    This guarantees the application is still fully functional end-to-end
    without any external API dependency."""
    pred = result["predicted_class"]
    conf = result["confidence"] * 100
    region = result["attention_region"]

    if pred == "PNEUMONIA":
        findings = (f"The model classifies this chest X-ray as consistent with PNEUMONIA "
                    f"with {conf:.1f}% confidence. Increased model attention (Grad-CAM) was "
                    f"concentrated in the {region}, which may correspond to areas of "
                    f"consolidation or opacity.")
        recommendation = ("Clinical correlation with patient symptoms, vital signs, and "
                           "laboratory findings (e.g. CRP, WBC count) is recommended. "
                           "Follow-up imaging or physician evaluation is advised.")
    else:
        findings = (f"The model classifies this chest X-ray as NORMAL with {conf:.1f}% "
                    f"confidence. No strong localized attention region indicative of "
                    f"consolidation was identified (max attention: {region}).")
        recommendation = ("No acute findings suggested by the model. Routine clinical "
                           "correlation is still recommended as this tool cannot rule out "
                           "conditions outside its trained scope.")

    return f"""**Findings**
{findings}

**Impression**
{pred.title()} pattern suggested by automated deep-learning classifier.

**Recommendation**
{recommendation}

**Disclaimer**
**This report was generated automatically by an AI system for an educational/portfolio project. \
It is NOT a medical diagnosis and must be reviewed and confirmed by a licensed radiologist or \
physician before any clinical decision is made.**"""

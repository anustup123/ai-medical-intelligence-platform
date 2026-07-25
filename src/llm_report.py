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

Requires an Anthropic API key set as the ANTHROPIC_API_KEY environment
variable. Get one at https://console.anthropic.com/
"""
import os

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MAX_TOKENS

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
    Calls the Anthropic Messages API to turn the structured prediction result
    into a natural-language draft report. Falls back to a deterministic
    templated report if no API key is configured, so the rest of the
    pipeline (API, DB, frontend) can still be demoed/tested without an LLM key.
    """
    if not ANTHROPIC_API_KEY:
        return _fallback_template_report(result)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(result)}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
    except Exception as e:  # network / auth / rate-limit errors
        fallback = _fallback_template_report(result)
        return f"{fallback}\n\n[Note: LLM call failed ({e}); showing templated report instead.]"


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

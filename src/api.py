"""
api.py

PURPOSE OF THIS FILE:
This is the backend server — the piece the Streamlit dashboard (and, in
theory, any other external system) talks to. It exposes three clean,
separate endpoints matching the architecture we planned:

  /predict  -> given telemetry values, return a failure risk score + RUL
  /explain  -> given telemetry values, return the SHAP driver breakdown
  /dispatch -> given a device's risk + drivers, return an LLM-generated
               SOP work order

WHY THESE ARE THREE SEPARATE ENDPOINTS RATHER THAN ONE BIG ONE:
Each endpoint does ONE job. This is what makes the architecture
"reusable" — another system could call just /predict without needing the
explanation or SOP generation, or just /explain on its own. Keeping them
separate also means each one is independently testable and swappable.

RUN THIS WITH:  uvicorn src.api:app --reload --port 8000
Then open http://127.0.0.1:8000/docs for FastAPI's automatic interactive
documentation page — useful to show judges "here's how the backend
actually works" without extra effort on your part.
"""

import os
import json
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from .rul_triage import load_triage_weights, build_priority_alert, FEATURE_DISPLAY_NAMES
from .llm_dispatch import configure_llm, generate_sop, generate_sop_offline_fallback

app = FastAPI(title="Medical Equipment Failure Prediction Agent")

# ---------------------------------------------------------------------
# STARTUP: load all trained models and config ONCE when the server
# starts, rather than reloading them on every single request. Loading
# a model from disk is relatively slow — doing it once at startup and
# keeping it in memory is standard practice for any ML-serving API.
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

classifier_realistic = joblib.load(os.path.join(MODELS_DIR, "classifier_realistic.pkl"))
classifier_optimized = joblib.load(os.path.join(MODELS_DIR, "classifier_optimized.pkl"))
category_encoder = joblib.load(os.path.join(MODELS_DIR, "category_encoder_realistic.pkl"))
feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns_realistic.pkl"))
triage_weights = load_triage_weights(os.path.join(CONFIG_DIR, "triage_weights.json"))

# Load the device schema so the API can determine, PER CATEGORY, which
# telemetry fields genuinely apply — this must match the exact same
# logic used when the synthetic training data was generated, or
# predictions get built inconsistently with how the model was trained.
# BUG WE CAUGHT: build_feature_row() previously used "value == 0.0" as a
# stand-in for "this field was missing" — but a real 0.0 reading (e.g.
# zero vibration on a healthy, still device) is a legitimate value, not
# a missing one. Using the schema instead fixes this properly.
with open(os.path.join(CONFIG_DIR, "device_schema.json")) as f:
    DEVICE_SCHEMA = json.load(f)
DEVICE_SCHEMA.pop("_comment", None)

MODELS = {"realistic": classifier_realistic, "optimized": classifier_optimized}

import shap
EXPLAINERS = {name: shap.TreeExplainer(model) for name, model in MODELS.items()}

# ---------------------------------------------------------------------
# LLM CONFIGURATION: try to load the Gemini API key and configure the
# LLM client. We check two sources, in order:
#   1. The GEMINI_API_KEY environment variable (standard, works everywhere)
#   2. A .env file in the project root (convenient for local development)
#
# If neither source has a key, /dispatch gracefully falls back to the
# offline template-based SOP generator — the dashboard never crashes,
# but LLM-generated SOPs are richer and device-specific.
# ---------------------------------------------------------------------
def _load_api_key_from_dotenv() -> str | None:
    """
    Reads the .env file from the project root and extracts
    GEMINI_API_KEY if present. This is a lightweight parser that
    doesn't require python-dotenv — just reads KEY=VALUE lines.
    """
    dotenv_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(dotenv_path):
        return None
    with open(dotenv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "GEMINI_API_KEY":
                return value.strip()
    return None


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or _load_api_key_from_dotenv()
LLM_AVAILABLE = False
LLM_ERROR = None
if GEMINI_API_KEY:
    try:
        configure_llm(GEMINI_API_KEY)
        LLM_AVAILABLE = True
        print(f"[LLM] Gemini configured successfully — LLM SOP generation is ACTIVE")
    except Exception as exc:
        LLM_ERROR = str(exc)
        LLM_AVAILABLE = False
        print(f"[LLM] Gemini configuration failed: {LLM_ERROR}")
        print(f"[LLM] Falling back to offline SOP generation")
else:
    LLM_ERROR = (
        "GEMINI_API_KEY not found. Set it as an environment variable or "
        "add it to the .env file in the project root."
    )
    print(f"[LLM] {LLM_ERROR}")
    print(f"[LLM] Using offline SOP generation fallback")


class TelemetryInput(BaseModel):
    """
    Defines exactly what a /predict or /explain request must contain.
    Using Pydantic here means FastAPI automatically validates incoming
    data — e.g. it will reject a request where 'temperature' is sent as
    text instead of a number, without us writing that check manually.
    This is one of the concrete reasons FastAPI was chosen over Flask.

    model_variant: which trained model to use — "realistic" (~95%
    accuracy, some built-in label randomness, more defensible under
    technical questioning) or "optimized" (~99% accuracy, deterministic
    label, hits an aggressive accuracy target). Defaults to "realistic".
    """
    device_name: str
    classification: str
    ward_criticality: str  # "ICU", "General Ward", or "Storage"
    age_fraction: float
    temperature: float = 0.0
    vibration: float = 0.0
    voltage: float = 0.0
    hours_used: float = 0.0
    manufacturer_risk_tier: str = "low"  # "low", "medium", or "high" — real
    # historical recall-frequency tier for this device's manufacturer,
    # computed from the actual events.csv + manufacturers.csv data by
    # real_data_features.py. Defaults to "low" if unknown.
    model_variant: str = "realistic"
    model_config = ConfigDict(protected_namespaces=())


def build_feature_row(payload: TelemetryInput):
    """
    Converts an incoming API request into the exact feature format the
    trained model expects (same columns, same order, same missing-flag
    logic as build_features() in train_model.py).

    WHY THIS DUPLICATES SOME LOGIC FROM train_model.py:
    In a larger production system, this transformation logic would live
    in one shared module imported by both training and serving code, to
    guarantee they never drift out of sync. For this hackathon-scale
    project, keeping it explicit and visible here makes it easier to
    read end-to-end during a live demo/code review.
    """
    telemetry_fields = ["temperature", "vibration", "voltage", "hours_used"]
    row = {}

    # Look up which fields THIS device's category actually tracks, per
    # device_schema.json — this is the correct source of truth for
    # "missing", matching exactly how the training data was built.
    category_fields = DEVICE_SCHEMA.get(payload.classification, {}).get("fields", [])

    for field in telemetry_fields:
        value = getattr(payload, field)
        field_applies = field in category_fields
        row[f"{field}_was_missing"] = 0 if field_applies else 1
        # If the field doesn't apply to this category, force it to 0
        # regardless of what was sent — matching training, where these
        # fields were never generated at all for that category, not just
        # coincidentally zero.
        row[field] = value if field_applies else 0.0

    row["age_fraction"] = payload.age_fraction

    tier_map = {"low": 0, "medium": 1, "high": 2}
    row["manufacturer_risk_tier_encoded"] = tier_map.get(payload.manufacturer_risk_tier, 0)

    # Encode the category using the SAME encoder fitted during training,
    # so category numbers match what the model actually learned. If a
    # brand-new category not seen during training is passed in, this
    # would raise an error — appropriate here, since the model genuinely
    # has no learned pattern for a category it never trained on.
    row["classification_encoded"] = category_encoder.transform([payload.classification])[0]

    return np.array([[row[col] for col in feature_columns]])


@app.post("/predict")
def predict(payload: TelemetryInput):
    """
    Returns the failure risk score and a priority alert (including RUL
    and triage-adjusted priority) for one device, using whichever model
    variant was requested ("realistic" or "optimized").
    """
    model = MODELS.get(payload.model_variant, classifier_realistic)
    explainer = EXPLAINERS.get(payload.model_variant, EXPLAINERS["realistic"])

    X = build_feature_row(payload)
    risk_score = float(model.predict_proba(X)[0][1])  # probability of class 1 (failed)

    shap_values = explainer.shap_values(X)[0]
    driver_pairs = list(zip(feature_columns, shap_values))
    top_3 = sorted(driver_pairs, key=lambda x: -abs(x[1]))[:3]
    top_drivers = [{"feature": name, "impact": round(float(val), 4)} for name, val in top_3]

    alert = build_priority_alert(
        device_name=payload.device_name,
        classification=payload.classification,
        raw_risk_score=risk_score,
        ward_criticality=payload.ward_criticality,
        triage_weights=triage_weights,
        top_drivers=top_drivers,
    )
    alert["model_variant_used"] = payload.model_variant
    return alert


@app.get("/model_metrics")
def model_metrics():
    """
    Returns the saved accuracy/precision/recall/F1 for BOTH model
    variants side by side — this is what the dashboard's comparison
    view calls, so judges can see the honest tradeoff between the two
    approaches rather than a single cherry-picked number.
    """
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/explain")
def explain(payload: TelemetryInput):
    """
    Returns the FULL SHAP driver breakdown for one device — every
    feature's impact, not just the top 3 shown in /predict. Useful for a
    detailed "why" view in the dashboard.
    """
    explainer = EXPLAINERS.get(payload.model_variant, EXPLAINERS["realistic"])
    X = build_feature_row(payload)
    shap_values = explainer.shap_values(X)[0]
    full_breakdown = [
        {"feature": FEATURE_DISPLAY_NAMES.get(name, name), "impact": round(float(val), 4)}
        for name, val in zip(feature_columns, shap_values)
    ]
    full_breakdown.sort(key=lambda x: -abs(x["impact"]))
    return {"device_name": payload.device_name, "drivers": full_breakdown}


class DispatchInput(BaseModel):
    """Defines what a /dispatch request needs: enough info to write an SOP."""
    device_name: str
    classification: str
    risk_score_pct: float
    top_drivers: list
    telemetry: dict[str, float] = Field(default_factory=dict)
    ward_criticality: str = "Unknown"
    estimated_days_remaining: float | None = None
    recommended_action: str | None = None


@app.post("/dispatch")
def dispatch(payload: DispatchInput):
    """
    Generates a maintenance SOP work order from the device's risk score
    and SHAP drivers, using the LLM if available, or a plain offline
    fallback template if not — this is the resilience behavior in action.
    """
    if LLM_AVAILABLE:
        try:
            sop_text, model_used = generate_sop(
                payload.device_name, payload.classification,
                payload.risk_score_pct, payload.top_drivers,
                payload.telemetry, payload.ward_criticality,
                payload.estimated_days_remaining, payload.recommended_action
            )
            return {"sop": sop_text, "source": "llm", "model": model_used}
        except Exception as exc:
            # If the LLM call fails at request time (e.g. ALL fallback
            # models quota-exhausted mid-demo), fall back gracefully
            # instead of returning an error.
            return {
                "sop": generate_sop_offline_fallback(
                    payload.device_name, payload.classification,
                    payload.risk_score_pct, payload.top_drivers
                ),
                "source": "offline_fallback",
                "llm_error": str(exc),
            }

    sop_text = generate_sop_offline_fallback(
        payload.device_name, payload.classification,
        payload.risk_score_pct, payload.top_drivers
    )
    return {"sop": sop_text, "source": "offline_fallback", "llm_error": LLM_ERROR}


@app.get("/")
def health_check():
    """Simple endpoint to confirm the API is running — useful when demoing."""
    return {
        "status": "running",
        "llm_available": LLM_AVAILABLE,
        "llm_provider": "Gemini" if LLM_AVAILABLE else None,
        "llm_error": LLM_ERROR,
    }


@app.get("/test_llm")
def test_llm():
    """
    Sends a simple test prompt to the Gemini LLM to verify the API
    connection works. Returns the LLM's response or an error message.
    Useful for debugging — hit this endpoint after starting the server
    to confirm LLM is live before running the full dashboard.
    """
    if not LLM_AVAILABLE:
        return {
            "status": "LLM not available",
            "error": LLM_ERROR,
            "hint": "Set GEMINI_API_KEY as an environment variable or "
                    "in the .env file, then restart the API server.",
        }
    try:
        from . import llm_dispatch
        test_prompt = (
            "You are a hospital maintenance assistant. "
            "Reply with exactly one sentence confirming you are working. "
            "Do not add anything else."
        )
        # Try models in the fallback chain until one works
        last_err = None
        for model in llm_dispatch.MODEL_FALLBACK_CHAIN:
            try:
                response = llm_dispatch._client.models.generate_content(
                    model=model, contents=test_prompt
                )
                return {
                    "status": "LLM is working",
                    "response": response.text,
                    "model": model,
                }
            except Exception as exc:
                last_err = exc
                continue
        return {
            "status": "LLM call failed",
            "error": str(last_err),
            "hint": "Check your API key and internet connection.",
        }
    except Exception as exc:
        return {
            "status": "LLM call failed",
            "error": str(exc),
            "hint": "Check your API key and internet connection.",
        }

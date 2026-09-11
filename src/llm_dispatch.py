"""
llm_dispatch.py — LLM-Powered SOP Work Order Generation
=========================================================

WHAT THIS MODULE DOES:
Takes the SHAP explanation (a list of technical feature names and numeric
impact values, e.g. "vibration: +0.25") and turns it into a structured,
plain-English Standard Operating Procedure (SOP) work order that a
maintenance technician could actually read and act on.

This is the "Agentic AI" / GenAI layer of the solution — the part that
demonstrates how large language models add genuine value beyond what a
traditional ML pipeline provides. The ML model says "this device is at
risk and here's why (SHAP drivers)"; the LLM translates that into "here's
what to do about it, in language a human can follow."

WHY AN LLM RATHER THAN HARDCODED TEXT TEMPLATES:
If we tried to write "if vibration is the top driver, say X; if age is
the top driver, say Y" for every possible combination of drivers and
risk levels, we'd need to manually anticipate hundreds of combinations.
An LLM generalizes to combinations we never explicitly wrote for, while
still following a consistent structure because we control that through
the prompt.

WHY WE STILL VALIDATE / STRUCTURE ITS OUTPUT:
LLMs can occasionally produce inconsistent formatting or drift off-topic.
We ask for a specific structure in the prompt (Summary, Root Causes,
Recommended Steps, Urgency) so the output is predictable enough to
display cleanly in the dashboard and export as a document.

RESILIENCE: THE OFFLINE FALLBACK:
The Gemini API might not be available during a demo (no internet, API
quota hit, wrong API key, SDK not installed). Rather than crashing or
showing a blank screen, we generate a DETAILED template-based SOP that
covers the same sections. It's deliberately less polished than the real
LLM output — but it's still genuinely useful, not just a "placeholder"
message. This is part of the broader resilience story: the core risk
score and SHAP drivers should always be actionable, even if the GenAI
layer can't be reached.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GEMINI SDK IMPORT: gracefully handle the case where it's not installed
# ---------------------------------------------------------------------------
# We use try/except rather than a hard import so the rest of the module
# (especially the offline fallback) still works even if google-genai
# isn't installed. This is the "fail gracefully, don't fail entirely"
# pattern — the dashboard stays usable even without the LLM.
# ---------------------------------------------------------------------------
try:
    from google import genai
except ImportError:
    genai = None

_client = None

# Model fallback chain. Each Gemini model has its OWN free-tier per-day
# quota (e.g. 20 requests/day for gemini-3.5-flash). If one model is
# exhausted ("RESOURCE_EXHAUSTED" / HTTP 429), rendered unavailable
# (503 temporary high demand), or retired by Google (404 NOT_FOUND), we
# rotate to the next available model in the chain instead of falling
# back offline. Retired 1.x/2.x models are NOT listed — they just add
# wasted latency while the chain waits for a 404 before moving on.
DEFAULT_MODEL = "gemini-3.5-flash"
MODEL_FALLBACK_CHAIN = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


def sanitize_sop_text(sop_text: str) -> str:
    """Normalize raw LLM output to clean plain text.

    Gemini sometimes returns HTML (or HTML-escaped text) instead of the
    plain text the prompt asks for. If that raw markup reaches the
    dashboard renderer it can display as visible "<div>" tags. This
    converts HTML back to plain text markers (**bold**, "- " bullets)
    so every downstream consumer (Streamlit renderer, .txt download)
    receives consistent, renderable text.

    Idempotent: running it on already-clean plain text is a no-op.
    """
    import re
    if re.search(r"<(?:div|p|span|ul|ol|li|strong|em|br|h[1-6])[\s>]", sop_text, re.IGNORECASE):
        sop_text = re.sub(r"<br\s*/?>", "\n", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"</(?:div|p|li|h[1-6])>", "\n", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<(?:div|p|ul|ol)[^>]*>", "", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<li[^>]*>", "- ", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<(strong|b)>", "**", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"</(strong|b)>", "**", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<(em|i)>", "*", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"</(em|i)>", "*", sop_text, flags=re.IGNORECASE)
        sop_text = re.sub(r"<[^>]+>", "", sop_text)
        sop_text = (sop_text.replace("&amp;", "&")
                            .replace("&lt;", "<")
                            .replace("&gt;", ">")
                            .replace("&nbsp;", " ")
                            .replace("&#8217;", "'")
                            .replace("&#8220;", '"')
                            .replace("&#8221;", '"'))
    return sop_text


def configure_llm(api_key: str):
    """
    Sets up the Gemini client with an API key.

    WHY KEPT AS ITS OWN FUNCTION: if we ever swap providers (Gemini ->
    AWS Bedrock, OpenAI, etc.), only this one function and generate_sop()
    below need to change — nothing else in the pipeline touches the LLM
    directly. The module-level _client variable lets generate_sop()
    reuse the configured client without the key being passed around.

    RAISES RuntimeError if the SDK isn't installed, so the caller
    (api.py) knows to fall back to offline mode rather than silently
    producing bad output.
    """
    if genai is None:
        raise RuntimeError(
            "Gemini SDK is not installed. "
            "Install it with: pip install google-genai"
        )

    global _client
    _client = genai.Client(
        api_key=api_key,
        http_options={
            # NOTE: do NOT add a "timeout" key here — in google-genai it is
            # passed to the TLS layer and makes every handshake time out
            # ("_ssl.c:993 handshake operation timed out") on this SDK
            # version. Keep the client on its default transport timeouts.
            # retry_options attempts=1 makes quota/retired/5xx errors return
            # immediately instead of the SDK sleeping up to ~60s on a 429
            # "retry in 52s" response (which blew past the dashboard's 20s
            # health-probe timeout and made the LLM look offline). Failing
            # fast lets our own MODEL_FALLBACK_CHAIN rotate to the next
            # model instead of hanging.
            "retry_options": {"attempts": 1, "max_delay": 1.0},
        },
    )


def build_prompt(device_name: str,
                 classification: str,
                 risk_score_pct: float,
                 top_drivers: list,
                 telemetry: dict | None = None,
                 ward_criticality: str | None = None,
                 estimated_days_remaining: float | None = None,
                 recommended_action: str | None = None) -> str:
    """
    Builds the exact instruction text sent to the LLM.

    WHY THE PROMPT IS STRUCTURED THIS WAY:
    - We explicitly list the SHAP drivers as plain facts, not asking the
      LLM to guess or invent causes — it should only reason about the
      drivers we actually computed, keeping the output grounded and
      avoiding hallucinated causes that have no basis in the model's
      actual explanation.
    - We request a fixed section structure (Summary / Root Causes /
      Recommended Steps / Urgency) so every SOP looks consistent
      regardless of which device or drivers are involved — this makes
      the dashboard display and any PDF export predictable.
    - We explicitly tell it NOT to invent facts beyond what's given —
      this reduces the risk of the LLM fabricating a cause (e.g.
      "condensation buildup") when the actual SHAP drivers show voltage
      instability and high age fraction.
    """
    drivers_text = "\n".join(
        [f"- {d['feature']}: impact {d['impact']}" for d in top_drivers]
    )
    telemetry_text = "\n".join(
        f"- {name}: {value}" for name, value in (telemetry or {}).items()
    ) or "- No raw telemetry was supplied."

    prompt = f"""You are a hospital maintenance assistant. Based ONLY on the data below,
write a short, structured maintenance work order. Do not invent any facts
not given here.

Device: {device_name}
Category: {classification}
Ward criticality: {ward_criticality or "Unknown"}
Predicted failure risk: {risk_score_pct}%
Estimated days remaining: {estimated_days_remaining if estimated_days_remaining is not None else "Unknown"}
Recommended triage action: {recommended_action or "Use the risk and drivers to determine urgency."}

Device-specific telemetry:
{telemetry_text}

Top contributing factors (from model explainability):
{drivers_text}

Write the work order with exactly these sections:
1. Device Information (device name, category, ward criticality, failure risk, estimated days remaining)
2. Risk Summary (1-2 sentences summarizing the biggest contributing factors)
3. Likely Root Causes (bullet points, based only on the factors above)
4. Recommended Maintenance Steps (bullet points, practical maintenance actions)
5. Safety Considerations (bullet points, hospital-safety precautions)
6. Required Inspection (what inspection is needed and when)
7. Urgency Level (Low / Medium / High, with a one-line justification)
8. Expected Action (one line: the recommended next action for the maintenance team)

CRITICAL FORMATTING RULES:
- Output ONLY plain text. Do NOT use any HTML tags (no <div>, <p>, <strong>, <span>, <ul>, <li>, <br>, or any other HTML markup).
- Use plain text bold with **asterisks** (e.g. **Device Name:** Value) instead of HTML tags.
- Use plain text bullet points with - or • characters, not HTML lists.
- Section titles should be plain text on their own line, not wrapped in any HTML tags.
"""
    return prompt


def generate_sop(device_name: str,
                 classification: str,
                 risk_score_pct: float,
                 top_drivers: list,
                 telemetry: dict | None = None,
                 ward_criticality: str | None = None,
                 estimated_days_remaining: float | None = None,
                 recommended_action: str | None = None,
                 model_name: str = DEFAULT_MODEL) -> tuple:
    """
    Calls the Gemini LLM and returns (generated_sop_text, model_used).

    WHY gemini-3.5-flash: as of 2026, this is the current stable
    Gemini Flash model — fast, high quality for structured short-text
    generation, and the dashboard doesn't feel sluggish when a judge
    clicks "generate SOP" live during a demo. Previous models
    (gemini-2.0-flash, gemini-2.5-flash, gemini-3-flash-preview) have
    been retired by Google by default but can still serve as quota
    fallbacks.

    WHY THE MODEL FALLBACK CHAIN: the free tier caps each model at ~20
    requests/day. Mid-demo that quota can be exhausted, which previously
    caused an abrupt drop to the offline template. Now, if the first
    model raises RESOURCE_EXHAUSTED (429), we automatically retry the
    next model in MODEL_FALLBACK_CHAIN so LLM-generated SOPs keep
    working. (Each model has its own independent free-tier quota.)

    RAISES RuntimeError if the LLM hasn't been configured yet, so the
    caller knows to use the offline fallback rather than crashing. Raises
    the last model error only if EVERY model in the chain fails.
    """
    if _client is None:
        raise RuntimeError(
            "LLM not configured — call configure_llm(api_key) first."
        )
    prompt = build_prompt(
        device_name, classification, risk_score_pct, top_drivers,
        telemetry, ward_criticality, estimated_days_remaining,
        recommended_action,
    )

    chain = [model_name] + [m for m in MODEL_FALLBACK_CHAIN if m != model_name]
    last_error: Exception | None = None
    for candidate in chain:
        try:
            response = _client.models.generate_content(
                model=candidate, contents=prompt
            )
            return sanitize_sop_text(response.text), candidate
        except Exception as exc:  # noqa: BLE001 — try every fallback model
            last_error = exc
            err_str = str(exc).upper()
            if any(tag in err_str for tag in (
                "RESOURCE_EXHAUSTED", "429", "NOT_FOUND", "404",
                "503", "UNAVAILABLE",           # quota / retired / high-demand
                "TIMEOUT", "HANDSHAKE", "CONNECTION", "CONNECT", "SSLError",
                "TRANSPORT", "NETWORK",
            )):
                # Quota, retired-model, transient high-demand, or network
                # blip: try the next model in the chain.
                continue
            # Genuine API errors (auth, invalid key) would fail every
            # model — surface them immediately.
            raise last_error

    raise last_error if last_error is not None else RuntimeError("LLM call failed")


# ===========================================================================
# OFFLINE FALLBACK: detailed, driver-aware SOP generation
# ===========================================================================
# This generates a genuinely useful work order WITHOUT the LLM. It's not
# as polished as LLM output, but it's far more useful than a generic
# "LLM unavailable" placeholder — it includes driver-specific guidance,
# risk-tier-aware urgency, and practical next steps. The goal is that
# even in offline mode, a maintenance technician could read this and know
# what to do.
# ===========================================================================

# Maps SHAP feature names — in BOTH raw and human-readable form — to
# plain-English maintenance implications. top_drivers arriving via the
# API have already been translated by rul_triage._translate_driver_names
# to display names ("Device Age"), so we key on those, but also accept
# the raw internal names as aliases so the fallback works even if a
# caller passes untranslated drivers.
_DRIVER_MAINTENANCE_HINTS = {
    "Device Age": (
        "The device is near the end of its expected service life. "
        "Evaluate whether replacement is more cost-effective than repair."
    ),
    "age_fraction": (
        "The device is near the end of its expected service life. "
        "Evaluate whether replacement is more cost-effective than repair."
    ),
    "Temperature": (
        "Abnormal temperature readings suggest cooling system degradation "
        "or internal component overheating. Inspect cooling fans, "
        "heat sinks, and ventilation pathways."
    ),
    "temperature": (
        "Abnormal temperature readings suggest cooling system degradation "
        "or internal component overheating. Inspect cooling fans, "
        "heat sinks, and ventilation pathways."
    ),
    "Vibration": (
        "Elevated vibration indicates mechanical wear — check bearings, "
        "motor mounts, and moving parts for looseness or damage."
    ),
    "vibration": (
        "Elevated vibration indicates mechanical wear — check bearings, "
        "motor mounts, and moving parts for looseness or damage."
    ),
    "Voltage": (
        "Voltage instability suggests power supply degradation. Inspect "
        "power circuits, capacitors, and voltage regulators."
    ),
    "voltage": (
        "Voltage instability suggests power supply degradation. Inspect "
        "power circuits, capacitors, and voltage regulators."
    ),
    "Usage Hours": (
        "High cumulative usage hours correlate with general wear across "
        "all components. Perform a comprehensive multi-point inspection."
    ),
    "hours_used": (
        "High cumulative usage hours correlate with general wear across "
        "all components. Perform a comprehensive multi-point inspection."
    ),
    "Device Category": (
        "The device category itself is a significant risk factor — "
        "check manufacturer-specific maintenance bulletins for this "
        "device type."
    ),
    "classification_encoded": (
        "The device category itself is a significant risk factor — "
        "check manufacturer-specific maintenance bulletins for this "
        "device type."
    ),
    "Manufacturer Recall History": (
        "This device's manufacturer has a notable recall history. "
        "Cross-reference with recent recall notices for this model."
    ),
    "manufacturer_risk_tier_encoded": (
        "This device's manufacturer has a notable recall history. "
        "Cross-reference with recent recall notices for this model."
    ),
}


def _get_risk_level(risk_score_pct: float) -> tuple:
    """
    Returns (level_name, urgency_text, color_hint) based on risk score.
    Used by the offline fallback to generate tier-appropriate language.
    """
    if risk_score_pct >= 70:
        return (
            "HIGH",
            "This device requires IMMEDIATE attention. Schedule inspection "
            "within 24-48 hours.",
            "critical",
        )
    elif risk_score_pct >= 40:
        return (
            "MEDIUM",
            "This device should be inspected within the next scheduled "
            "maintenance window (within 1-2 weeks).",
            "warning",
        )
    else:
        return (
            "LOW",
            "No immediate action required. Continue routine monitoring "
            "at the next scheduled maintenance cycle.",
            "routine",
        )


def _get_driver_guidance(top_drivers: list) -> str:
    """
    Builds driver-specific maintenance guidance by matching each top
    SHAP driver against our hint dictionary. This makes the offline
    SOP genuinely specific to this device's risk profile, not a
    generic template.
    """
    lines = []
    for driver in top_drivers:
        feature = driver["feature"]
        impact = driver["impact"]
        hint = _DRIVER_MAINTENANCE_HINTS.get(feature)

        if hint:
            lines.append(f"  - [{feature}] (impact: {impact:+.4f}): {hint}")
        else:
            lines.append(
                f"  - [{feature}] (impact: {impact:+.4f}): "
                f"Investigate this factor further during inspection."
            )

    return "\n".join(lines) if lines else "  - No specific driver guidance available."


def generate_sop_offline_fallback(device_name: str,
                                  classification: str,
                                  risk_score_pct: float,
                                  top_drivers: list) -> str:
    """
    Generates a DETAILED, driver-aware SOP work order without the LLM.

    WHY THIS IS MORE THAN A PLACEHOLDER:
        An earlier version of this function returned a generic 3-line
        "LLM unavailable" message. That's technically correct but
        practically useless — a maintenance technician staring at that
        screen still doesn't know what to do. This version generates a
        full structured work order with:
          - Risk-tier-appropriate urgency language
          - Driver-specific maintenance guidance
          - Practical next steps based on the risk level
          - Clear sections matching the LLM output structure

        The result is less polished than what the LLM produces, but
        it's genuinely actionable — the dashboard and the maintenance
        team can use it immediately, even offline.

    WHY WE DON'T CAP THIS AT 100 or add fake confidence:
        The offline fallback is honest about what it is: a template-based
        approximation, not an AI-generated narrative. We label it clearly
        as "[OFFLINE MODE]" so users know the LLM wasn't involved, but
        we still make it useful rather than just a warning message.
    """
    level_name, urgency_text, _ = _get_risk_level(risk_score_pct)
    driver_guidance = _get_driver_guidance(top_drivers)

    # Build a driver summary line for the header
    driver_names = [d["feature"] for d in top_drivers]
    driver_summary = ", ".join(driver_names) if driver_names else "None identified"

    return sanitize_sop_text(f"""{'='*60}
  SOP WORK ORDER (Offline Mode)
  Generated: Template-based — LLM was not available
{'='*60}

DEVICE INFORMATION
  Name:         {device_name}
  Category:     {classification}
  Failure Risk: {risk_score_pct}%

RISK SUMMARY
  The device shows elevated predicted failure risk, primarily
  associated with: {driver_summary}.

LIKELY ROOT CAUSES
{driver_guidance}

RECOMMENDED MAINTENANCE STEPS:
  1. Pull the device maintenance history from the hospital CMMS.
  2. Perform a visual inspection of all components related to the
     factors listed above.
  3. Run diagnostic tests specific to the device category
     ({classification}).
  4. Document all findings and update the device's risk status.
  5. If the device is in a critical ward, coordinate with the ward
     supervisor before any shutdown for maintenance.

SAFETY CONSIDERATIONS
  - Follow the facility's lockout/tagout procedure before any
    maintenance work on powered equipment.
  - Isolate the device from clinical use during inspection and
    notify the ward supervisor of the estimated downtime.
  - Use manufacturer-approved replacement parts and follow the
    original equipment service manual.

REQUIRED INSPECTION
  Periodic inspection on the next scheduled maintenance window for
  all components tied to the contributing factors above.

URGENCY LEVEL: {level_name}
  {urgency_text}

{'='*60}
  NOTE: This work order was generated offline. A more detailed,
  narrative SOP can be generated once the LLM service is available.
{'='*60}
""")

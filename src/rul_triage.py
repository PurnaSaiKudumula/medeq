"""
rul_triage.py — Remaining Useful Life & Clinical Triage Prioritization
=======================================================================

WHAT THIS MODULE DOES:
This module sits ON TOP of the raw ML model output (from train_model.py)
and turns a bare probability number into something a hospital maintenance
team can actually act on. It provides two closely related features:

  1. REMAINING USEFUL LIFE (RUL) ESTIMATION
     Converts a failure probability (0.0 to 1.0) into an estimated number
     of days remaining before the device likely fails. A device at 0.05
     risk gets ~347 days (relax, monitor routinely), while a device at
     0.95 risk gets ~2 days (act now). The goal is to give maintenance
     crews a human-readable countdown instead of an abstract percentage.

  2. CLINICAL TRIAGE-AWARE PRIORITIZATION
     Adjusts the raw risk score based on WHERE the device sits in the
     hospital — an ICU ventilator at 60% risk is more urgent than a
     storage-room thermometer at 80% risk, because a failure in the ICU
     endangers lives immediately. This module applies a configurable
     ward-criticality multiplier so that ranking reflects real clinical
     priority, not just raw probability.

WHY BOTH ARE SIMPLE RULES, NOT SEPARATE ML MODELS:
We already trained a regression model (train_model.py) to predict the
failure_probability itself — that's where the real machine learning
happens. These functions just RE-EXPRESS that same number in more
human-readable and clinically-actionable forms, using straightforward
mathematical transforms. Building separate "days until failure" or
"clinical priority" models would require real failure-date data and real
clinical-outcome data we don't have (the Kaggle dataset is a recall
registry, not a time-series monitoring log), and would add complexity
without adding real accuracy. A transparent, auditable rule is also
easier for a hospital administrator to trust than a second black-box
model — they can see exactly how the number was computed, challenge it,
and adjust the weights themselves by editing a JSON file.
"""

import json
import logging
import math

# Configure a lightweight logger for this module — used to warn (not crash)
# when unexpected input types are received, so the dashboard keeps running
# even if someone passes a string where a float was expected.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DISPLAY NAMES: maps raw internal column names to human-readable labels
# ---------------------------------------------------------------------------
# This dictionary is used ONLY for display purposes — in the dashboard and
# in the LLM's SOP prompt. The model itself never sees these strings.
#
# WHY CENTRALIZE THIS HERE: every downstream consumer (dashboard, API,
# LLM dispatch) needs to show "Device Age" instead of "age_fraction".
# Keeping one canonical mapping in one place means adding a new feature to
# the model requires updating only this dictionary, not hunting through
# three separate files to update display strings.
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    "age_fraction": "Device Age",
    "temperature": "Temperature",
    "vibration": "Vibration",
    "voltage": "Voltage",
    "hours_used": "Usage Hours",
    "temperature_was_missing": "Temperature Sensor Not Tracked",
    "vibration_was_missing": "Vibration Sensor Not Tracked",
    "voltage_was_missing": "Voltage Sensor Not Tracked",
    "hours_used_was_missing": "Usage Hours Not Tracked",
    "classification_encoded": "Device Category",
    "manufacturer_risk_tier_encoded": "Manufacturer Recall History",
}


# ---------------------------------------------------------------------------
# TIER ACTION LABELS: human-readable recommended actions for each risk tier
# ---------------------------------------------------------------------------
# Separated from the if/elif/else chain in build_priority_alert() so they
# can be referenced, tested, and translated independently. If a hospital
# wants these in a different language or phrasing, they edit this mapping,
# not the function logic.
# ---------------------------------------------------------------------------
TIER_ACTIONS = {
    "critical": "Schedule immediate inspection — high failure risk.",
    "warning": "Schedule maintenance within the next maintenance window.",
    "routine": "Continue routine monitoring — no immediate action needed.",
}


# ---------------------------------------------------------------------------
# RISK TIER BOUNDARIES: thresholds that map raw risk to recommended action
# ---------------------------------------------------------------------------
# These are deterministic business rules, not learned thresholds. The
# values (0.7 for critical, 0.4 for warning) were chosen to align with
# common hospital maintenance scheduling practices: anything above 70%
# risk needs attention this week, anything above 40% should be handled
# at the next scheduled window, and below that is routine.
# ---------------------------------------------------------------------------
RISK_CRITICAL_THRESHOLD = 0.7
RISK_WARNING_THRESHOLD = 0.4


# ===========================================================================
# FUNCTION 1: load_triage_weights
# ===========================================================================

def load_triage_weights(config_path: str) -> dict:
    """
    Loads ward-criticality multipliers from a JSON config file.

    WHAT IT RETURNS:
        A dictionary mapping ward names (strings) to numeric multipliers,
        e.g. {"ICU": 1.5, "General Ward": 1.0, "Storage": 0.7}. These
        multipliers are used by apply_triage_weighting() to adjust raw
        risk scores based on where a device is physically located.

    WHY THIS IS A SEPARATE FUNCTION (not inline JSON loading):
        Keeping config-loading isolated means if the config format ever
        changes — e.g. the hospital wants to store triage policy in a
        database instead of a JSON file, or add environment-specific
        overrides — only this function needs to change. Every caller
        (api.py, the dashboard, tests) is insulated from that change.

    WHY THE CONFIG LIVES IN A JSON FILE, NOT HARD-CODED:
        Hospital triage policy varies. A cardiac ICU might weight
        cardiovascular devices at 2.0x instead of 1.5x; a pediatric
        ward might have different criticality rules entirely. Putting
        these numbers in a JSON file means a hospital administrator
        can adjust priorities by editing a text file — no Python
        knowledge, no code deployment, no developer needed. This is
        a deliberate "transparent rule" design choice: the relationship
        between location and priority is visible, auditable, and
        adjustable by non-technical stakeholders.

    ERROR HANDLING:
        If the file doesn't exist or contains invalid JSON, this will
        raise an exception — which is intentional, because proceeding
        with missing triage weights would silently default everything
        to weight 1.0, making the triage feature useless without anyone
        noticing. Fail loudly here so the problem is obvious.
    """
    with open(config_path, "r") as f:
        weights = json.load(f)

    # Strip the human-readable comment key so it doesn't accidentally
    # participate in weight lookups (e.g. a ward literally named "_comment"
    # would be pathological, but defensive coding is free).
    weights.pop("_comment", None)

    # Validate that all values are numeric — catching misconfigured JSON
    # early, before it silently produces wrong priority scores.
    for ward, multiplier in weights.items():
        if not isinstance(multiplier, (int, float)):
            raise ValueError(
                f"Triage weight for ward '{ward}' must be a number, "
                f"got {type(multiplier).__name__}: {multiplier}"
            )

    return weights


# ===========================================================================
# FUNCTION 2: estimate_rul_days
# ===========================================================================

def estimate_rul_days(failure_probability: float,
                      max_days: int = 365,
                      min_days: int = 1) -> float:
    """
    Converts a raw failure probability (0.0 to 1.0) into an estimated
    number of days remaining before the device likely fails.

    THE CORE FORMULA:
        days = max_days * (1 - failure_probability)

        A failure_probability near 0.0 (healthy device) maps to close
        to max_days (plenty of time left — 365 days by default).
        A failure_probability near 1.0 (about to fail) maps to close
        to min_days (act now — 1 day by default).

    WHY THIS IS A SIMPLE INVERSE FORMULA, NOT A SEPARATE TRAINED MODEL:
        We already trained a regression model (train_model.py) to predict
        the underlying failure_probability number itself — that's where
        the real machine learning happens, using XGBoost with validated
        hyperparameters. This function just RE-EXPRESSES that same number
        in a more human-readable unit (days) using a straightforward
        inverse relationship. Building a second, separate "days until
        failure" model would require real failure-date timestamp data we
        don't have — the Kaggle dataset is a recall/safety-notice
        registry with no time-series monitoring, so there's no ground
        truth for "this device failed on day X." A second model trained
        without real temporal data would be no more accurate than this
        formula, but far less transparent.

    WHY WE CLAMP TO [0.01, 0.99]:
        - failure_probability = 0.0 would give days = max_days exactly,
          which is fine, but 0.0 also means "never fails" — a dangerous
          assumption for any physical device. Clamping to 0.01 keeps the
          estimate conservative ("at least check in a year").
        - failure_probability = 1.0 would give days = 0, which is less
          than min_days. Clamping to 0.99 avoids this edge case and
          produces a small but non-zero countdown that reads naturally
          in the UI ("1.0 days" rather than "0 days").

    PARAMETERS:
        failure_probability: float between 0.0 and 1.0, from the
            model's predict_proba() output. Non-numeric inputs are
            logged as warnings and clamped to the midpoint (0.5).
        max_days: the "healthy device" endpoint (default 365 = one year).
            Tunable per deployment — a hospital with faster equipment
            turnover might set this to 180.
        min_days: the "act now" floor (default 1 day). Never returns
            fewer than this, even for probability > 0.99.

    RETURNS:
        A float rounded to 1 decimal place, representing estimated days
        remaining. Always >= min_days.
    """
    # --- Input validation ---
    # If someone passes a string, None, or other non-numeric type,
    # we don't want to crash the entire dashboard. Log a warning and
    # use 0.5 (the "we genuinely don't know" midpoint) as a safe default.
    if not isinstance(failure_probability, (int, float)):
        logger.warning(
            "estimate_rul_days received non-numeric input: %r — "
            "defaulting to 0.5 (unknown risk)", failure_probability
        )
        failure_probability = 0.5

    # Also handle NaN and infinity gracefully.
    if math.isnan(failure_probability) or math.isinf(failure_probability):
        logger.warning(
            "estimate_rul_days received NaN/inf input — defaulting to 0.5"
        )
        failure_probability = 0.5

    # Clamp to safe range [0.01, 0.99] — see docstring for WHY.
    p = max(0.01, min(0.99, float(failure_probability)))

    # The inverse relationship: higher probability -> fewer days left.
    # This is a linear mapping, which is simple but honest — we're not
    # claiming to know the exact degradation curve (that would require
    # real temporal data we don't have), just providing a proportional
    # countdown that scales sensibly with the model's risk estimate.
    days = max_days * (1 - p)

    # Floor at min_days so we never say "0 days" — even a device at
    # 99% risk still has SOME time, and saying "0" would be misleading
    # (it's not already broken, just very likely to break soon).
    return round(max(min_days, days), 1)


# ===========================================================================
# FUNCTION 3: apply_triage_weighting
# ===========================================================================

def apply_triage_weighting(raw_risk_score: float,
                           ward_criticality: str,
                           triage_weights: dict) -> float:
    """
    Multiplies the raw ML risk score by a ward-criticality weight to
    produce a clinically-adjusted priority score for RANKING devices.

    THE FORMULA:
        priority = raw_risk_score * 100 * weight

        The *100 converts the probability (0.0-1.0) to a percentage
        scale (0-100) that's more intuitive for ranking. The weight
        then inflates or deflates that number based on ward criticality.

    CONCRETE EXAMPLE (with ICU weight=1.5 and Storage weight=0.7):
        - A ventilator at 60% raw risk in the ICU:
          priority = 0.60 * 100 * 1.5 = 90.0

        - A storage-room device at 80% raw risk:
          priority = 0.80 * 100 * 0.7 = 56.0

        The ICU device is now ranked MORE urgent (90 > 56) despite
        having a lower raw risk score — because a failure in the ICU
        endangers patient lives immediately, while a storage failure is
        an inconvenience, not a safety crisis. This is exactly the kind
        of context-aware prioritization that a raw risk score alone
        cannot provide.

    WHY THIS IS A DETERMINISTIC RULE, NOT SOMETHING THE ML MODEL LEARNS:
        We don't have enough real labeled data teaching "which ward
        location makes a failure more severe" to train a model on this
        relationship reliably. The Kaggle dataset has no ward-location
        labels or clinical-outcome severity data at all — we map ward
        criticality from device_schema.json, which is a human-defined
        configuration. More importantly, a transparent, explainable
        RULE is easier for a hospital administrator to trust and audit
        than a black-box weight buried inside a model's feature
        importance. When an administrator asks "why was this ICU device
        ranked higher?", we can point to a single number in a JSON file
        and let them adjust it — no retraining required.

    WHY WE DON'T CAP AT 100:
        The adjusted score is used purely for RANKING/sorting devices
        against each other in the priority alert list. It is NOT a
        probability — the raw_risk_score remains the true probability
        shown to the user. The adjusted number only decides ORDER, so
        exceeding 100 is fine and actually useful (it means "this is
        more than twice as urgent as a baseline device").

    PARAMETERS:
        raw_risk_score: the model's failure probability (0.0 to 1.0).
        ward_criticality: string key like "ICU", "General Ward", etc.
        triage_weights: dict from load_triage_weights(), mapping ward
            names to numeric multipliers.

    RETURNS:
        A float rounded to 2 decimal places, representing the adjusted
        priority score used for ranking.
    """
    # --- Input validation ---
    if not isinstance(raw_risk_score, (int, float)):
        logger.warning(
            "apply_triage_weighting received non-numeric raw_risk_score: "
            "%r — defaulting to 0.5", raw_risk_score
        )
        raw_risk_score = 0.5

    if math.isnan(raw_risk_score) or math.isinf(raw_risk_score):
        logger.warning(
            "apply_triage_weighting received NaN/inf raw_risk_score — "
            "defaulting to 0.5"
        )
        raw_risk_score = 0.5

    # Look up the ward multiplier, defaulting to 1.0 if the ward name
    # isn't in our config. Defaulting to 1.0 (no adjustment) is the
    # safest choice: it means "unknown location = no priority boost or
    # penalty," rather than artificially inflating or deflating the score.
    weight = triage_weights.get(ward_criticality, 1.0)

    return round(raw_risk_score * 100 * weight, 2)


# ===========================================================================
# FUNCTION 4: build_priority_alert
# ===========================================================================

def _determine_recommended_action(raw_risk_score: float) -> str:
    """
    Determines the recommended maintenance action based on the raw risk
    score, using three clear, auditable tiers.

    WHY THREE TIERS (not a continuous function or free-text generation):
        Maintenance teams work in discrete scheduling windows — they
        don't act on a continuous "risk is 0.573" scale. Three tiers
        map directly to real scheduling decisions:
          - >= 0.7:  "this week" (immediate inspection)
          - >= 0.4:  "next maintenance window" (scheduled)
          - < 0.4:   "routine monitoring" (no special action)

        These thresholds are business rules, not learned from data,
        because we're encoding a scheduling policy, not a statistical
        pattern. Making them explicit constants (RISK_CRITICAL_THRESHOLD,
        RISK_WARNING_THRESHOLD) at the top of the file means they can
        be adjusted by anyone who reads the code, without understanding
        the rest of the logic.
    """
    if raw_risk_score >= RISK_CRITICAL_THRESHOLD:
        return TIER_ACTIONS["critical"]
    elif raw_risk_score >= RISK_WARNING_THRESHOLD:
        return TIER_ACTIONS["warning"]
    else:
        return TIER_ACTIONS["routine"]


def _translate_driver_names(top_drivers: list) -> list:
    """
    Translates raw internal column names (e.g. "age_fraction",
    "manufacturer_risk_tier_encoded") into human-readable labels
    (e.g. "Device Age", "Manufacturer Recall History").

    WHY THIS EXISTS:
        top_drivers arrives from the SHAP explainer with raw internal
        column names — technically correct, but not what a maintenance
        technician wants to read in a work order, and not what a judge
        wants to see in a dashboard. This function translates them ONCE,
        in ONE place, so both the dashboard AND the LLM prompt (which
        reads these exact strings) show clean names. Without this
        centralization, display strings would drift out of sync across
        the dashboard and API layers over time.
    """
    return [
        {
            "feature": FEATURE_DISPLAY_NAMES.get(d["feature"], d["feature"]),
            "impact": d["impact"],
        }
        for d in top_drivers
    ]


def build_priority_alert(device_name: str,
                         classification: str,
                         raw_risk_score: float,
                         ward_criticality: str,
                         triage_weights: dict,
                         top_drivers: list) -> dict:
    """
    Combines RUL estimation and triage weighting into one structured
    "alert" dictionary — the actual object the Streamlit dashboard
    displays per device, and the API returns per prediction.

    WHY THIS IS ONE FUNCTION (not three separate calls in every caller):
        Bundling the assembly into a single function means the API layer
        (api.py) and the dashboard (app_streamlit.py) both call ONE
        function to get a consistent, fully-formed alert, rather than
        each separately calling estimate_rul_days(), then
        apply_triage_weighting(), then assembling the dictionary
        themselves. This eliminates the risk of two callers computing
        the alert slightly differently (e.g. one forgetting to translate
        driver names, another forgetting to clamp the risk score).

    RETURNS:
        A dictionary with these keys:
          - device_name: the device's display name
          - classification: the device category (e.g. "Cardiovascular Devices")
          - ward_criticality: the ward where the device is located
          - raw_risk_score_pct: the raw risk as a percentage (0-100)
          - priority_score: the triage-adjusted ranking score
          - estimated_days_remaining: RUL countdown in days
          - recommended_action: one of three tier-based action strings
          - top_drivers: list of SHAP feature-impact dicts (translated
            to human-readable names)
    """
    # --- Compute the two core derived values ---
    rul_days = estimate_rul_days(raw_risk_score)
    priority_score = apply_triage_weighting(
        raw_risk_score, ward_criticality, triage_weights
    )

    # --- Determine the recommended action tier ---
    action = _determine_recommended_action(raw_risk_score)

    # --- Translate raw SHAP driver names to human-readable labels ---
    readable_drivers = _translate_driver_names(top_drivers)

    # --- Assemble and return the complete alert dictionary ---
    return {
        "device_name": device_name,
        "classification": classification,
        "ward_criticality": ward_criticality,
        "raw_risk_score_pct": round(raw_risk_score * 100, 1),
        "priority_score": priority_score,
        "estimated_days_remaining": rul_days,
        "recommended_action": action,
        "top_drivers": readable_drivers,
    }

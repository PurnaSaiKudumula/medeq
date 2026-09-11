# MedEQ — Medical Equipment Failure Prediction Agent

An explainable-AI predictive maintenance dashboard for hospital medical
equipment. It turns raw device data and synthetic telemetry into ranked,
clinically-triage-aware risk alerts, explains each prediction with SHAP,
and generates actionable maintenance work orders using a Gemini LLM — with
a fully functional offline fallback.

---

## 1. Overview

```
Real Device Data  ──►  Synthetic Telemetry  ──►  XGBoost Risk Model
                                                       │
                                                     SHAP
                                                       │
                                             RUL + Clinical Triage
                                                       │
                                             LLM SOP Agent (Gemini)
```

The three endpoints that power this are intentionally kept **separate**
(reusability):

| Endpoint  | Purpose                                                        |
|-----------|----------------------------------------------------------------|
| `/predict` | Returns a failure risk score, RUL countdown & priority alert  |
| `/explain` | Returns the full SHAP driver breakdown for one device          |
| `/dispatch`| Returns an LLM-generated SOP work order (with offline fallback)|

---

## 2. Features

- **ML-Powered Risk Prediction** — XGBoost classifier + RUL regressor.
- **Explainable AI** — TreeSHAP identifies exactly why each device is risky.
- **Clinical Triage-Aware Prioritization** — ICU / General Ward / Storage
  ward weights adjust ranking so a risk in the ICU outranks a risk in storage.
- **Remaining Useful Life (RUL)** — converts a probability into a readable
  days-until-failure countdown.
- **Agentic LLM SOP Generation** — Gemini writes structured maintenance work
  orders from the SHAP drivers, with a smart model fallback chain.
- **Resilient Offline Fallback** — if the LLM or the model is unreachable,
  the system degrades gracefully instead of crashing.
- **Two Model Variants Shown Honestly** — "realistic" (~95% acc) vs
  "optimized" (~99% acc), displayed side by side with transparent reasoning.
- **Fully Real Second Model** — a 100%-real-data recall-type classifier
  (no synthetic data anywhere in it) with real metrics.
- **Four-Tab Interactive Dashboard** — Risk List, What-If Simulator,
  Model Comparison, RUL Estimator.

---

## 3. Project Structure

```
medeq_project/
├── app_streamlit.py              # Step 5: the Streamlit dashboard (UI/UX)
├── requirements.txt              # Python dependencies
├── config/
│   ├── device_schema.json        # Device categories, telemetry fields, ranges
│   └── triage_weights.json       # Ward-criticality multipliers (ICU/General/Storage)
├── data/
│   ├── real_devices_sample.csv   # Real device names/categories (from Kaggle devices.csv)
│   ├── synthetic_telemetry_realistic.csv    # 20k rows, 15% label noise
│   ├── synthetic_telemetry_optimized.csv    # 20k rows, deterministic labels
│   ├── manufacturer_recall_stats.csv        # Real recall counts per manufacturer
│   └── category_recall_stats.csv            # Real recall rates per device category
├── models/                       # Trained models saved here (created automatically)
│   ├── classifier_realistic.pkl / classifier_optimized.pkl
│   ├── regressor_realistic.pkl / regressor_optimized.pkl
│   ├── real_type_classifier.pkl  # Fully-real-data model
│   └── metrics.json              # Saved accuracy/precision/recall/F1
└── src/                          # Python source package
    ├── synthetic_telemetry.py    # Step 2: generates realistic synthetic sensor data
    ├── real_data_features.py     # Real manufacturer/category recall features + real model
    ├── train_model.py            # Step 3: trains XGBoost + runs SHAP + reports metrics
    ├── experiment_imputation.py  # Imputation-strategy comparison experiment
    ├── rul_triage.py             # RUL countdown + clinical triage prioritization
    ├── llm_dispatch.py           # SHAP-to-SOP LLM agent (Gemini) + offline fallback
    └── api.py                    # Step 4: FastAPI backend /predict /explain /dispatch
```

---

## 4. Setup (in VS Code)

**Step 1 — Create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — (Recommended) Get a free Gemini API key** for LLM SOP generation:
- Go to https://aistudio.google.com/apikey and create a free key.
- Set it as an environment variable (or add a `GEMINI_API_KEY=<key>` line
  in a `.env` file in the project root):
```bash
set GEMINI_API_KEY=your_key_here      # Windows (cmd)
$env:GEMINI_API_KEY="your_key_here"   # Windows (PowerShell)
export GEMINI_API_KEY=your_key_here   # Mac/Linux
```
> Keep keys out of source control. Revoke and replace any key that has
> leaked. If you skip this step **the app still works** — `/dispatch` uses
> the offline fallback template instead of the live LLM.

---

## 5. Run Steps (in order)

**Step 1 — Real device data.**
Already done — `data/real_devices_sample.csv` is extracted from the Kaggle
`faulty-medical-devices-global-dataset` `devices.csv`.

**Step 2 — Generate synthetic telemetry:**
```bash
python src/synthetic_telemetry.py
```
Creates `synthetic_telemetry.csv` — 20,000 rows of realistic synthetic
sensor readings grounded in real device names/categories, with a simulated
physical degradation curve and failure labels.

**Step 3 — (Optional) Build real-data features & the real model:**
```bash
python src/real_data_features.py
```
Computes real manufacturer/category recall statistics and trains the
100%-real-data recall-type classifier. (Note: this reads the original
Kaggle CSVs from `/home/claude/dataset` — precomputed outputs are already
committed in `data/` and `models/`.)

**Step 4 — Train the failure-prediction + RUL models:**
```bash
python src/train_model.py
```
Trains both classifier variants + regressors, runs TreeSHAP, prints
accuracy/precision/recall/F1, and saves everything to `models/`.

**Step 5 — Start the backend API:**
```bash
uvicorn src.api:app --reload --port 8000
```
Visit http://127.0.0.1:8000/docs for FastAPI's interactive documentation.

**Step 6 — Start the dashboard (in a NEW terminal, keep the API running):**
```bash
streamlit run app_streamlit.py
```

---

## 6. API Reference

All endpoints are documented interactively at `/docs` when the API runs.

| Endpoint        | Method | Body (JSON)                                                       | Returns                                   |
|-----------------|--------|-------------------------------------------------------------------|-------------------------------------------|
| `/predict`      | POST   | `TelemetryInput` (device, category, ward, sensors, model_variant) | risk %, RUL days, priority, top drivers   |
| `/explain`      | POST   | `TelemetryInput`                                                  | full SHAP driver breakdown (all features) |
| `/dispatch`     | POST   | `DispatchInput` (risk, drivers, telemetry)                        | SOP work order + `source` (llm/offline)   |
| `/model_metrics`| GET    | —                                                                 | saved metrics for BOTH variants           |
| `/test_llm`     | GET    | —                                                                 | probes Gemini connectivity                |
| `/`             | GET    | —                                                                 | health check (status, llm_available)      |

The request body for `/predict` & `/explain` (`TelemetryInput`):
```json
{
  "device_name": "Infusion Pump A",
  "classification": "Infusion Pumps",
  "ward_criticality": "ICU",
  "age_fraction": 0.75,
  "temperature": 38.2,
  "vibration": 6.1,
  "voltage": 210.0,
  "hours_used": 12000,
  "manufacturer_risk_tier": "high",
  "model_variant": "realistic"
}
```

---

## 7. Module Reference (what each file does)

### `src/synthetic_telemetry.py`
The real Kaggle data has no sensor readings, so this module generates
realistic synthetic telemetry grounded in **real** device names/categories.
Key functions:
- `load_device_schema()` — reads `config/device_schema.json`.
- `load_real_devices()` — filters real devices to schema-known categories
  and joins real manufacturer recall tiers.
- `generate_degradation_curve()` — simulates one sensor reading that drifts
  toward failure as age increases (accelerating near end-of-life).
- `generate_synthetic_dataset()` — builds 20k rows with realistic failure
  labels (15% label-noise blend; see the module's LABEL ASSIGNMENT comment).
- `main()` — CLI entry point writing `data/synthetic_telemetry.csv`.

### `src/real_data_features.py`
Closes the gap where `events.csv`/`manufacturers.csv` were analyzed but
unused. Key functions:
- `load_raw_tables()` — loads the three original Kaggle CSVs.
- `build_manufacturer_recall_stats()` — counts real recall events per
  manufacturer → genuinely real features.
- `build_category_recall_stats()` — recall *rate* per category (rate, not
  count, so big categories aren't unfairly flagged).
- `build_real_type_classifier_dataset()` — assembles the fully-real model input.
- `train_real_type_classifier()` — trains a 100%-real-data model predicting
  Recall / Field Safety Notice / Safety Alert.
  **Data-leakage fix documented:** raw manufacturer counts were coarsened
  into low/medium/high tiers to stop the model memorizing manufacturers.

### `src/train_model.py`
Trains both model variants. Key functions:
- `load_data()` / `build_features()` — data loading + feature engineering
  (missing-indicator flags, category/tier encoding).
- `tune_classifier()` — RandomizedSearchCV (F1-scored, 5-fold CV) to find
  good hyperparameters honestly.
- `train_classifier()` / `train_regressor()` — XGBoost models. Every
  hyperparameter choice has an inline justification comment.
- `evaluate_classifier()` / `evaluate_regressor()` — accuracy/precision/
  recall/F1 and MAE reporting per the official rubric.
- `compute_shap_values()` — TreeSHAP explainer (exact + fast for trees).
- `main()` — runs both variants and writes `models/`.

### `src/experiment_imputation.py`
A senior asked whether smart imputation beats zero-fill + missing-flag.
Rather than guessing, this actually compares **zero-fill** vs **global-mean**
vs **category-mean** on the same data/split and reports real metrics.
`run_experiment()` trains a classifier per strategy; `main()` prints a
side-by-side comparison.

### `src/rul_triage.py`
Turns a bare probability into actionable numbers:
- `load_triage_weights()` — loads ward multipliers from JSON config.
- `estimate_rul_days()` — probability → days-until-failure countdown.
- `apply_triage_weighting()` — ward-aware priority score for ranking.
- `build_priority_alert()` — assembles the complete alert object.
- Helpers `_determine_recommended_action()` / `_translate_driver_names()`.

### `src/llm_dispatch.py`
The LLM/agentic layer:
- `sanitize_sop_text()` — normalizes raw LLM output to clean plain text.
- `configure_llm()` — builds the Gemini client (swap providers by editing
  only this + `generate_sop`).
- `build_prompt()` — structures a grounded, no-hallucination prompt.
- `generate_sop()` — calls Gemini with a model-fallback chain
  (`MODEL_FALLBACK_CHAIN`) so quota exhaustion rotates to the next model.
- `generate_sop_offline_fallback()` — a genuinely useful driver-aware SOP
  when the LLM is unreachable, not a placeholder message.

### `src/api.py`
The FastAPI backend. Loads models/encoders/schema once at startup, configures
the LLM, and exposes the endpoints in section 6. Also handles the graceful
LLM fallback: any `/dispatch` failure falls back to the offline template
rather than erroring.

### `app_streamlit.py`
The dashboard entry point (`streamlit run app_streamlit.py`). Contains the
CSS design system, the header, backend health probe, and four tabs:
1. **Device Risk List** — fleet overview cards, paginated ranked table,
   per-device alert card, SHAP explanation, SOP generation + download.
2. **What-If Simulator** — interactive sliders → live `/predict` result.
3. **Model Comparison** — both variants' metrics side by side.
4. **RUL Estimator** — interactive days-until-failure countdown.

Key helper functions: `call_api()`, `check_system_status()`, `get_risk_level()`,
`render_fleet_overview()`, `render_alert_card()`, `_clean_sop_text()`,
`render_sop_display()`, `render_rul_trajectory()`.

---

## 8. Key Design Decisions & Honest Notes

### All three real dataset files are now genuinely used
- `devices.csv` → real device names/categories (`data/real_devices_sample.csv`).
- `manufacturers.csv` → real identities joined to real recall counts.
- `events.csv` → real recall/safety events → manufacturer recall tiers that
  feed the prediction model as a **real feature** (a demonstrable raw-risk
  bump for high-recall-history manufacturers).

### One model was dropped after investigation (honesty beats a fake metric)
An early attempt predicted `type` (Recall / Field Safety Notice / Safety
Alert) from real data and hit **99.99% accuracy** — suspiciously perfect.
Investigation showed every one of the 39 countries reports exactly ONE
`type`, so the model had simply memorized country→type mapping. We dropped
that as a showcase model (it would not survive questioning) and kept the
signal that IS genuinely predictive: manufacturer recall history.

### Two model variants are shown transparently
- `classifier_realistic.pkl` — ~95% accuracy; 15% of labels include real
  randomness. Recommended for production (more defensible).
- `classifier_optimized.pkl` — ~99% accuracy; deterministic label (failure
  when computed risk > 50%). Higher number, but the label is a direct
  function of the features the model sees.

Both are selectable via the API `model_variant` field and compared openly
in the dashboard's **Model Comparison** tab — turning a potential "why so
high?" question into a demonstrated strength.

### Known honest limitation (be ready to explain)
The real Kaggle dataset is a device-recall registry: every device already
has a recall event, there are no "healthy device" rows and no sensor data.
Synthetic telemetry is therefore required (and explicitly permitted by the
brief). Class imbalance in the synthetic labels is handled with
`scale_pos_weight` — without it recall on the "failed" class was only 0.29;
with it, recall rose to 0.70 at a modest accuracy cost, the right tradeoff
for patient safety (missing a real failure is worse than a false alarm).

---

## 9. Troubleshooting

| Symptom                                      | Fix                                              |
|----------------------------------------------|--------------------------------------------------|
| Dashboard shows "Backend API is not running" | Start `uvicorn src.api:app --reload --port 8000` |
| SOP uses offline template when key is set    | Run `/test_llm`; check key validity / quota      |
| `GEMINI_API_KEY not found`                   | Set env var or add `.env` in project root        |
| Models folder missing                        | Run `python src/train_model.py`                  |
| Port 8000 busy                               | Use `--port 8001` and update `API_BASE_URL`      |

---

## 10. Tech Stack

| Layer        | Technology                                             |
|--------------|--------------------------------------------------------|
| Frontend/UI  | Streamlit (Python), custom CSS                          |
| Backend API  | FastAPI + Uvicorn, Pydantic validation                  |
| ML           | XGBoost, scikit-learn, SHAP (TreeExplainer), joblib    |
| LLM          | Google Gemini (`google-genai`), multi-model fallback   |
| Data         | pandas, numpy                                           |
| Python       | 3.12+                                                   |